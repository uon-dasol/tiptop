import asyncio
import json
import os
import subprocess
import sys

import pytest

from tiptop.task_worker import TaskWorker, serve_worker, forward_if_requested


def parse(argv):
    from tiptop.scripts.run_known_pick_place import parse_args
    return parse_args(argv)


async def connect(path, argv, *, cwd=None):
    reader, writer = await asyncio.open_unix_connection(str(path))
    writer.write((json.dumps(dict(argv=argv, cwd=cwd or os.getcwd()))+'\n').encode())
    await writer.drain()
    return reader, writer


async def result(reader):
    logs = []
    while True:
        event = json.loads(await asyncio.wait_for(reader.readline(), timeout=5))
        if event['event'] == 'result':
            return event['result'], ''.join(logs)
        logs.append(event['text'])


def test_arbitrary_tasks_share_worker_but_each_request_gets_fresh_options_and_logs(tmp_path):
    async def check():
        calls, prepared = [], []
        async def run(args):
            calls.append((args.task, args.execute, args.max_approach_tilt_deg))
            print('TASK_OUTPUT '+args.task)
            return dict(success=True, task=args.task, process_id=os.getpid())
        worker = TaskWorker(parse, run, before_task=lambda: prepared.append(True))
        path = tmp_path/'worker.sock'
        async with await asyncio.start_unix_server(worker.handle, path=str(path)):
            for name, extra in [('C', ['--max-approach-tilt-deg','20']), ('A', []), ('C', [])]:
                reader, writer = await connect(path, ['--task',name,'--perception-mode','sam3_reference',*extra])
                data, logs = await result(reader)
                assert data['process_id'] == os.getpid() and data['task'] == name
                assert 'TASK_OUTPUT '+name in logs
                writer.close(); await writer.wait_closed()
            await worker.close()
        assert calls == [('C',False,20),('A',False,3),('C',False,3)]
        assert len(prepared) == 3
    asyncio.run(check())


def test_busy_request_is_rejected_without_queueing_or_cancelling_active_task(tmp_path):
    async def check():
        started, release = asyncio.Event(), asyncio.Event()
        calls = []
        async def run(args):
            calls.append(args.task); started.set()
            await release.wait()
            return dict(success=True)
        worker = TaskWorker(parse, run)
        path = tmp_path/'worker.sock'
        async with await asyncio.start_unix_server(worker.handle, path=str(path)):
            first, one = await connect(path, ['--task','A','--perception-mode','sam3_reference'])
            await asyncio.wait_for(started.wait(), timeout=5)
            second, two = await connect(path, ['--task','B','--perception-mode','sam3_reference'])
            rejected, _ = await result(second)
            assert rejected['success'] is False and 'busy' in rejected['error'] and worker.busy
            assert calls == ['A']
            two.close(); await two.wait_closed()
            release.set(); assert (await result(first))[0]['success']
            one.close(); await one.wait_closed()
            await worker.close()
        assert not worker.busy
    asyncio.run(check())


def test_disconnect_waits_for_cancel_cleanup_before_accepting_another_task(tmp_path):
    async def check():
        started, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def run(args):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
                await release.wait()
        worker = TaskWorker(parse, run)
        path = tmp_path/'worker.sock'
        async with await asyncio.start_unix_server(worker.handle, path=str(path)):
            _, writer = await connect(path, ['--task','A','--perception-mode','sam3_reference'])
            await asyncio.wait_for(started.wait(), timeout=5)
            writer.close(); await writer.wait_closed()
            await asyncio.wait_for(cancelled.wait(), timeout=5)
            assert worker.busy
            release.set()
            for _ in range(100):
                if not worker.busy:
                    break
                await asyncio.sleep(.001)
            assert not worker.busy
            await worker.close()
    asyncio.run(check())


@pytest.mark.parametrize('argv,cwd', [(['--check'],None), (['--interactive'],None),
    (['--max-approach-tilt-deg','91'],None), (['--task','A'],'/different/project')])
def test_invalid_requests_never_start_a_task(tmp_path, argv, cwd):
    async def check():
        calls=[]
        async def run(args):
            calls.append(args)
        worker=TaskWorker(parse,run)
        path=tmp_path/'worker.sock'
        async with await asyncio.start_unix_server(worker.handle,path=str(path)):
            reader, writer=await connect(path,[*argv,'--perception-mode','sam3_reference'],cwd=cwd)
            assert (await result(reader))[0]['success'] is False
            writer.close(); await writer.wait_closed()
            await worker.close()
        assert not calls
    asyncio.run(check())


def test_thin_cli_skips_heavy_imports_and_does_not_retry_a_rejected_request(tmp_path):
    async def check():
        calls=[]
        async def run(args):
            calls.append(args.task)
            return dict(success=False,status='intentional failure',task=args.task)
        path=tmp_path/'worker.sock'
        worker=TaskWorker(parse,run)
        async with await asyncio.start_unix_server(worker.handle,path=str(path)):
            proc=await asyncio.to_thread(subprocess.run,[sys.executable,'-X','importtime','-m',
                'tiptop.scripts.run_known_pick_place','--worker-socket',str(path),'--task','B',
                '--perception-mode','sam3_reference'],capture_output=True,text=True,timeout=10)
            assert proc.returncode == 1 and 'intentional failure' in proc.stdout
            assert '\033[1;31m[Result] 작업 실패' in proc.stdout
            assert '\033[1;36m[Worker Wall]' in proc.stdout
            assert '[Worker Wall]' in proc.stdout and 'torch' not in proc.stderr and 'numpy' not in proc.stderr
            await worker.close()
        assert calls == ['B']
    asyncio.run(check())


def test_private_socket_ready_and_shutdown_cleanup(tmp_path, capsys):
    async def check():
        private=tmp_path/'private'; private.mkdir(mode=0o700)
        path=private/'task.sock'
        ready=asyncio.Event()
        async def run(args):
            return dict(success=True)
        worker=TaskWorker(parse,run)
        def prepared():
            ready.set()
            return dict(objects=[{},{}])
        server=asyncio.create_task(serve_worker(path,worker,ready=prepared))
        await asyncio.wait_for(ready.wait(),timeout=5)
        assert path.stat().st_mode & 0o777 == 0o600
        reader,writer=await connect(path,['--task','A','--perception-mode','sam3_reference'])
        assert (await result(reader))[0]['success']
        writer.close(); await writer.wait_closed()
        server.cancel(); await asyncio.gather(server,return_exceptions=True)
        assert not path.exists()
    asyncio.run(check())
    assert '\033[1;32mKNOWN_TASK_WORKER_READY 2/2 물체 준비 완료' in capsys.readouterr().out


def test_preparation_failure_never_announces_worker_ready(tmp_path, capsys):
    async def check():
        private=tmp_path/'private'; private.mkdir(mode=0o700)
        path=private/'task.sock'
        def fail():
            raise ValueError('bad asset')
        worker=TaskWorker(parse,None)
        with pytest.raises(ValueError,match='bad asset'):
            await serve_worker(path,worker,ready=fail)
        assert not path.exists()
    asyncio.run(check())
    assert 'KNOWN_TASK_WORKER_READY' not in capsys.readouterr().out


def test_readiness_and_preparation_still_run_locally_with_worker_environment(monkeypatch):
    monkeypatch.setenv('TIPTOP_TASK_WORKER_SOCKET','/nonexistent-worker.sock')
    for flag in ('--check','--prepare-only','--interactive','--help'):
        assert forward_if_requested([flag]) is None


def test_disconnect_reaches_existing_physical_stop_and_drains_executor(tmp_path, monkeypatch):
    import threading
    from tiptop.doosan import client, execution
    from tiptop.scripts import run_known_pick_place as runner

    started, stopped, drained = threading.Event(), threading.Event(), threading.Event()
    class FakeClient:
        def __init__(self, url):
            pass
        def stop(self):
            stopped.set()
    def execute(*args, cancelled, **kwargs):
        started.set()
        assert cancelled.wait(5)
        drained.set()
        return dict(success=False)
    monkeypatch.setattr(client,'DoosanUONClient',FakeClient)
    monkeypatch.setattr(execution,'execute_measured_plan',execute)
    async def check():
        async def run(args):
            return await runner.execute_physical({},'fake-gateway')
        worker=TaskWorker(parse,run)
        path=tmp_path/'worker.sock'
        async with await asyncio.start_unix_server(worker.handle,path=str(path)):
            _,writer=await connect(path,['--task','A','--perception-mode','sam3_reference'])
            assert await asyncio.to_thread(started.wait,5)
            writer.close(); await writer.wait_closed()
            for _ in range(100):
                if not worker.busy:
                    break
                await asyncio.sleep(.01)
            assert stopped.is_set() and drained.is_set() and not worker.busy
            await worker.close()
    asyncio.run(check())
