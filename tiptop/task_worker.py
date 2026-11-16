"""One-at-a-time local task RPC. Only this Unix socket's owner can submit tasks."""

import asyncio
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import logging
import os
from pathlib import Path
import socket
import time

from tiptop.console import task_result_text, timing_line
from tiptop.task_logging import LOG_OWNER_ENV


def default_socket():
    return Path(f'/tmp/tiptop-{os.getuid()}/known-task.sock')


def forward_if_requested(argv):
    """Runs before NumPy/CUDA imports, retaining the ordinary --task CLI."""
    if any(arg in {'--help', '-h', '--check', '--prepare-only', '--interactive'} for arg in argv):
        return None
    path = os.environ.get('TIPTOP_TASK_WORKER_SOCKET')
    forwarded = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == '--worker-socket':
            index += 1
            if index == len(argv):
                raise ValueError('--worker-socket requires a path')
            path = argv[index]
        elif arg.startswith('--worker-socket='):
            path = arg.split('=', 1)[1]
        else:
            forwarded.append(arg)
        index += 1
    if not path:
        return None
    started = time.perf_counter()
    # No fallback/retry after submission: a dropped connection must not repeat motion.
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(os.path.expanduser(path))
        client.sendall((json.dumps(dict(argv=forwarded, cwd=os.getcwd(),
                                       terminal_log_external=os.environ.get(LOG_OWNER_ENV) == "launcher"))+'\n').encode())
        with client.makefile('rb') as stream:
            for line in stream:
                event = json.loads(line)
                if event['event'] == 'log':
                    print(event['text'], end='', flush=True)
                elif event['event'] == 'result':
                    result = event['result']
                    print(task_result_text(result), flush=True)
                    print(timing_line(f'[Worker Wall] 요청부터 결과 수신까지 {time.perf_counter()-started:.2f}s',
                                      primary=True), flush=True)
                    return 1 if result.get('success') is False else 0
    raise RuntimeError('Task worker disconnected; the task was not resubmitted')


class TaskWorker:
    def __init__(self, parse_args, run_task, *, before_task=None):
        self.parse_args, self.run_task, self.before_task = parse_args, run_task, before_task
        self.busy = False
        self.handlers = set()

    async def handle(self, reader, writer):
        handler = asyncio.current_task()
        self.handlers.add(handler)
        job = disconnected = None
        acquired = False
        loop = asyncio.get_running_loop()

        def send(event):
            if not writer.is_closing():
                writer.write((json.dumps(event, ensure_ascii=False)+'\n').encode())

        class Output(io.TextIOBase):
            def write(self, text):
                if text:
                    loop.call_soon_threadsafe(send, dict(event='log', text=text))
                return len(text)

            def flush(self):
                pass

        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=5)
            request = json.loads(raw)
            if self.busy:
                raise ValueError('Task worker is busy; no task was queued')
            if request.get('cwd') != os.getcwd():
                raise ValueError('Run the task CLI from the same project directory as the worker')
            argv = request['argv']
            if not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
                raise ValueError('Expected CLI arguments')
            # Parse fresh options and reload config before every job. Never
            # inherit a preceding task's --execute, thresholds, URLs or ROIs.
            self.busy = acquired = True
            request_started = time.perf_counter()
            if self.before_task is not None:
                self.before_task()
            output = Output()
            with redirect_stdout(output), redirect_stderr(output):
                args = self.parse_args(argv)
                if args.check or args.interactive or args.prepare_only or args.worker_socket:
                    raise ValueError('Worker accepts single --task commands only')
                if args.perception_mode != 'sam3_reference':
                    raise ValueError('Worker currently requires --perception-mode sam3_reference')
                args._cli_started_at = request_started
                args._worker_request = True
                args._terminal_log_external = request.get("terminal_log_external") is True
                streams = [(h, h.stream) for h in logging.getLogger().handlers
                           if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
                for h, _ in streams:
                    h.setStream(output)
                try:
                    job = asyncio.create_task(self.run_task(args))
                    disconnected = asyncio.create_task(reader.read(1))
                    done, _ = await asyncio.wait([job, disconnected], return_when=asyncio.FIRST_COMPLETED)
                    if disconnected in done:
                        job.cancel()
                    result = await job
                finally:
                    if job is not None and not job.done():
                        job.cancel()
                        await asyncio.gather(job, return_exceptions=True)
                    await asyncio.gather(*getattr(args, '_background', []), return_exceptions=True)
                    for h, stream in streams:
                        h.setStream(stream)
            # Run queued log writes before the final response.
            await asyncio.sleep(0)
            send(dict(event='result', result=result))
        except (Exception, SystemExit) as error:
            await asyncio.sleep(0)
            send(dict(event='result', result=dict(success=False, status='worker_rejected', error=str(error))))
        finally:
            if disconnected is not None:
                disconnected.cancel()
                await asyncio.gather(disconnected, return_exceptions=True)
            if acquired:
                self.busy = False
            try:
                await writer.drain()
            except (ConnectionError, BrokenPipeError):
                pass
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, BrokenPipeError):
                pass
            finally:
                self.handlers.discard(handler)

    async def close(self):
        jobs = list(self.handlers)
        for task in jobs:
            task.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)


async def serve_worker(path, worker, *, ready=None):
    import fcntl
    import stat

    path = Path(path).expanduser().absolute()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent = path.parent.stat()
    if parent.st_uid != os.getuid() or stat.S_IMODE(parent.st_mode) & 0o077:
        raise ValueError('Worker socket requires a private directory owned by the current user (mode 700)')
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists() or path.is_symlink():
            old = path.lstat()
            if old.st_uid != os.getuid() or not stat.S_ISSOCK(old.st_mode):
                raise ValueError('Refusing to replace a non-socket or another user\'s socket')
            path.unlink()
        server = None
        try:
            server = await asyncio.start_unix_server(worker.handle, path=str(path), limit=65536)
            path.chmod(0o600)
            started = time.perf_counter()
            preparation = None
            if ready is not None:
                preparation = ready()
            summary = ''
            if isinstance(preparation, dict) and 'objects' in preparation:
                count = len(preparation['objects'])
                summary = f"{count}/{count} 물체 준비 완료 ({time.perf_counter()-started:.2f}s) | "
            print(f'\033[1;32mKNOWN_TASK_WORKER_READY {summary}작업 서버 준비 완료 | socket={path}\033[0m', flush=True)
            async with server:
                await server.serve_forever()
        finally:
            if server is not None:
                server.close()
                await server.wait_closed()
            await worker.close()
            path.unlink(missing_ok=True)
