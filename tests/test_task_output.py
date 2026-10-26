"""Output reservations preserve old attempts and support repeated CLI use."""
import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from tiptop.scripts import run_known_pick_place as runner


def test_nested_number_increments_and_preserves_existing_records(tmp_path):
    requested = tmp_path / 'sam3-test-02' / 'task'
    requested.mkdir(parents=True)
    old = requested / 'task-result.json'
    old.write_text('{"status":"failed"}')
    first = runner.reserve_output_directory(requested)
    second = runner.reserve_output_directory(requested)
    assert first == tmp_path / 'sam3-test-03/task'
    assert second == tmp_path / 'sam3-test-04/task'
    assert old.read_text() == '{"status":"failed"}'


def test_even_interrupted_gemini_attempt_is_preserved(tmp_path):
    requested = tmp_path / 'attempt'
    requested.mkdir()
    marker = requested / 'gemini-call-consumed.json'
    marker.write_text('{"maximum_calls":1}')
    selected = runner.reserve_output_directory(requested)
    assert selected == tmp_path / 'attempt-001'
    assert marker.read_text() == '{"maximum_calls":1}'


def test_concurrent_commands_reserve_distinct_outputs(tmp_path):
    requested = tmp_path / 'task-001'
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(runner.reserve_output_directory, [requested] * 8))
    assert len(set(paths)) == 8
    assert all((path / '.task-reserved').is_file() for path in paths)


def test_result_and_console_use_actual_incremented_path(monkeypatch, tmp_path, capsys):
    requested = tmp_path / 'run-009/task'
    requested.mkdir(parents=True)
    (requested / 'observation.npz').write_bytes(b'previous observation')

    async def fake(args):
        assert args.output == tmp_path / 'run-010/task'
        return dict(status='planned', success=None)

    monkeypatch.setattr(runner, 'run', fake)
    args = SimpleNamespace(output=requested, task='test', perception_mode='sam3_reference')
    result = asyncio.run(runner.run_task(args))
    assert result['output'] == str(args.output)
    assert json.loads((args.output / 'task-result.json').read_text())['output'] == str(args.output)
    assert f'TASK_OUTPUT {args.output}' in capsys.readouterr().out
    assert (requested / 'observation.npz').read_bytes() == b'previous observation'


def test_output_argument_is_optional(monkeypatch):
    called = []

    async def fake(args):
        called.append(str(args.output))

    monkeypatch.setattr(runner, 'run_task', fake)
    monkeypatch.setattr(sys, 'argv', ['run', '--perception-mode', 'sam3_reference'])
    runner.main()
    assert called == ['tiptop_outputs/m1013/tasks/task-001']
