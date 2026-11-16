"""Keep known-task imports/assets/planners alive for arbitrary --task requests."""

import argparse
import asyncio
import logging

from tiptop.task_worker import TaskWorker, default_socket, serve_worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket', default=str(default_socket()))
    parser.add_argument('--preload-objects', nargs='+', metavar='CLASS_ID')
    args = parser.parse_args()
    print('\033[33mKNOWN_TASK_WORKER_STARTING 공통 모듈 로딩 중\033[0m', flush=True)
    from tiptop.scripts import run_known_pick_place as runner
    from tiptop import tiptop_run  # Warm shared imports; no device or gateway requests.
    from tiptop.config import get_tiptop_cfg_path, set_tiptop_cfg_from_file

    console = logging.StreamHandler()
    console.setFormatter(runner._TimingFormatter('%(levelname)s:%(name)s:%(message)s'))
    console.addFilter(runner._TiptopProgressFilter())
    logging.basicConfig(level=logging.INFO, handlers=[console])
    runner.quiet_noisy_loggers()
    config_path = get_tiptop_cfg_path()

    def prepare():
        if args.preload_objects:
            from tiptop.object_catalog import sequence_objects
            from tiptop.perception.sequence import prepare_local_assets
            return prepare_local_assets(sequence_objects(args.preload_objects))

    worker = TaskWorker(runner.parse_args, runner.run_task,
                        before_task=lambda: set_tiptop_cfg_from_file(config_path))
    try:
        asyncio.run(serve_worker(args.socket, worker, ready=prepare))
    except KeyboardInterrupt:
        pass
    except Exception as error:
        parser.exit(2, f'\033[31mKNOWN_TASK_WORKER_FAILED {error}\033[0m\n')


if __name__ == '__main__':
    main()
