"""Monotonic task timings; phases are sequential, details may overlap them."""

import time
from contextlib import contextmanager


class TaskTimings:
    def __init__(self):
        self.started = time.perf_counter()
        self._name = None
        self._phase_started = self.started
        self.phases = []
        self.details = []

    def phase(self, name):
        self.finish()
        self._name = name
        self._phase_started = time.perf_counter()

    def finish(self, *, status="completed"):
        if self._name is not None:
            ended = time.perf_counter()
            self.phases.append(dict(
                name=self._name, start_s=self._phase_started - self.started,
                duration_s=ended - self._phase_started, status=status,
            ))
            self._name = None

    @contextmanager
    def measure(self, name):
        started = time.perf_counter()
        status = "completed"
        try:
            yield
        except BaseException:
            status = "failed"
            raise
        finally:
            self.details.append(dict(
                name=name, start_s=started - self.started,
                duration_s=time.perf_counter() - started, status=status,
            ))

    def snapshot(self):
        return dict(
            task_wall_seconds=time.perf_counter() - self.started,
            phases=list(self.phases), details=list(self.details),
            scope="run_task entry through background drain and task-result write; excludes timings.json write",
            details_overlap_phases=True,
        )
