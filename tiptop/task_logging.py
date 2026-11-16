"""Persist task terminal output while retaining the existing console streams."""

import logging
import re
import sys
import threading
from contextlib import contextmanager, redirect_stderr, redirect_stdout


LOG_OWNER_ENV = "TIPTOP_TERMINAL_LOG_OWNER"
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class TerminalLog:
    def __init__(self, path):
        self.file = path.open("a", encoding="utf-8")
        self.lock = threading.RLock()
        self.failed = False
        self.errors = sys.stderr

    def write(self, text):
        with self.lock:
            if self.failed:
                return
            try:
                self.file.write(_ANSI.sub("", text))
                self.file.flush()
            except OSError as error:
                self.failed = True
                self.errors.write(f"Terminal log saving failed: {error}\n")

    def close(self):
        with self.lock:
            try:
                self.file.close()
            except OSError as error:
                self.errors.write(f"Terminal log closing failed: {error}\n")


class _Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        result = self.stream.write(text)
        self.log.write(text)
        return result

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextmanager
def capture_terminal_log(path):
    """Capture Python stdout/stderr and console logging for a standalone/worker task."""
    try:
        log = TerminalLog(path)
    except OSError as error:
        sys.stderr.write(f"Terminal log unavailable: {error}\n")
        yield
        return
    stdout, stderr = _Tee(sys.stdout, log), _Tee(sys.stderr, log)
    consoles = (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__)
    loggers = [logging.getLogger(), *logging.Logger.manager.loggerDict.values()]
    handlers = {handler for logger in loggers if isinstance(logger, logging.Logger)
                for handler in logger.handlers if isinstance(handler, logging.StreamHandler)
                and not isinstance(handler, (logging.FileHandler, type(logging.lastResort)))
                and any(handler.stream is stream for stream in consoles)}
    streams = [(handler, handler.stream) for handler in handlers]
    try:
        for handler, stream in streams:
            handler.setStream(_Tee(stream, log))
        with redirect_stdout(stdout), redirect_stderr(stderr):
            yield
    finally:
        for handler, stream in streams:
            handler.setStream(stream)
        log.close()
