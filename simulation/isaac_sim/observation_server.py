"""Deliver current Isaac observations on the Kit thread, with no stored-frame input."""

from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from queue import Empty, Queue
from threading import Thread
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


def fixture_target_index(fixture, class_id):
    """Resolve exactly one execution target; never fall back to the first item."""
    matches = [i for i, entry in enumerate(fixture) if entry["class"] == class_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one fixture instance for class_id {class_id!r}; found {len(matches)}")
    return matches[0]


def require_current_result(observation_id, latest_id, current_q, observed_q, *, tolerance=0.01):
    """Reject another capture's result or a robot pose changed since acquisition."""
    import numpy as np

    current, observed = np.asarray(current_q, float), np.asarray(observed_q, float)
    if not latest_id or observation_id != latest_id:
        raise ValueError("Result belongs to a different observation")
    if (
        current.shape != observed.shape
        or current.shape != (6,)
        or not np.isfinite(current).all()
        or not np.isfinite(observed).all()
        or np.max(np.abs(current - observed)) > tolerance
    ):
        raise ValueError("Robot moved since observation; acquire a new frame")


class ObservationServer:
    def __init__(self, *, port=13345, result_path=None, executions=None):
        self.queue = Queue()
        queue = self.queue

        class Handler(BaseHTTPRequestHandler):
            def dispatch(self):
                route = urlsplit(self.path)
                if route.path == "/result" and (result_path is not None or executions is not None):
                    # Reading a saved result needs no Kit/PhysX calls, so polling
                    # remains responsive while the main thread executes motion.
                    try:
                        if executions is not None:
                            observation_id = parse_qs(route.query).get("observation_id", [None])[0]
                            content = json.dumps(executions.result(observation_id)).encode()
                        else:
                            path = Path(result_path)
                            content = path.read_bytes() if path.exists() else b'{"finished":false}'
                        json.loads(content)
                    except (OSError, ValueError):
                        content = b'{"finished":false}'
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                    return
                future = Future()
                size = int(self.headers.get("Content-Length", 0))
                if not 0 <= size <= 32 * 1024 * 1024:
                    self.send_error(413)
                    return
                body = self.rfile.read(size)
                queue.put((route.path, body, future))
                try:
                    content = future.result(timeout=30.0)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                except Exception as error:
                    future.cancel()
                    content = json.dumps(dict(error=str(error))).encode()
                    self.send_response(503)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

            do_GET = dispatch
            do_POST = dispatch

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def poll(self, callback):
        try:
            path, body, future = self.queue.get_nowait()
        except Empty:
            return
        if future.cancelled():
            return
        try:
            future.set_result(callback(path, body))
        except Exception as error:
            future.set_exception(error)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        while not self.queue.empty():
            _, _, future = self.queue.get_nowait()
            if not future.done():
                future.set_exception(RuntimeError("Isaac observation server closed"))
