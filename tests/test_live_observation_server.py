"""CPU-only HTTP/Kit-thread handoff: every request obtains a new observation."""

from concurrent.futures import ThreadPoolExecutor
import threading
import time
from urllib.request import urlopen

from simulation.isaac_sim.observation_server import ObservationServer


def test_each_http_request_captures_on_the_polling_thread():
    server = ObservationServer(port=0)
    polling_thread = threading.get_ident()
    calls = []

    def capture(path, body):
        assert path == "/observe"
        assert threading.get_ident() == polling_thread
        calls.append(len(calls) + 1)
        return f"current-frame-{calls[-1]}".encode()

    def request():
        with urlopen(f"http://127.0.0.1:{server.server.server_port}/observe", timeout=3.0) as response:
            return response.read()

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            for index in [1, 2]:
                pending = pool.submit(request)
                deadline = time.monotonic() + 3.0
                while not pending.done() and time.monotonic() < deadline:
                    server.poll(capture)
                    time.sleep(0.005)
                assert pending.result(timeout=1.0) == f"current-frame-{index}".encode()
        assert calls == [1, 2]
    finally:
        server.close()
