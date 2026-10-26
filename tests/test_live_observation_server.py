"""CPU-only HTTP/Kit-thread handoff: every request obtains a new observation."""

from concurrent.futures import ThreadPoolExecutor
import threading
import time
from urllib.request import urlopen

import pytest

from simulation.isaac_sim.observation_server import ObservationServer, fixture_target_index


def test_execution_target_resolves_all_classes_and_rejects_ambiguous_instances():
    import json
    from pathlib import Path

    fixture = json.loads(Path("simulation/isaac_sim/config/three_objects.json").read_text())
    for index, entry in enumerate(fixture):
        assert fixture_target_index(fixture, entry["class"]) == index
    for unknown in [None, "unknown"]:
        with pytest.raises(ValueError, match="found 0"):
            fixture_target_index(fixture, unknown)
    with pytest.raises(ValueError, match="found 2"):
        fixture_target_index([fixture[1], fixture[1]], fixture[1]["class"])


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
