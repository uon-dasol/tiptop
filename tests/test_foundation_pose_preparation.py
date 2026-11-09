"""Cache/HTTP checks with recording builders; no model weights or devices."""

import importlib.util
import io
import json
import threading
from contextlib import contextmanager
from http.server import HTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

FP = Path(__file__).resolve().parents[1] / "FoundationPose"


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, FP / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


assets = module("fp_assets_test", "tiptop_server/assets.py")
server = module("fp_server_test", "tiptop_server/server.py")


def item(root, name="one"):
    path = root / name / "edited" / (name+".obj")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("mesh")
    return dict(mesh_path=str(path), mesh_scale=.001, class_id=name)


def test_preparation_and_requests_share_versioned_estimators(tmp_path):
    calls = []
    store = assets.EstimatorStore(tmp_path, lambda **kw: calls.append(kw) or object())
    first, second = item(tmp_path), item(tmp_path, "two")
    report = store.prepare([first, second, first])
    assert report["success"] and report["inference_performed"] is False
    assert len(calls) == 2 and len(report["objects"]) == 2
    a, hit = store.get(first)
    assert hit
    assert all(row["cache_hit"] for row in store.prepare([first, second])["objects"])
    texture = Path(first["mesh_path"]).parent.parent / "texture.png"
    texture.write_bytes(b"new texture")
    b, hit = store.get(first)
    assert not hit and a is not b and len(calls) == 3
    Path(first["mesh_path"]).with_suffix(".mtl").write_text("new material")
    assert store.get(first)[1] is False
    Path(first["mesh_path"]).write_text("new geometry")
    assert store.get(first)[1] is False


def test_validation_finishes_before_any_build_and_partial_failure_is_not_ready(tmp_path):
    first, second = item(tmp_path), item(tmp_path, "two")
    calls = []

    def build(**kw):
        calls.append(kw["class_id"])
        if kw["class_id"] == "two":
            raise RuntimeError("invalid texture")
        return object()

    store = assets.EstimatorStore(tmp_path, build)
    with pytest.raises(ValueError, match="scale"):
        store.prepare([first, dict(second, mesh_scale=float("nan"))])
    assert not calls
    report = store.prepare([first, second])
    assert report["success"] is False and report["objects"][-1]["error"] == "invalid texture"
    assert store.loaded_classes == ["one"]
    with pytest.raises(ValueError, match="Conflicting"):
        store.prepare([first, dict(first, mesh_scale=.002)])


def test_changed_during_build_or_outside_root_never_enters_cache(tmp_path):
    root = tmp_path / "allowed"
    original = item(root)

    def changing(**kw):
        Path(kw["mesh_path"]).write_text("changed during build")
        return object()

    store = assets.EstimatorStore(root, changing)
    with pytest.raises(RuntimeError, match="changed while preparing"):
        store.get(original)
    assert not store.entries
    outside = item(tmp_path / "outside")
    with pytest.raises(ValueError, match="asset root"):
        store.get(outside)


def test_http_prepare_runs_no_estimation_and_each_estimate_is_fresh(tmp_path):
    selected, calls = item(tmp_path), []
    store = assets.EstimatorStore(tmp_path, lambda **kw: object())

    def estimate(body):
        calls.append(body)
        assert store.get(selected)[1] is True
        return dict(frame=body.decode(), count=len(calls))

    http = HTTPServer(("127.0.0.1", 0), server.make_handler(store, estimate))
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{http.server_port}"

    def post(path, data):
        with urlopen(Request(base+path, data=data, method="POST"), timeout=5) as response:
            return json.load(response)

    try:
        assert post("/prepare", json.dumps(dict(objects=[selected])).encode())["success"]
        assert not calls
        with urlopen(base+"/health", timeout=5) as response:
            health = json.load(response)
        assert health["preparation_supported"] and health["loaded_classes"] == ["one"]
        assert post("/estimate", b"new-frame-1")["count"] == 1
        assert post("/estimate", b"new-frame-2")["count"] == 2
        with pytest.raises(HTTPError):
            post("/prepare", b'{"objects":[]}')
        assert calls == [b"new-frame-1", b"new-frame-2"]
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=5)


def test_startup_preload_shares_weights_and_only_announces_ready_after_success(tmp_path, monkeypatch, capsys):
    import sys

    selected = [item(tmp_path), item(tmp_path, "two")]
    built, model_loads, servers, frames = [], [], [], []

    @contextmanager
    def mesh(*args, **kwargs):
        yield SimpleNamespace(vertices=np.zeros((3, 3)), vertex_normals=np.ones((3, 3)))

    def weights(kind):
        model_loads.append(kind)
        return object()

    def estimator(**kwargs):
        built.append(kwargs)
        def register(**inputs):
            frames.append(int(inputs["rgb"][0, 0, 0]))
            pose = np.eye(4)
            pose[0, 3] = frames[-1]
            return pose
        return SimpleNamespace(diameter=np.float64(.1), scores=np.array([.9]), register=register)

    monkeypatch.setitem(sys.modules, "nvdiffrast", SimpleNamespace(torch=SimpleNamespace(RasterizeCudaContext=lambda: object())))
    monkeypatch.setitem(sys.modules, "nvdiffrast.torch", sys.modules["nvdiffrast"].torch)
    monkeypatch.setitem(sys.modules, "custom_datareader", SimpleNamespace(load_mesh_readonly=mesh))
    monkeypatch.setitem(sys.modules, "estimater", SimpleNamespace(
        FoundationPose=estimator, ScorePredictor=lambda: weights("score"), PoseRefinePredictor=lambda: weights("refine")))
    monkeypatch.setitem(sys.modules, "tiptop_server.assets", assets)

    class HTTP:
        def __init__(self, address, handler):
            servers.append(handler)
            self.handler = handler

        def serve_forever(self):
            # Exercise the actual estimate-body closure with two fresh packets.
            http = HTTPServer(("127.0.0.1", 0), self.handler)
            thread = threading.Thread(target=http.serve_forever, daemon=True)
            thread.start()
            try:
                for pixel in (1, 2):
                    packet = io.BytesIO()
                    np.savez_compressed(packet, rgb=np.full((4, 4, 3), pixel, np.uint8), depth=np.ones((4, 4)),
                        mask=np.ones((4, 4), bool), intrinsics=np.eye(3), **selected[0])
                    with urlopen(Request(f"http://127.0.0.1:{http.server_port}/estimate", data=packet.getvalue()),
                                 timeout=5) as response:
                        result = json.load(response)
                    assert result["asset_cache_hit"] is True and result["camera_from_cad"][0][3] == pixel
                    assert "asset_load_s" in result and "register_s" in result
            finally:
                http.shutdown()
                http.server_close()
                thread.join(timeout=5)
            raise KeyboardInterrupt

        def server_close(self):
            pass

    monkeypatch.setattr(server, "HTTPServer", HTTP)
    server.serve(tmp_path, 0, preload_objects=selected)
    assert model_loads == ["score", "refine"] and len(built) == 2
    assert built[0]["scorer"] is built[1]["scorer"] and built[0]["refiner"] is built[1]["refiner"]
    assert built[0]["glctx"] is built[1]["glctx"]
    assert frames == [1, 2]  # Neither startup preparation nor cache hits reuse an old pose.
    output = capsys.readouterr().out
    assert "FOUNDATIONPOSE_READY" in output
    assert "\033[33mFOUNDATIONPOSE_PREPARE 1/2 " in output
    assert "\033[33mFOUNDATIONPOSE_PREPARE 2/2 " in output
    assert "\033[1;32mFOUNDATIONPOSE_PRELOAD_COMPLETE 2/2 물체 로딩 완료" in output
    assert output.index("FOUNDATIONPOSE_READY") < output.index("FOUNDATIONPOSE_PRELOAD_COMPLETE")

    def fail(**kwargs):
        raise ValueError("bad CAD")

    sys.modules["estimater"].FoundationPose = fail
    with pytest.raises(RuntimeError, match="preload failed"):
        server.serve(tmp_path, 0, preload_objects=selected)
    output = capsys.readouterr().out
    assert len(servers) == 1 and "FOUNDATIONPOSE_READY" not in output
    assert "FOUNDATIONPOSE_PRELOAD_COMPLETE" not in output
    assert "\033[31mFOUNDATIONPOSE_PREPARE 1/2 " in output
