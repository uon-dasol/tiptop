"""Serve FoundationPose (uonrobotics fork, dasol branch) in its own Python environment.

Start it with `bash scripts/run_foundation_pose_server.sh` (foundationpose conda Python) or launch this script
with that Python. `--check` only reports missing prerequisites. External sources and CAD assets are read only;
model/debug scratch stays under /tmp.
"""

import argparse
import importlib.machinery
import importlib.util
import io
import json
import os
import socket
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# README_CUSTOM.md clones FoundationPose into the project root; an existing sibling data_gen checkout also works
FOUNDATIONPOSE_CANDIDATES = (ROOT / "FoundationPose", ROOT.parent / "data_gen" / "FoundationPose")
DEFAULT_PORT = 13346
CATALOG = ROOT / "tiptop/config/known_objects.json"
ENV_HINT = "see README_CUSTOM.md, '6. FoundationPose 설치'"
# Run directories loaded by the upstream predict_score.py and predict_pose_refine.py
WEIGHT_RUNS = ("2024-01-11-20-02-45", "2023-10-28-18-33-37")
SOURCE_FILES = ("estimater.py", "custom_datareader.py", "Utils.py")
# Imported by this script itself, and by the FoundationPose sources
SERVER_MODULES = ("numpy", "torch", "nvdiffrast")
SOURCE_MODULES = ("pytorch3d", "trimesh", "cv2", "scipy", "omegaconf", "open3d", "imageio", "transformations")

Finding = tuple[str, str]  # (level, message) with level "ok", "warn" or "error"


def default_foundationpose_root() -> Path:
    return next((path for path in FOUNDATIONPOSE_CANDIDATES if path.is_dir()), FOUNDATIONPOSE_CANDIDATES[0])


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # The same option HTTPServer binds with
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def cad_findings(asset_root: Path) -> list[Finding]:
    """Check the registered CAD assets: OBJ, MTL and the textures the server resolves by file name."""
    if not CATALOG.is_file():
        return [("warn", f"Registered-object catalog not found: {CATALOG}; CAD files were not checked")]
    findings = []
    for entry in json.loads(CATALOG.read_text())["objects"]:
        name, obj = entry["class_id"], (ROOT / entry["cad"]).resolve()
        problems = []
        if not obj.is_file():
            problems.append(f"CAD for {name} not found: {obj} (expected under data/<object>/3d_model/edited/)")
        else:
            if not obj.is_relative_to(asset_root.resolve()):
                problems.append(f"CAD for {name} is outside the asset root {asset_root}; the server will reject it")
            mtl = obj.with_suffix(".mtl")
            if not mtl.is_file():
                problems.append(f"MTL for {name} not found next to the OBJ: {mtl}")
            else:
                for line in mtl.read_text(errors="ignore").splitlines():
                    parts = line.split(maxsplit=1)
                    if parts[0:1] == ["map_Kd"] and len(parts) == 2:
                        texture = Path(parts[1].strip()).name
                        if not any((root / texture).is_file() for root in (obj.parent, obj.parent.parent)):
                            problems.append(f"Texture {texture} of {name} not found in {obj.parent} or its parent")
        findings += [("warn", problem) for problem in problems] or [("ok", f"CAD for {name}: {obj.name}")]
    return findings


def preflight(foundationpose_root: Path, asset_root: Path, port: int, port_busy_level: str) -> list[Finding]:
    """Report what is missing before serving. Level "error" means serving cannot work; imports no GPU library."""
    root = foundationpose_root
    if not root.is_dir():
        return [
            (
                "error",
                f"FoundationPose source not found: {root}. Clone it into the project root "
                f"(git clone -b dasol git@github.com:uonrobotics/FoundationPose.git; {ENV_HINT}), "
                "or pass --foundationpose-root / set FOUNDATIONPOSE_ROOT",
            )
        ]
    findings = [("ok", f"FoundationPose source: {root}")]
    for name in SOURCE_FILES:
        if not (root / name).is_file():
            hint = "; only the 'dasol' branch has it" if name == "custom_datareader.py" else ""
            findings.append(("error", f"FoundationPose source file missing: {root / name}{hint}"))
    missing_weights = [
        root / "weights" / run / name
        for run in WEIGHT_RUNS
        for name in ("model_best.pth", "config.yml")
        if not (root / "weights" / run / name).is_file()
    ]
    findings += [
        (
            "error",
            f"FoundationPose weights missing: {path}; download the checkpoints ({ENV_HINT})",
        )
        for path in missing_weights
    ] or [("ok", f"FoundationPose weights: {', '.join(WEIGHT_RUNS)}")]
    build = root / "mycpp" / "build"
    if any((build / f"mycpp{suffix}").is_file() for suffix in importlib.machinery.EXTENSION_SUFFIXES):
        findings.append(("ok", f"mycpp extension built for Python {sys.version.split()[0]}"))
    elif importlib.util.find_spec("mycpp") is None:
        findings.append(
            (
                "error",
                f"mycpp extension not built for Python {sys.version.split()[0]} in {build}; "
                f"run `bash build_all_conda.sh` in {root} with the foundationpose env active ({ENV_HINT})",
            )
        )
    for name in SERVER_MODULES:
        if importlib.util.find_spec(name) is None:
            findings.append(("error", f"Python module '{name}' not found for {sys.executable}; {ENV_HINT}"))
    missing = [name for name in SOURCE_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        findings.append(("warn", f"Python modules not found: {', '.join(missing)}; {ENV_HINT}"))
    elif all(importlib.util.find_spec(name) for name in SERVER_MODULES):
        findings.append(("ok", f"Python modules present for {sys.executable}"))
    if not asset_root.is_dir():
        findings.append(("warn", f"Asset root not found: {asset_root}; every mesh request would be rejected"))
    else:
        findings += cad_findings(asset_root)
    if not port_is_free(port):
        findings.append(
            (
                port_busy_level,
                f"Port {port} is in use; a server may already be running "
                f"(curl http://127.0.0.1:{port}/health) or pass --port",
            )
        )
    return findings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--foundationpose-root",
        type=Path,
        default=Path(os.environ.get("FOUNDATIONPOSE_ROOT", default_foundationpose_root())),
        help="FoundationPose sources and weights (default: ./FoundationPose, else ../data_gen/FoundationPose)",
    )
    parser.add_argument(
        "--asset-root", type=Path, default=ROOT / "data", help="Only OBJ meshes under this directory are served"
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--check", action="store_true", help="Report missing prerequisites and exit; no GPU use")
    args = parser.parse_args()
    findings = preflight(args.foundationpose_root, args.asset_root, args.port, "warn" if args.check else "error")
    for level, message in findings:
        if level != "ok" or args.check:
            print(f"[{level.upper()}] {message}", flush=True)
    failed = any(level == "error" for level, _ in findings)
    if args.check:
        print(f"FoundationPose check: {'FAILED' if failed else 'OK'}", flush=True)
    if args.check or failed:
        sys.exit(1 if failed else 0)
    sys.path.insert(0, str(args.foundationpose_root.resolve()))
    import numpy as np
    import nvdiffrast.torch as dr
    from custom_datareader import load_mesh_readonly
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor

    scorer, refiner = ScorePredictor(), PoseRefinePredictor()
    context = dr.RasterizeCudaContext()
    estimators = {}
    asset_root = args.asset_root.resolve()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, result):
            body = json.dumps(result, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(200 if self.path == "/health" else 404, dict(status="healthy", model_loaded=True))

        def do_POST(self):
            if self.path != "/estimate":
                return self.reply(404, dict(error="Unknown endpoint"))
            try:
                started = time.perf_counter()
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size < 32 * 1024 * 1024:
                    raise ValueError("Invalid pose packet size")
                packet = np.load(io.BytesIO(self.rfile.read(size)), allow_pickle=False)
                path = Path(str(packet["mesh_path"])).resolve()
                if not path.is_relative_to(asset_root) or path.suffix != ".obj":
                    raise ValueError("Mesh must be an OBJ in the configured asset root")
                class_id, scale = str(packet["class_id"]), float(packet["mesh_scale"])
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError("Invalid mesh scale")
                key = (str(path), scale, class_id)
                if key not in estimators:
                    with load_mesh_readonly(
                        path,
                        identity_name=class_id,
                        known_names=[class_id],
                        mesh_scale=scale,
                        texture_roots=[path.parent, path.parent.parent],
                        max_texture_size=2048,
                    ) as mesh:
                        estimators[key] = FoundationPose(
                            model_pts=mesh.vertices,
                            model_normals=mesh.vertex_normals,
                            mesh=mesh,
                            scorer=scorer,
                            refiner=refiner,
                            glctx=context,
                            debug=0,
                            debug_dir="/tmp/tiptop-foundationpose-runtime",
                        )
                estimator = estimators[key]
                # The existing data_gen demo normalizes this scalar too: a numpy
                # float64 makes the crop-offset tensor double beside float poses.
                estimator.diameter = float(estimator.diameter)
                rgb, depth, mask, k = (packet[name] for name in ["rgb", "depth", "mask", "intrinsics"])
                if (
                    rgb.dtype != np.uint8
                    or rgb.shape != (*depth.shape, 3)
                    or mask.shape != depth.shape
                    or k.shape != (3, 3)
                    or not np.isfinite(k).all()
                    or min(k[0, 0], k[1, 1]) <= 0
                    or np.count_nonzero(mask & np.isfinite(depth) & (depth > 0)) < 10
                ):
                    raise ValueError("Invalid aligned RGB/depth/mask/intrinsics")
                pose = estimator.register(
                    K=k.astype(np.float32), rgb=rgb, depth=depth.astype(np.float32), ob_mask=mask, iteration=5
                )
                self.reply(
                    200,
                    dict(
                        camera_from_cad=pose.tolist(),
                        class_id=class_id,
                        elapsed_s=time.perf_counter() - started,
                        ranking_score=float(estimator.scores[0]),
                        score_is_probability=False,
                    ),
                )
            except Exception as error:
                self.reply(400, dict(error_type=type(error).__name__, error=str(error)))

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"FOUNDATIONPOSE_READY port={args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
