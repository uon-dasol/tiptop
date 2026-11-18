"""Serve task RGB/depth snapshots locally, and optionally launch run_known_pick_place from the page.

Launching is enabled unless --read-only is given; those requests need the token embedded in the served page.
"""

import argparse
import hmac
import json
import os
import secrets
import socket
import stat
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

from tiptop.launcher import FIELDS, KINDS, MODES, PRESETS_PATH, load_presets, save_preset
from tiptop.object_catalog import demo_order, load_catalog
from tiptop.task_result_log import set_task_mark, task_log_rows
from tiptop.run_manager import RunManager


def service_up(service):
    try:
        if "socket" in service:
            return stat.S_ISSOCK(os.stat(service["socket"]).st_mode)
        with socket.create_connection((service["host"], service["port"]), timeout=0.25):
            return True
    except OSError:
        return False


def snapshot_target_label(data, task_output):
    """Read explicit target identity, including snapshots made by older workers."""
    labels = {box["label"] for box in data.get("boxes", [])}
    if data.get("target_label") in labels:
        return data["target_label"]
    for filename in ("semantic-task.json", "registered-pose.json"):
        try:
            saved = json.loads((Path(task_output) / filename).read_text())
            candidates = [atom["args"][0] for atom in saved.get("grounded_atoms", []) if atom.get("args")]
            candidates.append(saved.get("label"))
            for label in candidates:
                if label in labels:
                    return label
        except (OSError, ValueError, KeyError):
            continue
    return None


@lru_cache(maxsize=32)
def _saved_rgb_camera(path, modified_ns, width, height):
    import numpy as np
    from tiptop.perception_preview import rgb_camera_view

    with np.load(path, allow_pickle=False) as packet:
        return rgb_camera_view(packet["world_from_cam"], packet["intrinsics"], width, height)


def snapshot_rgb_camera(data, task_output):
    """Recover this observation's rotated RGB calibration for older workers."""
    camera = data.get("obstacle_scene", {}).get("camera")
    if camera is not None:
        return camera
    path = Path(task_output) / "observation.npz"
    try:
        return _saved_rgb_camera(path, path.stat().st_mtime_ns, data["width"], data["height"])
    except (OSError, ValueError, KeyError):
        return None


def make_server(root, *, port=8091, task_output=None, launcher=None, presets_path=PRESETS_PATH):
    root = Path(root).resolve()
    fixed = Path(task_output).resolve() if task_output else None
    if fixed is not None and not fixed.is_relative_to(root):
        raise ValueError("Task output must be inside the viewer root")
    page = Path(__file__).parents[1] / "ui/perception_viewer.html"
    token = secrets.token_urlsafe(16) if launcher is not None else ""

    def handle_api(method, path, query, body):
        if method == "GET" and path == "/api/presets":
            return 200, dict(fields=FIELDS, modes=list(MODES), kinds=list(KINDS), **load_presets(presets_path))
        if method == "GET" and path == "/api/task-log":
            return 200, dict(rows=task_log_rows())
        if method == "POST" and path == "/api/task-log/mark":
            set_task_mark(int(body["task"]), body["field"], body["value"])
            return 200, dict(saved=True)
        if method == "GET" and path == "/api/objects":
            demo = demo_order()
            return 200, dict(objects=[dict(class_id=item["class_id"], name_ko=item["name_ko"],
                                           demo=demo.get(item["class_id"]))
                                      for item in load_catalog()["objects"]])
        if method == "GET" and path == "/api/status":
            services = [dict(name=item["name"], modes=item["modes"], up=service_up(item))
                        for item in load_presets(presets_path)["services"]]
            key = bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
            return 200, dict(**launcher.state(), services=services, google_api_key=key)
        if method == "GET" and path == "/api/log":
            return 200, dict(**launcher.log(int(query.get("since", ["0"])[0])), **launcher.state())
        if method == "POST" and path == "/api/command":
            from tiptop.launcher import display_command

            return 200, dict(command=display_command(launcher.build(body["mode"], body["values"], body["kind"])))
        if method == "POST" and path == "/api/run":
            try:
                return 200, dict(command=launcher.start(body["mode"], body["values"], body["kind"]))
            except RuntimeError as error:
                return 409, dict(error=str(error))
        if method == "POST" and path == "/api/stop":
            return 200, dict(stopped=launcher.stop())
        if method == "POST" and path == "/api/presets/save":
            save_preset(body["mode"], body["values"], presets_path)
            return 200, dict(saved=True)
        return 404, dict(error="not found")

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def host_ok(self):
            return self.headers.get("Host", "").rsplit(":", 1)[0] in ("127.0.0.1", "localhost")

        def authorized(self):
            origin = self.headers.get("Origin")
            same_origin = not origin or urlsplit(origin).netloc == self.headers.get("Host")
            return (launcher is not None and self.host_ok() and same_origin
                    and hmac.compare_digest(self.headers.get("X-Launcher-Token", ""), token))

        def api(self, method, route):
            if not self.authorized():
                return self.reply(403, b'{"error": "forbidden"}', "application/json")
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length > 65536:
                    raise ValueError("Request body too large")
                body = json.loads(self.rfile.read(length) or b"{}") if method == "POST" else {}
                status, payload = handle_api(method, route.path, parse_qs(route.query), body)
            except KeyError as error:
                status, payload = 400, dict(error=f"missing field {error}")
            except (ValueError, TypeError) as error:
                status, payload = 400, dict(error=str(error))
            self.reply(status, json.dumps(payload, ensure_ascii=False).encode(), "application/json")

        def do_POST(self):
            route = urlsplit(self.path)
            if route.path.startswith("/api/"):
                return self.api("POST", route)
            self.send_error(501, "Unsupported method ('POST')")

        def do_GET(self):
            route = urlsplit(self.path)
            if route.path.startswith("/api/"):
                return self.api("GET", route)
            try:
                if route.path == "/":
                    html = page.read_text(encoding="utf-8").replace("__LAUNCHER_TOKEN__", token if self.host_ok() else "")
                    return self.reply(200, html.encode(), "text/html; charset=utf-8")
                if route.path == "/state":
                    paths = [fixed / "viewer/state.json"] if fixed else list(root.rglob("viewer/state.json"))
                    paths = [p for p in paths if p.is_file() and p.resolve().is_relative_to(root)]
                    if not paths:
                        return self.reply(200, b'{"waiting":true}', "application/json")
                    path = max(paths, key=lambda p: p.stat().st_mtime_ns)
                    data = json.loads(path.read_text())
                    if data.get("detector") == "sam3_reference":
                        data["target_label"] = snapshot_target_label(data, path.parent.parent)
                    if data.get("obstacle_scene") is not None:
                        data["obstacle_scene"]["camera"] = snapshot_rgb_camera(data, path.parent.parent)
                    data["task_directory"] = str(path.parent.parent.relative_to(root))
                    data["images"] = {
                        key: "/image?path=" + quote(str((path.parent / value).relative_to(root)), safe="")
                        for key, value in data["images"].items()
                    }
                    return self.reply(200, json.dumps(data, ensure_ascii=False).encode(), "application/json")
                if route.path == "/image":
                    relative = parse_qs(route.query).get("path", [""])[0]
                    path = (root / relative).resolve()
                    if not path.is_relative_to(root) or path.parent.name != "viewer" or path.suffix != ".png":
                        return self.reply(404, b"Not found", "text/plain")
                    return self.reply(200, path.read_bytes(), "image/png")
                self.reply(404, b"Not found", "text/plain")
            except (OSError, ValueError, KeyError):
                self.reply(404, b"Snapshot unavailable", "text/plain")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.launcher_token = token
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("tiptop_outputs/m1013/tasks"))
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--task-output", type=Path, help="Review a saved task instead of following new tasks")
    parser.add_argument("--read-only", action="store_true", help="Disable the run panel; only show task snapshots")
    args = parser.parse_args()
    if args.task_output:
        from tiptop.perception_preview import preview_saved_task

        preview_saved_task(args.task_output)
    launcher = None if args.read_only else RunManager(Path(__file__).resolve().parents[2])
    server = make_server(args.root, port=args.port, task_output=args.task_output, launcher=launcher)
    print(f"PERCEPTION_VIEWER_READY http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
