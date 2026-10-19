"""Show a cuTAMP pepper plan in existing RViz, optionally run the virtual arm.

The existing executor/client are unchanged. The observer only attaches/releases
the RViz object after successful gripper commands, following actual model TF.
"""
import argparse
import json
import logging
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiptop.doosan.client import DoosanUONClient
from tiptop.execute_plan import execute_cutamp_plan


def obj_geometry(path):
    # Geometry-only parsing avoids the source MTL's unsupported 'aniso' keyword.
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == "v":
            vertices.append([float(v) for v in fields[1:4]])
        elif fields and fields[0] == "f":
            indices = [int(v.split("/")[0]) for v in fields[1:]]
            indices = [v-1 if v > 0 else len(vertices)+v for v in indices]
            faces.extend([indices[0], indices[i], indices[i+1]] for i in range(1, len(indices)-1))
    return np.asarray(vertices), np.asarray(faces, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--execute", action="store_true", help="Execute through the existing virtual-only gateway")
    parser.add_argument("--url", default="http://127.0.0.1:8090")
    args = parser.parse_args()
    args.plan = args.plan.resolve()
    if not args.plan.is_relative_to(ROOT):
        parser.error("Use a project plan; execution outputs must not be written to source-data directories")
    plan = json.loads(args.plan.read_text())
    report = json.loads(args.plan.with_name("result.json").read_text())
    if not report.get("success") or plan.get("version") != "1.0.0":
        parser.error("Expected a successfully verified cuTAMP plan")
    # Generate only the 1.6MB visualization mesh; never alter source OBJ/MTL.
    import trimesh
    vertices, faces = obj_geometry(Path(report["cad"]))
    mesh = trimesh.Trimesh(vertices=vertices*.001, faces=faces, process=False)
    cache = ROOT / "ros2_ws/build/tiptop_grasp_preview"
    cache.mkdir(parents=True, exist_ok=True)
    mesh.export(cache / "pepper.stl")
    scene = {"object_pose": report.get("object_pose", report["object_xyz"]+[1., 0., 0., 0.]),
             "target_xyz": report["target_xyz"]}
    worker = Path(__file__).with_name("m1013_rviz_scene.py").read_text()
    command = "source /opt/ros/jazzy/setup.bash\npython3 -u -c " + shlex.quote(worker)
    process = subprocess.Popen(["docker", "exec", "-i", "--user", "ros", "ros2-dev", "bash", "-lc", command],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    process.stdin.write(json.dumps(scene)+"\n")
    process.stdin.flush()
    events = []

    def receive(expected):
        while True:
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("RViz visualization worker exited before " + expected)
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                print(line.rstrip(), flush=True)
                continue
            if event.get("event") != expected:
                raise RuntimeError(f"Expected {expected}, got {event}")
            events.append(event)
            print(json.dumps(event), flush=True)
            return event

    def send(event):
        process.stdin.write(json.dumps({"event": event})+"\n")
        process.stdin.flush()
        return receive(event)

    class ObservedClient(DoosanUONClient):
        def close_gripper(self, **kwargs):
            result = super().close_gripper(**kwargs)
            if result.get("success"):
                send("attach")
            return result

        def open_gripper(self, **kwargs):
            result = super().open_gripper(**kwargs)
            if result.get("success"):
                send("detach")
            return result

    result = {"success": False, "visual_attachment_only": True, "physical_grasp_verified": False,
              "sent_to_robot": False, "plan": str(args.plan.resolve())}
    try:
        receive("ready")
        if not args.execute:
            print("RViz scene ready. Ctrl+C to stop; no motion commands sent.", flush=True)
            process.wait()
            return
        client = ObservedClient(args.url)
        state = client.get_state()
        if not state.get("success") or state.get("motion_status") != 0:
            raise RuntimeError(f"Gateway is not idle: {state}")
        if np.max(np.abs(np.array(state["joint_positions"])-plan["q_init"])) > np.radians(1):
            raise RuntimeError("Plan q_init differs from current joints by more than 1 degree")
        if state["gripper"]["commanded_position_rad"] > .01:
            raise RuntimeError("Open the model gripper before this pick/place demonstration")
        logging.basicConfig(level=logging.INFO)
        started = time.perf_counter()
        result["sent_to_robot"] = True
        execute_cutamp_plan(plan["steps"], client)
        final = send("quit")
        result.update(success=True, completed_steps=len(plan["steps"]),
                      elapsed_s=time.perf_counter()-started, final_state=client.get_state(),
                      final_world_from_object=final["world_from_object"])
    finally:
        if process.poll() is None:
            process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
        if args.execute:
            result["events"] = events
            args.plan.with_name("rviz-execution.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
