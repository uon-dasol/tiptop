"""Read-only Orbbec/M1013 HTTP observations for run_known_pick_place.

This supplies /observe and /display using measured or explicitly assumed fixed optical poses and
stationary robot feedback. /execute is deliberately unavailable. Camera SDK,
ROS gateway and inference processes remain separate; no ROS package is imported.
"""

import argparse
import importlib.util
import io
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from simulation.isaac_sim.observation_server import ObservationServer, require_current_result
from tiptop.doosan.client import DoosanUONClient
from tiptop.doosan.observation import capture_stationary, stationary_state, validate_camera_calibration
from tiptop.serialization import savez_fast


class ReadOnlyObservationBackend:
    def __init__(self, camera, client, calibration, output, source="foundation_stereo"):
        self.transform = validate_camera_calibration(calibration, allow_assumed=True)
        if camera.serial != calibration["camera_serial"]:
            raise ValueError("Connected camera serial differs from camera calibration")
        self.parent_link = calibration["parent_link"]
        self.image_rotation_deg = calibration.get("image_rotation_deg", 0)
        self.pose_source = calibration.get("pose_source", "measured")
        self.camera, self.client, self.output, self.source = camera, client, Path(output), source
        self.latest_id, self.latest_q = None, None
        self.output.mkdir(parents=True, exist_ok=True)

    def dispatch(self, path, body):
        if path == "/health":
            return json.dumps(
                dict(
                    success=True,
                    read_only=True,
                    camera_serial=self.camera.serial,
                    camera_parent_link=self.parent_link,
                    image_rotation_deg=self.image_rotation_deg,
                    camera_pose_source=self.pose_source,
                    depth_source=self.source,
                    pid=os.getpid(),
                )
            ).encode()
        if path == "/execute":
            raise ValueError("This physical observation backend is read-only; no arm or gripper commands are supported")
        if path == "/observe":
            stationary_state(self.client, expected_system=0)
            packet = capture_stationary(
                self.camera, self.client, parent_from_camera=self.transform, parent_link=self.parent_link,
                image_rotation_deg=self.image_rotation_deg, source=self.source,
            )
            packet["camera_pose_source"] = self.pose_source
            self.latest_id = uuid.uuid4().hex
            self.latest_q = packet["q_init"].copy()
            packet["observation_id"] = self.latest_id
            buffer = io.BytesIO()
            savez_fast(buffer, **packet)
            (self.output / (self.latest_id + ".npz")).write_bytes(buffer.getvalue())
            Image.fromarray(packet["rgb"]).save(self.output / (self.latest_id + ".png"), compress_level=1)
            return buffer.getvalue()
        if path == "/display":
            packet = np.load(io.BytesIO(body), allow_pickle=False)
            q, _ = stationary_state(self.client, expected_system=0)
            require_current_result(str(packet["observation_id"]), self.latest_id, q, self.latest_q)
            overlay = np.asarray(packet["overlay"])
            if overlay.dtype != np.uint8 or overlay.ndim != 3 or overlay.shape[2] != 3:
                raise ValueError("Display overlay must be RGB uint8")
            Image.fromarray(overlay).save(self.output / (self.latest_id + "-overlay.png"), compress_level=1)
            print(str(packet["status"]), flush=True)
            return b'{"success":true,"read_only":true}'
        raise ValueError("Unknown observation endpoint")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8090")
    parser.add_argument("--depth-source", choices=["foundation_stereo", "sensor"], default="foundation_stereo")
    parser.add_argument("--port", type=int, default=13345)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check", action="store_true", help="Check local inputs only; do not open camera or robot connection"
    )
    args = parser.parse_args()
    try:
        calibration = json.loads(args.calibration.read_text())
        validate_camera_calibration(calibration, allow_assumed=True)
        if importlib.util.find_spec("pyorbbecsdk") is None:
            raise ValueError("pyorbbecsdk is not installed in this environment")
        if args.check:
            print(json.dumps(dict(
                success=True, local_inputs_valid=True, hardware_probed=False, read_only=True,
                camera_pose_source=calibration.get("pose_source", "measured"),
            )))
            return
        if args.output is None:
            parser.error("--output is required when serving observations")
        client = DoosanUONClient(args.gateway_url)
        stationary_state(client, expected_system=0)
        # Reserve before opening the physical camera, so another backend cannot share this endpoint.
        server = ObservationServer(port=args.port)
        camera = None
        try:
            from tiptop.perception.cameras.orbbec_camera import OrbbecCamera

            camera = OrbbecCamera(
                serial=calibration["camera_serial"], enable_depth=args.depth_source == "sensor", undistort=True
            )
            backend = ReadOnlyObservationBackend(camera, client, calibration, args.output, args.depth_source)
            print(f"ORBBEC_OBSERVATION_READY port={args.port} read_only=true", flush=True)
            while True:
                server.poll(backend.dispatch)
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        finally:
            if camera is not None:
                camera.close()
            server.close()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(2, f"Observation setup failed: {error}\n")


if __name__ == "__main__":
    main()
