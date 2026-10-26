"""Prepare targets, collect read-only stationary samples, or solve saved M1013 calibration.

No subcommand moves the arm or actuates the gripper. Images from OrbbecCamera
are already undistorted. Hand-eye calibration is relative to the named parent
link, not an implicit Franka TCP. Outputs are candidates and never overwrite or
activate an existing hardware calibration.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tiptop.doosan.calibration import fit_rigid_datums, solve_hand_eye, solve_pivot
from tiptop.doosan.client import DoosanUONClient
from tiptop.doosan.observation import stationary_state, world_from_parent


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def make_board(spec):
    columns, rows = int(spec["columns"]), int(spec["rows"])
    square, marker = float(spec["square_m"]), float(spec["marker_m"])
    if min(columns, rows) < 3 or not np.isfinite([square, marker]).all() or not 0 < marker < square:
        raise ValueError("Board needs at least 3x3 squares and 0 < marker_m < square_m")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.aruco.CharucoBoard((columns, rows), square, marker, dictionary)


def detect_board(frame, spec):
    board = make_board(spec)
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(frame.rgb)
    if ids is None or len(ids) < 6:
        raise ValueError("At least six labelled ChArUco corners must be fully visible")
    object_points, pixels = board.matchImagePoints(corners, ids)
    # Frame.rgb already uses a distortion-free pinhole K; do not apply SDK lens distortion twice.
    ok, rvec, tvec = cv2.solvePnP(object_points, pixels, frame.intrinsics, np.zeros(5))
    if not ok or not np.isfinite(tvec).all() or tvec[2, 0] <= 0:
        raise ValueError("Board pose is invalid or behind the colour camera")
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, frame.intrinsics, np.zeros(5))
    rms = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - pixels.reshape(-1, 2)) ** 2, axis=1))))
    if rms > 0.8:
        raise ValueError(f"Board reprojection RMS {rms:.3f}px exceeds 0.8px")
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.reshape(3)
    return transform, rms, len(ids)


def capture(args):
    if args.output.exists() or args.output.with_suffix(".png").exists():
        raise FileExistsError("Use a new sample filename")
    client = DoosanUONClient(args.gateway_url)
    q, _ = stationary_state(client, expected_system=0)
    sample = dict(
        kind="pivot",
        source="gateway_read_only",
        parent_link=args.parent_link,
        world_frame="base_link",
        units="metres",
        q_rad=q.tolist(),
        world_from_parent=world_from_parent(q, args.parent_link).tolist(),
    )
    if args.command == "capture-hand-eye":
        from tiptop.perception.cameras.orbbec_camera import OrbbecCamera

        spec = json.loads(args.board.read_text())
        make_board(spec)
        camera = OrbbecCamera(serial=args.serial, enable_depth=False, undistort=True)
        try:
            # Warm-up can take time; bracket the actual read with a second idle-state check.
            before, _ = stationary_state(client, expected_system=0)
            frame = camera.read_camera()
            after, _ = stationary_state(client, expected_system=0)
            delta = float(np.max(np.abs(after - before)))
            if delta > 0.001:
                raise ValueError("Robot moved during image acquisition; discard this sample")
            board_pose, rms, count = detect_board(frame, spec)
            sample.update(
                kind="hand_eye",
                source="orbbec_and_gateway_read_only",
                q_rad=after.tolist(),
                world_from_parent=world_from_parent(after, args.parent_link).tolist(),
                camera_from_board=board_pose.tolist(),
                camera_serial=frame.serial,
                camera_frame="undistorted_color_optical",
                intrinsics=frame.intrinsics.tolist(),
                frame_timestamp_s=frame.timestamp,
                stationary_joint_delta_rad=delta,
                board=spec,
                corners=count,
                reprojection_rms_px=rms,
                image=args.output.with_suffix(".png").name,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.with_suffix(".png").open("xb") as stream:
                Image.fromarray(frame.rgb).save(stream, format="PNG")
        finally:
            camera.close()
    save_json(args.output, sample)
    return sample


def solve(args):
    samples = [json.loads(path.read_text()) for path in args.samples]
    expected = "hand_eye" if args.command == "solve-hand-eye" else "pivot"
    if any(sample.get("kind") != expected or sample.get("units") != "metres" for sample in samples):
        raise ValueError(f"Expected {expected} samples in metres")
    parents = {sample["parent_link"] for sample in samples}
    if len(parents) != 1 or any(sample.get("world_frame") != "base_link" for sample in samples):
        raise ValueError("All samples must use the same parent and robot base frame")
    poses = [sample["world_from_parent"] for sample in samples]
    if expected == "hand_eye":
        serials = {sample["camera_serial"] for sample in samples}
        boards = {json.dumps(sample["board"], sort_keys=True) for sample in samples}
        if (
            len(serials) != 1
            or len(boards) != 1
            or any(sample.get("camera_frame") != "undistorted_color_optical" for sample in samples)
        ):
            raise ValueError("Do not mix cameras, board sizes or image frames")
        result = solve_hand_eye(poses, [sample["camera_from_board"] for sample in samples])
        result.update(camera_serial=serials.pop(), camera_frame="undistorted_color_optical")
    else:
        result = solve_pivot(poses)
    result.update(
        parent_link=parents.pop(),
        world_frame="base_link",
        calibrated=False,
        status="candidate_requires_independent_measurement_check",
        sample_files=[str(p) for p in args.samples],
    )
    save_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    board = sub.add_parser("board", help="Generate a labelled printable target and its metre-scale specification")
    board.add_argument("--columns", type=int, default=7)
    board.add_argument("--rows", type=int, default=5)
    board.add_argument("--square-m", type=float, required=True)
    board.add_argument("--marker-m", type=float, required=True)
    board.add_argument(
        "--output", type=Path, required=True, help="New PNG filename; JSON specification is written beside it"
    )
    for name in ["capture-hand-eye", "capture-pivot"]:
        item = sub.add_parser(name, help="Record one stationary sample; no motion commands")
        item.add_argument("--gateway-url", default="http://127.0.0.1:8090")
        item.add_argument("--parent-link", default="link_6")
        item.add_argument("--output", type=Path, required=True)
        if name == "capture-hand-eye":
            item.add_argument("--board", type=Path, required=True, help="JSON specification from board command")
            item.add_argument("--serial")
    for name in ["solve-hand-eye", "solve-pivot"]:
        item = sub.add_parser(name, help="Offline solve from at least ten independent stationary samples")
        item.add_argument("samples", type=Path, nargs="+")
        item.add_argument("--output", type=Path, required=True)
    datums = sub.add_parser("fit-datums", help="Offline mounting/DB-frame rigid fit from named measured datum points")
    datums.add_argument("input", type=Path)
    datums.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "board":
            spec = dict(
                columns=args.columns,
                rows=args.rows,
                square_m=args.square_m,
                marker_m=args.marker_m,
                dictionary="DICT_4X4_50",
                print_width_m=args.columns * args.square_m,
                print_height_m=args.rows * args.square_m,
            )
            board = make_board(spec)
            if args.output.exists() or args.output.with_suffix(".json").exists():
                raise FileExistsError("Use new board output filenames")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("xb") as stream:
                Image.fromarray(board.generateImage((args.columns * 200, args.rows * 200))).save(stream, format="PNG")
            save_json(args.output.with_suffix(".json"), spec)
            result = spec
        elif args.command.startswith("capture-"):
            result = capture(args)
        elif args.command.startswith("solve-"):
            result = solve(args)
        else:
            document = json.loads(args.input.read_text())
            result = fit_rigid_datums(document["source_points_m"], document["target_points_m"])
            result.update(
                source_frame=document["source_frame"], target_frame=document["target_frame"], calibrated=False
            )
            save_json(args.output, result)
        print(json.dumps(result, indent=2, allow_nan=False))
        if result.get("consistent") is False:
            raise SystemExit(1)
    except (ValueError, KeyError, FileExistsError, RuntimeError, cv2.error) as error:
        parser.exit(2, f"Calibration failed: {error}\n")


if __name__ == "__main__":
    main()
