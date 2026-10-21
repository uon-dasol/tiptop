"""Hardware check for the Orbbec Gemini 336L adapter. Run: pixi run python -m tiptop.scripts.check_orbbec"""

import argparse
import sys
import time

import cv2
import numpy as np

from tiptop.perception.cameras.orbbec_camera import OrbbecCamera, OrbbecFrame, OrbbecIntrinsics
from tiptop.perception.cameras.rs_camera import _depth_ir_to_color

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


def check_stream(frames: list[OrbbecFrame], seconds: float) -> tuple[str, str]:
    rate = len(frames) / seconds
    increasing = bool(np.all(np.diff([f.timestamp for f in frames]) > 0))
    status = PASS if rate >= 10 and increasing else FAIL
    return (
        status,
        f"{len(frames)} frames in {seconds:.1f}s ({rate:.1f}/s), timestamps strictly increasing: {increasing}",
    )


def check_contract(frame: OrbbecFrame, with_depth: bool) -> tuple[str, str]:
    h, w = frame.rgb.shape[:2]
    problems = []
    if frame.rgb.dtype != np.uint8 or frame.rgb.shape != (h, w, 3) or frame.rgb.max() == 0:
        problems.append("rgb must be a non-black uint8 (h, w, 3) array")
    for name in ("ir_left", "ir_right"):
        ir = getattr(frame, name)
        if ir.dtype != np.uint8 or ir.shape != (h, w):
            problems.append(f"{name} must be uint8 {(h, w)}, got {ir.dtype} {ir.shape}")
    if frame.intrinsics.shape != (3, 3) or not np.isfinite(frame.intrinsics).all():
        problems.append("intrinsics must be a finite 3x3 matrix")
    if with_depth:
        depth = frame.depth
        if depth is None or depth.dtype != np.float32 or depth.shape != (h, w):
            problems.append("depth must be float32 on the rgb grid")
        elif depth.min() < 0 or depth.max() > 65.6:
            problems.append("depth must lie in [0, 65.6] m")
    return (FAIL, "; ".join(problems)) if problems else (PASS, f"rgb {w}x{h} uint8, ir_left/ir_right uint8, depth ok")


def check_calibration(intrinsics: OrbbecIntrinsics) -> tuple[str, str]:
    T = intrinsics.T_color_from_ir
    R, t = T[:3, :3], T[:3, 3]
    problems = []
    if not 0.09 <= intrinsics.baseline_ir <= 0.10:
        problems.append("baseline outside 90-100 mm")
    if np.abs(R.T @ R - np.eye(3)).max() > 1e-3 or abs(np.linalg.det(R) - 1.0) > 1e-3:
        problems.append("T_color_from_ir rotation is not a proper rotation")
    if not 0.015 <= np.linalg.norm(t) <= 0.04:
        problems.append("colour-IR offset outside 15-40 mm")
    if not (500 <= intrinsics.K_color[0, 0] <= 700 and 550 <= intrinsics.K_ir[0, 0] <= 700):
        problems.append("focal length outside 500-700 px")
    summary = (
        f"baseline {intrinsics.baseline_ir * 1000:.2f} mm, colour-IR offset {np.linalg.norm(t) * 1000:.1f} mm, "
        f"fx colour {intrinsics.K_color[0, 0]:.1f} / IR {intrinsics.K_ir[0, 0]:.1f}"
    )
    return (FAIL, "; ".join(problems) + f" ({summary})") if problems else (PASS, summary)


def check_freshness(cam: OrbbecCamera) -> tuple[str, str]:
    first = cam.read_camera()
    time.sleep(0.5)
    second = cam.read_camera()
    gap = second.timestamp - first.timestamp
    status = PASS if gap >= 0.4 else FAIL
    return status, f"frames are {gap:.2f}s apart after a 0.5s pause (a stale queued frame would be about 0.03s)"


def check_geometry(frame: OrbbecFrame, intrinsics: OrbbecIntrinsics) -> tuple[str, str]:
    """Stereo-matched IR depth warped to the colour grid must agree with the aligned sensor depth."""
    if frame.depth is None:
        return SKIP, "needs the sensor depth stream"
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=256,
        blockSize=7,
        P1=8 * 49,
        P2=32 * 49,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_HH,
    )
    disparity = sgbm.compute(frame.ir_left, frame.ir_right).astype(np.float32) / 16.0
    depth_ir = np.where(disparity > 2, intrinsics.K_ir[0, 0] * intrinsics.baseline_ir / np.maximum(disparity, 1e-6), 0)
    stereo = _depth_ir_to_color(
        depth_ir.astype(np.float32),
        intrinsics.K_ir,
        intrinsics.T_color_from_ir,
        intrinsics.K_color,
        frame.rgb.shape[:2],
    )
    both = (frame.depth > 0.3) & (frame.depth < 4.0) & (stereo > 0.3) & (stereo < 4.0)
    if both.sum() < 3000:
        return SKIP, f"only {int(both.sum())} px valid in both depths; aim at a textured scene 0.4-3 m away"
    ratio = frame.depth[both] / stereo[both]
    median, within = float(np.median(ratio)), float((np.abs(ratio - 1) < 0.05).mean())
    status = PASS if abs(median - 1) < 0.03 and within >= 0.8 else FAIL
    return status, f"sensor / stereo depth: median {median:.3f}, within 5%: {within:.2f} over {int(both.sum())} px"


def report_depth(frame: OrbbecFrame) -> None:
    """Informational numbers; the plane fit is only meaningful when the camera faces a flat surface."""
    depth = frame.depth
    valid = depth > 0
    print(f"[INFO] depth fill {valid.mean():.2f}, median {np.median(depth[valid]) * 1000:.0f} mm")
    h, w = depth.shape
    v, u = np.mgrid[h // 2 - 100 : h // 2 + 100, w // 2 - 100 : w // 2 + 100]
    roi = depth[h // 2 - 100 : h // 2 + 100, w // 2 - 100 : w // 2 + 100]
    mask = roi > 0
    if mask.sum() < 5000:
        print("[INFO] plane fit skipped: fewer than 5000 valid depth pixels in the centre 200x200 window")
        return
    K = frame.intrinsics
    z = roi[mask]
    points = np.stack([(u[mask] - K[0, 2]) * z / K[0, 0], (v[mask] - K[1, 2]) * z / K[1, 1], z], axis=1)
    centred = points - points.mean(axis=0)
    normal = np.linalg.svd(centred, full_matrices=False)[2][-1]
    print(f"[INFO] centre 200x200 plane-fit RMS {np.sqrt(np.mean((centred @ normal) ** 2)) * 1000:.1f} mm")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default=None, help="Camera serial; required when several are connected")
    parser.add_argument("--seconds", type=float, default=3.0, help="Duration of the streaming check")
    parser.add_argument("--no-depth", action="store_true", help="Skip the sensor depth stream and the geometry check")
    args = parser.parse_args()
    with_depth = not args.no_depth

    cam = OrbbecCamera(serial=args.serial, enable_depth=with_depth)
    try:
        frames = []
        end = time.monotonic() + args.seconds
        while time.monotonic() < end:
            frames.append(cam.read_camera())
        intrinsics = cam.get_intrinsics()
        results = {
            "stream": check_stream(frames, args.seconds),
            "contract": check_contract(frames[-1], with_depth),
            "calibration": check_calibration(intrinsics),
            "freshness": check_freshness(cam),
            "geometry": check_geometry(frames[-1], intrinsics),
        }
        print(f"Camera s/n {cam.serial}")
        for name, (status, message) in results.items():
            print(f"[{status}] {name}: {message}")
        if with_depth:
            report_depth(frames[-1])
    finally:
        cam.close()
    sys.exit(1 if any(status == FAIL for status, _ in results.values()) else 0)


if __name__ == "__main__":
    main()
