"""Camera-only RGB/optional sensor-depth preview; no robot or inference connection."""

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np


def main():
    import pyorbbecsdk as ob

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="CPC8563000Z2")
    parser.add_argument("--depth", action="store_true", help="Show aligned sensor depth and save a camera-only RGB-D snapshot")
    parser.add_argument("--ir", action="store_true", help="Include native left/right IR in camera-only snapshots")
    parser.add_argument("--output", type=Path, default=Path("/tmp/gemini336l-real-view"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ob.Context.set_logger_to_file(ob.OBLogLevel.NONE, "")
    ob.Context.set_logger_to_console(ob.OBLogLevel.ERROR)
    context = ob.Context()
    device = context.query_devices().get_device_by_serial_number(args.serial)
    pipeline = ob.Pipeline(device)
    profile = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR).get_video_stream_profile(
        1280, 720, ob.OBFormat.MJPG, 30)
    intrinsic, distortion = profile.get_intrinsic(), profile.get_distortion()
    k = np.array([[intrinsic.fx,0,1279-intrinsic.cx],
                  [0,intrinsic.fy,719-intrinsic.cy], [0,0,1]], float)
    d = np.array([getattr(distortion,name) for name in ("k1","k2","p1","p2","k3","k4","k5","k6")])
    d[2:4] *= -1
    maps = cv2.initUndistortRectifyMap(k, d, None, k, (1280,720), cv2.CV_32FC1)
    config = ob.Config()
    config.enable_stream(profile)
    stereo = {}
    if args.ir:
        from tiptop.perception.cameras.orbbec_camera import _k_matrix, _transform_from_extrinsic

        ir_profiles = []
        for sensor in (ob.OBSensorType.LEFT_IR_SENSOR, ob.OBSensorType.RIGHT_IR_SENSOR):
            ir_profile = pipeline.get_stream_profile_list(sensor).get_video_stream_profile(1280, 720, ob.OBFormat.Y8, 30)
            config.enable_stream(ir_profile)
            ir_profiles.append(ir_profile)
        stereo = dict(
            K_ir=_k_matrix(ir_profiles[0].get_intrinsic()),
            baseline_ir=float(np.linalg.norm(ir_profiles[0].get_extrinsic_to(ir_profiles[1]).transform)/1000.),
            T_color_from_ir=np.diag([-1., -1., 1., 1.]) @ _transform_from_extrinsic(ir_profiles[0].get_extrinsic_to(profile)),
        )
        config.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        pipeline.enable_frame_sync()
    align = None
    if args.depth:
        depth_width, depth_height = (1280, 720) if args.ir else (848, 480)
        depth_profile = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR).get_video_stream_profile(
            depth_width, depth_height, ob.OBFormat.Y16, 30)
        config.enable_stream(depth_profile)
        config.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        pipeline.enable_frame_sync()
        align = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
        align.set_config_value("TargetDistortion", 1.0)
    title = "Gemini 336L REAL " + ("RGB + SENSOR + FS" if args.ir else "RGB + DEPTH" if args.depth else "RGB") + " - 180 deg - Q/ESC to close"
    try:
        pipeline.start(config)
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 1280 if args.depth else 800, 360 if args.depth else 450)
        cv2.moveWindow(title, 40, 80)
        frames_seen, last_saved = 0, 0.
        fs_depth, fs_mtime = None, None
        print("REAL_RGB_PREVIEW_READY serial="+args.serial+" rotate180=true undistorted=true", flush=True)
        while True:
            frames = pipeline.wait_for_frames(2000)
            if frames is None or frames.get_color_frame() is None:
                continue
            if args.ir and (frames.get_left_ir_frame() is None or frames.get_right_ir_frame() is None):
                continue
            color = frames.get_color_frame()
            raw = cv2.imdecode(np.frombuffer(color.get_data(), np.uint8), cv2.IMREAD_COLOR)
            if raw is None:
                raise RuntimeError("Cannot decode the camera colour frame")
            pixels = cv2.remap(cv2.rotate(raw, cv2.ROTATE_180), *maps, cv2.INTER_LINEAR)
            depth = None
            display = pixels
            if args.depth:
                aligned = align.process(frames).as_frame_set().get_depth_frame()
                if aligned is None:
                    continue
                raw_depth = np.frombuffer(aligned.get_data(), np.uint16).reshape(
                    aligned.get_height(), aligned.get_width()).copy()
                depth = raw_depth.astype(np.float32) * (float(aligned.get_depth_scale()) / 1000.)
                depth[(raw_depth == 0) | (raw_depth == 65535)] = 0
                depth = cv2.remap(cv2.rotate(depth, cv2.ROTATE_180), *maps, cv2.INTER_NEAREST)
                colours = cv2.applyColorMap(np.clip(depth / 1.5 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                colours[depth <= 0] = 0
                display = np.concatenate([pixels, colours], axis=1)
            if args.ir:
                path = args.output / "foundation-depth.npy"
                if path.is_file() and path.stat().st_mtime_ns != fs_mtime:
                    fs_depth = np.load(path, allow_pickle=False)
                    fs_mtime = path.stat().st_mtime_ns
                fs_colours = np.zeros_like(pixels)
                if fs_depth is not None:
                    fs_colours = cv2.applyColorMap(np.clip(fs_depth/1.5*255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                    fs_colours[~np.isfinite(fs_depth) | (fs_depth <= 0)] = 0
                panels = [pixels.copy(), colours.copy() if depth is not None else np.zeros_like(pixels), fs_colours]
                for panel, label in zip(panels, ["RGB LIVE / 180 deg", "SENSOR DEPTH LIVE", "FOUNDATIONSTEREO SNAPSHOT" if fs_depth is not None else "FOUNDATIONSTEREO: WAITING"]):
                    cv2.rectangle(panel, (0,0), (1280,45), (0,0,0), -1)
                    cv2.putText(panel, label, (15,32), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2)
                display = np.concatenate(panels, axis=1)
            cv2.imshow(title, display)
            frames_seen += 1
            if time.monotonic()-last_saved > 2:
                cv2.imwrite(str(args.output / "latest.png"), pixels)
                status = dict(
                    frames=frames_seen, timestamp=time.time(), serial=args.serial,
                    rotated180=True, undistorted=True, intrinsics=k.tolist())
                if depth is not None:
                    cv2.imwrite(str(args.output / "depth.png"), colours)
                    snapshot = dict(rgb=cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB), depth=depth, intrinsics=k,
                                    camera_serial=args.serial, timestamp=color.get_timestamp_us()/1e6,
                                    image_rotation_deg=180, camera_only=True)
                    if args.ir:
                        snapshot.update(stereo)
                        for name, ir_frame in [("ir_left",frames.get_left_ir_frame()), ("ir_right",frames.get_right_ir_frame())]:
                            snapshot[name] = np.frombuffer(ir_frame.get_data(), np.uint8).reshape(
                                ir_frame.get_height(), ir_frame.get_width()).copy()
                    temporary = args.output / "latest-pending.npz"
                    np.savez_compressed(temporary, **snapshot)
                    temporary.replace(args.output / "latest.npz")
                    valid = depth > 0
                    status.update(depth_fill=float(valid.mean()),
                                  depth_median_m=float(np.median(depth[valid])) if valid.any() else None)
                (args.output / "status.json").write_text(json.dumps(status))
                last_saved = time.monotonic()
            if cv2.waitKey(1) & 0xff in (27, ord("q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE)<1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
