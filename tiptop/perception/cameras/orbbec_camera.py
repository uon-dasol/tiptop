"""Orbbec Gemini 336L: colour image, rectified IR stereo pair and optional aligned sensor depth.

Conventions of the data returned by `OrbbecCamera.read_camera`:
- Camera axes are x right, y down, z forward. Pixel origin is the top-left corner. Lengths are metres.
- `rgb` is RGB uint8, undistorted unless `undistort=False`; `Frame.intrinsics` is its pinhole K.
- `ir_left` / `ir_right` are uint8, row-aligned and distortion free. They share `K_ir` and are separated by
  `baseline_ir` along the left IR +x axis. The sensor depth frame is the left IR frame.
- `depth` (`enable_depth=True`) is float32 metres on the `rgb` pixel grid. 0 marks invalid pixels.
- `T_color_from_ir` maps left IR points to the colour frame: p_color = R @ p_ir + t.
"""

import logging
import time
from dataclasses import dataclass

import aiohttp
import cv2
import numpy as np
from jaxtyping import Float, UInt8

from tiptop.perception.cameras.frame import Frame
from tiptop.perception.cameras.rs_camera import rs_infer_depth_async

_log = logging.getLogger(__name__)

# Sensor depth is reported in millimetres; zero and the 16-bit maximum mark invalid pixels
_DEPTH_INVALID_MAX = 65535


@dataclass(frozen=True, kw_only=True)
class OrbbecFrame(Frame):
    """Frame from the Gemini 336L which also includes the rectified IR stereo pair."""

    ir_left: UInt8[np.ndarray, "h w"]
    ir_right: UInt8[np.ndarray, "h w"]


@dataclass(frozen=True)
class OrbbecIntrinsics:
    """Calibration of the Gemini 336L. Field names match `RealsenseIntrinsics`."""

    K_color: Float[np.ndarray, "3 3"]  # Pinhole matrix of the (undistorted) colour image
    K_ir: Float[np.ndarray, "3 3"]  # Shared matrix of the rectified IR pair and the depth frame
    baseline_ir: float  # Metres between the left and right IR cameras
    T_color_from_ir: Float[np.ndarray, "4 4"]  # Left IR frame to colour frame, translation in metres
    distortion_color: Float[np.ndarray, "8"]  # OpenCV order: k1, k2, p1, p2, k3, k4, k5, k6


def _k_matrix(intrinsic) -> np.ndarray:
    return np.array(
        [[intrinsic.fx, 0.0, intrinsic.cx], [0.0, intrinsic.fy, intrinsic.cy], [0.0, 0.0, 1.0]], dtype=np.float32
    )


def _distortion_vector(distortion) -> np.ndarray:
    names = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
    return np.array([getattr(distortion, name) for name in names], dtype=np.float64)


def _transform_from_extrinsic(extrinsic) -> np.ndarray:
    """4x4 transform in metres from an SDK extrinsic (row-major rotation, translation in millimetres)."""
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.asarray(extrinsic.rot, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(extrinsic.transform, dtype=np.float64) / 1000.0
    return transform


def _depth_to_metres(raw: np.ndarray, scale_mm: float) -> np.ndarray:
    """Convert raw uint16 depth to float32 metres, mapping invalid pixels to 0."""
    depth = raw.astype(np.float32) * (scale_mm / 1000.0)
    depth[(raw == 0) | (raw == _DEPTH_INVALID_MAX)] = 0.0
    return depth


def _frame_pixels(frame, dtype) -> np.ndarray:
    height, width = frame.get_height(), frame.get_width()
    return np.frombuffer(frame.get_data(), dtype, count=height * width).reshape(height, width).copy()


class OrbbecCamera:
    def __init__(
        self,
        serial: str | None = None,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        enable_depth: bool = False,
        undistort: bool = True,
        warmup_frames: int = 15,
        timeout_ms: int = 1000,
    ):
        import pyorbbecsdk as ob

        # The SDK would otherwise write a Log/ folder into the working directory
        ob.Context.set_logger_to_file(ob.OBLogLevel.NONE, "")
        ob.Context.set_logger_to_console(ob.OBLogLevel.WARNING)

        start_time = time.perf_counter()
        self._ob = ob
        self._enable_depth = enable_depth
        self._timeout_ms = timeout_ms
        self._ctx = ob.Context()  # Must outlive the device, otherwise the SDK dereferences a freed device manager
        self._device = self._open_device(serial)
        self.serial = self._device.get_device_info().get_serial_number()
        self._pipeline = ob.Pipeline(self._device)

        sensors = {
            "color": (ob.OBSensorType.COLOR_SENSOR, ob.OBFormat.MJPG),
            "ir_left": (ob.OBSensorType.LEFT_IR_SENSOR, ob.OBFormat.Y8),
            "ir_right": (ob.OBSensorType.RIGHT_IR_SENSOR, ob.OBFormat.Y8),
        }
        if enable_depth:
            sensors["depth"] = (ob.OBSensorType.DEPTH_SENSOR, ob.OBFormat.Y16)
        self._profiles = {
            name: self._select_profile(sensor, fmt, width, height, fps) for name, (sensor, fmt) in sensors.items()
        }

        config = ob.Config()
        for profile in self._profiles.values():
            config.enable_stream(profile)
        config.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        self._pipeline.enable_frame_sync()
        self._pipeline.start(config)
        try:
            # Keep the raw colour lens geometry so aligned depth shares the pixel grid of the distorted colour image
            self._align = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
            self._align.set_config_value("TargetDistortion", 1.0)
            for _ in range(warmup_frames):
                self._pipeline.wait_for_frames(timeout_ms)
            self._intrinsics = self._read_intrinsics()
        except Exception:
            self._pipeline.stop()
            raise

        self._undistort_maps = None
        if undistort:
            K = self._intrinsics.K_color
            dist = self._intrinsics.distortion_color
            self._undistort_maps = cv2.initUndistortRectifyMap(K, dist, None, K, (width, height), cv2.CV_16SC2)
        init_dur = time.perf_counter() - start_time
        _log.info(f"Orbbec camera (s/n: {self.serial}) initialization complete, took {init_dur:.2f}s")

    def _open_device(self, serial: str | None):
        device_list = self._ctx.query_devices()
        usb = [
            i
            for i in range(device_list.get_count())
            if device_list.get_device_connection_type_by_index(i) != "Ethernet"
        ]
        if not usb:
            raise RuntimeError("No USB Orbbec camera found; check the cable and the udev rule (see install-orbbec)")
        serials = [device_list.get_device_serial_number_by_index(i) for i in usb]
        if serial is None:
            if len(serials) > 1:
                raise RuntimeError(f"Multiple Orbbec cameras connected {serials}; set the camera serial")
            return device_list.get_device_by_index(usb[0])
        if serial not in serials:
            raise RuntimeError(f"Orbbec camera {serial} not found; connected cameras: {serials}")
        return device_list.get_device_by_serial_number(serial)

    def _select_profile(self, sensor, fmt, width: int, height: int, fps: int):
        try:
            return self._pipeline.get_stream_profile_list(sensor).get_video_stream_profile(width, height, fmt, fps)
        except self._ob.OBError as exc:
            raise ValueError(f"Orbbec {sensor} offers no {width}x{height}@{fps} {fmt} stream") from exc

    def _read_intrinsics(self) -> OrbbecIntrinsics:
        color, ir_left, ir_right = (self._profiles[name] for name in ("color", "ir_left", "ir_right"))
        baseline = np.linalg.norm(np.asarray(ir_left.get_extrinsic_to(ir_right).transform)) / 1000.0
        return OrbbecIntrinsics(
            K_color=_k_matrix(color.get_intrinsic()),
            K_ir=_k_matrix(ir_left.get_intrinsic()),
            baseline_ir=float(baseline),
            T_color_from_ir=_transform_from_extrinsic(ir_left.get_extrinsic_to(color)),
            distortion_color=_distortion_vector(color.get_distortion()),
        )

    def get_intrinsics(self) -> OrbbecIntrinsics:
        return self._intrinsics

    def read_camera(self, max_wait_s: float = 5.0) -> OrbbecFrame:
        """Return the first complete frame set received after this call, never an older queued one."""
        requested_at_us = time.time_ns() // 1000
        deadline = time.monotonic() + max_wait_s
        while time.monotonic() < deadline:
            frames = self._pipeline.wait_for_frames(self._timeout_ms)
            if frames is None:
                continue
            color, ir_left, ir_right = frames.get_color_frame(), frames.get_left_ir_frame(), frames.get_right_ir_frame()
            if color is None or ir_left is None or ir_right is None:
                continue
            if color.get_system_timestamp_us() < requested_at_us:
                continue
            bgr = cv2.imdecode(np.frombuffer(color.get_data(), np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            depth = None
            if self._enable_depth:
                aligned = self._align.process(frames).as_frame_set().get_depth_frame()
                if aligned is None:
                    continue
                depth = _depth_to_metres(_frame_pixels(aligned, np.uint16), float(aligned.get_depth_scale()))

            if self._undistort_maps is not None:
                rgb = cv2.remap(rgb, *self._undistort_maps, cv2.INTER_LINEAR)
                if depth is not None:
                    depth = cv2.remap(depth, *self._undistort_maps, cv2.INTER_NEAREST)
            if depth is not None and depth.shape != rgb.shape[:2]:
                raise RuntimeError(f"Aligned depth {depth.shape} does not match colour image {rgb.shape[:2]}")

            return OrbbecFrame(
                serial=self.serial,
                timestamp=color.get_timestamp_us() / 1e6,
                rgb=rgb,
                intrinsics=self._intrinsics.K_color,
                depth=depth,
                ir_left=_frame_pixels(ir_left, np.uint8),
                ir_right=_frame_pixels(ir_right, np.uint8),
            )
        raise RuntimeError(f"Orbbec camera {self.serial}: no complete fresh frame within {max_wait_s:.1f}s")

    def close(self):
        """Stop the camera pipeline."""
        self._pipeline.stop()


async def orbbec_infer_depth_async(
    session: aiohttp.ClientSession,
    frame: OrbbecFrame,
    intrinsics: OrbbecIntrinsics,
) -> Float[np.ndarray, "h w"]:
    """Estimate depth on the colour pixel grid with FoundationStereo, using the same pipeline as RealSense."""
    return await rs_infer_depth_async(session, frame, intrinsics)  # type: ignore[arg-type]
