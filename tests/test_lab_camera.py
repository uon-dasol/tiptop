"""SDK projections and both180-degree views retain identical scene geometry."""

import json

import cv2
import numpy as np
import pytest

from simulation.isaac_sim.lab_camera import FACTORY, camera_specs, align_depth180_to_rgb180


def project(body_points, spec):
    local = (np.linalg.inv(spec["body_from_optical"]) @ body_points.T).T[:, :3]
    uv, _ = cv2.projectPoints(local, np.zeros(3), np.zeros(3), spec["K"], spec["D"])
    return uv.reshape(-1, 2)


@pytest.mark.parametrize("raw,corrected", [("RGBRaw", "RGB180"), ("DepthRaw", "Depth180")])
def test_rotated_camera_matches_image_rotation_even_off_center_and_with_distortion(raw, corrected):
    specs = camera_specs(json.loads(FACTORY.read_text()))
    # Off-axis rays exercise the off-centre principal point and tangential distortion.
    points = np.array([[-.2, -.25, -1., 1], [.3, .2, -.8, 1], [.03, -.1, -1.2, 1]])
    original = project(points, specs[raw])
    rotated = project(points, specs[corrected])
    width, height = specs[raw]["resolution_wh"]
    np.testing.assert_allclose(rotated, [width-1, height-1] - original, atol=1e-7)


def test_native_sensors_keep_sdk_relative_geometry_and_positive_stereo_disparity():
    factory = json.loads(FACTORY.read_text())
    specs = camera_specs(factory)
    body_color = specs["RGBRaw"]["body_from_optical"]
    body_depth = specs["DepthRaw"]["body_from_optical"]
    np.testing.assert_allclose(np.linalg.inv(body_color) @ body_depth, factory["color_from_depth"], atol=1e-7)
    body_left = specs["IRLeft"]["body_from_optical"]
    body_right = specs["IRRight"]["body_from_optical"]
    np.testing.assert_allclose(np.linalg.inv(body_right) @ body_left, factory["ir_right_from_ir_left"], atol=1e-7)
    point = np.array([[0., 0., -1., 1.]])
    assert project(point, specs["IRLeft"])[0, 0] > project(point, specs["IRRight"])[0, 0]


def test_robot_side_is_below_in_raw_and_above_in_user_confirmed_view():
    specs = camera_specs(json.loads(FACTORY.read_text()))
    # Lab body is yaw180 at(0,.77,1.02); robot base is at the origin.
    world_from_body = np.diag([-1., -1., 1., 1.])
    world_from_body[:3, 3] = [0, .77, 1.02]
    body_point = (np.linalg.inv(world_from_body) @ np.array([0, 0, 0, 1])).reshape(1, 4)
    assert project(body_point, specs["RGBRaw"])[0, 1] > specs["RGBRaw"]["K"][1, 2]
    assert project(body_point, specs["RGB180"])[0, 1] < specs["RGB180"]["K"][1, 2]


def test_d2c_places_a_depth_sample_at_the_same_color_pixel_as_sdk_geometry():
    factory = json.loads(FACTORY.read_text())
    specs = camera_specs(factory)
    depth = np.zeros((480, 848), np.float32)
    depth[200, 420] = 1.1
    k = specs["Depth180"]["K"]
    p = np.array([(420-k[0, 2])/k[0, 0]*1.1, (200-k[1, 2])/k[1, 1]*1.1, 1.1, 1])
    body_point = (specs["Depth180"]["body_from_optical"] @ p).reshape(1, 4)
    expected = project(body_point, specs["RGB180"])[0]
    aligned = align_depth180_to_rgb180(depth, factory)
    y, x = np.nonzero(aligned)
    assert len(x) == 4
    np.testing.assert_allclose([x.mean(), y.mean()], np.floor(expected)+.5)
    assert aligned.shape == (720, 1280)
    assert np.isfinite(aligned).all()
