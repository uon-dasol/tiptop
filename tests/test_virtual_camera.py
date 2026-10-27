"""Fixed and wrist fixture geometry agree across observations and planning."""

import json
import subprocess
import sys
from copy import deepcopy

import numpy as np
import pytest

from tiptop.doosan.virtual_camera import (
    DEFAULT_PREPARATION,
    camera_planning_options,
    camera_preparation_id,
    load_camera_preparation,
    resolve_camera_preparation,
    validate_virtual_camera_plan,
    virtual_camera_obstacles,
)
from tiptop.perception.cameras.geometry import camera_world_pose


FIXED = DEFAULT_PREPARATION.with_name("fixed_camera.json")


def test_runtime_configuration_imports_without_camera_or_gpu_libraries():
    subprocess.run(
        [sys.executable, "-c", "import sys; import tiptop.doosan.virtual_camera; "
         "assert not {'torch', 'aiohttp', 'cv2', 'isaacsim'}.intersection(sys.modules)"],
        check=True,
    )


def test_legacy_wrist_keeps_its_mount_and_has_no_fixed_camera_boxes():
    original = json.loads(DEFAULT_PREPARATION.read_text())
    prepared = load_camera_preparation(DEFAULT_PREPARATION)
    assert "mount" not in original["profile"]
    assert prepared["profile"]["mount"] == "wrist"
    assert prepared["q_capture"] == original["q_capture"]
    assert virtual_camera_obstacles(prepared) == []
    options = camera_planning_options(prepared)
    assert options["camera_profile"] == prepared["profile"]
    assert options["fixed_obstacles"] == []
    validate_virtual_camera_plan({}, prepared)  # Existing wrist plans remain usable.
    with pytest.raises(ValueError, match="wrist camera preparation"):
        load_camera_preparation({key: value for key, value in original.items() if key != "success"})


def test_fixed_view_covers_table_and_is_independent_of_robot_motion():
    prepared = load_camera_preparation(FIXED)
    profile = prepared["profile"]
    transform = np.asarray(profile["parent_from_rig"], float)
    np.testing.assert_allclose(transform[:3, 2], [0, 0, -1])
    for x in [.25, 1.05]:
        for y in [-.4, .4]:
            # The actual perception image is the left stereo optical frame.
            left = transform.copy()
            left[:3, 3] -= transform[:3, 0] * profile["baseline_m"] / 2
            point = left[:3, :3].T @ (np.array([x, y, .4]) - left[:3, 3])
            half_width = point[2] * np.tan(np.deg2rad(profile["horizontal_fov_deg"]) / 2)
            half_height = half_width * profile["resolution_wh"][1] / profile["resolution_wh"][0]
            assert point[2] > 0 and abs(point[0]) < half_width and abs(point[1]) < half_height
    for translation in [[0, 0, 0], [1, 2, 3]]:
        fk = np.eye(4)
        fk[:3, 3] = translation
        np.testing.assert_array_equal(camera_world_pose(transform, mount="fixed", world_from_ee=fk), transform)


def test_fixed_camera_boxes_are_world_obstacles_and_do_not_attach_to_arm():
    prepared = load_camera_preparation(FIXED)
    obstacle = dict(name="stand", dims_m=[.05, .05, 1], base_from_obstacle=np.eye(4).tolist())
    prepared["fixed_obstacles"] = [obstacle]
    boxes = virtual_camera_obstacles(prepared)
    assert [b["name"] for b in boxes] == ["stand", "fixed_camera_body", "fixed_camera_bracket"]
    np.testing.assert_allclose(np.asarray(boxes[1]["base_from_obstacle"])[:3, 3], [.65, 0, 1.21385])
    np.testing.assert_allclose(np.asarray(boxes[2]["base_from_obstacle"])[:3, 3], [.59, 0, 1.23])
    options = camera_planning_options(prepared)
    assert options["camera_profile"] is None  # No wrist spheres are added in fixed mode.
    assert len(options["fixed_obstacles"]) == 3
    for actual, box in zip(options["fixed_obstacles"], boxes):
        np.testing.assert_allclose(actual.pose[:3], np.asarray(box["base_from_obstacle"])[:3, 3])
        np.testing.assert_allclose(actual.dims, box["dims_m"])
    assert prepared["fixed_obstacles"] == [obstacle]


def test_live_camera_geometry_is_authoritative_and_conflicting_cli_file_is_rejected():
    prepared = load_camera_preparation(FIXED)
    packet = dict(camera_preparation=np.array(json.dumps(prepared)))
    assert resolve_camera_preparation(packet) == prepared
    assert resolve_camera_preparation(packet, FIXED) == prepared
    with pytest.raises(ValueError, match="differs"):
        resolve_camera_preparation(packet, DEFAULT_PREPARATION)
    assert resolve_camera_preparation({})["profile"]["mount"] == "wrist"


def test_fixed_runtime_rejects_plans_for_another_mount_or_changed_structure():
    prepared = load_camera_preparation(FIXED)
    plan = dict(camera_preparation_sha256=camera_preparation_id(prepared))
    validate_virtual_camera_plan(plan, prepared)
    for invalid in [{}, dict(camera_preparation_sha256=camera_preparation_id(load_camera_preparation(DEFAULT_PREPARATION)))]:
        with pytest.raises(ValueError, match="geometry"):
            validate_virtual_camera_plan(invalid, prepared)
    changed = deepcopy(prepared)
    changed["profile"]["parent_from_rig"][0][3] += .01
    with pytest.raises(ValueError, match="geometry"):
        validate_virtual_camera_plan(plan, changed)


@pytest.mark.parametrize("change", [
    {"parent_link": "link_6"}, {"mount": "unknown"}, {"baseline_m": 0},
    {"horizontal_fov_deg": 180}, {"resolution_wh": [640, 0]}, {"body_dims_m": [0, .1, .1]},
])
def test_bad_fixed_geometry_fails_before_starting_runtime(change):
    prepared = load_camera_preparation(FIXED)
    prepared["profile"].update(change)
    with pytest.raises(ValueError):
        load_camera_preparation(prepared)


def test_fixed_structure_cannot_override_another_usd_path():
    prepared = load_camera_preparation(FIXED)
    prepared["fixed_obstacles"] = [dict(name="../Arm", dims_m=[1, 1, 1], base_from_obstacle=np.eye(4).tolist())]
    with pytest.raises(ValueError, match="names"):
        load_camera_preparation(prepared)
