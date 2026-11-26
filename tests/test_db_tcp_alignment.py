"""DB→TCP 정렬 검사의 좌표 규약: 손가락 연동과 접촉점 변환 방향."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from experiments.grasping import check_db_tcp_alignment as check

MESH = check.PROJECT_ROOT / "ros2_ws/src/description/uon_3f_description/meshes/finger1_01.stl"
pytestmark = pytest.mark.skipif(not MESH.exists(), reason="ros2_ws 서브모듈의 그리퍼 메쉬가 필요하다")


def test_fingers_turn_by_minus_spinner_angle_in_the_body_frame():
    spinner = 0.4
    before = check.gripper_fingers(0.0)["finger1"][0]
    after = check.gripper_fingers(spinner)["finger1"][0]
    far = int(np.argmax(np.linalg.norm(before[:, :2] - before[0, :2], axis=1)))
    a, b = before[far] - before[0], after[far] - after[0]
    # 스피너 +q와 손가락 연동 -2q가 합쳐져 몸체 기준으로 -q만큼 돈다.
    assert np.arctan2(b[1], b[0]) - np.arctan2(a[1], a[0]) == pytest.approx(-spinner, abs=1e-6)
    assert b[2] == pytest.approx(a[2], abs=1e-9)


def test_contacts_on_finger_surfaces_identify_the_declared_rotation():
    declared = check.rigid(Rotation.from_euler("z", -60, degrees=True).as_matrix())
    body_from_tcp = check.rigid(translation=(0.0, 0.0, 0.2368508))
    object_from_grasp = check.rigid(Rotation.from_euler("xyz", [0.3, -0.5, 1.1]).as_matrix(), (0.02, -0.01, 0.03))
    spinner = 0.7
    object_from_body = object_from_grasp @ declared @ np.linalg.inv(body_from_tcp)
    contacts = {name: check.transform(vertices[len(vertices) // 2], object_from_body)
                for name, (vertices, _) in check.gripper_fingers(spinner).items()}
    candidate = dict(spinner_rad=spinner, contacts=contacts, object_from_grasp=object_from_grasp)

    results = check.evaluate([candidate], body_from_tcp, check.hypotheses(declared))

    assert results["declared"]["same_finger"]["rms_mm"] < 0.01
    assert results["Rz(0°)"]["nearest_finger"]["rms_mm"] > 1.0
