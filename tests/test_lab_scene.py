"""Lab collision geometry retains USD transforms without duplicate camera boxes."""

from copy import deepcopy

import numpy as np
import pytest

from simulation.isaac_sim.lab_scene import LAB_HARDWARE_PROFILE, box_from_bounds, lab_preparation
from tiptop.doosan.virtual_camera import virtual_camera_obstacles


def test_oriented_box_contains_scaled_mesh_and_retains_its_world_centre():
    a = np.deg2rad(45)
    rotation = np.array([[1,0,0], [0,np.cos(a),-np.sin(a)], [0,np.sin(a),np.cos(a)]])
    matrix = np.eye(4)
    matrix[:3,:3] = rotation @ np.diag([.08,.08,.47])
    matrix[:3,3] = [0,1,1.1]
    lower, upper = np.array([-.5,-.5,-.4]), np.array([.5,.5,.5])
    row = box_from_bounds('brace', lower, upper, matrix)
    transform = np.array(row['base_from_obstacle'])
    np.testing.assert_allclose(row['dims_m'], [.08,.08,.423])
    np.testing.assert_allclose(transform[:3,:3], rotation)
    points = np.array([[x,y,z,1] for x in (lower[0],upper[0])
                       for y in (lower[1],upper[1]) for z in (lower[2],upper[2])])
    local = (np.linalg.inv(transform) @ matrix @ points.T).T[:,:3]
    np.testing.assert_allclose(np.max(np.abs(local),axis=0),np.array(row['dims_m'])/2)


def test_scene_camera_geometry_is_not_duplicated_at_the_optical_origin():
    environment, camera, _ = shared_lab_fixture()
    prepared = lab_preparation(environment, camera)
    prepared['profile']['collision_geometry_in_fixed_obstacles'] = True
    assert virtual_camera_obstacles(prepared) == prepared['fixed_obstacles']


def shared_lab_fixture():
    from types import SimpleNamespace
    from tiptop.doosan.hardware import load_hardware_profile

    hardware = load_hardware_profile(LAB_HARDWARE_PROFILE)
    workspace = hardware['workspace']
    table = next(row for row in workspace['fixed_obstacles'] if row['name'] == 'full_table')
    centre = np.asarray(table['base_from_obstacle'])[:3, 3]
    table_bounds = centre + np.array([-.5, .5])[:, None] * table['dims_m']
    environment = dict(
        fixed_obstacles=deepcopy([row for row in workspace['fixed_obstacles'] if row['name'] != 'full_table']),
        table_bounds=table_bounds, q_capture=[0.]*6, source='lab-test.usda',
        pick_place_bounds_xy_m=workspace['pick_place_bounds_xy_m'],
    )
    for name in ('body', 'bracket_left', 'bracket_right', 'bracket_top'):
        environment['fixed_obstacles'].append(dict(name='lab_gemini336l_mount_'+name,
            dims_m=[.01]*3, base_from_obstacle=np.eye(4).tolist()))
    optical = np.asarray(hardware['_camera_calibration']['parent_from_camera']) @ np.diag([-1.,-1.,1.,1.])
    camera = SimpleNamespace(width=1280, height=720, intrinsics=np.diag([900.,900.,1.]),
        specs={'RGB180': {'D': np.zeros(8)}},
        factory={'ir_right_from_ir_left': [[1,0,0,-.0953565],[0,1,0,0],[0,0,1,0],[0,0,0,1]]},
        world_from_cam=lambda: optical)
    return environment, camera, hardware


def test_lab_matches_physical_db_alignment_and_camera_collision_envelopes():
    from tiptop.camera_geometry import fixed_camera_envelopes

    environment, camera, hardware = shared_lab_fixture()
    result = lab_preparation(environment, camera)
    assert result['db_grasp_from_tcp'] == hardware['tool']['db_grasp_from_tcp']
    assert len(result['fixed_obstacles']) == 15
    assert virtual_camera_obstacles(result)[-2:] == fixed_camera_envelopes(hardware['camera_profile'])
    assert result['profile']['intrinsics'] == camera.intrinsics.tolist()
    # Collision rig remains the body frame; it is never replaced with RGB180.
    assert result['profile']['parent_from_rig'] == hardware['camera_profile']['parent_from_rig']
    assert not np.allclose(result['profile']['parent_from_rig'], camera.world_from_cam())
    assert len(environment['fixed_obstacles']) == 19  # Input is preserved.


@pytest.mark.parametrize('mismatch', ['camera', 'structure', 'table', 'work area'])
def test_lab_rejects_a_scene_that_drifted_from_physical_geometry(mismatch):
    environment, camera, _ = shared_lab_fixture()
    if mismatch == 'camera':
        wrong = camera.world_from_cam().copy()
        wrong[0,3] += .01
        camera.world_from_cam = lambda: wrong
    elif mismatch == 'structure':
        environment['fixed_obstacles'][0]['dims_m'][0] += .01
    elif mismatch == 'table':
        environment['table_bounds'][1,2] += .01
    else:
        environment['pick_place_bounds_xy_m'] = [[0,0],[1,1]]
    with pytest.raises(ValueError, match=mismatch):
        lab_preparation(environment, camera)


def test_usd_spinner_alignment_keeps_joint_anchors_coincident_and_body_unchanged():
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    from simulation.isaac_sim.m1013_uon import GRIPPER, GRIPPER_ASSET, align_gripper_spinner

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World/Gripper').GetPrim().GetReferences().AddReference(str(GRIPPER_ASSET))
    cache = UsdGeom.XformCache()
    before = {name: np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(GRIPPER+'/'+name)))
              for name in ('body', 'spinner', 'finger1', 'finger2', 'finger3')}
    align_gripper_spinner(stage)
    cache.Clear()
    rotation = np.diag([-1.,-1.,1.,1.])
    for name, original in before.items():
        actual = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(GRIPPER+'/'+name)))
        np.testing.assert_allclose(actual, original if name == 'body' else original @ rotation, atol=1e-7)
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint) or prim.GetName() == 'FixedJoint':
            continue
        joint = UsdPhysics.Joint(prim)
        anchors = []
        for targets, pos, rot in (
            (joint.GetBody0Rel(), joint.GetLocalPos0Attr(), joint.GetLocalRot0Attr()),
            (joint.GetBody1Rel(), joint.GetLocalPos1Attr(), joint.GetLocalRot1Attr()),
        ):
            anchor = Gf.Matrix4d().SetRotate(Gf.Quatd(rot.Get()))
            anchor.SetTranslateOnly(Gf.Vec3d(pos.Get()))
            anchors.append(np.asarray(anchor * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(targets.GetTargets()[0]))))
        np.testing.assert_allclose(*anchors, atol=1e-7)


def test_fixed_base_contact_filter_preserves_moving_links_objects_and_original_spheres():
    import torch
    from types import SimpleNamespace
    from tiptop.doosan.fixed_base_contact import filter_fixed_base_world_cost

    received = []
    config = SimpleNamespace(total_spheres=5,
        get_sphere_index_from_link_name=lambda _: torch.tensor([0,1]))
    cost = SimpleNamespace(forward=lambda spheres: received.append(spheres.clone()))
    filter_fixed_base_world_cost(cost, SimpleNamespace(kinematics_config=config))
    robot = torch.ones(1,2,5,4)
    cost.forward(robot)
    assert (robot == 1).all()
    assert (received[-1][...,:2,3] == -1).all()
    assert (received[-1][...,2:,:] == 1).all()
    obj = torch.ones(1,2,3,4)
    cost.forward(obj)
    assert torch.equal(received[-1], obj)


def test_pick_place_area_clips_only_endpoints_and_keeps_full_object_inside():
    from tiptop.perception.placement import restrict_pick_place_area

    table = np.array([[-.39,-1.25,-.03],[.39,1.25,0.]])
    area = np.array([[-.365,.470],[.365,1.225]])
    vertices = np.array([[.1,.6,0],[.2,.7,.1]])
    clipped = restrict_pick_place_area(table,vertices,area)
    np.testing.assert_allclose(clipped[:,:2],area)
    np.testing.assert_allclose(clipped[:,2],table[:,2])
    np.testing.assert_allclose(table[0],[-.39,-1.25,-.03])
    # Centre inside is insufficient: the full CAD footprint must fit.
    with pytest.raises(ValueError,match='footprint'):
        restrict_pick_place_area(table,np.array([[.35,.6,0],[.38,.7,.1]]),area)
    with pytest.raises(ValueError,match='footprint'):
        restrict_pick_place_area(table,np.array([[.1,.2,0],[.2,.3,.1]]),area)
