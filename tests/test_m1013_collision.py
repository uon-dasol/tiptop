"""Collision-model regressions; no robot connection or motion commands."""
import numpy as np
import pytest


@pytest.fixture(scope='module')
def planner():
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from curobo.geom.types import Cuboid, WorldConfig
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
    from tiptop.doosan.model import m1013_uon_curobo_cfg
    world = WorldConfig(cuboid=[Cuboid(name='placeholder', dims=[.01] * 3,
                                      pose=[5, 5, 5, 1, 0, 0, 0])])
    motion = MotionGen(MotionGenConfig.load_from_robot_config(
        robot_cfg=m1013_uon_curobo_cfg(), world_model=world, use_cuda_graph=False,
        trajopt_particle_opt=False, collision_activation_distance=0.,
        position_threshold=.01, rotation_threshold=.1))
    return motion, world


def test_clear_reference_and_showcase_poses(planner):
    from curobo.types.base import TensorDeviceType
    from curobo.types.state import JointState
    motion, _ = planner
    q = np.radians([5, -5, 5, 2.5, 12.5, 2.5])
    for positions in [np.zeros(6), q, q + np.radians([60, -30, 40, 15, -15, 25])]:
        valid, reason = motion.check_start_state(JointState.from_position(
            TensorDeviceType().to_device(positions[None])))
        assert valid, reason


@pytest.mark.parametrize('angle', [0., 1.5009832], ids=['open', 'closed'])
def test_locked_gripper_link_poses_match_urdf(angle):
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig
    from yourdfpy import URDF
    from tiptop.doosan.model import ASSETS, m1013_uon_curobo_cfg
    tensor = TensorDeviceType()
    model = CudaRobotModel(RobotConfig.from_dict(m1013_uon_curobo_cfg(angle)['robot_cfg'], tensor).kinematics)
    q = np.array([.1, -.3, .5, .1, .2, -.1])
    urdf = URDF.load(str(ASSETS / 'm1013_uon.urdf'), load_meshes=False)
    urdf.update_cfg(dict(zip(model.joint_names, q)) | {'gripper_spinner_joint': angle})
    state = model.get_state(tensor.to_device(q[None]))
    for index, link in enumerate(model.link_names):
        np.testing.assert_allclose(state.links_position[0, index].cpu(),
                                   urdf.get_transform(link, 'base_link')[:3, 3], atol=2e-6)


def test_world_rejection_preserves_self_collision_check_and_clear_recovery(planner):
    from curobo.geom.types import Cuboid, WorldConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.state import JointState
    motion, empty = planner
    tensor = TensorDeviceType()
    zero = JointState.from_position(tensor.to_device([[0.] * 6]))
    kin = motion.kinematics
    index = int(kin.kinematics_config.get_sphere_index_from_link_name('gripper_finger1')[0])
    center = kin.get_state(zero.position).link_spheres_tensor[0, index, :3].cpu().tolist()
    try:
        motion.update_world(WorldConfig(cuboid=[Cuboid(
            name='deliberate_finger_box', dims=[.04] * 3, pose=[*center, 1, 0, 0, 0])]))
        valid, reason = motion.check_start_state(zero)
        assert not valid and reason.name == 'INVALID_START_STATE_WORLD_COLLISION'
        assert motion.rollout_fn.robot_self_collision_constraint.enabled
        motion.update_world(empty)
        folded = JointState.from_position(tensor.to_device(
            np.radians([[0, 0, 140, 0, 90, 0]])))
        valid, reason = motion.check_start_state(folded)
        assert not valid and reason.name == 'INVALID_START_STATE_SELF_COLLISION'
        valid, reason = motion.check_start_state(zero)
        assert valid, reason
    finally:
        motion.update_world(empty)


def test_refined_spheres_cover_entire_collision_mesh_triangles():
    import trimesh
    import yaml
    from scripts.fit_m1013_collision_spheres import ASSETS, GRIPPER_MAX_EDGE, load_collision_mesh
    spheres = yaml.safe_load((ASSETS / 'm1013_uon.yml').read_text())['robot_cfg']['kinematics']['collision_spheres']
    for link in spheres:
        mesh = load_collision_mesh(ASSETS, link)
        vertices, faces = trimesh.remesh.subdivide_to_size(
            mesh.vertices, mesh.faces, max_edge=GRIPPER_MAX_EDGE if link.startswith('gripper_') else .02, max_iter=12)
        centers = np.array([s['center'] for s in spheres[link]])
        radii = np.array([s['radius'] for s in spheres[link]])
        for start in range(0, len(faces), 512):
            triangles = vertices[faces[start:start + 512]]
            distances = np.linalg.norm(triangles[:, None] - centers[None, :, None], axis=-1)
            # All three vertices in one sphere imply its entire triangle is covered.
            covered = (distances <= radii[None, :, None] - .0005).all(axis=-1).any(axis=-1)
            assert covered.all(), f'{link}: uncovered collision-mesh triangle'


def test_compact_model_uses_experimental_kernel_with_object_capacity():
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from copy import deepcopy
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig
    from tiptop.doosan.model import m1013_uon_curobo_cfg
    cfg = deepcopy(m1013_uon_curobo_cfg()['robot_cfg'])
    cfg['kinematics']['extra_collision_spheres']['attached_object'] = 32
    model = CudaRobotModel(RobotConfig.from_dict(cfg, TensorDeviceType()).kinematics)
    assert model.get_self_collision_config().experimental_kernel


def test_tamp_gripper_spheres_match_planning_geometry():
    import yaml
    from scripts.fit_m1013_collision_spheres import ASSETS, gripper_spheres_from_config
    cfg = yaml.safe_load((ASSETS / 'm1013_uon.yml').read_text())
    np.testing.assert_allclose(np.load(ASSETS / 'gripper_spheres_open.npy'),
                               gripper_spheres_from_config(cfg, ASSETS), atol=1e-7)


def test_reference_mesh_clearance_and_folded_mesh_intersection():
    import open3d as o3d
    from yourdfpy import URDF
    from scripts.fit_m1013_collision_spheres import ASSETS, load_collision_mesh
    urdf = URDF.load(ASSETS / 'm1013_uon.urdf', load_meshes=False)

    def mesh_at_pose(link):
        mesh = load_collision_mesh(ASSETS, link)
        mesh.apply_transform(urdf.get_transform(link, 'base_link'))
        mesh.merge_vertices()
        return mesh

    base, arm = mesh_at_pose('base_link'), mesh_at_pose('link_2')
    assert arm.bounds[0, 2] - base.bounds[1, 2] > .016
    urdf.update_cfg(dict(zip([f'joint_{i}' for i in range(1, 7)],
                            np.radians([0, 0, 140, 0, 90, 0]))))
    body, finger = mesh_at_pose('link_1'), mesh_at_pose('gripper_finger3')
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(body.vertices.astype(np.float32)),
                        o3d.core.Tensor(body.faces.astype(np.uint32)))
    edges = finger.vertices[finger.edges_unique]
    rays = np.concatenate([edges[:, 0], edges[:, 1] - edges[:, 0]], axis=1).astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
    assert ((hit > 1e-5) & (hit < 1 - 1e-5)).any(), 'Expected actual mesh surface crossing'
