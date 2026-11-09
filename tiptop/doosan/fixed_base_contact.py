"""Fixed robot mounting contact for lab planning, without changing self collision.

The stationary base is bolted to the tabletop. Its conservative spheres extend
below that mounting plane, so they cannot be treated as moving world geometry.
Only external collision queries omit base_link; all moving links and the full
self-collision model retain their original spheres. Object queries are unchanged.
"""


def check_fixed_base_clearance(kinematics, q, obstacles):
    """A fixed base needs one check against non-mounting static geometry."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    from curobo.types.base import TensorDeviceType

    indices = kinematics.kinematics_config.get_sphere_index_from_link_name("base_link")
    spheres = kinematics.get_state(TensorDeviceType().to_device(np.asarray(q)[None])).link_spheres_tensor[0]
    base = spheres[indices].detach().cpu().numpy()
    for box in obstacles:
        pose = np.asarray(box.pose)
        rotation = Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()
        points = (base[:,:3]-pose[:3]) @ rotation
        delta = np.abs(points)-np.asarray(box.dims)/2
        distance = np.linalg.norm(np.maximum(delta,0),axis=1)+np.minimum(delta.max(axis=1),0)
        if np.any((distance < base[:,3]) & (base[:,3]>0)):
            raise ValueError(f"Fixed robot base intersects non-mounting obstacle: {box.name}")


def filter_fixed_base_world_cost(cost, kinematics):
    config = kinematics.kinematics_config
    indices = config.get_sphere_index_from_link_name("base_link")
    count = config.total_spheres
    original = cost.forward

    def forward(spheres, *args, **kwargs):
        if spheres.shape[-2] == count:
            spheres = spheres.clone()
            spheres[..., indices, 3] = -1.
        return original(spheres, *args, **kwargs)

    cost.forward = forward
    return cost


def allow_fixed_base_mounting_contact(solver):
    """Apply to IK/MotionGen external costs only; leave self costs untouched."""
    seen = set()
    for rollout in solver.get_all_rollout_instances():
        model = rollout.dynamics_model.robot_model
        for name in ("primitive_collision_cost", "primitive_collision_constraint"):
            cost = getattr(rollout, name, None)
            if cost is not None and id(cost) not in seen:
                filter_fixed_base_world_cost(cost, model)
                seen.add(id(cost))
