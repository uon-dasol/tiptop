"""Camera extrinsics shared by wrist and fixed observation sources.

Transforms use column vectors: T_A_B maps B coordinates into A. Camera axes must
already use the optical convention of Frame intrinsics (+Z depth, +Y down).
"""

from tiptop.camera_geometry import _rigid_transform, camera_world_pose

__all__ = ["_rigid_transform", "camera_world_pose"]
