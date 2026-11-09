"""CPU screening of open-gripper spheres against a convex CAD hull."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CadGraspScreening:
    indices: list[int] = field(default_factory=list)
    available: int = 0
    checked: int = 0
    distance_queries: int = 0
    distance_points: int = 0

    @property
    def complete(self):
        return self.checked == self.available


def screen_cad_grasps(tcp, world_from_cad, tool, candidates, hull, *, limit=48, minimum_gap_m=0.0005):
    """Return the first `limit` clear grasps in the supplied score order.

    A convex hull lies behind each outward-facing plane. A sphere separated
    from any one plane by its radius plus the required gap therefore clears
    the whole hull. This is only a sufficient acceptance test: edge/corner
    cases still use the original point-to-triangle distance calculation.
    Centers inside the hull use the original 1 nm half-space tolerance.
    """
    import trimesh

    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("CAD candidate limit must be a positive integer")
    if not np.isfinite(minimum_gap_m) or minimum_gap_m <= 0:
        raise ValueError("CAD clearance must be finite and positive")
    result = CadGraspScreening(available=len(candidates))
    inverse = np.linalg.inv(world_from_cad)
    normals = hull.face_normals
    face_centers = hull.triangles_center
    normal_lengths = np.linalg.norm(normals, axis=1)
    if not np.isfinite(normal_lengths).all() or np.any(normal_lengths <= 0):
        raise ValueError("CAD hull must have finite nonzero face normals")
    radii = tool[:, 3]
    for index in candidates:
        result.checked += 1
        transform = inverse @ tcp[index]
        local = tool[:, :3] @ transform[:3, :3].T + transform[:3, 3]
        signed = np.einsum("sfj,fj->sf", local[:, None] - face_centers[None], normals)
        if (signed <= 1e-9).all(axis=1).any():
            continue

        # Plane separation is a lower bound on distance to the convex hull.
        # Near the threshold, defer to the original distance calculation.
        plane_distance = (signed / normal_lengths).max(axis=1)
        uncertain = plane_distance - radii < minimum_gap_m + 1e-9
        if uncertain.any():
            result.distance_queries += 1
            result.distance_points += int(uncertain.sum())
            _, distances, _ = trimesh.proximity.closest_point_naive(hull, local[uncertain])
            if np.any(distances - radii[uncertain] < minimum_gap_m):
                continue

        result.indices.append(int(index))
        if len(result.indices) == limit:
            break
    return result
