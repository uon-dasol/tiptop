"""Refine the M1013 base/lower-arm surface cover without removing collision pairs.

Run from the project root. Every subdivided collision-mesh triangle is wholly
contained in one fitted sphere, with 1 mm padding. Other links are unchanged.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.cluster.vq import kmeans2, vq
from scipy.spatial.transform import Rotation
import trimesh
from yourdfpy import URDF
import yaml

ASSETS = Path(__file__).resolve().parents[1] / 'tiptop/doosan/assets/m1013_uon'
PADDING = .001


def load_collision_mesh(assets, link):
    root = ET.parse(assets / 'm1013_uon.urdf').getroot()
    pieces = []
    for collision in root.find(f"link[@name='{link}']").findall('collision'):
        element = collision.find('geometry/mesh')
        mesh = trimesh.load(assets / element.get('filename'), force='mesh', process=False)
        mesh.apply_scale(np.fromstring(element.get('scale', '1 1 1'), sep=' '))
        origin = collision.find('origin')
        transform = np.eye(4)
        if origin is not None:
            transform[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            transform[:3, :3] = Rotation.from_euler(
                'xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
        mesh.apply_transform(transform)
        pieces.append(mesh)
    return trimesh.util.concatenate(pieces)


def fit(assets, clusters_per_link=(12, 12), max_radius=.04):
    path = assets / 'm1013_uon.yml'
    config = yaml.safe_load(path.read_text())
    spheres = config['robot_cfg']['kinematics']['collision_spheres']
    urdf = URDF.load(assets / 'm1013_uon.urdf', load_meshes=False)
    report = {}
    meshes, all_triangles, groups = {}, {}, {}
    for link, clusters in zip(['base_link', 'link_2'], clusters_per_link):
        mesh = load_collision_mesh(assets, link)
        meshes[link] = mesh
        vertices, faces = trimesh.remesh.subdivide_to_size(
            mesh.vertices, mesh.faces, max_edge=.02, max_iter=12)
        triangles = vertices[faces]
        all_triangles[link] = triangles
        centers = triangles.mean(axis=1)
        seeds = centers[np.linspace(0, len(centers) - 1, clusters, dtype=int)]
        cluster_centers, _ = kmeans2(centers, seeds, iter=20, minit='matrix', missing='raise')
        assignments, _ = vq(centers, cluster_centers)
        transform = urdf.get_transform(link, 'base_link')
        fitted = []
        groups[link] = []
        max_excess = -float('inf')

        def cover(indices):
            nonlocal max_excess
            points = triangles[indices].reshape(-1, 3)
            center = (points.min(axis=0) + points.max(axis=0)) / 2
            radius = np.linalg.norm(points - center, axis=1).max() + PADDING
            # Refine the base everywhere and the lower part of link_2 next to it.
            base_z = points @ transform[2, :3] + transform[2, 3]
            refine = link == 'base_link' or base_z.min() < .22
            if refine and radius > max_radius and len(indices) > 1:
                axis = np.ptp(centers[indices], axis=0).argmax()
                ordered = indices[np.argsort(centers[indices, axis], kind='stable')]
                split = len(ordered) // 2
                cover(ordered[:split])
                cover(ordered[split:])
            else:
                fitted.append({'center': center.tolist(), 'radius': float(radius)})
                groups[link].append(indices)
                max_excess = max(max_excess, float(
                    (np.linalg.norm(points - center, axis=1) - radius).max()))

        for cluster in range(clusters):
            indices = np.flatnonzero(assignments == cluster)
            if len(indices):
                cover(indices)
        report[link] = {'old_spheres': len(spheres[link]), 'new_spheres': len(fitted),
                        'triangles_covered': len(triangles),
                        'max_face_vertex_excess_m': max_excess}
        spheres[link] = fitted
    # Only original-mesh AABB separation authorizes refinement. No link pair
    # is excluded; every original triangle remains wholly covered with padding.
    witnesses = []
    for q1 in [-180, -120, -60, -30, 0, 30, 60, 120, 180]:
        for q2 in [-60, -30, 0, 30, 60]:
            urdf.update_cfg({'joint_1': np.radians(q1), 'joint_2': np.radians(q2)})
            transform = urdf.get_transform('link_2', 'base_link').copy()
            points = trimesh.transform_points(meshes['link_2'].vertices, transform)
            bounds = np.array([points.min(axis=0), points.max(axis=0)])
            base_bounds = meshes['base_link'].bounds
            gap = np.maximum(bounds[0] - base_bounds[1], base_bounds[0] - bounds[1]).max()
            if gap > 2 * PADDING:
                witnesses.append(transform)
    refinements = 0
    while True:
        base, arm = [np.array([[*s['center'], s['radius']] for s in spheres[link]])
                     for link in ['base_link', 'link_2']]
        worst, pair = 0., None
        for transform in witnesses:
            moved = trimesh.transform_points(arm[:, :3], transform)
            depth = base[:, None, 3] + arm[None, :, 3] - np.linalg.norm(
                base[:, None, :3] - moved[None], axis=-1)
            i, j = np.unravel_index(depth.argmax(), depth.shape)
            if depth[i, j] > worst:
                worst, pair = float(depth[i, j]), (i, j)
        if worst <= 1e-7:
            break
        candidates = [(spheres[link][i]['radius'], link, i) for link, i in
                      zip(['base_link', 'link_2'], pair) if len(groups[link][i]) > 1]
        if not candidates:
            raise ValueError('Triangle resolution cannot preserve witnessed mesh clearance')
        _, link, i = max(candidates)
        indices = groups[link].pop(i)
        spheres[link].pop(i)
        triangles = all_triangles[link]
        centers = triangles[indices].mean(axis=1)
        axis = np.ptp(centers, axis=0).argmax()
        ordered = indices[np.argsort(centers[:, axis], kind='stable')]
        for child in np.array_split(ordered, 2):
            points = triangles[child].reshape(-1, 3)
            center = (points.min(axis=0) + points.max(axis=0)) / 2
            radius = np.linalg.norm(points - center, axis=1).max() + PADDING
            spheres[link].append({'center': center.tolist(), 'radius': float(radius)})
            groups[link].append(child)
        refinements += 1
        if sum(len(v) for v in spheres.values()) + 4 >= 1024:
            raise ValueError('Refinement exceeds current cuRobo sphere limit')
    for link in groups:
        excess = []
        for sphere, indices in zip(spheres[link], groups[link]):
            points = all_triangles[link][indices].reshape(-1, 3)
            excess.append(float((np.linalg.norm(points - sphere['center'], axis=1)
                                 - sphere['radius']).max()))
        report[link]['new_spheres'] = len(spheres[link])
        report[link]['max_face_vertex_excess_m'] = max(excess)
    report['mesh_separated_witness_poses'] = len(witnesses)
    report['local_refinements'] = refinements
    total = sum(len(values) for values in spheres.values()) + 4
    if total >= 1024:
        raise ValueError(f'{total} spheres exceeds the current cuRobo kernel limit: {report}')
    report['total_spheres_including_attachment'] = total
    return config, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='update the planning YAML')
    args = parser.parse_args()
    config, report = fit(ASSETS)
    if args.write:
        (ASSETS / 'm1013_uon.yml').write_text(yaml.safe_dump(config, sort_keys=False))
    output = Path('tiptop_outputs/m1013/collision-sphere-coverage.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
