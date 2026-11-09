"""Fit M1013 collision spheres without removing self-collision pairs.

Run from the project root. Every subdivided collision-mesh triangle is wholly
contained in one fitted sphere, with 1 mm padding. Sparse link covers retain
local base/lower-arm clearance.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.cluster.vq import kmeans2, vq
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation
import trimesh
from yourdfpy import URDF
import yaml

ASSETS = Path(__file__).resolve().parents[1] / 'tiptop/doosan/assets/m1013_uon'
PADDING = .001
GRIPPER_MAX_EDGE = .01

# Sparse link models; refine only the base/lower-arm clearance region below.
COMPACT_COUNTS = {'base_link': 5, 'link_1': 3, 'link_2': 10, 'link_3': 3,
                  'link_4': 8, 'link_5': 3, 'link_6': 1, 'gripper_body': 10,
                  'gripper_spinner': 4, 'gripper_finger1': 12,
                  'gripper_finger2': 12, 'gripper_finger3': 12}


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


def fit(assets):
    """Sparse link cover with local half-plane constraints at the base joint.

    Each assigned triangle is enclosed by one sphere with 1 mm padding. Unlike
    uniform fine subdivision, only clusters violating the base-joint clearance
    planes are split. No self-collision pair is removed.
    """
    config = yaml.safe_load((assets / 'm1013_uon.yml').read_text())
    spheres = config['robot_cfg']['kinematics']['collision_spheres']
    meshes = {link: load_collision_mesh(assets, link) for link in COMPACT_COUNTS}
    urdf = URDF.load(assets / 'm1013_uon.urdf', load_meshes=False)
    # Original meshes have >16 mm vertical AABB clearance at these reference
    # poses. Allow at most 7 mm sphere protrusion per side, retaining >2 mm gap.
    normals, limits = [], []
    for q2 in [-60, -30, 0, 30, 60]:
        urdf.update_cfg({'joint_2': np.radians(q2)})
        transform = urdf.get_transform('link_2', 'base_link')
        points = trimesh.transform_points(meshes['link_2'].vertices, transform)
        if points[:, 2].min() - meshes['base_link'].bounds[1, 2] <= .014:
            raise ValueError('Mesh geometry no longer supports the base-joint clearance planes')
        normals.append(np.r_[-transform[2, :3], 1.])
        limits.append(transform[2, 3] - points[:, 2].min() + .007)
    rng = np.random.default_rng(2026)
    report = {}
    for link, count in COMPACT_COUNTS.items():
        mesh = meshes[link]
        constrained = link in {'base_link', 'link_2'}
        if constrained:
            vertices, faces = trimesh.remesh.subdivide_to_size(
                mesh.vertices, mesh.faces, max_edge=.02, max_iter=12)
            triangles = vertices[faces]
            centers = triangles.mean(axis=1)
            count = 12 if link == 'base_link' else 16
            seeds = centers[np.linspace(0, len(centers) - 1, count, dtype=int)]
            _, assignment = kmeans2(centers, seeds, iter=20, minit='matrix', missing='raise')
            planes = np.array([[0., 0., 1., 1.]]) if link == 'base_link' else np.array(normals)
            bounds = np.array([mesh.bounds[1, 2] + .007]) if link == 'base_link' else np.array(limits)
        else:
            triangles = mesh.triangles
            if link.startswith('gripper_'):
                vertices, faces = trimesh.remesh.subdivide_to_size(
                    mesh.vertices, mesh.faces, max_edge=GRIPPER_MAX_EDGE, max_iter=12)
                triangles = vertices[faces]
            centers = triangles.mean(axis=1)
            areas = trimesh.triangles.area(triangles)
            samples = rng.choice(len(centers), size=min(4096, len(centers)),
                                 p=areas / areas.sum(), replace=True)
            seeds, _ = kmeans2(centers[samples], count, iter=30, minit='++', seed=2026, missing='raise')
            assignment, _ = vq(centers, seeds)
        fitted, excesses = [], []

        def cover(indices):
            points = np.unique(triangles[indices].reshape(-1, 3), axis=0)
            center = (points.min(axis=0) + points.max(axis=0)) / 2
            radius = np.linalg.norm(points - center, axis=1).max() + PADDING
            if constrained:
                try:
                    support = points[ConvexHull(points, qhull_options='QJ').vertices]
                except QhullError:
                    support = points

                def enclosure(x):
                    return x[3] - PADDING - np.linalg.norm(support - x[:3], axis=1)

                def enclosure_jac(x):
                    delta = support - x[:3]
                    return np.c_[delta / np.maximum(np.linalg.norm(delta, axis=1)[:, None], 1e-12),
                                 np.ones(len(delta))]

                solution = minimize(lambda x: x[3], np.r_[center, radius],
                                    jac=lambda x: np.array([0., 0., 0., 1.]), method='SLSQP',
                                    constraints=[{'type': 'ineq', 'fun': enclosure, 'jac': enclosure_jac},
                                                 {'type': 'ineq', 'fun': lambda x: bounds - planes @ x,
                                                  'jac': lambda x: -planes}],
                                    bounds=[(None, None)] * 3 + [(.001, .12)],
                                    options={'maxiter': 100, 'ftol': 1e-10})
                center = solution.x[:3]
                # Recheck every original point, regardless of solver status or hull reduction.
                radius = np.linalg.norm(points - center, axis=1).max() + PADDING
                if radius > .120001 or np.max(planes @ np.r_[center, radius] - bounds) > 1e-7:
                    if len(indices) <= 1:
                        raise ValueError(f'{link}: cannot cover a triangle within clearance planes')
                    axis = np.ptp(centers[indices], axis=0).argmax()
                    ordered = indices[np.argsort(centers[indices, axis], kind='stable')]
                    for child in np.array_split(ordered, 2):
                        cover(child)
                    return
            fitted.append({'center': center.tolist(), 'radius': float(radius)})
            excesses.append(float((np.linalg.norm(points - center, axis=1) - radius).max()))

        for cluster in range(count):
            indices = np.flatnonzero(assignment == cluster)
            if len(indices):
                cover(indices)
        report[link] = {'old_spheres': len(spheres[link]), 'new_spheres': len(fitted),
                        'triangles_covered': len(triangles), 'max_face_vertex_excess_m': max(excesses)}
        spheres[link] = fitted
    report['total_spheres_including_attachment'] = sum(map(len, spheres.values())) + 4
    return config, report


def gripper_spheres_from_config(config, assets):
    """Use the same sparse gripper geometry in the cuTAMP tool-frame sphere file."""
    urdf = URDF.load(assets / 'm1013_uon.urdf', load_meshes=False)
    rows = []
    for link, values in config['robot_cfg']['kinematics']['collision_spheres'].items():
        if link.startswith('gripper_'):
            transform = urdf.get_transform(link, 'gripper_tcp')
            for sphere in values:
                center = transform[:3, :3] @ sphere['center'] + transform[:3, 3]
                rows.append([*center, sphere['radius']])
    return np.asarray(rows, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='update the planning YAML')
    parser.add_argument('--output', type=Path, help='save a candidate YAML without replacing the active model')
    args = parser.parse_args()
    if args.output and not args.output.resolve().is_relative_to(ASSETS.parents[3]):
        parser.error('--output must stay inside the TiPToP repository')
    config, report = fit(ASSETS)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(yaml.safe_dump(config, sort_keys=False))
    if args.write:
        (ASSETS / 'm1013_uon.yml').write_text(yaml.safe_dump(config, sort_keys=False))
        np.save(ASSETS / 'gripper_spheres_open.npy', gripper_spheres_from_config(config, ASSETS))
    output = Path('tiptop_outputs/m1013/collision-sphere-coverage.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
