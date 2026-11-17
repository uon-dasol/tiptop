"""Task choices independent of the perception backend; positions are in base_link metres."""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

TASK_PRESETS_PATH = Path(__file__).parent / "config/task_presets.json"
TASK_PRESETS = json.loads(TASK_PRESETS_PATH.read_text())
FIXED_BOX_NAME = "preset_fixed_box"
_BOX = TASK_PRESETS["fixed_box"]


@dataclass(frozen=True)
class TaskSettings:
    preset: str = "in_place_return"
    release_height_m: float | None = None
    fixed_box_present: bool = _BOX.get("present", False)
    box_center_x_m: float = _BOX["box_center_x_m"]
    box_center_y_m: float = _BOX["box_center_y_m"]
    box_bottom_z_m: float = _BOX["box_bottom_z_m"]
    box_size_x_m: float = _BOX["box_size_x_m"]
    box_size_y_m: float = _BOX["box_size_y_m"]
    box_size_z_m: float = _BOX["box_size_z_m"]

    def __post_init__(self):
        if self.preset not in TASK_PRESETS:
            raise ValueError(f"Unknown task preset: {self.preset}")
        if self.release_height_m is None:
            object.__setattr__(self, "release_height_m", TASK_PRESETS[self.preset]["release_height_m"])
        if not isinstance(self.fixed_box_present, bool):
            raise ValueError("fixed_box_present must be a boolean")
        # Installed geometry belongs to the workspace, regardless of the task.
        # Old UI payloads cannot disable a box declared present in configuration.
        if self.pick_only or _BOX.get("present", False):
            object.__setattr__(self, "fixed_box_present", True)
        values = [v for k, v in asdict(self).items() if k != "preset"]
        if not np.isfinite(values).all() or self.release_height_m < 0:
            raise ValueError("Task positions and release height must be finite; release height must not be negative")
        if min(self.box_size_x_m, self.box_size_y_m, self.box_size_z_m) <= 0:
            raise ValueError("Box dimensions must be positive metres")

    @property
    def pick_only(self):
        """Presets whose destination is fixed, so perception only has to find the object to pick."""
        return self.preset in ("fixed_box", "home_drop")

    @property
    def holding_goal(self):
        """Presets planned as a single Pick; the fixed destination is handled after the grasp."""
        return self.preset in ("in_place_return", "home_drop")

    def validate_perception_mode(self, mode):
        allowed = TASK_PRESETS[self.preset].get("perception_modes")
        if allowed and mode not in allowed:
            raise ValueError("인식한 목적지에 놓기는 Gemini 인식이 필요합니다. SAM3에서는 고정 박스를 선택하세요.")

    def perception_instruction(self, instruction):
        """Fixed destinations never ask the vision backend to find a box."""
        if not self.pick_only:
            return instruction
        match = re.fullmatch(r"\s*(.+?)\s+집(?:고|어서)\s*.+", instruction)
        if match is None:
            raise ValueError(f"{TASK_PRESETS[self.preset]['label']} 지시문: <물체명> {TASK_PRESETS[self.preset]['action']}")
        return match[1] + " 집어"

    def box_spec(self):
        size = [self.box_size_x_m, self.box_size_y_m, self.box_size_z_m]
        center = [self.box_center_x_m, self.box_center_y_m, self.box_bottom_z_m + size[2] / 2]
        return dict(name=FIXED_BOX_NAME, dims=size, pose=[*center, 1, 0, 0, 0])

    def box_parts(self):
        """Cuboids registered as the box: one solid block, or four walls when the held object is dropped inside."""
        solid = self.box_spec()
        if self.preset != "home_drop":
            return [solid]
        thickness = TASK_PRESETS["home_drop"]["box_wall_thickness_m"]
        (sx, sy, sz), (cx, cy, cz) = solid["dims"], solid["pose"][:3]
        walls = [("front", [sx, thickness, sz], [cx, cy - (sy - thickness) / 2, cz]),
                 ("back", [sx, thickness, sz], [cx, cy + (sy - thickness) / 2, cz]),
                 ("left", [thickness, sy - 2 * thickness, sz], [cx - (sx - thickness) / 2, cy, cz]),
                 ("right", [thickness, sy - 2 * thickness, sz], [cx + (sx - thickness) / 2, cy, cz])]
        return [dict(name=f"{FIXED_BOX_NAME}_wall_{side}", dims=dims, pose=[*center, 1, 0, 0, 0])
                for side, dims, center in walls]

    def drop_region(self):
        """Free footprint inside the walls and the box top, where a dropped object must end up."""
        region = self.box_region()
        thickness = TASK_PRESETS["home_drop"]["box_wall_thickness_m"]
        region["size_xy_m"] = [size - 2 * thickness for size in region["size_xy_m"]]
        return region

    def box_region(self):
        box = self.box_spec()
        top = self.box_bottom_z_m + self.box_size_z_m
        return dict(source="fixed_box", center_xy_m=box["pose"][:2], size_xy_m=box["dims"][:2],
                    support_z_m=top, box=asdict(self), position_source=_BOX["position_source"],
                    destination=box["name"], release_height_m=self.release_height_m,
                    release_bottom_z_m=top + self.release_height_m)


def task_defaults():
    """Flat defaults shared by the CLI and launcher, including editable box geometry."""
    return dict(task_preset="home_drop", fixed_box_present=_BOX.get("present", False), **{k: v for k, v in _BOX.items()
                                                if k.startswith("box_") or k == "release_height_m"})


def register_fixed_box(env, settings, fixed_obstacles):
    """Physical geometry can remain present while a different task is selected."""
    from curobo.geom.types import Cuboid

    box = Cuboid(**settings.box_spec())
    full_table = [item for item in fixed_obstacles if item.name == "configured_full_table"]
    if len(full_table) != 1:
        raise ValueError("Fixed box needs exactly one configured_full_table; a cropped pick ROI is not the destination table")
    table = full_table[0]
    lower = np.asarray(table.pose[:3]) - np.asarray(table.dims) / 2
    upper = np.asarray(table.pose[:3]) + np.asarray(table.dims) / 2
    center, half = np.asarray(box.pose[:3]), np.asarray(box.dims) / 2
    if (np.any(center[:2] - half[:2] < lower[:2]) or np.any(center[:2] + half[:2] > upper[:2])
            or abs(settings.box_bottom_z_m - upper[2]) > 1e-4):
        raise ValueError("Fixed box must fit on the configured full tabletop")
    previous = [item for item in [*env.movables, *env.statics] if item.name == FIXED_BOX_NAME]
    if any(not isinstance(item, Cuboid) or not np.allclose(item.pose, box.pose)
           or not np.allclose(item.dims, box.dims) for item in previous):
        raise ValueError("Fixed box name collides with another scene object")
    env.statics = [item for item in env.statics if not item.name.startswith(FIXED_BOX_NAME)] + [
        Cuboid(**part) for part in settings.box_parts()]
    return box


def prepare_task_environment(env, scene, settings, *, fixed_obstacles=()):
    """Register a fixed surface or preserve the perceived On goal. Never change the source ROI."""
    from copy import deepcopy

    from cutamp.tamp_domain import HandEmpty, Holding, On

    env = deepcopy(env)
    if len(scene.grasps) != 1:
        raise ValueError("A task preset requires exactly one picked object")
    name = next(iter(scene.grasps))
    box = register_fixed_box(env, settings, fixed_obstacles) if settings.fixed_box_present else None
    if settings.holding_goal:
        env.goal_state = frozenset({Holding.ground(name)})
        env.type_to_objects["Surface"] = [scene.table_cuboid]
        return env, scene.perception_results.get("placement_region")

    if settings.preset == "fixed_box":
        env.type_to_objects["Surface"] = [scene.table_cuboid, box]
        env.goal_state = frozenset({On.ground(name, box.name), HandEmpty.ground()})
        region = settings.box_region()
    else:
        goals = [goal for goal in env.goal_state if goal.name == "On"]
        if len(goals) != 1 or set(env.goal_state) != {goals[0], HandEmpty.ground()} or goals[0].values[0] != name:
            raise ValueError("인식한 목적지는 한 물체의 On(물체, 목적지) 목표가 필요합니다.")
        surface_name = goals[0].values[1]
        surfaces = [item for item in env.type_to_objects["Surface"] if item.name == surface_name]
        if len(surfaces) != 1:
            raise ValueError("Perceived On destination is missing from the surface/collision scene")
        if surface_name == scene.table_cuboid.name:
            # Perception lowers its table collision box by 20 mm. A Place goal
            # needs the measured support plane, unlike that pick-clearance box.
            from tiptop.perception.placement import TABLE_BOX_LOWERING_M

            surface = deepcopy(scene.table_cuboid)
            surface.pose[2] += TABLE_BOX_LOWERING_M
            env.statics = [surface if item.name == surface_name else item for item in env.statics]
            env.type_to_objects["Surface"] = [surface if item.name == surface_name else item
                                               for item in env.type_to_objects["Surface"]]
            surfaces = [surface]
        mesh = surfaces[0].get_trimesh_mesh(process=False)
        from scipy.spatial.transform import Rotation

        pose = np.asarray(surfaces[0].pose)
        rotation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
        vertices = np.asarray(mesh.vertices) @ rotation.T + pose[:3]
        low, high = vertices.min(0), vertices.max(0)
        region = dict(source="perceived_surface", center_xy_m=((low + high) / 2)[:2].tolist(),
                      size_xy_m=(high - low)[:2].tolist(), support_z_m=float(high[2]),
                      destination=surface_name)
    region.update(release_height_m=settings.release_height_m,
                  release_bottom_z_m=region["support_z_m"] + settings.release_height_m,
                  height_reference="object bottom above destination top; cuTAMP collision clearance also applies")
    from tiptop.perception.placement import table_support_z

    region["source_support_z_m"] = table_support_z(scene.table_cuboid.pose[2] + scene.table_cuboid.dims[2] / 2)
    scene.perception_results["placement_region"] = region
    return env, region
