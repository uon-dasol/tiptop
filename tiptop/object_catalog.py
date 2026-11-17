"""Shared CAD/DB paths without importing perception models or device libraries."""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "tiptop/config/known_objects.json"
DEMO_OBJECTS = ROOT / "tiptop/config/demo_objects.json"


def demo_order(path=DEMO_OBJECTS):
    """Class ids shown in demo mode, mapped to their 1-based position in that list."""
    return {class_id: number for number, class_id in enumerate(json.loads(Path(path).read_text()), 1)}


def load_catalog(catalog_path=DEFAULT_CATALOG, *, asset_root=ROOT):
    catalog = json.loads(Path(catalog_path).read_text())
    ids = [entry["class_id"] for entry in catalog["objects"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate registered class IDs")
    seen = set(ids)
    for dataset in catalog.get("additional_datasets", []):
        scale = float(dataset["cad_scale_to_m"])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid dataset CAD scale")
        for entry in _discover_databases([dataset["root"]], asset_root):
            class_id = entry["class_id"]
            if class_id in seen:
                continue
            directory = entry["db"].parent.parent
            catalog["objects"].append(dict(
                class_id=class_id, name_ko=class_id.replace("_", " "),
                aliases=[class_id.replace("_", " ")],
                cad=str(directory / "edited" / f"{class_id}.obj"), cad_scale_to_m=scale,
                db=str(entry["db"]),
            ))
            seen.add(class_id)
    return catalog


def asset_path(catalog, entry, key, *, asset_root=ROOT):
    """Resolve CAD and DB against separate roots; retain legacy catalog support.

    A catalog without cad_root/db_root keeps project-relative paths. Explicit
    absolute paths also work. Missing external files never fall back to data/.
    """
    root_key = {"cad": "cad_root", "db": "db_root"}[key]
    root = Path(asset_root) / Path(catalog.get(root_key, "")).expanduser()
    return (root / Path(entry[key]).expanduser()).resolve()


def _discover_databases(roots, asset_root):
    for value in roots:
        root = (Path(asset_root) / Path(value).expanduser()).resolve()
        for directory in sorted(root.glob("*/UON_3finger_gripper")):
            class_id = directory.parent.name
            paths = [directory / f"{class_id}{suffix}_merged_grasp_zero_pose.json"
                     for suffix in ("", "_UON_3finger_gripper")]
            for path in paths:
                if path.is_file():
                    yield dict(class_id=class_id, db=path.resolve())
                    break


def _database_entries(catalog, asset_root):
    entries = {
        entry["class_id"]: dict(class_id=entry["class_id"], db=asset_path(catalog, entry, "db", asset_root=asset_root))
        for entry in catalog["objects"]
    }
    for entry in _discover_databases(catalog.get("additional_db_roots", []), asset_root):
        # Explicit registrations, then the first additional root, take precedence.
        entries.setdefault(entry["class_id"], entry)
    return entries


def database_entries(*, catalog_path=DEFAULT_CATALOG, asset_root=ROOT):
    """List registered and discovered DBs without requiring CAD files or a metadata CSV.

    Additional roots use <class>/UON_3finger_gripper/ and either merged JSON
    filename convention. Missing optional roots contribute no entries. Discovery
    under additional_db_roots does not register CADs. additional_datasets also
    registers each discovered DB's edited/<class>.obj with its configured scale.
    """
    return list(_database_entries(load_catalog(catalog_path, asset_root=asset_root), asset_root).values())


def registered_database(class_id, *, catalog_path=DEFAULT_CATALOG, asset_root=ROOT, check_files=True):
    """Resolve a grasp DB independently of the perception object's CAD registration."""
    entries = _database_entries(load_catalog(catalog_path, asset_root=asset_root), asset_root)
    if class_id not in entries:
        raise ValueError(f"Unregistered grasp database: {class_id}")
    entry = entries[class_id]
    if check_files and not entry["db"].is_file():
        raise FileNotFoundError(entry["db"])
    return entry


def registered_object(class_id, *, catalog_path=DEFAULT_CATALOG, asset_root=ROOT, check_files=True):
    catalog = load_catalog(catalog_path, asset_root=asset_root)
    matches = [entry for entry in catalog["objects"] if entry["class_id"] == class_id]
    if len(matches) != 1:
        raise ValueError(f"Unregistered class: {class_id}")
    entry = dict(matches[0])
    for key in ("cad", "db"):
        entry[key] = asset_path(catalog, entry, key, asset_root=asset_root)
        if check_files and not entry[key].is_file():
            raise FileNotFoundError(entry[key])
    return entry


def sequence_objects(class_ids, *, catalog_path=DEFAULT_CATALOG, check_files=True):
    """Resolve a canonical preparation list, preserving order and removing repeats."""
    if not isinstance(class_ids, (list, tuple)) or not class_ids:
        raise ValueError("Provide at least one canonical class_id, or all")
    if "all" in class_ids:
        if list(class_ids) != ["all"]:
            raise ValueError("Use all alone, or list individual class IDs")
        class_ids = [item["class_id"] for item in load_catalog(catalog_path)["objects"]]
    if not class_ids:
        raise ValueError("No registered objects to prepare")
    if not all(isinstance(value, str) and value for value in class_ids):
        raise ValueError("Class IDs must be nonempty strings")
    return [registered_object(name, catalog_path=catalog_path, check_files=check_files)
            for name in dict.fromkeys(class_ids)]
