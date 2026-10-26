"""Shared CAD/DB paths without importing perception models or device libraries."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "tiptop/config/known_objects.json"


def load_catalog(catalog_path=DEFAULT_CATALOG):
    catalog = json.loads(Path(catalog_path).read_text())
    ids = [entry["class_id"] for entry in catalog["objects"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate registered class IDs")
    return catalog


def asset_path(catalog, entry, key, *, asset_root=ROOT):
    """Resolve CAD and DB against separate roots; retain legacy catalog support.

    A catalog without cad_root/db_root keeps project-relative paths. Explicit
    absolute paths also work. Missing external files never fall back to data/.
    """
    root_key = {"cad": "cad_root", "db": "db_root"}[key]
    root = Path(asset_root) / Path(catalog.get(root_key, "")).expanduser()
    return (root / Path(entry[key]).expanduser()).resolve()


def registered_object(class_id, *, catalog_path=DEFAULT_CATALOG, asset_root=ROOT, check_files=True):
    catalog = load_catalog(catalog_path)
    matches = [entry for entry in catalog["objects"] if entry["class_id"] == class_id]
    if len(matches) != 1:
        raise ValueError(f"Unregistered class: {class_id}")
    entry = dict(matches[0])
    for key in ("cad", "db"):
        entry[key] = asset_path(catalog, entry, key, asset_root=asset_root)
        if check_files and not entry[key].is_file():
            raise FileNotFoundError(entry[key])
    return entry
