"""Launch the FoundationPose pose server (FoundationPose/tiptop_server) with the TiPToP CAD checks.

Start it with `bash scripts/run_foundation_pose_server.sh` (foundationpose conda Python) or launch this script
with that Python. `--check` only reports missing prerequisites.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tiptop.object_catalog import DEFAULT_CATALOG, asset_path, load_catalog

FOUNDATIONPOSE_ROOT = ROOT / "FoundationPose"
CATALOG = DEFAULT_CATALOG

Finding = tuple[str, str]  # (level, message) with level "ok", "warn" or "error"


def cad_findings(asset_root: Path) -> list[Finding]:
    """Check the registered CAD assets: OBJ, MTL and the textures the server resolves by file name."""
    if not CATALOG.is_file():
        return [("warn", f"Registered-object catalog not found: {CATALOG}; CAD files were not checked")]
    findings = []
    catalog = load_catalog(CATALOG)
    allowed_roots = [asset_root.resolve(), *additional_asset_roots()]
    for entry in catalog["objects"]:
        name, obj = entry["class_id"], asset_path(catalog, entry, "cad")
        problems = []
        if not obj.is_file():
            problems.append(f"CAD for {name} not found: {obj}")
        else:
            if not any(obj.is_relative_to(root) for root in allowed_roots):
                problems.append(f"CAD for {name} is outside the asset root {asset_root}; the server will reject it")
            mtl = obj.with_suffix(".mtl")
            if not mtl.is_file():
                problems.append(f"MTL for {name} not found next to the OBJ: {mtl}")
            else:
                # Like load_mesh_readonly, ignore leftover materials that the
                # OBJ never uses. Export history can reference absent textures.
                used_materials = set()
                with obj.open(errors="replace") as stream:
                    for line in stream:
                        parts = line.split(maxsplit=1)
                        if parts[0:1] == ["usemtl"] and len(parts) == 2:
                            used_materials.add(parts[1].strip())
                material = None
                for line in mtl.read_text(errors="ignore").splitlines():
                    parts = line.split(maxsplit=1)
                    if len(parts) != 2:
                        continue
                    if parts[0].lower() == "newmtl":
                        material = parts[1].strip()
                    elif parts[0].lower() == "map_kd" and material in used_materials:
                        texture = Path(parts[1].split()[-1]).name
                        if not any((root / texture).is_file() for root in (obj.parent, obj.parent.parent)):
                            problems.append(f"Texture {texture} of {name} not found in {obj.parent} or its parent")
        findings += [("warn", problem) for problem in problems] or [("ok", f"CAD for {name}: {obj.name}")]
    return findings


def default_asset_root():
    """Use the same CAD root as the client; --asset-root remains an override."""
    catalog = load_catalog(CATALOG)
    return (ROOT / Path(catalog.get("cad_root", "data")).expanduser()).resolve()


def additional_asset_roots():
    """Allow bundled CADs only under explicitly configured additional datasets."""
    return [(ROOT / Path(dataset["root"]).expanduser()).resolve()
            for dataset in load_catalog(CATALOG).get("additional_datasets", [])]


def resolve_preload_objects(class_ids):
    from tiptop.object_catalog import sequence_objects

    return [dict(class_id=item["class_id"], mesh_path=str(item["cad"]), mesh_scale=item["cad_scale_to_m"])
            for item in sequence_objects(class_ids, catalog_path=CATALOG, check_files=False)]


def main():
    server = FOUNDATIONPOSE_ROOT / "tiptop_server" / "server.py"
    if not server.is_file():
        sys.exit(
            f"[ERROR] FoundationPose server not found: {server}. Run `git submodule update --init FoundationPose` "
            "in the project root (see README_CUSTOM.md, '3. FoundationPose')"
        )
    sys.path.insert(0, str(FOUNDATIONPOSE_ROOT))
    from tiptop_server.server import main as serve

    serve(default_asset_root=default_asset_root(), extra_findings=cad_findings,
          resolve_preload_objects=resolve_preload_objects, default_additional_asset_roots=additional_asset_roots())


if __name__ == "__main__":
    main()
