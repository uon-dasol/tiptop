"""Launch the FoundationPose pose server (FoundationPose/tiptop_server) with the TiPToP CAD checks.

Start it with `bash scripts/run_foundation_pose_server.sh` (foundationpose conda Python) or launch this script
with that Python. `--check` only reports missing prerequisites.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOUNDATIONPOSE_ROOT = ROOT / "FoundationPose"
CATALOG = ROOT / "tiptop/config/known_objects.json"

Finding = tuple[str, str]  # (level, message) with level "ok", "warn" or "error"


def cad_findings(asset_root: Path) -> list[Finding]:
    """Check the registered CAD assets: OBJ, MTL and the textures the server resolves by file name."""
    if not CATALOG.is_file():
        return [("warn", f"Registered-object catalog not found: {CATALOG}; CAD files were not checked")]
    findings = []
    for entry in json.loads(CATALOG.read_text())["objects"]:
        name, obj = entry["class_id"], (ROOT / entry["cad"]).resolve()
        problems = []
        if not obj.is_file():
            problems.append(f"CAD for {name} not found: {obj} (expected under data/<object>/3d_model/edited/)")
        else:
            if not obj.is_relative_to(asset_root.resolve()):
                problems.append(f"CAD for {name} is outside the asset root {asset_root}; the server will reject it")
            mtl = obj.with_suffix(".mtl")
            if not mtl.is_file():
                problems.append(f"MTL for {name} not found next to the OBJ: {mtl}")
            else:
                for line in mtl.read_text(errors="ignore").splitlines():
                    parts = line.split(maxsplit=1)
                    if parts[0:1] == ["map_Kd"] and len(parts) == 2:
                        texture = Path(parts[1].strip()).name
                        if not any((root / texture).is_file() for root in (obj.parent, obj.parent.parent)):
                            problems.append(f"Texture {texture} of {name} not found in {obj.parent} or its parent")
        findings += [("warn", problem) for problem in problems] or [("ok", f"CAD for {name}: {obj.name}")]
    return findings


def main():
    server = FOUNDATIONPOSE_ROOT / "tiptop_server" / "server.py"
    if not server.is_file():
        sys.exit(
            f"[ERROR] FoundationPose server not found: {server}. Run `git submodule update --init FoundationPose` "
            "in the project root (see README_CUSTOM.md, '6. FoundationPose 설치')"
        )
    sys.path.insert(0, str(FOUNDATIONPOSE_ROOT))
    from tiptop_server.server import main as serve

    serve(default_asset_root=ROOT / "data", extra_findings=cad_findings)


if __name__ == "__main__":
    main()
