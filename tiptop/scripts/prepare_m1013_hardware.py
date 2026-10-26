"""Check measured hardware inputs or build a derived model; never open or move devices."""

import argparse
import json
from pathlib import Path

from tiptop.doosan.hardware import load_hardware_profile, prepare_hardware_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument(
        "--output", type=Path, help="Build derived URDF/collision/tool model; requires the planning GPU environment"
    )
    args = parser.parse_args()
    try:
        profile = load_hardware_profile(args.profile)
        if args.output:
            prepared = prepare_hardware_model(profile, args.output)
            result = dict(
                success=True, model_built=True, tool_spheres=len(prepared["gripper_spheres"]), output=str(args.output)
            )
        else:
            result = dict(success=True, measured_inputs_valid=True, model_built=False)
        result.update(devices_opened=0, commands_sent=0, physical_validation_performed=False)
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"Hardware preparation incomplete: {error}\n")


if __name__ == "__main__":
    main()
