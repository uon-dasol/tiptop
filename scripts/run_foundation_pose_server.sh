#!/bin/bash
# Start the FoundationPose pose server (default port 13346) from the project root:
#   bash scripts/run_foundation_pose_server.sh [--check] [--port N] [--asset-root DIR]
# Uses the foundationpose conda env; the server finds ./FoundationPose (README_CUSTOM.md) or ../data_gen/FoundationPose.
# --check only reports missing prerequisites (no GPU). Override with FOUNDATIONPOSE_PYTHON and FOUNDATIONPOSE_ROOT.

set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FOUNDATIONPOSE_PYTHON="${FOUNDATIONPOSE_PYTHON:-$HOME/miniforge3/envs/foundationpose/bin/python}"

if [ ! -x "$FOUNDATIONPOSE_PYTHON" ]; then
    echo "ERROR: FoundationPose Python not found: $FOUNDATIONPOSE_PYTHON"
    echo "Create the 'foundationpose' conda env (README_CUSTOM.md, '6. FoundationPose 설치'),"
    echo "or set FOUNDATIONPOSE_PYTHON to an existing interpreter."
    exit 1
fi

# Keep bytecode out of the FoundationPose sources
export PYTHONDONTWRITEBYTECODE=1
cd "$ROOT"
exec "$FOUNDATIONPOSE_PYTHON" tiptop/scripts/foundation_pose_server.py "$@"
