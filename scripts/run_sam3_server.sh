#!/bin/bash
# Dedicated SAM3 environment; the source always comes from this checkout.
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAM3_PYTHON="${SAM3_PYTHON:-$HOME/miniforge3/envs/sam3/bin/python}"
if [ ! -x "$SAM3_PYTHON" ]; then
    echo "SAM3 Python missing: $SAM3_PYTHON. See docs/sam3-reference.md."
    exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
cd "$ROOT"
exec "$SAM3_PYTHON" tiptop/scripts/sam3_server.py "$@"
