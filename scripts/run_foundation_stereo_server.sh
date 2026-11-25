#!/bin/bash
# Use the existing local FoundationStereo checkout and weights; no downloads.
set -eo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export FOUNDATION_STEREO_ROOT="${FOUNDATION_STEREO_ROOT:-$TASK_ROOT/FoundationStereo}"
export HF_HUB_OFFLINE=1
exec "$FOUNDATION_STEREO_ROOT/.pixi/envs/default/bin/python" - "$@" <<'PY'
import importlib.util
import os
import sys
from pathlib import Path
import uvicorn
root = Path(os.environ['FOUNDATION_STEREO_ROOT'])
os.chdir(root)
sys.path.insert(0, str(root))
spec = importlib.util.spec_from_file_location('foundation_stereo_server', root / 'scripts/server.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
# Upstream server resolves weights relative to code_dir, not scripts/.
module.code_dir = str(root)
# Return unused inference buffers to other GPU processes (SAM3/pose/Isaac).
# Keep model weights loaded; the returned depth is already a CPU numpy array.
original_inference = module.run_inference

def infer_and_release_cache(*args, **kwargs):
    try:
        return original_inference(*args, **kwargs)
    finally:
        module.torch.cuda.empty_cache()

module.run_inference = infer_and_release_cache
uvicorn.run(module.app, host='127.0.0.1', port=int(sys.argv[1]) if len(sys.argv) > 1 else 13344)
PY
