"""Recording must not create a GUI in its worker or crash at process exit."""

import os
from pathlib import Path
import subprocess
import sys


def test_detection_recording_exits_normally_with_gui_backend_selected(tmp_path):
    program = """
from concurrent.futures import ThreadPoolExecutor
import sys
import numpy as np
from PIL import Image
from matplotlib._pylab_helpers import Gcf
from tiptop.perception.visualization import visualize_detections

with ThreadPoolExecutor(max_workers=1) as pool:
    pixels = pool.submit(visualize_detections, Image.new('RGB', (32, 32)),
        [dict(label='pepper', box_2d=[100, 100, 900, 900])], sys.argv[1], False).result()
assert pixels.shape == (800, 1200, 3)
assert pixels.dtype == np.uint8 and pixels.flags.owndata
assert not Gcf.get_all_fig_managers(), 'Recording created a GUI figure manager'
assert Image.open(sys.argv[1]).size == (1200, 800)
print('recording-and-shutdown-ok', flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", program, str(tmp_path / "bboxes.png")],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "MPLBACKEND": "QtAgg", "MPLCONFIGDIR": str(tmp_path / "matplotlib")},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "recording-and-shutdown-ok" in result.stdout
    assert "Starting a Matplotlib GUI outside" not in result.stderr
