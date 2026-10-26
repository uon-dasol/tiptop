"""Thin launcher for the server in the independent SAM3 checkout."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "SAM3"
if not (ROOT / "tiptop_server/server.py").is_file():
    sys.exit("SAM3 server missing. Initialize SAM3 and include its tiptop_server changes.")
sys.path.insert(0, str(ROOT))
from tiptop_server.server import main

if __name__ == "__main__":
    main()
