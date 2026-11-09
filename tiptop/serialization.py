"""Lossless NumPy packets with inexpensive compression for local services."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np


def savez_fast(file, *arrays, **named):
    """Write a standard NPZ at DEFLATE level 1; keep array values and dtypes."""
    values = dict(named)
    for index, array in enumerate(arrays):
        key = f"arr_{index}"
        if key in values:
            raise ValueError(f"Cannot use un-named variables and keyword {key}")
        values[key] = array
    if isinstance(file, (str, Path)):
        file = str(file)
        if not file.endswith(".npz"):
            file += ".npz"
    with ZipFile(file, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
        for name, array in values.items():
            with archive.open(name + ".npy", "w", force_zip64=True) as member:
                np.lib.format.write_array(member, np.asanyarray(array), allow_pickle=False)
