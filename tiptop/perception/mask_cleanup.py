"""Mask cleanup shared by the detection client and the pose client."""

import numpy as np
from scipy import ndimage

STRAY_PIECE_FRACTION = 0.05  # a mask piece smaller than this share of its largest piece is dropped


def drop_stray_pieces(mask, min_fraction=STRAY_PIECE_FRACTION):
    """Remove the pieces of one mask that are far smaller than its largest piece.

    SAM3 sometimes leaves a few pixels away from the object; they stretch the mask's bounding box, which the pose
    estimator uses to place its first guess.
    """
    labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
    if count < 2:
        return mask
    areas = np.bincount(labels.ravel())[1:]
    return np.isin(labels, np.flatnonzero(areas >= min_fraction * areas.max()) + 1)
