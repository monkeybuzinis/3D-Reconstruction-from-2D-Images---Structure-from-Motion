"""
Parse Strecha-format `<image>.camera` sidecar files (K, radial distortion,
R, t) -- used by this lab's benchmark datasets (Fountain-P11, Rathaus, and
others). When present, this is real calibration, and should always be
preferred over the EXIF-based guess in sfm/io/calibration.py, which exists
specifically for the case where nothing better is available.

Documented projection convention (from the dataset's own README):

    x = K [R^T | -R^T t] X

i.e. R is the camera's camera-to-world orientation and t IS the camera
center directly -- the opposite convention from sfm/map.py's Camera, which
stores world-to-camera (R, t) such that x_cam = R @ X + t. Converted here:

    R_ours = R_theirs.T
    t_ours = -R_ours @ t_theirs
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def load_camera_file(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (K, R, t) in this project's own (world-to-camera) convention."""
    values = [float(v) for v in path.read_text().split()[:24]]
    if len(values) < 24:
        raise ValueError(f"Expected >= 24 numbers (K, distortion, R, t) in {path}, got {len(values)}")

    K = np.array(values[0:9]).reshape(3, 3)
    # values[9:12] are radial distortion params -- ignored. Per the
    # dataset's own README: "if zero, the images are corrected for radial
    # distortion", which is the case for the datasets this project uses.
    R_theirs = np.array(values[12:21]).reshape(3, 3)
    t_theirs = np.array(values[21:24])

    R = R_theirs.T
    t = -R @ t_theirs
    return K, R, t


def find_camera_file(image_path: Path) -> Optional[Path]:
    """The Strecha convention is `<image_filename>.camera` (e.g. rdimage.000.ppm.camera)."""
    sidecar = image_path.parent / f"{image_path.name}.camera"
    return sidecar if sidecar.is_file() else None
