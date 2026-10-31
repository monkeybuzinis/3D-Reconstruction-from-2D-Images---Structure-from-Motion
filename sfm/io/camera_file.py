"""
Read the calibration a dataset ships with it, in whichever format it ships.

Two formats appear in this project. Strecha-format `<image>.camera` sidecars
(Fountain-P11, Rathaus, Herz-Jesus-P8/P25) and COLMAP's text `cameras.txt`
(ETH3D). Both are real calibration and both should beat the EXIF-based guess
in sfm/io/calibration.py, which exists for the case where nothing better is
available. The 55-degree fallback at the bottom of that ladder is a guess
about an ordinary lens: on the ETH3D statue set, whose true horizontal field
of view is 84.6 degrees, it overestimates the focal length by 75 per cent.

Parse Strecha-format `<image>.camera` sidecar files (K, radial distortion,
R, t) -- used by this lab's benchmark datasets (Fountain-P11, Rathaus, and
others). When present, this is real calibration, and should always be
preferred over the EXIF-based guess in sfm/io/calibration.py, which exists
specifically for the case where nothing better is available.

Documented projection convention (from the dataset's own README):

    

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


# ---------------------------------------------------------- COLMAP text model
# cameras.txt holds one line per camera:
#     CAMERA_ID MODEL WIDTH HEIGHT PARAMS...
# ETH3D distributes its benchmark calibration this way. The parameter layout
# depends on the model, and only the focal length and principal point are
# read: this project projects through a distortion-free pinhole, so radial
# terms are dropped here exactly as they are for the Strecha sidecars above.
_COLMAP_FOCAL_LAYOUT = {
    "SIMPLE_PINHOLE": ("f", "cx", "cy"),
    "PINHOLE": ("fx", "fy", "cx", "cy"),
    "SIMPLE_RADIAL": ("f", "cx", "cy"),
    "RADIAL": ("f", "cx", "cy"),
    "OPENCV": ("fx", "fy", "cx", "cy"),
}


def find_cameras_txt(image_path: Path) -> Optional[Path]:
    """COLMAP's text model sits beside the images, not per-image."""
    cameras = image_path.parent / "cameras.txt"
    return cameras if cameras.is_file() else None


def load_cameras_txt(path: Path, width_px: int, height_px: int) -> np.ndarray:
    """
    K from the first camera in a COLMAP cameras.txt.

    One shared camera is assumed, which is what every dataset in this project
    uses and what the rest of the pipeline is built around. If the file
    declares a different image size from the one actually loaded, the
    calibration is rescaled: datasets are often distributed downsampled while
    their calibration still describes the original frame, and a focal length
    read at the wrong scale is worse than no calibration at all.
    """
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        model, declared_w, declared_h = fields[1], int(fields[2]), int(fields[3])
        layout = _COLMAP_FOCAL_LAYOUT.get(model)
        if layout is None:
            raise ValueError(f"Unsupported COLMAP camera model {model!r} in {path}")
        params = dict(zip(layout, (float(v) for v in fields[4:])))
        fx = params.get("fx", params.get("f"))
        fy = params.get("fy", fx)
        cx, cy = params["cx"], params["cy"]

        if (declared_w, declared_h) != (width_px, height_px):
            scale = width_px / declared_w
            fx, fy, cx, cy = fx * scale, fy * scale, cx * scale, cy * scale
            print(f"  {path.name} describes {declared_w}x{declared_h}; "
                  f"rescaled by {scale:.4f} to match {width_px}x{height_px}")
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    raise ValueError(f"No camera line found in {path}")
