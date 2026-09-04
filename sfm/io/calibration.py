"""
Approximate camera intrinsics (K) for arbitrary photos, since the Fountain
dataset's hardcoded K (sfm/io/fountain.py) obviously doesn't apply to a new
camera. This is inherently an approximation -- a real calibration
(checkerboard) is more accurate -- but EXIF-derived K is the standard,
good-enough starting point most casual photogrammetry tools use, and it's
what actually makes "any folder of photos" reconstruction possible without
asking the user to run a calibration procedure first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import ExifTags, Image

_FULL_FRAME_WIDTH_MM = 36.0  # the reference width FocalLengthIn35mmFilm is defined relative to


def _read_exif_ifd(path: Path) -> dict:
    """Merge the top-level EXIF dict and the Exif sub-IFD (where focal length tags usually live)."""
    img = Image.open(path)
    exif = img.getexif()
    tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    try:
        exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
        tags.update({ExifTags.TAGS.get(k, k): v for k, v in exif_ifd.items()})
    except (KeyError, AttributeError):
        pass
    return tags


def _focal_px_from_exif(path: Path, width_px: int) -> Optional[float]:
    """
    Try, in order of reliability:
      1. FocalLengthIn35mmFilm -- doesn't need the camera's actual sensor
         size, which is frequently missing from phone EXIF.
      2. FocalLength (mm) + FocalPlaneXResolution/Unit -- needs both tags
         present and consistent, which many cameras omit.
    Returns None if neither is usable.
    """
    tags = _read_exif_ifd(path)

    focal_35mm = tags.get("FocalLengthIn35mmFilm")
    if focal_35mm:
        return (float(focal_35mm) / _FULL_FRAME_WIDTH_MM) * width_px

    focal_mm = tags.get("FocalLength")
    plane_res = tags.get("FocalPlaneXResolution")
    plane_unit = tags.get("FocalPlaneResolutionUnit")  # 2 = inches, 3 = cm
    if focal_mm and plane_res and plane_unit:
        focal_mm = float(focal_mm)
        plane_res = float(plane_res)
        unit_mm = {2: 25.4, 3: 10.0}.get(int(plane_unit))
        if unit_mm and plane_res > 0:
            sensor_width_mm = width_px / (plane_res / unit_mm)
            if sensor_width_mm > 0:
                return (focal_mm / sensor_width_mm) * width_px

    return None


def estimate_intrinsics(
    image_path: Path,
    width: int,
    height: int,
    fallback_fov_deg: float = 55.0,
    override_focal_px: Optional[float] = None,
) -> np.ndarray:
    """
    Build an approximate K for one photo (assumed representative of the
    whole shoot -- same camera/lens throughout, which is the normal case
    for a single capture session).

    Falls back to assuming a `fallback_fov_deg` horizontal field of view
    (55 degrees is a reasonable default for a typical phone's main camera)
    if EXIF has neither FocalLengthIn35mmFilm nor a usable
    FocalLength+FocalPlaneResolution combination -- common on
    screenshots, edited photos, or cameras that strip EXIF.
    """
    if override_focal_px is not None:
        focal_px = override_focal_px
    else:
        focal_px = _focal_px_from_exif(image_path, width)
        if focal_px is None:
            focal_px = (width / 2) / np.tan(np.radians(fallback_fov_deg) / 2)

    return np.array(
        [
            [focal_px, 0.0, width / 2],
            [0.0, focal_px, height / 2],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
