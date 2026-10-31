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

# A 35 mm-equivalent focal length states what lens on a 36 x 24 mm frame
# would give the same field of view. That equivalence is conventionally
# defined over the DIAGONAL -- it is how crop factor is defined -- not over
# either edge, because only the diagonal is independent of aspect ratio and
# of which way the camera was held. Pairing it with an edge is right only
# when the sensor has the same 3:2 shape as 35 mm film, which phone sensors
# (4:3) do not. See _focal_px_from_exif.
_FULL_FRAME_DIAGONAL_MM = float(np.hypot(36.0, 24.0))   # 43.267 mm


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


def _focal_px_from_exif(path: Path, width_px: int, height_px: int) -> Optional[float]:
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
        # Scale by the image diagonal against the 35 mm frame diagonal. Two
        # earlier versions of this line were wrong in different ways: using
        # width alone shortened the focal length by the aspect ratio on every
        # portrait photo (1536/2048 = 0.75), and using the long edge still
        # left a systematic 5% because a 4:3 sensor is not the shape of a
        # 3:2 frame. Neither error is detectable downstream -- the solve
        # converges and still reports a good residual.
        return (float(focal_35mm) / _FULL_FRAME_DIAGONAL_MM) * float(np.hypot(width_px, height_px))

    focal_mm = tags.get("FocalLength")
    plane_res = tags.get("FocalPlaneXResolution")
    plane_unit = tags.get("FocalPlaneResolutionUnit")  # 2 = inches, 3 = cm
    if focal_mm and plane_res and plane_unit:
        focal_mm = float(focal_mm)
        plane_res = float(plane_res)
        unit_mm = {2: 25.4, 3: 10.0}.get(int(plane_unit))
        if unit_mm and plane_res > 0:
            # width_px cancels out of this branch -- the result reduces to
            # focal_mm * plane_res / unit_mm -- so it is already independent
            # of which way the camera was held.
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
    across the image's own width (55 degrees is a reasonable default for a
    typical phone's main camera held in portrait) if EXIF has neither FocalLengthIn35mmFilm nor a
    usable FocalLength+FocalPlaneResolution combination -- common on
    screenshots, edited photos, or cameras that strip EXIF.
    """
    if override_focal_px is not None:
        focal_px = override_focal_px
        print(f"Using the focal length supplied on the command line: {focal_px:.1f} px")
    else:
        focal_px = _focal_px_from_exif(image_path, width, height)
        if focal_px is not None:
            print(f"Using EXIF metadata to estimate the focal length: {focal_px:.1f} px")
        if focal_px is None:
            # Unlike the 35 mm-equivalent tag above, this constant is not a
            # convention to be interpreted -- it is a guess about one angle,
            # and it is calibrated across the image's own width. The three
            # datasets here that carry no EXIF bear that out: COLMAP, which
            # refines the focal length from the photographs, implies a
            # horizontal field of view of 56.7, 57.6 and 53.2 degrees on them.
            # Reading the same 55 degrees across the diagonal instead would
            # overestimate the focal length by about 70%.
            focal_px = (width / 2) / np.tan(np.radians(fallback_fov_deg) / 2)
            print(f"No usable EXIF: assuming a {fallback_fov_deg:.0f} degree field of view, "
                  f"which gives {focal_px:.1f} px. This is a guess, not a measurement.")

    return np.array(
        [
            [focal_px, 0.0, width / 2],
            [0.0, focal_px, height / 2],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
