"""
Load an arbitrary folder of photos (not the fixed Fountain-P11 layout) into
the same {camera_id: image} dict every other sfm module expects, plus an
estimated shared K. This is what makes the pipeline usable on new,
self-captured data instead of only the one benchmark dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from sfm.io.calibration import estimate_intrinsics
from sfm.io.camera_file import find_camera_file, load_camera_file

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".ppm", ".pgm"}


def load_image_folder(folder: Path) -> Tuple[Dict[int, np.ndarray], list]:
    """
    Load every image in `folder`, sorted by filename, as camera id 0..N-1
    (in filename order). Filename order matters: the incremental pipeline's
    "consecutive cameras have the smallest baseline" assumption (used for
    both the two-view seed and MVS pair selection) only holds if photos are
    numbered/named in roughly the order they were taken while moving around
    the object -- the normal way anyone would name a casual photo sequence.
    """
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)
    if len(paths) < 2:
        raise ValueError(f"Need >= 2 images in {folder}, found {len(paths)}")

    images: Dict[int, np.ndarray] = {}
    for i, path in enumerate(paths):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"OpenCV failed to read {path}")
        images[i] = img
    return images, paths


def load_scene(
    folder: Path,
    focal_mm: Optional[float] = None,
    sensor_width_mm: Optional[float] = None,
    fallback_fov_deg: float = 55.0,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """
    Load images + determine a shared K, all in one call.

    K is resolved in order of trust:
      1. An explicit focal_mm + sensor_width_mm override.
      2. A Strecha-style `<image>.camera` sidecar file, if the dataset ships
         one (e.g. Fountain-P11, Rathaus) -- this is real calibration, not
         a guess, so it always wins over EXIF when present.
      3. EXIF-based estimation (sfm/io/calibration.py), falling back to an
         assumed FOV if EXIF has no usable focal length either.
    """
    images, paths = load_image_folder(folder)
    h, w = images[0].shape[:2]

    if focal_mm is not None and sensor_width_mm is not None:
        focal_px = (focal_mm / sensor_width_mm) * w
        return images, estimate_intrinsics(paths[0], w, h, override_focal_px=focal_px)

    camera_file = find_camera_file(paths[0])
    if camera_file is not None:
        K, _R, _t = load_camera_file(camera_file)
        print(f"Using real calibration from {camera_file.name} (not an EXIF/FOV guess)")
        return images, K

    return images, estimate_intrinsics(paths[0], w, h, fallback_fov_deg=fallback_fov_deg)
