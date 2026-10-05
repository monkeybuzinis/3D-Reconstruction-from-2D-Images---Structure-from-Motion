"""
Load the Fountain-P11 dataset (images, shared intrinsics, and a reference
sparse cloud) into the Map data model. This is the only file that knows
about this specific dataset's file layout; everything downstream (features,
geometry, recon) works purely off the Map/image dicts this hands back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from sfm.map import Camera, Map, Point3D

# Fountain-P11 intrinsics: same K on every VERTEX_CAM line in
# Fountain/solution.graph, i.e. all 11 photos were taken with the same
# camera/lens/zoom, so one 3x3 K matrix covers every view.
FOUNTAIN_K = np.array(
    [
        [2759.48, 0.0, 1520.69],
        [0.0, 2759.48, 1006.81],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

N_IMAGES = 11


def default_fountain_dir() -> Path:
    """The Fountain/ folder under dataset/ at the repo root, resolved relative to this file so it works from any cwd."""
    return Path(__file__).resolve().parents[2] / "dataset" / "Fountain"


def image_paths(fountain_dir: Path) -> List[Path]:
    """0000.png .. 0010.png, in camera-id order. Fails loudly if any are missing rather than silently reconstructing from fewer views."""
    paths = [fountain_dir / f"{i:04d}.png" for i in range(N_IMAGES)]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(f"Missing Fountain images in {fountain_dir}: {names}")
    return paths


def load_images(fountain_dir: Path) -> Dict[int, np.ndarray]:
    """Load all 11 RGB images as BGR uint8 arrays, keyed by camera id 0..10 (OpenCV's native channel order, kept as-is since SIFT/FLANN don't care)."""
    images: Dict[int, np.ndarray] = {}
    for i, path in enumerate(image_paths(fountain_dir)):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"OpenCV failed to read {path}")
        images[i] = img
    return images


def load_reference_points(fountain_dir: Path) -> np.ndarray:
    """
    Sparse reference XYZ from the BA solution pack (solution.txt).

    This is NOT the Strecha dense laser/MVS ground truth used for real
    Accuracy/Completeness/F-score benchmarking (that's a separate download,
    see Fountain/Readme.txt) -- it's ~59k points from someone else's already
    bundle-adjusted solve. Useful here only as a "does this look like the
    right scene" sanity check for week 1, not as final evaluation ground
    truth.

    File layout: NOT a clean "11 header lines then points" structure --
    only 2 lines actually have the 6-field camera-parameter format; the
    other 9 header-ish lines are real 3-field XYZ points (verified against
    solution.graph's explicit VERTEX_CAM/VERTEX_XYZ tags, which unlike
    solution.txt actually label each line's type). Filtering by field count
    rather than by line position is what avoids silently discarding those
    9 points.
    """
    path = fountain_dir / "solution.txt"
    if not path.is_file():
        raise FileNotFoundError(f"No reference cloud at {path}")

    xyz: List[np.ndarray] = []
    with path.open() as f:
        for line in f:
            parts = line.split()
            if len(parts) != 3:
                continue  # not a plain "X Y Z" line (e.g. a 6-field camera line)
            xyz.append(np.array([float(v) for v in parts], dtype=np.float64))
    if not xyz:
        raise RuntimeError(f"No XYZ points parsed from {path}")
    return np.stack(xyz, axis=0)


def load_fountain_dataset(
    fountain_dir: Optional[Path] = None,
    *,
    include_reference_points: bool = True,
) -> Tuple[Map, Dict[int, np.ndarray]]:
    """
    Week-1 loader: builds a Map with 11 cameras (K + image size set, pose
    left empty -- solving for pose is what later stages do) and, optionally,
    the reference cloud as a first population of Point3D for a visual sanity
    check.

    Camera ids match filenames: id i <-> {i:04d}.png.
    """
    fountain_dir = Path(fountain_dir) if fountain_dir else default_fountain_dir()
    if not fountain_dir.is_dir():
        raise FileNotFoundError(f"Fountain directory not found: {fountain_dir}")

    images = load_images(fountain_dir)
    scene = Map()

    for cam_id, img in images.items():
        h, w = img.shape[:2]
        scene.add_camera(
            Camera(
                id=cam_id,
                K=FOUNTAIN_K.copy(),
                image_name=f"{cam_id:04d}.png",
                width=w,
                height=h,
            )
        )

    if include_reference_points:
        xyz = load_reference_points(fountain_dir)
        for i, p in enumerate(xyz):
            scene.add_point(Point3D(id=i, xyz=p))

    return scene, images
