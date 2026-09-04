"""Minimal PLY export, so reconstructed clouds can be opened in MeshLab/CloudCompare for a visual check."""

from pathlib import Path

import numpy as np


def write_xyz_ply(path: Path, xyz: np.ndarray) -> None:
    """Write an ASCII PLY with XYZ only (no color/normals -- geometry-only clouds don't need them)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) array, got {xyz.shape}")

    path = Path(path)
    n = xyz.shape[0]
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with path.open("w") as f:
        f.write(header)
        for row in xyz:
            f.write(f"{row[0]:.9f} {row[1]:.9f} {row[2]:.9f}\n")


def write_xyzrgb_ply(path: Path, xyz: np.ndarray, colors: np.ndarray) -> None:
    """Write an ASCII PLY with XYZ + RGB (colors from sfm.io.colorize.colorize_points, 0-255 uint8)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) xyz array, got {xyz.shape}")
    if colors.shape != xyz.shape:
        raise ValueError(f"Expected colors shape {xyz.shape}, got {colors.shape}")

    path = Path(path)
    n = xyz.shape[0]
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with path.open("w") as f:
        f.write(header)
        for row, c in zip(xyz, colors):
            f.write(f"{row[0]:.9f} {row[1]:.9f} {row[2]:.9f} {c[0]} {c[1]} {c[2]}\n")
