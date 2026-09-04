"""Fuse dense stereo results from multiple camera pairs into one dense colored cloud."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from sfm.mvs.stereo import dense_stereo_pair


def dense_reconstruct(
    images: Dict[int, np.ndarray],
    cam_ids: Sequence[int],
    Ks: np.ndarray,
    Rs: np.ndarray,
    ts: np.ndarray,
    min_depth: float,
    max_depth: float,
    downscale: int = 2,
    num_disparities: int = 256,
    pairs: Sequence[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run dense_stereo_pair over consecutive camera pairs (or an explicit
    `pairs` list) and concatenate the results.

    Consecutive pairs are used by default because, on this dataset, they're
    the smallest available baseline (see the baseline table computed while
    developing this) -- smaller baseline means less perspective distortion
    between the two images, which is what classical block-matching stereo
    depends on for reliable correspondences.

    Returns (points_world (N,3), colors (N,3) uint8) -- unfiltered/unfused
    beyond what dense_stereo_pair itself already discards; voxel
    downsampling and outlier removal happen separately (see mesh.py) since
    those need Open3D, not raw numpy.
    """
    cam_ids = list(cam_ids)
    if pairs is None:
        pairs = [(cam_ids[k], cam_ids[k + 1]) for k in range(len(cam_ids) - 1)]

    all_points: List[np.ndarray] = []
    all_colors: List[np.ndarray] = []

    for i, j in pairs:
        idx_i, idx_j = cam_ids.index(i), cam_ids.index(j)
        pts, colors, _disp = dense_stereo_pair(
            images[i], images[j], Ks[idx_i], Rs[idx_i], ts[idx_i], Rs[idx_j], ts[idx_j],
            min_depth, max_depth, num_disparities=num_disparities, downscale=downscale,
        )
        all_points.append(pts)
        all_colors.append(colors)

    return np.concatenate(all_points, axis=0), np.concatenate(all_colors, axis=0)
