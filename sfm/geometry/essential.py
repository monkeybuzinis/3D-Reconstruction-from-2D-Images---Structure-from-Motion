"""
Essential matrix: recovering the calibrated relative pose [R|t] between two
cameras. Where Fundamental matrix F works purely in pixels, the Essential
matrix E = K^T F K works in calibrated (ray) coordinates and can actually be
decomposed into a rotation + translation -- this is what turns "two images
match" into "camera 1 is over here, rotated like this, relative to camera 0".
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from sfm.geometry.triangulation import triangulate_dlt


def essential_from_fundamental(F: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    E = K^T F K.

    A valid Essential matrix has two equal non-zero singular values and one
    zero (unlike F, which only needs rank 2). Noise in F breaks this, so we
    re-project onto the nearest valid E by averaging the top two singular
    values and zeroing the third -- the calibrated analogue of the rank-2
    cleanup done for F.
    """
    E = K.T @ F @ K
    U, S, Vt = np.linalg.svd(E)
    m = (S[0] + S[1]) / 2.0
    return U @ np.diag([m, m, 0.0]) @ Vt


def decompose_essential(E: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Decompose E into the 4 candidate (R, t) pairs (Hartley & Zisserman Sec. 9.6.2).

    E only determines t up to sign and R up to a two-way ambiguity, so there
    are exactly 4 algebraically valid (R, t) combinations from one E: this
    is the "twisted pair" ambiguity. All 4 satisfy the epipolar geometry
    equally well -- only one places the observed points in front of both
    cameras, which cheirality_check() below resolves.
    """
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]  # translation direction is the left null vector of E, up to sign

    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]


def cheirality_check(
    K: np.ndarray,
    candidates: List[Tuple[np.ndarray, np.ndarray]],
    pts0: np.ndarray,
    pts1: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pick the (R, t) candidate that puts the most triangulated points in front
    of both cameras ("positive depth" = cheirality). Camera 0 is fixed at
    the world origin (identity pose); this establishes the world coordinate
    frame for the whole reconstruction.

    Physically, only one of the 4 algebraic candidates corresponds to a real
    camera arrangement where the scene is actually visible to both cameras
    -- the other 3 would require the scene to be behind one or both cameras.
    Triangulating a few points under each candidate and counting how many
    end up in front is the standard, simple way to tell them apart.

    Returns (R, t, xyz) for the winning candidate; xyz covers ALL input
    points (including ones that end up behind a camera even under the
    winning pose -- the caller filters those out separately).
    """
    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])

    best_count = -1
    best = None
    for R, t in candidates:
        P1 = K @ np.hstack([R, t.reshape(3, 1)])
        xyz = triangulate_dlt(P0, P1, pts0, pts1)

        depth0 = xyz[:, 2]  # depth in camera 0's frame (camera 0 = identity, so this is just Z)
        depth1 = (R @ xyz.T + t.reshape(3, 1))[2, :]  # depth in camera 1's frame
        count = int(np.sum((depth0 > 0) & (depth1 > 0)))

        if count > best_count:
            best_count = count
            best = (R, t, xyz)

    if best is None:
        raise RuntimeError("Cheirality check failed to select a pose")
    return best
