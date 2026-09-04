"""
Fundamental matrix estimation: Hartley normalization, the normalized 8-point
algorithm, and RANSAC. This is the first hand-written geometry step in the
pipeline: F encodes the epipolar constraint between two uncalibrated views
(x1' F x0 = 0 for every true correspondence), and is what the Essential
matrix (sfm/geometry/essential.py) gets built from once K is known.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def normalize_points(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hartley normalization: shift the centroid to the origin and rescale so
    the average point is at distance sqrt(2) from it.

    Why this matters: the 8-point algorithm below builds a matrix out of
    products like x*x', x*y', ... of raw pixel coordinates (which can be in
    the thousands). Those wildly different magnitudes make the SVD
    ill-conditioned and the resulting F inaccurate. Normalizing first (and
    un-normalizing F afterward) is the standard fix and is what makes the
    "8-point algorithm" numerically usable at all, as opposed to just
    theoretically correct.
    """
    centroid = np.mean(pts, axis=0)
    shifted = pts - centroid
    mean_dist = np.mean(np.linalg.norm(shifted, axis=1))
    scale = np.sqrt(2) / mean_dist

    T = np.array(
        [
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    pts_homo = np.column_stack([pts, np.ones(len(pts))])
    pts_norm = (T @ pts_homo.T).T
    return pts_norm[:, :2], T


def compute_fundamental_8point(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """
    Normalized 8-point algorithm.

    Each correspondence (x1,y1) <-> (x2,y2) gives one linear equation in the
    9 unknown entries of F (from x2' F x1 = 0, expanded out). Stack >=8 such
    equations into a matrix A and the least-squares F is the right singular
    vector of A for its smallest singular value (the "solve Af=0" trick).

    A true Fundamental matrix has rank 2 (its third singular value should be
    exactly zero), but the raw SVD solution above generally comes out full
    rank due to noise. The second SVD below re-projects it onto the nearest
    rank-2 matrix by zeroing that smallest singular value -- without this
    step, epipolar lines from F wouldn't actually all pass through a common
    epipole.
    """
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)

    x1, y1 = pts1_norm[:, 0], pts1_norm[:, 1]
    x2, y2 = pts2_norm[:, 0], pts2_norm[:, 1]

    A = np.column_stack(
        [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, np.ones(len(pts1))]
    )

    _, _, Vt = np.linalg.svd(A)
    F_norm = Vt[-1].reshape(3, 3)  # nullspace vector of A, reshaped to 3x3

    U, S, Vt_f = np.linalg.svd(F_norm)
    S[2] = 0.0  # enforce rank-2
    F_norm = U @ np.diag(S) @ Vt_f

    F = T2.T @ F_norm @ T1  # undo the Hartley normalization
    return F / F[2, 2]  # fix the arbitrary homogeneous scale for readability


def sampson_distance(pts1: np.ndarray, pts2: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Sampson distance: a first-order approximation of the geometric distance
    from each point to its epipolar line, cheap to compute and what RANSAC
    below uses to decide inlier vs. outlier for a candidate F.
    """
    N = len(pts1)
    p1 = np.column_stack([pts1, np.ones(N)])
    p2 = np.column_stack([pts2, np.ones(N)])

    F_p1 = (F @ p1.T).T
    FT_p2 = (F.T @ p2.T).T

    p2_F_p1 = np.sum(p2 * F_p1, axis=1)  # this is x2' F x1, ~0 for a perfect correspondence

    denom = F_p1[:, 0] ** 2 + F_p1[:, 1] ** 2 + FT_p2[:, 0] ** 2 + FT_p2[:, 1] ** 2
    return (p2_F_p1**2) / (denom + 1e-8)


def ransac_fundamental_matrix(
    pts1: np.ndarray,
    pts2: np.ndarray,
    threshold: float = 1.0,
    max_iters: int = 2000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    RANSAC wrapper around the 8-point algorithm.

    SIFT+FLANN+ratio-test matches (sfm/features/matching.py) still contain
    wrong correspondences -- RANSAC is what makes F estimation robust to
    those: repeatedly fit F from a random minimal sample of 8 points, count
    how many of ALL points agree with it (Sampson distance under
    `threshold`), and keep the F with the most agreement. The final F is
    then re-fit using every inlier, not just the winning 8, for a cleaner
    least-squares result.
    """
    best_inliers = np.array([], dtype=int)
    best_F = None
    N = len(pts1)

    rng = np.random.default_rng(seed)
    for _ in range(max_iters):
        indices = rng.choice(N, 8, replace=False)
        sample1, sample2 = pts1[indices], pts2[indices]

        try:
            F_cand = compute_fundamental_8point(sample1, sample2)
            errors = sampson_distance(pts1, pts2, F_cand)
            inliers = np.where(errors < threshold)[0]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_F = F_cand
        except np.linalg.LinAlgError:
            continue  # degenerate sample (e.g. near-collinear points); just try another one

    if best_F is None:
        raise RuntimeError("RANSAC failed to find any valid Fundamental matrix")

    if len(best_inliers) >= 8:
        best_F = compute_fundamental_8point(pts1[best_inliers], pts2[best_inliers])

    return best_F, best_inliers
