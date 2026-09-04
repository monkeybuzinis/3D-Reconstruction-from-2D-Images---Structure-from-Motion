"""
Calibrated PnP (Perspective-n-Point / camera resectioning): given a set of
already-triangulated 3D points and where they show up in a NEW image, solve
for that new camera's pose. This is what lets incremental SfM add cameras
one at a time after the initial two-view seed (Essential-matrix pose
recovery only works for the first pair; every camera after that is added via
PnP instead, since we already have 3D points to resect against).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def reprojection_errors_pnp(
    K: np.ndarray, R: np.ndarray, t: np.ndarray, pts3d: np.ndarray, pts2d: np.ndarray
) -> np.ndarray:
    """Pixel reprojection error of each 3D point under pose (R, t) -- the metric RANSAC uses to score a candidate pose below."""
    P = K @ np.hstack([R, t.reshape(3, 1)])
    Xh = np.column_stack([pts3d, np.ones(len(pts3d))])
    proj = (P @ Xh.T).T
    uv = proj[:, :2] / proj[:, 2:3]
    return np.linalg.norm(uv - pts2d, axis=1)


def linear_pnp_dlt(K: np.ndarray, pts3d: np.ndarray, pts2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calibrated DLT camera resectioning.

    Given >=6 3D-2D correspondences and known intrinsics K, solve for the
    world-to-camera pose (R, t) such that x_cam = R @ X + t.

    The approach mirrors triangulate_dlt but solves for the CAMERA instead
    of the POINT: convert pixels to calibrated rays via K^-1 (so the unknown
    projection is just [R|t], not K[R|t]), then set up the same
    "cross-multiply the parallel-vectors condition" linear system and solve
    its nullspace via SVD. That gives [R|t] up to an unknown homogeneous
    scale and sign, which is then fixed in two steps:
      1. Frobenius-norm rescale so the 3x3 part has the size of an actual
         rotation matrix.
      2. A determinant check (a proper rotation has det=+1; flipping the
         overall sign of the solution flips the determinant's sign too, so
         exactly one of the two sign choices is physically valid).
    This is the same "nearest proper rotation" idea used for Fundamental
    (rank-2 projection) and Essential (equal-singular-value projection), just
    applied to the "must be an orthogonal matrix" constraint instead.
    """
    pts3d = np.asarray(pts3d, dtype=np.float64)
    pts2d = np.asarray(pts2d, dtype=np.float64)
    n = pts3d.shape[0]
    if n < 6:
        raise ValueError(f"Linear PnP needs >= 6 correspondences, got {n}")

    # Calibrated rays: xn = K^-1 [u, v, 1], normalized to z=1.
    Kinv = np.linalg.inv(K)
    uv1 = np.column_stack([pts2d, np.ones(n)])
    rays = (Kinv @ uv1.T).T
    xn = rays[:, 0] / rays[:, 2]
    yn = rays[:, 1] / rays[:, 2]

    # Hartley-style normalization of the 3D points (centroid to origin,
    # mean distance sqrt(3) -- the 3D analogue of normalize_points() in
    # fundamental.py) for a well-conditioned DLT.
    centroid = pts3d.mean(axis=0)
    shifted = pts3d - centroid
    mean_dist = np.mean(np.linalg.norm(shifted, axis=1))
    if mean_dist < 1e-12:
        raise np.linalg.LinAlgError("Degenerate (coincident) 3D points for PnP")
    scale = np.sqrt(3) / mean_dist
    U = np.array(
        [
            [scale, 0, 0, -scale * centroid[0]],
            [0, scale, 0, -scale * centroid[1]],
            [0, 0, scale, -scale * centroid[2]],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    Xh = np.column_stack([pts3d, np.ones(n)])
    Xn = (U @ Xh.T).T  # normalized homogeneous 3D points

    # Each correspondence gives 2 rows of the same "x*(row3.X) - row1.X = 0"
    # form as the Fundamental/triangulation DLTs, but with the 12 unknowns
    # of P = [R|t] (3x4) instead of a 3x3 or a 3-vector.
    A = np.zeros((2 * n, 12))
    for i in range(n):
        row = Xn[i]
        A[2 * i, 0:4] = row
        A[2 * i, 8:12] = -xn[i] * row
        A[2 * i + 1, 4:8] = row
        A[2 * i + 1, 8:12] = -yn[i] * row

    _, _, Vt = np.linalg.svd(A)
    P_norm = Vt[-1].reshape(3, 4)
    P = P_norm @ U  # undo the 3D normalization

    R_raw = P[:, :3]
    t_raw = P[:, 3]

    # R_raw = lambda * R_true for some unknown scalar lambda (could be
    # negative). Its magnitude is recovered from the Frobenius norm (a
    # rotation matrix always has ||R||_F = sqrt(3)); the sign is then fixed
    # by requiring det(R) = +1.
    lambda_abs = np.linalg.norm(R_raw, ord="fro") / np.sqrt(3)
    if lambda_abs < 1e-12:
        raise np.linalg.LinAlgError("Degenerate PnP solution (near-zero scale)")

    R_candidate = R_raw / lambda_abs
    t_candidate = t_raw / lambda_abs
    if np.linalg.det(R_candidate) < 0:
        R_candidate = -R_candidate
        t_candidate = -t_candidate

    # Final polish: project onto the nearest exactly-orthogonal matrix (small
    # residual noise keeps R_candidate from being perfectly orthogonal).
    Ur, _, Vtr = np.linalg.svd(R_candidate)
    R = Ur @ Vtr
    if np.linalg.det(R) < 0:  # rare numerical edge case
        Vtr[-1, :] *= -1
        R = Ur @ Vtr

    return R, t_candidate


def ransac_pnp(
    K: np.ndarray,
    pts3d: np.ndarray,
    pts2d: np.ndarray,
    threshold_px: float = 8.0,
    max_iters: int = 2000,
    seed: int = 0,
    min_samples: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RANSAC over linear_pnp_dlt, for the same reason RANSAC wraps the 8-point
    algorithm: the 2D-3D correspondences (tracks that survived to this
    camera) still contain some bad matches or badly-triangulated points, so
    fit from a random minimal sample, score by reprojection error, and keep
    re-fitting on the winning inlier set.

    Returns (R, t, inlier_indices).
    """
    pts3d = np.asarray(pts3d, dtype=np.float64)
    pts2d = np.asarray(pts2d, dtype=np.float64)
    n = len(pts3d)
    if n < min_samples:
        raise ValueError(f"Need >= {min_samples} correspondences for RANSAC PnP, got {n}")

    rng = np.random.default_rng(seed)
    best_inliers = np.array([], dtype=int)
    best_pose = None

    for _ in range(max_iters):
        idx = rng.choice(n, min_samples, replace=False)
        try:
            R, t = linear_pnp_dlt(K, pts3d[idx], pts2d[idx])
        except np.linalg.LinAlgError:
            continue  # e.g. a near-planar minimal sample; just try another one

        errs = reprojection_errors_pnp(K, R, t, pts3d, pts2d)
        inliers = np.where(errs < threshold_px)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_pose = (R, t)

    if best_pose is None or len(best_inliers) < min_samples:
        raise RuntimeError("RANSAC PnP failed to find a valid pose")

    # Re-fit using every inlier, not just the winning minimal sample. This
    # step can make things WORSE, not just better, and needs a safeguard:
    # linear/DLT resectioning is a known-degenerate method when the 3D
    # points are (near-)coplanar -- a flat building facade being the classic
    # case, and worse the more points from that facade you feed it, not
    # better, since coplanarity is a structural rank deficiency of the
    # linear system, not noise that averages out with more data. The search
    # phase can still succeed despite this: a random 6-point minimal sample
    # occasionally has enough incidental non-planar variation (parallax
    # noise, points off the main facade) to be well-conditioned, even when
    # the full best_inliers set is overwhelmingly coplanar. So: only trust
    # the refit if it's actually at least as good as what the search phase
    # already validated; otherwise keep the search-phase result.
    R_refit, t_refit = linear_pnp_dlt(K, pts3d[best_inliers], pts2d[best_inliers])
    errs_refit = reprojection_errors_pnp(K, R_refit, t_refit, pts3d, pts2d)
    inliers_refit = np.where(errs_refit < threshold_px)[0]

    if len(inliers_refit) >= len(best_inliers):
        return R_refit, t_refit, inliers_refit
    return best_pose[0], best_pose[1], best_inliers
