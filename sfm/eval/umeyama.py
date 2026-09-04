"""
Umeyama alignment: the closed-form similarity transform (scale + rotation +
translation) that best maps one set of corresponding 3D points onto another.

Why this is needed at all: monocular SfM has no way to know true scale or
absolute orientation (camera 0 is arbitrarily fixed at the world origin --
see sfm/ba/scipy_ba.py's gauge-freedom note). This project's reconstruction
and the reference solve (sfm/io/reference.py) are two independent SfM runs
of the same physical scene, each in its own arbitrary coordinate frame.
Umeyama finds the one similarity transform that reconciles them, using the
11 camera centers as known correspondences (the same physical camera
position, reconstructed twice) -- which is what makes evaluating the point
cloud against a reference meaningful at all.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def umeyama_alignment(source: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Find (scale, R, t) minimizing sum_i || target_i - (scale * R @ source_i + t) ||^2,
    via Umeyama (1991)'s closed-form solution.

    source, target: (N, 3) arrays of N corresponding points (N >= 3, not
    collinear). Returns (scale, R (3,3), t (3,)) such that
    target ~= scale * (R @ source.T).T + t.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n = source.shape[0]
    if n < 3:
        raise ValueError(f"Umeyama alignment needs >= 3 correspondences, got {n}")

    mu_source = source.mean(axis=0)
    mu_target = target.mean(axis=0)
    src_centered = source - mu_source
    tgt_centered = target - mu_target

    # Variance of the source cloud -- the denominator that turns the
    # cross-covariance's singular values into an actual scale factor.
    var_source = np.mean(np.sum(src_centered**2, axis=1))

    # Cross-covariance between target and source.
    cov = (tgt_centered.T @ src_centered) / n

    U, D, Vt = np.linalg.svd(cov)

    # Reflection correction: an unconstrained U @ Vt can be a reflection
    # (det = -1) rather than a proper rotation. Flipping the sign of the
    # smallest-singular-value axis fixes this without disturbing the
    # dominant alignment direction -- the same "nearest proper rotation"
    # idea used throughout sfm/geometry (Essential decomposition, PnP).
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    scale = np.trace(np.diag(D) @ S) / var_source
    t = mu_target - scale * (R @ mu_source)

    return scale, R, t


def apply_similarity(scale: float, R: np.ndarray, t: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply the (scale, R, t) transform to an (N, 3) point array."""
    return scale * (R @ np.asarray(points, dtype=np.float64).T).T + t
