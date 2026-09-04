"""
Hand-written so(3) rotation math: the skew-symmetric cross-product matrix and
the Rodrigues exponential map. This replaces cv2.Rodrigues (used as a
convenience in sfm/ba/scipy_ba.py) with our own implementation, since the
plan calls for the axis-angle parameterization itself to be hand-derived,
not just borrowed from OpenCV.
"""

from __future__ import annotations

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric matrix [v]_x such that [v]_x @ w == v cross w, for a single 3-vector."""
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def skew_batch(v: np.ndarray) -> np.ndarray:
    """Vectorized skew(): v is (N, 3), returns (N, 3, 3)."""
    n = v.shape[0]
    S = np.zeros((n, 3, 3))
    S[:, 0, 1] = -v[:, 2]
    S[:, 0, 2] = v[:, 1]
    S[:, 1, 0] = v[:, 2]
    S[:, 1, 2] = -v[:, 0]
    S[:, 2, 0] = -v[:, 1]
    S[:, 2, 1] = v[:, 0]
    return S


def rodrigues(omega: np.ndarray) -> np.ndarray:
    """
    so(3) exponential map: axis-angle vector -> 3x3 rotation matrix.

    R = I + sin(theta) K + (1 - cos(theta)) K^2, where theta = |omega| is the
    rotation angle and K = skew(omega / theta) is the skew matrix of the unit
    rotation axis. This is the same formula cv2.Rodrigues implements; having
    our own means the LM solver below (and its finite-difference check) don't
    depend on OpenCV for the one piece of math the plan specifically asks to
    be hand-derived.

    Falls back to the first-order approximation R ~= I + skew(omega) for
    tiny angles, where dividing by theta would be unstable.
    """
    theta = np.linalg.norm(omega)
    if theta < 1e-12:
        return np.eye(3) + skew(omega)

    k = omega / theta
    K = skew(k)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
