"""
DLT triangulation: given two posed cameras and a 2D correspondence, recover
the 3D point. This is the "turn matched pixels into an actual 3D point"
step used both by the two-view seed and by every later camera addition
(sfm/recon/incremental.py).
"""

from __future__ import annotations

import numpy as np


def triangulate_dlt(P0: np.ndarray, P1: np.ndarray, pts0: np.ndarray, pts1: np.ndarray) -> np.ndarray:
    """
    Direct Linear Transform triangulation.

    A 3D point X projects to pixel x in camera P as x = P X (homogeneous),
    i.e. x is parallel to (not equal to) P X. Cross-multiplying that
    parallel-vectors condition for both cameras gives 4 linear equations in
    the unknown X (2 independent ones per view); stacking them into a matrix
    A and solving Av=0 via SVD (the smallest singular vector) gives the
    least-squares 3D point, the same "solve a nullspace problem" trick used
    for the Fundamental matrix.

    P0, P1: (3, 4) camera projection matrices K [R | t].
    pts0, pts1: (N, 2) pixel correspondences, row i of pts0 seen by P0
        corresponds to row i of pts1 seen by P1.
    Returns (N, 3) XYZ in the world frame P0/P1 are expressed in.
    """
    pts0 = np.asarray(pts0, dtype=np.float64)
    pts1 = np.asarray(pts1, dtype=np.float64)
    n = pts0.shape[0]
    xyz = np.empty((n, 3), dtype=np.float64)

    for i in range(n):
        x0, y0 = pts0[i]
        x1, y1 = pts1[i]
        A = np.stack(
            [
                x0 * P0[2] - P0[0],
                y0 * P0[2] - P0[1],
                x1 * P1[2] - P1[0],
                y1 * P1[2] - P1[1],
            ],
            axis=0,
        )
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]  # nullspace vector = homogeneous 3D point, up to scale
        xyz[i] = X[:3] / X[3]  # de-homogenize

    return xyz
