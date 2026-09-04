"""
Analytic reprojection-error Jacobian, vectorized over every observation in
the scene at once. This is the piece scipy computed for us via finite
differences (sfm/ba/scipy_ba.py just hands scipy a residual function and a
sparsity pattern); here we derive and hand-code the actual derivatives, so
the LM solver in sfm/ba/lm.py never calls scipy at all.

Derivation (left/global perturbation convention: R' = Exp(omega) @ R, i.e.
the rotation update is applied in world coordinates, not the camera's local
frame -- this is what makes the compact -skew(Y) formula below valid, and it
determines how updates get composed back in lm.py):

    Y  = R @ X                (rotate world point into camera frame, no translation yet)
    Xc = Y + t                (add translation -> point in camera coordinates)
    u  = fx * Xc.x / Xc.z + cx
    v  = fy * Xc.y / Xc.z + cy

For a small rotation perturbation omega (applied as R' = Exp(omega) @ R):
    Xc(omega) ~= Xc - skew(Y) @ omega     =>   dXc/domega = -skew(Y)
    dXc/dt = I_3                          (translation shifts Xc directly)
    dXc/dX = R                            (point moves directly through the fixed rotation)

and the pinhole projection derivative:
    d(u,v)/dXc = [[fx/z,   0,   -fx*x/z^2],
                  [  0,  fy/z,  -fy*y/z^2]]

Chaining these gives the 2x6 camera Jacobian (columns: 3 rotation + 3
translation) and 2x3 point Jacobian returned below.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from sfm.ba.so3 import skew_batch
from sfm.map import Map


def batch_project_and_jacobian(
    Ks: np.ndarray, Rs: np.ndarray, ts: np.ndarray, X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized pinhole projection + analytic Jacobians for many independent
    (camera, point) observations at once.

    Ks, Rs: (N, 3, 3); ts, X: (N, 3). Assumes zero-skew K (true for
    Fountain-P11 and every camera this project uses).

    Returns:
      uv_pred:  (N, 2)    predicted pixel for each observation
      J_cam:    (N, 2, 6) d(uv)/d[omega | t], the pose Jacobian
      J_point:  (N, 2, 3) d(uv)/dX, the point Jacobian
    """
    Y = np.einsum("nij,nj->ni", Rs, X)  # rotated point, before translation
    Xc = Y + ts
    x, y, z = Xc[:, 0], Xc[:, 1], Xc[:, 2]

    fx = Ks[:, 0, 0]
    fy = Ks[:, 1, 1]
    cx = Ks[:, 0, 2]
    cy = Ks[:, 1, 2]

    uv_pred = np.stack([fx * x / z + cx, fy * y / z + cy], axis=1)

    n = X.shape[0]
    duv_dXc = np.zeros((n, 2, 3))
    duv_dXc[:, 0, 0] = fx / z
    duv_dXc[:, 0, 2] = -fx * x / z**2
    duv_dXc[:, 1, 1] = fy / z
    duv_dXc[:, 1, 2] = -fy * y / z**2

    skew_Y = skew_batch(Y)  # (N, 3, 3)
    J_omega = -np.einsum("nij,njk->nik", duv_dXc, skew_Y)  # (N, 2, 3)
    J_t = duv_dXc  # dXc/dt = I, so this chain is just duv_dXc itself
    J_cam = np.concatenate([J_omega, J_t], axis=2)  # (N, 2, 6)

    J_point = np.einsum("nij,njk->nik", duv_dXc, Rs)  # (N, 2, 3)

    return uv_pred, J_cam, J_point


def scene_residuals_and_jacobians(
    scene: Map,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper: pull every observation's camera/point out of `scene`
    at its CURRENT pose/position and compute residuals + Jacobians for all
    of them in one batched call.

    Returns (residuals (M,2), J_cam (M,2,6), J_point (M,2,3), obs_camera_ids
    (M,), obs_point_ids (M,)) where M = number of observations.
    """
    obs_camera_ids = np.array([o.camera_id for o in scene.observations])
    obs_point_ids = np.array([o.point_id for o in scene.observations])
    obs_uv = np.array([o.uv for o in scene.observations])

    Ks = np.stack([scene.cameras[cid].K for cid in obs_camera_ids])
    Rs = np.stack([scene.cameras[cid].R for cid in obs_camera_ids])
    ts = np.stack([scene.cameras[cid].t for cid in obs_camera_ids])
    X = np.stack([scene.points[pid].xyz for pid in obs_point_ids])

    uv_pred, J_cam, J_point = batch_project_and_jacobian(Ks, Rs, ts, X)
    residuals = uv_pred - obs_uv
    return residuals, J_cam, J_point, obs_camera_ids, obs_point_ids
