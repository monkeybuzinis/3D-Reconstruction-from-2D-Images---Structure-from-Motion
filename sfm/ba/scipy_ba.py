"""
Bundle Adjustment via scipy.optimize.least_squares.

BA is the step that jointly refines every camera pose AND every 3D point at
once to minimize total reprojection error -- everything upstream (two-view
pose, PnP, DLT triangulation) only ever looked at a small local subset of
the data, so small errors accumulate; BA is what cleans that up using every
observation simultaneously.

This is the "safe library" BA called for in the plan: get the whole scene
optimizing correctly first, using scipy's general-purpose nonlinear
least-squares solver as the optimizer backend. The hand-written
Levenberg-Marquardt solver (analytic Jacobian, so(3) parameterization,
Schur complement) replaces this in a later stage -- this file is the
"does the overall BA formulation even work" checkpoint before writing that
by hand.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from sfm.map import Map


def _project(Ks: np.ndarray, Rs: np.ndarray, ts: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Vectorized pinhole projection over many (camera, point) pairs at once. Ks/Rs: (N,3,3); ts/X: (N,3). Returns (N,2) pixel coords."""
    Xc = np.einsum("kij,kj->ki", Rs, X) + ts  # world point -> camera-frame point
    proj_h = np.einsum("kij,kj->ki", Ks, Xc)  # camera-frame point -> homogeneous pixel
    return proj_h[:, :2] / proj_h[:, 2:3]  # de-homogenize


def bundle_adjust(
    scene: Map,
    fixed_camera_ids: Sequence[int] = (0,),
    max_nfev: int = 100,
    verbose: int = 0,
) -> Dict[str, float]:
    """
    Refine every free camera pose and every 3D point to minimize total
    reprojection error.

    fixed_camera_ids stay fixed to remove the 6-DOF pose gauge freedom
    (rigidly rotating/translating the whole scene doesn't change any
    reprojection error, so without a fixed reference the optimizer has no
    way to pick one specific answer out of infinitely many equally-good
    ones). A 1-DOF overall-scale gauge freedom remains too (uniformly
    scaling all translations and points also leaves reprojection error
    unchanged -- this is the classic "monocular SfM can't know true scale"
    fact) -- harmless for least_squares, which simply doesn't move along
    that null direction.

    Mutates `scene` in place (updates camera R/t and point xyz) and returns
    the total reprojection cost before/after, mostly for logging/diagnostics.
    """
    all_camera_ids = sorted(scene.cameras)
    free_camera_ids = [cid for cid in all_camera_ids if cid not in fixed_camera_ids]
    point_ids = sorted(scene.points)

    cam_pos = {cid: i for i, cid in enumerate(all_camera_ids)}
    free_pos = {cid: i for i, cid in enumerate(free_camera_ids)}
    point_pos = {pid: i for i, pid in enumerate(point_ids)}

    # Precompute, once, which camera/point each observation refers to as
    # plain integer indices -- this is what lets the residual function below
    # be fully vectorized (no per-observation Python loop) even though the
    # scene has tens of thousands of observations.
    n_obs = len(scene.observations)
    obs_cam_idx = np.array([cam_pos[o.camera_id] for o in scene.observations])
    obs_point_idx = np.array([point_pos[o.point_id] for o in scene.observations])
    obs_uv = np.array([o.uv for o in scene.observations])
    obs_free_cam = np.array([free_pos.get(o.camera_id, -1) for o in scene.observations])

    Ks_all = np.stack([scene.cameras[cid].K for cid in all_camera_ids])

    n_cam_params = len(free_camera_ids) * 6  # 3 rotation (axis-angle) + 3 translation, per free camera
    n_point_params = len(point_ids) * 3

    def pack() -> np.ndarray:
        """Flatten current free-camera poses + all point positions into scipy's one big parameter vector."""
        cam_block = []
        for cid in free_camera_ids:
            cam = scene.cameras[cid]
            rvec, _ = cv2.Rodrigues(cam.R)  # rotation matrix -> axis-angle (3 numbers instead of 9, minimal + unconstrained)
            cam_block.append(np.concatenate([rvec.ravel(), cam.t]))
        cam_block = np.concatenate(cam_block) if cam_block else np.zeros(0)
        pts_block = np.concatenate([scene.points[pid].xyz for pid in point_ids])
        return np.concatenate([cam_block, pts_block])

    def unpack_poses(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Rebuild (R, t) for every camera -- fixed ones from the scene, free ones from the current parameter vector x."""
        Rs = np.empty((len(all_camera_ids), 3, 3))
        ts = np.empty((len(all_camera_ids), 3))
        for cid in all_camera_ids:
            i = cam_pos[cid]
            if cid in free_pos:
                j = free_pos[cid]
                block = x[j * 6 : j * 6 + 6]
                R, _ = cv2.Rodrigues(block[:3])  # axis-angle -> rotation matrix
                Rs[i], ts[i] = R, block[3:6]
            else:
                cam = scene.cameras[cid]
                Rs[i], ts[i] = cam.R, cam.t
        return Rs, ts

    def residuals(x: np.ndarray) -> np.ndarray:
        """The vector scipy drives to zero: (predicted pixel - observed pixel) for every observation, stacked."""
        Rs, ts = unpack_poses(x)
        pts = x[n_cam_params:].reshape(-1, 3)

        uv_proj = _project(Ks_all[obs_cam_idx], Rs[obs_cam_idx], ts[obs_cam_idx], pts[obs_point_idx])
        return (uv_proj - obs_uv).ravel()

    def sparsity() -> lil_matrix:
        """
        Which residual depends on which parameter -- BA's Jacobian is huge
        (2*n_observations x n_params) but extremely sparse, since each
        observation's residual only depends on ONE camera's 6 params and
        ONE point's 3 params, not all of them. Telling scipy this pattern up
        front lets it use a sparse finite-difference scheme instead of
        perturbing every one of the thousands of parameters one at a time,
        which is the difference between this running in seconds vs. hours.
        """
        n_residuals = 2 * n_obs
        n_params = n_cam_params + n_point_params
        J = lil_matrix((n_residuals, n_params), dtype=int)
        rows = np.arange(n_obs)
        free_mask = obs_free_cam >= 0
        for c in range(6):
            cols = obs_free_cam[free_mask] * 6 + c
            J[2 * rows[free_mask], cols] = 1
            J[2 * rows[free_mask] + 1, cols] = 1
        for c in range(3):
            cols = n_cam_params + obs_point_idx * 3 + c
            J[2 * rows, cols] = 1
            J[2 * rows + 1, cols] = 1
        return J

    x0 = pack()
    cost0 = 0.5 * float(np.sum(residuals(x0) ** 2))

    result = least_squares(
        residuals,
        x0,
        jac_sparsity=sparsity(),
        method="trf",  # Trust Region Reflective: scipy's method for large sparse least-squares problems
        max_nfev=max_nfev,
        verbose=verbose,
    )

    # Write the optimized poses/points back into the Map.
    Rs, ts = unpack_poses(result.x)
    pts = result.x[n_cam_params:].reshape(-1, 3)

    for cid in free_camera_ids:
        i = cam_pos[cid]
        scene.cameras[cid].R = Rs[i]
        scene.cameras[cid].t = ts[i]
    for i, pid in enumerate(point_ids):
        scene.points[pid].xyz = pts[i]

    cost1 = 0.5 * float(np.sum(residuals(result.x) ** 2))
    return {"cost_before": cost0, "cost_after": cost1, "n_observations": n_obs}
