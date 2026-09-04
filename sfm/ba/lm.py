"""
Hand-written Levenberg-Marquardt Bundle Adjustment, using the analytic
Jacobian from sfm/ba/jacobian.py and a Schur-complement linear solve. This
is what sfm/ba/scipy_ba.py was standing in for: same optimization problem
(minimize total reprojection error over every camera pose and 3D point),
but every piece -- Jacobian, normal equations, damping, the linear solve --
is derived and coded here instead of handed to scipy.

Why the Schur complement specifically: each residual touches exactly one
camera's 6 parameters and one point's 3 parameters, so J^T J has a very
particular sparse block structure:

    [ B   E ] [delta_cam  ]   [ b ]
    [ E^T C ] [delta_point] = [ bp]

B (camera-camera) and C (point-point) are each BLOCK-DIAGONAL -- no residual
ever touches two different cameras, or two different points, at once. That
means C's 3x3 point blocks are trivial to invert one at a time, which is
exactly what lets us eliminate delta_point analytically (the "Schur
complement") and end up solving a MUCH smaller dense system in delta_cam
alone (6 x n_free_cameras, e.g. 60x60 here instead of ~25,000x25,000):

    S = B - E C^-1 E^T          (Schur complement)
    S @ delta_cam = b - E C^-1 bp

then back-substitute for delta_point once delta_cam is known. This is the
standard trick that makes Bundle Adjustment tractable at all with thousands
of points -- without it, the normal equations would be a dense system in
every parameter at once.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from sfm.ba.jacobian import scene_residuals_and_jacobians
from sfm.ba.so3 import rodrigues
from sfm.map import Map


def _total_cost(scene: Map) -> float:
    residuals, _, _, _, _ = scene_residuals_and_jacobians(scene)
    return 0.5 * float(np.sum(residuals**2))


def _build_normal_equations(
    scene: Map,
    free_camera_ids: List[int],
    point_ids: List[int],
    damping: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    """
    Assemble the Schur complement S, its right-hand side, and everything
    needed for back-substitution afterward.

    Returns (S, rhs, point_backsub) where point_backsub[point_idx] =
    (C_p_inv, list of (free_cam_idx, E_block)) -- exactly what's needed to
    recover delta_point once delta_cam has been solved for.
    """
    residuals, J_cam, J_point, obs_camera_ids, obs_point_ids = scene_residuals_and_jacobians(scene)

    cam_pos = {cid: i for i, cid in enumerate(free_camera_ids)}
    point_pos = {pid: i for i, pid in enumerate(point_ids)}
    n_free = len(free_camera_ids)
    n_points = len(point_ids)

    obs_free_idx = np.array([cam_pos.get(cid, -1) for cid in obs_camera_ids])
    obs_point_idx = np.array([point_pos[pid] for pid in obs_point_ids])
    free_mask = obs_free_idx >= 0

    # Per-observation normal-equation contributions, fully vectorized.
    JcT_Jc = np.einsum("nai,naj->nij", J_cam, J_cam)  # (M, 6, 6)
    JpT_Jp = np.einsum("nai,naj->nij", J_point, J_point)  # (M, 3, 3)
    JcT_r = np.einsum("nai,na->ni", J_cam, residuals)  # (M, 6)
    JpT_r = np.einsum("nai,na->ni", J_point, residuals)  # (M, 3)
    JcT_Jp = np.einsum("nai,naj->nij", J_cam, J_point)  # (M, 6, 3), only meaningful where camera is free

    # B, b: block-diagonal per free camera. C, bp: block-diagonal per point.
    B = np.zeros((n_free, 6, 6))
    b = np.zeros((n_free, 6))
    np.add.at(B, obs_free_idx[free_mask], JcT_Jc[free_mask])
    np.add.at(b, obs_free_idx[free_mask], -JcT_r[free_mask])

    C = np.zeros((n_points, 3, 3))
    bp = np.zeros((n_points, 3))
    np.add.at(C, obs_point_idx, JpT_Jp)
    np.add.at(bp, obs_point_idx, -JpT_r)

    # LM damping: inflate the diagonal, which both improves conditioning and
    # (for large damping) shrinks the step toward plain gradient descent.
    for i in range(n_free):
        B[i] += damping * np.diag(np.diag(B[i]))
    for p in range(n_points):
        C[p] += damping * np.diag(np.diag(C[p]))

    # Group per-point (free_camera, E_block) pairs -- this is the piece that
    # can't be vectorized away, since different points are observed by
    # different (and different numbers of) free cameras.
    point_free_obs: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    free_obs_indices = np.where(free_mask)[0]
    for k in free_obs_indices:
        p = int(obs_point_idx[k])
        ci = int(obs_free_idx[k])
        point_free_obs.setdefault(p, []).append((ci, JcT_Jp[k]))

    S = np.zeros((n_free * 6, n_free * 6))
    for i in range(n_free):
        S[i * 6 : i * 6 + 6, i * 6 : i * 6 + 6] += B[i]
    rhs = b.reshape(-1).copy()

    point_backsub: Dict[int, Tuple[np.ndarray, List[Tuple[int, np.ndarray]]]] = {}
    for p, obs_list in point_free_obs.items():
        Cp_inv = np.linalg.inv(C[p])
        point_backsub[p] = (Cp_inv, obs_list)

        for ci, Eci in obs_list:
            rhs[ci * 6 : ci * 6 + 6] -= Eci @ (Cp_inv @ bp[p])
            for cj, Ecj in obs_list:
                S[ci * 6 : ci * 6 + 6, cj * 6 : cj * 6 + 6] -= Eci @ Cp_inv @ Ecj.T

    # Points with no free-camera observation at all (shouldn't happen given
    # every point needs >=2 distinct cameras and only one camera is fixed,
    # but handled defensively): they just don't couple into S.
    for p in range(n_points):
        if p not in point_backsub:
            point_backsub[p] = (np.linalg.inv(C[p]), [])

    return S, rhs, point_backsub, bp


def _solve_and_apply(
    scene: Map,
    free_camera_ids: List[int],
    point_ids: List[int],
    S: np.ndarray,
    rhs: np.ndarray,
    point_backsub: Dict[int, Tuple[np.ndarray, List[Tuple[int, np.ndarray]]]],
    bp: np.ndarray,
) -> None:
    """Solve S @ delta_cam = rhs, back-substitute for delta_point, and apply both updates to `scene` in place."""
    delta_cam = np.linalg.solve(S, rhs)

    for i, cid in enumerate(free_camera_ids):
        domega = delta_cam[i * 6 : i * 6 + 3]
        dt = delta_cam[i * 6 + 3 : i * 6 + 6]
        cam = scene.cameras[cid]
        cam.R = rodrigues(domega) @ cam.R  # left-multiply: matches the perturbation convention the Jacobian was derived under
        cam.t = cam.t + dt

    for p, pid in enumerate(point_ids):
        Cp_inv, obs_list = point_backsub[p]
        rhs_p = bp[p].copy()
        for ci, Eci in obs_list:
            rhs_p -= Eci.T @ delta_cam[ci * 6 : ci * 6 + 6]
        delta_X = Cp_inv @ rhs_p
        scene.points[pid].xyz = scene.points[pid].xyz + delta_X


def hand_bundle_adjust(
    scene: Map,
    fixed_camera_ids: Sequence[int] = (0,),
    max_iters: int = 30,
    damping0: float = 1e-3,
    damping_up: float = 10.0,
    damping_down: float = 10.0,
    cost_tol: float = 1e-6,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Hand-written Levenberg-Marquardt Bundle Adjustment.

    Same interface/contract as sfm.ba.scipy_ba.bundle_adjust: mutates
    `scene` in place, fixes fixed_camera_ids for gauge freedom, and returns
    before/after cost. Internally uses the analytic Jacobian + Schur
    complement described in this module's docstring, plus a standard LM
    accept/reject loop: try a step, keep it and loosen damping if cost
    dropped, otherwise undo it and tighten damping (making the step more
    like gradient descent, which is more reliable near a bad linearization)
    and try again.
    """
    free_camera_ids = [cid for cid in sorted(scene.cameras) if cid not in fixed_camera_ids]
    point_ids = sorted(scene.points)

    cost0 = _total_cost(scene)
    cost = cost0
    damping = damping0

    for iteration in range(max_iters):
        S, rhs, point_backsub, bp = _build_normal_equations(scene, free_camera_ids, point_ids, damping)

        # Snapshot state so a rejected step can be undone exactly.
        snapshot = {
            "R": {cid: scene.cameras[cid].R.copy() for cid in free_camera_ids},
            "t": {cid: scene.cameras[cid].t.copy() for cid in free_camera_ids},
            "xyz": {pid: scene.points[pid].xyz.copy() for pid in point_ids},
        }

        try:
            _solve_and_apply(scene, free_camera_ids, point_ids, S, rhs, point_backsub, bp)
        except np.linalg.LinAlgError:
            damping *= damping_up
            continue

        new_cost = _total_cost(scene)

        if new_cost < cost:
            improvement = cost - new_cost
            cost = new_cost
            damping = max(damping / damping_down, 1e-12)
            if verbose:
                print(f"  iter {iteration:3d}: accepted, cost={cost:.4f}, damping={damping:.2e}")
            if improvement < cost_tol * max(1.0, cost):
                break
        else:
            for cid in free_camera_ids:
                scene.cameras[cid].R = snapshot["R"][cid]
                scene.cameras[cid].t = snapshot["t"][cid]
            for pid in point_ids:
                scene.points[pid].xyz = snapshot["xyz"][pid]
            damping *= damping_up
            if verbose:
                print(f"  iter {iteration:3d}: rejected, damping={damping:.2e}")
            if damping > 1e12:
                break

    return {"cost_before": cost0, "cost_after": cost, "n_observations": len(scene.observations)}
