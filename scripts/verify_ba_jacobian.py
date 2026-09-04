"""
Finite-difference validation of the analytic BA Jacobian (sfm/ba/jacobian.py).

The plan explicitly calls for this check BEFORE the hand-written Jacobian is
trusted inside the LM solver (sfm/ba/lm.py): pick real observations from an
actual reconstructed scene, perturb each parameter by a small epsilon, and
compare the resulting numerical derivative against what
batch_project_and_jacobian() computes analytically. A real bug (sign error,
wrong chain-rule term, mixed-up convention) shows up here as a large
discrepancy -- silently wrong analytic derivatives would otherwise still let
LM "run" while converging to nonsense or not converging at all, which is a
much harder failure to diagnose after the fact.

Run from the repo root:
    python scripts/verify_ba_jacobian.py
"""

from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfm.ba.jacobian import batch_project_and_jacobian
from sfm.ba.so3 import rodrigues
from sfm.io.fountain import FOUNTAIN_K, default_fountain_dir
from sfm.recon.two_view import reconstruct_two_view


def numeric_derivative_camera(K, R, t, X, obs_uv, param_idx, eps=1e-6):
    """
    Central-difference derivative of the residual (2,) w.r.t. one of the 6
    camera params: 0-2 are the rotation perturbation omega (composed as
    R_pert = Exp(omega) @ R, matching jacobian.py's left-perturbation
    convention exactly), 3-5 are translation.
    """

    def residual_at(omega, t_):
        R_pert = rodrigues(omega) @ R
        uv_pred, _, _ = batch_project_and_jacobian(K[None], R_pert[None], t_[None], X[None])
        return uv_pred[0] - obs_uv

    if param_idx < 3:
        dw = np.zeros(3)
        dw[param_idx] = eps
        r_plus = residual_at(dw, t)
        r_minus = residual_at(-dw, t)
    else:
        dt = np.zeros(3)
        dt[param_idx - 3] = eps
        r_plus = residual_at(np.zeros(3), t + dt)
        r_minus = residual_at(np.zeros(3), t - dt)
    return (r_plus - r_minus) / (2 * eps)


def numeric_derivative_point(K, R, t, X, obs_uv, param_idx, eps=1e-6):
    """Central-difference derivative of the residual w.r.t. one of the 3 point coordinates."""
    dX = np.zeros(3)
    dX[param_idx] = eps
    uv_plus, _, _ = batch_project_and_jacobian(K[None], R[None], t[None], (X + dX)[None])
    uv_minus, _, _ = batch_project_and_jacobian(K[None], R[None], t[None], (X - dX)[None])
    return ((uv_plus[0] - obs_uv) - (uv_minus[0] - obs_uv)) / (2 * eps)


def main() -> None:
    # A real (non-synthetic) but small, fast scene: two-view reconstruction
    # already gives non-trivial R/t and 1000+ real triangulated points.
    fountain_dir = default_fountain_dir()
    img0 = cv2.imread(str(fountain_dir / "0000.png"), cv2.IMREAD_COLOR)
    img1 = cv2.imread(str(fountain_dir / "0001.png"), cv2.IMREAD_COLOR)
    scene, _ = reconstruct_two_view(img0, img1, FOUNTAIN_K)

    cam1 = scene.cameras[1]  # camera 0 is identity/fixed; camera 1 has a real, non-trivial pose
    K = cam1.K

    rng = np.random.default_rng(0)
    point_ids = rng.choice(sorted(scene.points), size=30, replace=False)

    max_cam_err = 0.0
    max_point_err = 0.0
    for pid in point_ids:
        pt = scene.points[pid]
        obs = next(o for o in scene.track(pid) if o.camera_id == 1)

        _, J_cam, J_point = batch_project_and_jacobian(
            K[None], cam1.R[None], cam1.t[None], pt.xyz[None]
        )
        J_cam, J_point = J_cam[0], J_point[0]

        for k in range(6):
            num = numeric_derivative_camera(K, cam1.R, cam1.t, pt.xyz, obs.uv, k)
            max_cam_err = max(max_cam_err, np.abs(num - J_cam[:, k]).max())

        for k in range(3):
            num = numeric_derivative_point(K, cam1.R, cam1.t, pt.xyz, obs.uv, k)
            max_point_err = max(max_point_err, np.abs(num - J_point[:, k]).max())

    print("Max |analytic - finite-difference| over 30 points x (6 camera + 3 point) params:")
    print(f"  camera Jacobian (rotation + translation): {max_cam_err:.3e}")
    print(f"  point Jacobian:                           {max_point_err:.3e}")

    tol = 1e-4
    ok = max_cam_err < tol and max_point_err < tol
    print(f"\n{'PASS' if ok else 'FAIL'} (tolerance {tol:.0e})")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
