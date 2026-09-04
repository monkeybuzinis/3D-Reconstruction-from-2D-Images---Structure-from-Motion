"""
Two-view seed check: SIFT match -> RANSAC F -> E -> pose -> DLT triangulation -> Map.

Sanity check for the two-view seed stage: reconstructs just the 0000/0001
pair and reports the numbers that tell you whether F/E/pose/triangulation
are all mutually consistent (median reprojection error should be well under
1px if everything's wired correctly).

Run from the repo root:
    python scripts/verify_two_view.py
"""

from pathlib import Path
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfm.features.matching import sift_match
from sfm.io.fountain import FOUNTAIN_K, default_fountain_dir
from sfm.io.ply import write_xyz_ply
from sfm.recon.two_view import reconstruct_two_view


def reprojection_errors(scene) -> np.ndarray:
    """Pixel reprojection error for every observation currently in the Map -- the key correctness metric for any reconstruction stage."""
    errs = []
    for obs in scene.observations:
        cam = scene.cameras[obs.camera_id]
        pt = scene.points[obs.point_id]
        P = cam.projection_matrix()
        X = np.append(pt.xyz, 1.0)
        proj = P @ X
        uv = proj[:2] / proj[2]
        errs.append(np.linalg.norm(uv - obs.uv))
    return np.array(errs)


def main() -> None:
    fountain_dir = default_fountain_dir()
    name0, name1 = "0000.png", "0001.png"
    img0 = cv2.imread(str(fountain_dir / name0), cv2.IMREAD_COLOR)
    img1 = cv2.imread(str(fountain_dir / name1), cv2.IMREAD_COLOR)
    if img0 is None or img1 is None:
        raise RuntimeError(f"Failed to read {name0}/{name1} from {fountain_dir}")

    scene, inlier_idx = reconstruct_two_view(img0, img1, FOUNTAIN_K, name0=name0, name1=name1)

    print(scene.summary())
    cam1 = scene.cameras[1]
    print("\nRecovered camera 1 pose:")
    print("R =\n", np.round(cam1.R, 6))
    print("t =", np.round(cam1.t, 6))
    print("camera center C =", np.round(cam1.camera_center(), 6))

    errs = reprojection_errors(scene)
    print(f"\nRANSAC inliers: {len(inlier_idx)}")
    print(f"Points triangulated in front of both cameras: {len(scene.points)}")
    print(f"Reprojection error (px): median={np.median(errs):.4f}  mean={errs.mean():.4f}  max={errs.max():.4f}")

    out_dir = ROOT / "outputs" / "two_view"
    out_dir.mkdir(parents=True, exist_ok=True)

    xyz = scene.points_xyz()
    ply_path = out_dir / "fountain_0000_0001_sparse.ply"
    write_xyz_ply(ply_path, xyz)
    print(f"\nWrote {xyz.shape[0]} triangulated points -> {ply_path}")

    # A handful of far/noisy points (typical for a small-baseline pair near
    # the epipole) can dominate the axis range. Clip the *view* robustly;
    # the saved PLY still has every triangulated point, unclipped.
    lo, hi = np.percentile(xyz, [1, 99], axis=0)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=0.6, c="k", alpha=0.6)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_title("Two-view seed: 0000/0001 triangulated cloud")
    ax.view_init(elev=15, azim=-70)
    fig.tight_layout()
    fig_path = out_dir / "fountain_0000_0001_sparse.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"Wrote preview -> {fig_path}")

    print("\nTwo-view seed OK.")


if __name__ == "__main__":
    main()
