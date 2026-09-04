"""
Scene-expansion check: build tracks over all 11 Fountain images, seed a
two-view reconstruction, incrementally register the rest via PnP, triangulate
newly-visible points along the way, then run scipy Bundle Adjustment.

This is the sanity check for the whole "scene expansion" stage -- it
exercises every module written for it (tracks, PnP, incremental
registration, BA) together on the full 11-image dataset, not just a pair.
Watch the registration log and the reprojection-error numbers: if a camera
gets very few PnP inliers relative to what was "available", or the max
error stays huge after BA, that's the pipeline telling you something
upstream (tracks/triangulation/PnP) is unreliable for that camera.

Run from the repo root:
    python scripts/verify_incremental.py
"""

from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfm.ba.scipy_ba import bundle_adjust
from sfm.features.tracks import build_tracks
from sfm.io.fountain import FOUNTAIN_K, default_fountain_dir, load_images
from sfm.io.ply import write_xyz_ply
from sfm.recon.incremental import filter_outlier_observations, run_incremental_sfm


def reprojection_errors(scene) -> np.ndarray:
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


def report(label: str, scene) -> None:
    errs = reprojection_errors(scene)
    print(
        f"{label}: {scene.summary()}\n"
        f"  reprojection error (px): median={np.median(errs):.4f} "
        f"mean={errs.mean():.4f} max={errs.max():.4f} p95={np.percentile(errs, 95):.4f}"
    )


def main() -> None:
    fountain_dir = default_fountain_dir()
    images = load_images(fountain_dir)

    # Step 1: multi-view tracks (all-pairs matching + geometric verification
    # + union-find chaining). One-time cost that everything else reuses.
    t0 = time.time()
    keypoints, tracks = build_tracks(images)
    print(f"Built {len(tracks)} tracks over {len(images)} images in {time.time() - t0:.1f}s")

    # Step 2: seed + incremental PnP registration + triangulation, with
    # periodic BA baked into run_incremental_sfm itself (see the docstring
    # there for why that's needed rather than BA-once-at-the-end).
    t0 = time.time()
    scene, log = run_incremental_sfm(images, FOUNTAIN_K, keypoints, tracks, seed=(0, 1))
    print(f"Incremental registration done in {time.time() - t0:.1f}s\n")

    print("Registration order (camera, correspondences available, PnP inliers):")
    for cid, n_avail, n_inliers in log:
        print(f"  camera {cid:2d}: {n_avail:4d} available -> {n_inliers:4d} inliers")
    n_registered = 2 + len(log)  # +2 for the two seed cameras, which aren't in `log`
    print(f"\nRegistered {n_registered}/{len(images)} cameras")

    report("Before BA", scene)

    # Step 3: one more, more thorough BA pass (higher max_nfev than the
    # quick per-camera passes during registration) now that every camera
    # is in.
    t0 = time.time()
    stats = bundle_adjust(scene, fixed_camera_ids=(0,), max_nfev=100, verbose=0)
    print(f"\nBundle adjustment done in {time.time() - t0:.1f}s")
    print(f"  cost before={stats['cost_before']:.2f}  cost after={stats['cost_after']:.2f}")

    report("After BA", scene)

    # Step 4: drop the handful of catastrophic-outlier points that dominate
    # BA's squared-error cost (see filter_outlier_observations' docstring),
    # then re-optimize -- this is usually where the numbers improve the most.
    n_before = len(scene.points)
    scene = filter_outlier_observations(scene, max_error_px=4.0)
    print(f"\nOutlier filter (>4px): kept {len(scene.points)}/{n_before} points")

    stats = bundle_adjust(scene, fixed_camera_ids=(0,), max_nfev=100, verbose=0)
    print(f"Re-optimized after filtering: cost before={stats['cost_before']:.2f}  cost after={stats['cost_after']:.2f}")
    report("After filter + re-BA", scene)

    out_dir = ROOT / "outputs" / "incremental"
    out_dir.mkdir(parents=True, exist_ok=True)

    xyz = scene.points_xyz()
    ply_path = out_dir / "fountain_incremental_sparse.ply"
    write_xyz_ply(ply_path, xyz)
    print(f"\nWrote {xyz.shape[0]} points -> {ply_path}")

    # Robust axis limits again (see verify_two_view.py) -- purely a
    # visualization concern, doesn't touch the saved data.
    lo, hi = np.percentile(xyz, [1, 99], axis=0)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=0.4, c="k", alpha=0.5)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Incremental SfM: {n_registered}/{len(images)} cameras, {xyz.shape[0]} points")
    ax.view_init(elev=15, azim=-70)
    fig.tight_layout()
    fig_path = out_dir / "fountain_incremental_sparse.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"Wrote preview -> {fig_path}")

    print("\nIncremental SfM + scipy BA OK." if n_registered == len(images) else "\nWARNING: not all cameras registered.")


if __name__ == "__main__":
    main()
