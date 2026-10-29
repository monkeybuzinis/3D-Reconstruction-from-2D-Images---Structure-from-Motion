"""
Ablation for the resectioning refit guard (report Section 8.1).

Why this exists. The "11/23 cameras before the fix, 21/23 after" figure was
recorded as a development note while the defect was being fixed, before the
project was under version control -- sfm/geometry/pnp.py has a single commit,
so the pre-fix code is not recoverable from git and no log or scene cache of
the failing run was kept. A before/after claim in a report needs evidence
that can be regenerated, so this script reconstructs the experiment.

Method. Tracks are built once and both variants run on the identical track
set, so feature matching cannot differ between the two arms. Both RANSAC
stages in this pipeline are already seeded (fundamental: seed=42,
resectioning: seed=0), so with shared tracks the only difference between the
arms is the guard itself.

The two arms:
  with guard     -- current behaviour: accept the all-inlier refit only if
                    it is at least as good as the pose the search phase
                    already validated.
  without guard  -- the pre-fix behaviour: always return the refit.

Run from the repo root (slow -- builds tracks on full-resolution images):
    .venv/bin/python3 scripts/ablate_pnp_guard.py dataset/herzjesu
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfm.features.tracks import build_tracks
from sfm.geometry import pnp as pnp_mod
from sfm.io.generic import load_scene
from sfm.ba.lm import hand_bundle_adjust
from sfm.recon import incremental as inc_mod
from sfm.recon.incremental import filter_outlier_observations, run_incremental_sfm


def reprojection_errors(scene) -> np.ndarray:
    """Same measure scripts/reconstruct.py reports, duplicated here so the
    ablation does not depend on that script's internals."""
    errs = []
    for obs in scene.observations:
        cam = scene.cameras[obs.camera_id]
        pt = scene.points[obs.point_id]
        proj = cam.projection_matrix() @ np.append(pt.xyz, 1.0)
        uv = proj[:2] / proj[2]
        errs.append(np.linalg.norm(uv - obs.uv))
    return np.array(errs)


def ransac_pnp_no_guard(K, pts3d, pts2d, threshold_px=8.0, max_iters=2000, seed=0, min_samples=6):
    """
    The pre-fix implementation: identical to the current one except that the
    all-inlier refit is returned unconditionally, with no check that it is
    any better than the pose the search phase validated.
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
            R, t = pnp_mod.linear_pnp_dlt(K, pts3d[idx], pts2d[idx])
        except np.linalg.LinAlgError:
            continue
        errs = pnp_mod.reprojection_errors_pnp(K, R, t, pts3d, pts2d)
        inliers = np.where(errs < threshold_px)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_pose = (R, t)
    if best_pose is None or len(best_inliers) < min_samples:
        raise RuntimeError("RANSAC PnP failed to find a valid pose")

    R_refit, t_refit = pnp_mod.linear_pnp_dlt(K, pts3d[best_inliers], pts2d[best_inliers])
    errs_refit = pnp_mod.reprojection_errors_pnp(K, R_refit, t_refit, pts3d, pts2d)
    inliers_refit = np.where(errs_refit < threshold_px)[0]
    return R_refit, t_refit, inliers_refit  # the defect: returned unconditionally


def run_arm(label, images, K, keypoints, tracks, seed, outlier_px):
    print(f"\n--- {label} ---", flush=True)
    t0 = time.time()
    try:
        scene, log = run_incremental_sfm(
            images, K, keypoints, tracks, seed=seed,
            ba_max_nfev=30, bundle_adjust_fn=hand_bundle_adjust,
        )
    except RuntimeError as e:
        print(f"  reconstruction failed: {e}")
        return {"label": label, "registered": 0, "n_images": len(images), "failed": str(e)}

    n_reg = 2 + len(log)
    hand_bundle_adjust(scene, fixed_camera_ids=(seed[0],), max_iters=50)
    n_before = len(scene.points)
    scene = filter_outlier_observations(scene, max_error_px=outlier_px)
    hand_bundle_adjust(scene, fixed_camera_ids=(seed[0],), max_iters=50)
    keep = len(scene.points)
    errs = reprojection_errors(scene)

    r = {
        "label": label,
        "registered": n_reg,
        "n_images": len(images),
        "points_before_filter": n_before,
        "points_after_filter": keep,
        "keep_rate_pct": 100.0 * keep / n_before if n_before else float("nan"),
        "median_reproj_px": float(np.median(errs)),
        "seconds": time.time() - t0,
    }
    print(f"  registered {n_reg}/{len(images)} cameras in {r['seconds']:.0f}s")
    print(f"  outlier filter kept {keep}/{n_before} points ({r['keep_rate_pct']:.1f}%)")
    print(f"  median reprojection error {r['median_reproj_px']:.3f} px")
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", type=Path)
    ap.add_argument("--seed-pair", type=int, nargs=2, default=(0, 1))
    ap.add_argument("--outlier-px", type=float, default=4.0)
    args = ap.parse_args()

    print(f"Loading {args.images} ...", flush=True)
    images, K = load_scene(args.images)
    print(f"{len(images)} images loaded.", flush=True)

    # Built once, shared by both arms: matching is the only unseeded stage in
    # this pipeline, so sharing the tracks removes it as a confound entirely.
    t0 = time.time()
    keypoints, tracks = build_tracks(images)
    print(f"Built {len(tracks)} tracks in {time.time()-t0:.0f}s (shared by both arms)", flush=True)

    original = pnp_mod.ransac_pnp
    results = []
    try:
        inc_mod.ransac_pnp = original
        results.append(run_arm("with the guard (current)", images, K, keypoints, tracks,
                               tuple(args.seed_pair), args.outlier_px))
        inc_mod.ransac_pnp = ransac_pnp_no_guard
        results.append(run_arm("without the guard (pre-fix)", images, K, keypoints, tracks,
                               tuple(args.seed_pair), args.outlier_px))
    finally:
        inc_mod.ransac_pnp = original

    print("\n" + "=" * 72)
    print(f"{'arm':30s} {'registered':>12s} {'keep rate':>11s} {'median px':>11s}")
    print("=" * 72)
    for r in results:
        reg = f"{r['registered']}/{r['n_images']}"
        kr = f"{r['keep_rate_pct']:.1f}%" if "keep_rate_pct" in r else "—"
        mp = f"{r['median_reproj_px']:.3f}" if "median_reproj_px" in r else "—"
        print(f"{r['label']:30s} {reg:>12s} {kr:>11s} {mp:>11s}")

    import json
    out = ROOT / "evaluation" / "metrics" / f"ablation_pnp_guard_{args.images.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
