"""
Evaluation check: Umeyama-align our reconstruction to the reference solve
(Fountain/solution.graph) via corresponding camera centers, then compute
Accuracy/Completeness/F-score of our point cloud against the reference
point cloud.

IMPORTANT caveat, repeated here deliberately: solution.graph is an
independent SfM/BA solve of the same dataset, not the true Strecha
laser-scanned ground truth (that dataset's hosting service is no longer
reachable -- verified while building this stage). This evaluates
"how well does our reconstruction agree with another SfM solve of the same
photos," which is a meaningful and standard sanity check, but is a weaker
claim than "how accurate is this versus a laser scan."

Uses the cached scene from scripts/build_scene_cache.py -- run that first
if outputs/scene_cache.npz doesn't exist yet.

Run from the repo root:
    .venv/bin/python3 scripts/verify_evaluation.py
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfm.eval.metrics import evaluate_reconstruction
from sfm.eval.umeyama import apply_similarity, umeyama_alignment
from sfm.io.fountain import default_fountain_dir
from sfm.io.reference import load_reference_camera_centers, load_reference_points


def main() -> None:
    fountain_dir = default_fountain_dir()
    cache = np.load(ROOT / "outputs" / "scene_cache.npz")
    cam_ids, Rs, ts = cache["cam_ids"], cache["Rs"], cache["ts"]
    our_points = cache["xyz"]

    our_centers = np.stack([-Rs[i].T @ ts[i] for i in range(len(cam_ids))])

    ref_centers_by_id = load_reference_camera_centers(fountain_dir)
    ref_centers = np.stack([ref_centers_by_id[cid] for cid in cam_ids])

    print("Aligning our 11 camera centers to the reference solve's 11 camera centers...")
    scale, R, t = umeyama_alignment(our_centers, ref_centers)
    print(f"  recovered scale: {scale:.4f}")

    aligned_centers = apply_similarity(scale, R, t, our_centers)
    center_errors = np.linalg.norm(aligned_centers - ref_centers, axis=1)
    print(f"  camera-center alignment residual (should be small): "
          f"mean={center_errors.mean():.5f} max={center_errors.max():.5f}")

    aligned_points = apply_similarity(scale, R, t, our_points)

    ref_points = load_reference_points(fountain_dir)
    print(f"\nOur points: {len(aligned_points):,}  Reference points: {len(ref_points):,}")

    # Thresholds scaled to the reference frame's own baseline (consecutive
    # reference cameras are ~0.15-0.2 apart -- see sfm/io/reference.py),
    # so "5% of baseline" etc. means something dataset-appropriate rather
    # than an arbitrary absolute number.
    ref_baseline = np.median(np.linalg.norm(np.diff(ref_centers, axis=0), axis=1))
    thresholds = tuple(round(ref_baseline * f, 5) for f in (0.05, 0.1, 0.25, 0.5))
    print(f"Reference camera baseline: {ref_baseline:.4f}  ->  thresholds: {thresholds}")

    metrics = evaluate_reconstruction(aligned_points, ref_points, thresholds=thresholds)

    print("\nAccuracy (our points -> nearest reference point):")
    print(f"  mean={metrics['accuracy_mean']:.5f}  median={metrics['accuracy_median']:.5f}")
    print("Completeness (reference points -> nearest our point):")
    print(f"  mean={metrics['completeness_mean']:.5f}  median={metrics['completeness_median']:.5f}")
    print("\nPrecision / Recall / F-score at each threshold:")
    for tau in thresholds:
        p, r, f = metrics[f"precision@{tau}"], metrics[f"recall@{tau}"], metrics[f"f_score@{tau}"]
        print(f"  tau={tau:.5f}:  precision={p:.3f}  recall={r:.3f}  f_score={f:.3f}")

    out_dir = ROOT / "outputs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ref_sample = ref_points[np.random.default_rng(0).choice(len(ref_points), min(20000, len(ref_points)), replace=False)]
    ax.scatter(ref_sample[:, 0], ref_sample[:, 1], ref_sample[:, 2], s=0.4, alpha=0.3, c="gray", label="reference")
    ax.scatter(aligned_points[:, 0], aligned_points[:, 1], aligned_points[:, 2], s=2, alpha=0.8, c="red", label="ours (aligned)")
    ax.scatter(aligned_centers[:, 0], aligned_centers[:, 1], aligned_centers[:, 2], s=40, c="blue", marker="^", label="our cameras (aligned)")
    ax.scatter(ref_centers[:, 0], ref_centers[:, 1], ref_centers[:, 2], s=40, c="green", marker="v", label="reference cameras")
    lo, hi = np.percentile(ref_points, [1, 99], axis=0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.view_init(elev=15, azim=-70)
    ax.legend()
    ax.set_title("Umeyama-aligned reconstruction vs. reference solve")
    fig.tight_layout()
    fig_path = out_dir / "alignment_overlay.png"
    fig.savefig(fig_path, dpi=120)
    print(f"\nWrote alignment overlay -> {fig_path}")

    print("\nEvaluation OK.")


if __name__ == "__main__":
    main()
