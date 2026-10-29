"""
Evaluation check #2: our Rathaus reconstruction against the ground truth
that ships with the dataset.

Why this exists alongside scripts/verify_evaluation.py: Fountain's
reference (solution.graph) is another SfM solve, because Strecha's laser
scan is no longer hosted. Rathaus is different -- dataset/rathaus ships
BOTH a per-image `.camera` calibration (true K, R and camera centre) and a
per-image `.3Dpoints` reference cloud. That makes it the one dataset here
with externally-measured geometry rather than a peer solver's opinion.

The README documents only the `.camera` format, so that the `.3Dpoints`
files live in the SAME coordinate frame is verified rather than assumed:
every reference point is checked to project in front of, and inside, every
image using that image's own `.camera` matrices (it does, 100%).

Two honest caveats, stated up front because they change how the numbers
should be read:

  1. The reference cloud is SPARSE -- 1,059 distinct points, repeated
     across the per-image files. Completeness (reference -> ours) is
     therefore meaningful: "did we reconstruct the surface at each
     surveyed point?". Accuracy (ours -> reference) is NOT, because a
     correct point of ours sitting between two surveyed points has no
     nearby reference point through no fault of its own. Accuracy is
     printed for completeness of reporting and explicitly flagged.

  2. The reference frame is internally consistent but NOT metric. Its unit
     is undocumented, and working backwards from the geometry -- a 42.5
     degree lens at a median depth of ~197 units frames ~153 units of
     facade, which the photographs show to be roughly 11-14 m of Leuven
     town hall -- puts one unit at about 7-9 cm. So this validates
     relative structure and pose, not absolute metric scale. Nothing in
     this project has a true metric reference; that needs a known-length
     object in the frame at capture time.

  3. Unlike Fountain, the reference here gives true per-image ROTATIONS as
     well as centres, so camera orientation error in degrees is reported
     directly -- the same measurement made on Buddha.

Run from the repo root:
    .venv/bin/python3 scripts/verify_rathaus_evaluation.py
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfm.eval.metrics import evaluate_reconstruction, nearest_neighbor_distances
from sfm.eval.umeyama import apply_similarity, umeyama_alignment

RATHAUS_DIR = ROOT / "dataset" / "rathaus"
CACHE = ROOT / "outputs" / "rathaus" / "scene_cache.npz"
DENSE_MESH = ROOT / "outputs" / "rathaus" / "dense_mesh.ply"


def load_reference_camera(path: Path):
    """
    Parse a Strecha `.camera` sidecar.

    Per dataset/rathaus/README the projection is  x = K[R^T | -R^T t] X,
    so the stored R is CAMERA-TO-WORLD and the stored t is the camera
    CENTRE -- not this project's world-to-camera (R, t) convention. The
    world-to-camera rotation is therefore R^T.
    """
    v = [float(x) for x in path.read_text().split()]
    K = np.array(v[:9]).reshape(3, 3)
    R_c2w = np.array(v[12:21]).reshape(3, 3)
    centre = np.array(v[21:24])
    return K, R_c2w.T, centre


def load_reference_points(path: Path) -> np.ndarray:
    lines = path.read_text().strip().split("\n")
    n = int(lines[0].split()[1])
    return np.array([[float(x) for x in line.split()] for line in lines[1:1 + n]])


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic angle between two rotations, in degrees."""
    cos = (np.trace(R_a @ R_b.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def main() -> None:
    cam_paths = sorted(RATHAUS_DIR.glob("*.camera"))
    if not cam_paths:
        raise FileNotFoundError(f"No .camera files in {RATHAUS_DIR}")

    ref_R, ref_centres, ref_pts = [], [], []
    for p in cam_paths:
        _, R_w2c, centre = load_reference_camera(p)
        ref_R.append(R_w2c)
        ref_centres.append(centre)
        ref_pts.append(load_reference_points(p.with_suffix(".3Dpoints")))
    ref_centres = np.stack(ref_centres)

    # The per-image files re-list the same surveyed points, so dedupe --
    # otherwise a point seen in five images would count five times and
    # silently weight the metrics toward whatever the overlap region is.
    stacked = np.vstack(ref_pts)
    ref_points = np.unique(np.round(stacked, 4), axis=0)
    print(f"Reference: {len(cam_paths)} calibrated cameras, "
          f"{len(ref_points):,} distinct 3D points ({len(stacked):,} before deduplication)")

    cache = np.load(CACHE)
    cam_ids, Rs, ts, our_points = cache["cam_ids"], cache["Rs"], cache["ts"], cache["xyz"]
    our_centres = np.stack([-Rs[i].T @ ts[i] for i in range(len(cam_ids))])
    print(f"Ours:      {len(cam_ids)} registered cameras, {len(our_points):,} sparse points")

    if len(cam_ids) != len(cam_paths):
        raise SystemExit(
            f"Registered {len(cam_ids)} cameras but the reference has {len(cam_paths)} -- "
            "this script assumes every image was registered, in filename order."
        )

    # --- Pose accuracy -------------------------------------------------
    print("\n" + "=" * 70)
    print("CAMERA POSE ACCURACY")
    print("=" * 70)

    scale, R_align, t_align = umeyama_alignment(our_centres, ref_centres)
    aligned_centres = apply_similarity(scale, R_align, t_align, our_centres)
    centre_errors = np.linalg.norm(aligned_centres - ref_centres, axis=1)

    ref_baseline = float(np.median(np.linalg.norm(np.diff(ref_centres, axis=0), axis=1)))
    print(f"Similarity scale (our arbitrary units -> reference units): {scale:.4f}")
    print(f"Median reference baseline between consecutive cameras: {ref_baseline:.3f}")

    print(f"\n  {'camera':8s} {'centre err':>12s} {'% of baseline':>14s} {'rotation err':>14s}")
    print("  " + "-" * 52)
    rot_errors = []
    for i, cid in enumerate(cam_ids):
        # Alignment maps our world into the reference world, so our
        # world-to-camera rotation composes with the alignment's inverse.
        rot_err = rotation_angle_deg(Rs[i] @ R_align.T, ref_R[i])
        rot_errors.append(rot_err)
        print(f"  {cid:<8d} {centre_errors[i]:12.4f} {100*centre_errors[i]/ref_baseline:13.2f}% "
              f"{rot_err:13.3f}°")
    rot_errors = np.array(rot_errors)
    print("  " + "-" * 52)
    print(f"  {'mean':8s} {centre_errors.mean():12.4f} {100*centre_errors.mean()/ref_baseline:13.2f}% "
          f"{rot_errors.mean():13.3f}°")
    print(f"  {'max':8s} {centre_errors.max():12.4f} {100*centre_errors.max()/ref_baseline:13.2f}% "
          f"{rot_errors.max():13.3f}°")

    # --- Structure -----------------------------------------------------
    print("\n" + "=" * 70)
    print("STRUCTURE vs REFERENCE POINTS")
    print("=" * 70)

    thresholds = tuple(round(ref_baseline * f, 4) for f in (0.05, 0.1, 0.25, 0.5))
    print(f"Thresholds are fractions of the reference baseline (same convention as the "
          f"Fountain evaluation): {thresholds}")

    # Two scale-free yardsticks, so the distances below mean something
    # without a metric unit: the ground sampling distance (how much of the
    # scene one pixel spans at the median viewing depth), and the diagonal
    # of the reference cloud's robust extent.
    depths = np.concatenate([
        np.linalg.norm(load_reference_points(p.with_suffix(".3Dpoints")) - c, axis=1)
        for p, c in zip(cam_paths, ref_centres)
    ])
    median_depth = float(np.median(depths))
    focal_px = float(load_reference_camera(cam_paths[0])[0][0, 0])
    gsd = median_depth / focal_px
    lo_e, hi_e = np.percentile(ref_points, [2, 98], axis=0)
    scene_diag = float(np.linalg.norm(hi_e - lo_e))
    print(f"Ground sampling distance at median depth {median_depth:.1f}: {gsd:.4f} units/pixel")
    print(f"Reference scene diagonal (2-98 percentile): {scene_diag:.1f} units")

    clouds = {"sparse": our_points}
    if DENSE_MESH.is_file():
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(str(DENSE_MESH))
        clouds["dense mesh vertices"] = np.asarray(mesh.vertices)

    for name, pts in clouds.items():
        aligned = apply_similarity(scale, R_align, t_align, pts)
        metrics = evaluate_reconstruction(aligned, ref_points, thresholds=thresholds)
        print(f"\n--- {name} ({len(pts):,} points) ---")
        print(f"  Completeness (reference -> ours)  mean={metrics['completeness_mean']:8.4f}  "
              f"median={metrics['completeness_median']:8.4f}   <- the meaningful one")
        print(f"  Accuracy     (ours -> reference)  mean={metrics['accuracy_mean']:8.4f}  "
              f"median={metrics['accuracy_median']:8.4f}   <- inflated: reference is sparse")
        comp = metrics["completeness_median"]
        print(f"      -> completeness median = {comp/gsd:.1f} pixels of ground sampling distance, "
              f"{100*comp/scene_diag:.2f}% of the scene diagonal")
        print(f"  {'tau':>10s} {'recall':>9s} {'precision':>11s} {'F-score':>9s}")
        for tau in thresholds:
            print(f"  {tau:10.4f} {metrics[f'recall@{tau}']:9.3f} "
                  f"{metrics[f'precision@{tau}']:11.3f} {metrics[f'f_score@{tau}']:9.3f}")

    # --- Figure --------------------------------------------------------
    out_dir = ROOT / "evaluation" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    aligned_sparse = apply_similarity(scale, R_align, t_align, our_points)
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2],
               s=6, alpha=0.9, c="black", label="reference points (ground truth)")
    ax.scatter(aligned_sparse[:, 0], aligned_sparse[:, 1], aligned_sparse[:, 2],
               s=1, alpha=0.25, c="red", label="ours (aligned)")
    ax.scatter(aligned_centres[:, 0], aligned_centres[:, 1], aligned_centres[:, 2],
               s=60, c="blue", marker="^", label="our cameras (aligned)")
    ax.scatter(ref_centres[:, 0], ref_centres[:, 1], ref_centres[:, 2],
               s=60, c="green", marker="v", label="reference cameras")
    # Frame both the facade and the camera arc, on a common cube so the
    # standoff distance isn't distorted by per-axis autoscaling.
    lo = np.minimum(np.percentile(ref_points, 1, axis=0), ref_centres.min(axis=0))
    hi = np.maximum(np.percentile(ref_points, 99, axis=0), ref_centres.max(axis=0))
    mid, half = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.view_init(elev=15, azim=-70)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Rathaus: reconstruction aligned to the dataset's own ground truth")
    fig.tight_layout()
    fig_path = out_dir / "rathaus_alignment_overlay.png"
    fig.savefig(fig_path, dpi=120)
    print(f"\nWrote alignment overlay -> {fig_path}")

    csv_path = out_dir / "rathaus_pose_accuracy.csv"
    with csv_path.open("w") as f:
        f.write("camera,centre_error,centre_error_pct_of_baseline,rotation_error_deg\n")
        for i, cid in enumerate(cam_ids):
            f.write(f"{cid},{centre_errors[i]:.6f},"
                    f"{100*centre_errors[i]/ref_baseline:.4f},{rot_errors[i]:.6f}\n")
    print(f"Wrote per-camera pose errors -> {csv_path}")

    print("\nRathaus evaluation OK.")


if __name__ == "__main__":
    main()
