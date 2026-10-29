"""
Evaluation against the ETH3D high-resolution benchmark scene `statue`.

What this dataset adds. dataset/statue ships a COLMAP-format reference model
-- cameras.txt, images.txt, points3D.txt -- with image names of the form
dslr_images_undistorted/DSC_xxxx.JPG, the ETH3D convention. It supplies true
pinhole intrinsics for the undistorted images (fx 3410.99, fy 3412.37,
principal point 3129.34, 2059.72 on a 6208x4138 frame), reference poses for
all 11 images, and 6,812 reference 3D points.

Two things to be precise about when citing this:

  - The reference POSES come from the benchmark's own calibration and
    registration pipeline and are the strongest pose reference available in
    this project after Buddha's projection matrices.
  - points3D.txt is a sparse triangulation carrying per-point ERROR and
    track lists, not the benchmark's laser scan (which is distributed
    separately and is not present here). Structure agreement measured
    against it is agreement with a peer triangulation, not with measurement
    -- the same caveat that applies to Fountain-P11.

Orientation. The images as distributed carry no EXIF at all and are stored
landscape 6208x4138, matching the reference model, whose own 2D observations
span x in [-29, 6212]. An earlier working copy of this dataset had been
re-saved with an EXIF Orientation=6 tag, which cv2.imread honours by
rotating the frame to portrait; that copy did not match the reference frame
and produced a systematically wrong result. Verify orientation before
trusting any comparison against a shipped reference.

Run from the repo root, after reconstructing dataset/statue-eth3d:
    .venv/bin/python3 scripts/verify_statue_evaluation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfm.eval.metrics import evaluate_reconstruction
from sfm.eval.umeyama import apply_similarity, umeyama_alignment

REFERENCE = ROOT / "dataset" / "statue"
IMAGES = ROOT / "dataset" / "statue"
CACHE = ROOT / "outputs" / "_variants" / "statue-eth3d" / "scene_cache.npz"


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    cos = (np.trace(R_a @ R_b.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def load_reference():
    """Reference poses and points, keyed by image basename."""
    import pycolmap
    pycolmap.logging.minloglevel = 3
    rec = pycolmap.Reconstruction(str(REFERENCE))
    poses = {}
    for im in rec.images.values():
        if not im.has_pose:
            continue
        cfw = im.cam_from_world()
        poses[Path(im.name).name] = (
            np.array(cfw.rotation.matrix()),
            np.array(im.projection_center()),
        )
    pts = np.stack([np.array(p.xyz) for p in rec.points3D.values()])
    cam = list(rec.cameras.values())[0]
    return poses, pts, rec, cam


def main() -> None:
    if not CACHE.is_file():
        raise SystemExit(
            f"No reconstruction at {CACHE}.\n"
            "Run:  .venv/bin/python3 scripts/reconstruct.py dataset/statue "
            "--focal-mm 19.7802 --sensor-width-mm 36.0 --output outputs/statue-eth3d"
        )

    ref_poses, ref_pts, rec, cam = load_reference()
    names = sorted(p.name for p in IMAGES.glob("*.JPG"))

    print("=" * 74)
    print("ETH3D `statue` — reference model")
    print("=" * 74)
    print(f"  camera model      : {cam.model.name}  {cam.width}x{cam.height}")
    print(f"  intrinsics        : {np.round(cam.params, 2)}")
    print(f"  reference poses   : {len(ref_poses)} of {len(names)} images")
    print(f"  reference points  : {len(ref_pts):,}")

    cache = np.load(CACHE)
    cam_ids, Rs, ts, our_pts = cache["cam_ids"], cache["Rs"], cache["ts"], cache["xyz"]
    ours = {}
    for i, cid in enumerate(cam_ids):
        if int(cid) < len(names):
            ours[names[int(cid)]] = (Rs[i], -Rs[i].T @ ts[i])
    print(f"\n  our reconstruction: {len(ours)}/{len(names)} cameras, {len(our_pts):,} points")

    common = sorted(set(ours) & set(ref_poses))
    if len(common) < 3:
        raise SystemExit(f"Only {len(common)} cameras in common — cannot align.")

    A = np.stack([ours[n][1] for n in common])
    B = np.stack([ref_poses[n][1] for n in common])
    scale, R_align, t_align = umeyama_alignment(A, B)
    centre_err = np.linalg.norm(apply_similarity(scale, R_align, t_align, A) - B, axis=1)
    baseline = float(np.median(np.linalg.norm(np.diff(B, axis=0), axis=1)))
    rot_err = np.array([rotation_angle_deg(ours[n][0] @ R_align.T, ref_poses[n][0]) for n in common])

    print("\n" + "=" * 74)
    print("CAMERA POSE ACCURACY vs the benchmark reference")
    print("=" * 74)
    print(f"  similarity scale  : {scale:.4f}")
    print(f"  median baseline   : {baseline:.4f}")
    print(f"\n  {'image':22s} {'centre err':>12s} {'% baseline':>12s} {'rotation':>12s}")
    print("  " + "-" * 60)
    for n, ce, re_ in zip(common, centre_err, rot_err):
        print(f"  {n:22s} {ce:12.4f} {100*ce/baseline:11.2f}% {re_:11.3f}°")
    print("  " + "-" * 60)
    print(f"  {'mean':22s} {centre_err.mean():12.4f} {100*centre_err.mean()/baseline:11.2f}% "
          f"{rot_err.mean():11.3f}°")
    print(f"  {'median':22s} {np.median(centre_err):12.4f} "
          f"{100*np.median(centre_err)/baseline:11.2f}% {np.median(rot_err):11.3f}°")
    print(f"  {'max':22s} {centre_err.max():12.4f} {100*centre_err.max()/baseline:11.2f}% "
          f"{rot_err.max():11.3f}°")

    print("\n" + "=" * 74)
    print("STRUCTURE vs the reference triangulation")
    print("=" * 74)
    aligned = apply_similarity(scale, R_align, t_align, our_pts)
    thresholds = tuple(round(baseline * f, 5) for f in (0.05, 0.1, 0.25, 0.5))
    m = evaluate_reconstruction(aligned, ref_pts, thresholds=thresholds)
    print(f"  thresholds (fractions of the reference baseline): {thresholds}")
    print(f"  completeness (reference → ours)  mean={m['completeness_mean']:.4f}  "
          f"median={m['completeness_median']:.4f}")
    print(f"  accuracy     (ours → reference)  mean={m['accuracy_mean']:.4f}  "
          f"median={m['accuracy_median']:.4f}")
    print(f"\n  {'tau':>10s} {'recall':>9s} {'precision':>11s} {'F-score':>9s}")
    for tau in thresholds:
        print(f"  {tau:10.5f} {m[f'recall@{tau}']:9.3f} {m[f'precision@{tau}']:11.3f} "
              f"{m[f'f_score@{tau}']:9.3f}")
    print("\n  Note: points3D.txt is a sparse triangulation, not the benchmark's laser")
    print("  scan, so these are agreement figures rather than accuracy against measurement.")

    out = ROOT / "evaluation" / "metrics" / "statue_pose_accuracy.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("image,centre_error,centre_error_pct_of_baseline,rotation_error_deg\n")
        for n, ce, re_ in zip(common, centre_err, rot_err):
            f.write(f"{n},{ce:.6f},{100*ce/baseline:.4f},{re_:.6f}\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
