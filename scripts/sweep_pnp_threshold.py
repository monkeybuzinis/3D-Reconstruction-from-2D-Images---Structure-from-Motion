"""
What should the resectioning acceptance threshold be?

Before a new camera is accepted, its PnP pose must be supported by at least
`min_inliers` correspondences and an inlier RATIO of at least
`min_inlier_ratio`. The current values, 30 and 0.25, are COLMAP's mapper
defaults. They are also the single change with the widest effect on this
project's results: eight datasets register fewer cameras than before it was
added, and on most of them the cameras that remain are far closer to the
external baseline, because the ones removed were being placed wrongly.

Two datasets go the other way, and the reason is a cascade rather than a
judgement about any one camera. On the ETH3D statue the first camera after
the seed pair has an inlier ratio of 0.229 against a 0.25 cut. Rejecting it
leaves nothing for the next three to resect against, so one borderline
decision costs four cameras. A threshold that is right on average can still
be wrong at the one place where it is consulted first.

This sweep runs several thresholds over the same tracks, so matching cannot
differ between arms, and scores each result against COLMAP on the images both
systems placed. The question it answers is whether any single threshold is
better everywhere, or whether the trade-off is real and has to be reported
as one.

Run from the repo root (slow; writes results incrementally):
    .venv/bin/python3 scripts/sweep_pnp_threshold.py
    .venv/bin/python3 scripts/sweep_pnp_threshold.py office statue
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfm.ba.lm import hand_bundle_adjust
from sfm.eval.umeyama import apply_similarity, umeyama_alignment
from sfm.features.tracks import build_tracks
from sfm.io.generic import load_scene
from sfm.recon.incremental import filter_outlier_observations, run_incremental_sfm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".tif", ".tiff"}

# Cheapest first, so a run cut short still answers something.
DATASETS = ["open-mouth", "statue", "church", "office", "child-statue",
            "fire-hydrant", "herzjesu", "woodlog", "buffalo-crop", "guitar", "buffalo"]

# (label, min_inliers, min_inlier_ratio). "off" is the behaviour before the
# test existed, and is what every pre-29-October result in this project used.
POLICIES = [("off", 0, 0.0), ("0.10", 30, 0.10), ("0.15", 30, 0.15), ("0.25", 30, 0.25)]


def dataset_dir(name: str) -> Path:
    folder = ROOT / "dataset" / name
    if any(p.suffix.lower() in IMAGE_EXTENSIONS for p in folder.iterdir()):
        return folder
    subdirs = [p for p in folder.iterdir() if p.is_dir()]
    return subdirs[0] if len(subdirs) == 1 else folder


def image_names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir()
                  if p.suffix.lower() in IMAGE_EXTENSIONS and "vertex_colors" not in p.name)


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    cos = (np.trace(R_a @ R_b.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def colmap_poses(name: str):
    import pycolmap
    pycolmap.logging.minloglevel = 3
    model = ROOT / "colmap" / name / "sparse" / "0"
    if not model.is_dir():
        return None
    rec = pycolmap.Reconstruction(str(model))
    return {Path(im.name).name: (np.array(im.cam_from_world().rotation.matrix()),
                                 np.array(im.projection_center()))
            for im in rec.images.values() if im.has_pose}


def agreement(ours: dict, theirs: dict):
    common = sorted(set(ours) & set(theirs))
    if len(common) < 3:
        return None
    A = np.stack([ours[n][1] for n in common])
    B = np.stack([theirs[n][1] for n in common])
    scale, R_align, t_align = umeyama_alignment(A, B)
    centre_err = np.linalg.norm(apply_similarity(scale, R_align, t_align, A) - B, axis=1)
    baseline = float(np.median(np.linalg.norm(np.diff(B, axis=0), axis=1)))
    rot_err = np.array([rotation_angle_deg(ours[n][0] @ R_align.T, theirs[n][0])
                        for n in common])
    return (float(100 * centre_err.mean() / baseline), float(rot_err.mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=None)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "evaluation" / "metrics" / "pnp_threshold_sweep.json")
    args = ap.parse_args()
    names = args.datasets or DATASETS

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(args.out.read_text()) if args.out.is_file() else {}

    for name in names:
        folder = dataset_dir(name)
        files = image_names(folder)
        baseline_poses = colmap_poses(name)
        images, K = load_scene(folder)

        t0 = time.time()
        keypoints, tracks = build_tracks(images, rng_seed=42)
        print(f"\n{name}: {len(images)} images, {len(tracks):,} tracks "
              f"({time.time() - t0:.0f}s)", flush=True)

        results.setdefault(name, {})
        for label, min_inliers, min_ratio in POLICIES:
            t0 = time.time()
            scene, log = run_incremental_sfm(
                images, K, keypoints, tracks, seed=(0, 1), ba_max_nfev=30,
                bundle_adjust_fn=hand_bundle_adjust, rng_seed=0,
                min_inliers=min_inliers, min_inlier_ratio=min_ratio)
            hand_bundle_adjust(scene, fixed_camera_ids=(0,), gauge_pair=(0, 1), max_iters=50)
            scene = filter_outlier_observations(scene, max_error_px=4.0)
            if scene.observations:
                hand_bundle_adjust(scene, fixed_camera_ids=(0,), gauge_pair=(0, 1), max_iters=50)

            ours = {files[cid]: (cam.R, -cam.R.T @ cam.t)
                    for cid, cam in scene.cameras.items() if cid < len(files)}
            scored = agreement(ours, baseline_poses) if baseline_poses else None
            row = {"cameras": len(ours), "images": len(files), "points": len(scene.points),
                   "seconds": round(time.time() - t0, 1),
                   "pos_pct": None if scored is None else round(scored[0], 2),
                   "rot_deg": None if scored is None else round(scored[1], 3)}
            results[name][label] = row
            cell = "  too few" if scored is None else f"{scored[0]:7.1f}% {scored[1]:7.2f}°"
            print(f"  ratio {label:>4s}: {row['cameras']:3d}/{row['images']:<3d} cameras, "
                  f"{row['points']:7,d} points, {cell}   ({row['seconds']:.0f}s)", flush=True)
            args.out.write_text(json.dumps(results, indent=2))

    print(f"\nWrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
