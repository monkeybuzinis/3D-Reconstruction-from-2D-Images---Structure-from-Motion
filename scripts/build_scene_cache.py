"""
Run the full incremental SfM + hand BA pipeline once and cache the result
(camera poses, points) to outputs/scene_cache.npz, so downstream experiments
(e.g. MVS) don't have to re-run the ~90s pipeline every time.

Run from the repo root:
    .venv/bin/python3 scripts/build_scene_cache.py
"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfm.ba.lm import hand_bundle_adjust
from sfm.features.tracks import build_tracks
from sfm.io.colorize import colorize_points
from sfm.io.fountain import FOUNTAIN_K, default_fountain_dir, load_images
from sfm.recon.incremental import filter_outlier_observations, run_incremental_sfm


def main() -> None:
    images = load_images(default_fountain_dir())
    keypoints, tracks = build_tracks(images)
    scene, _ = run_incremental_sfm(
        images, FOUNTAIN_K, keypoints, tracks, seed=(0, 1),
        ba_max_nfev=30, bundle_adjust_fn=hand_bundle_adjust,
    )
    hand_bundle_adjust(scene, fixed_camera_ids=(0,), max_iters=50)
    scene = filter_outlier_observations(scene, max_error_px=4.0)
    hand_bundle_adjust(scene, fixed_camera_ids=(0,), max_iters=50)
    colorize_points(scene, images)

    cam_ids = sorted(scene.cameras)
    Ks = np.stack([scene.cameras[c].K for c in cam_ids])
    Rs = np.stack([scene.cameras[c].R for c in cam_ids])
    ts = np.stack([scene.cameras[c].t for c in cam_ids])
    widths = np.array([scene.cameras[c].width for c in cam_ids])
    heights = np.array([scene.cameras[c].height for c in cam_ids])

    point_ids = sorted(scene.points)
    xyz = np.stack([scene.points[p].xyz for p in point_ids])
    colors = np.stack([scene.points[p].color for p in point_ids])

    out_path = ROOT / "outputs" / "scene_cache.npz"
    np.savez(
        out_path,
        cam_ids=np.array(cam_ids), Ks=Ks, Rs=Rs, ts=ts, widths=widths, heights=heights,
        xyz=xyz, colors=colors,
    )
    print(f"Cached {len(cam_ids)} cameras + {len(point_ids)} points -> {out_path}")


if __name__ == "__main__":
    main()
