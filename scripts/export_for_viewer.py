"""
Export the final hand-BA reconstruction (points + camera poses) as compact
JSON for the browser-based interactive viewer artifact. Not part of the
core pipeline -- a one-off convenience for visualizing the result.

Run from the repo root:
    python scripts/export_for_viewer.py
"""

from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfm.ba.lm import hand_bundle_adjust
from sfm.features.tracks import build_tracks
from sfm.io.colorize import colorize_points
from sfm.io.fountain import FOUNTAIN_K, default_fountain_dir, load_images
from sfm.io.ply import write_xyzrgb_ply
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

    points = scene.points_xyz().astype(np.float32)
    point_ids_sorted = sorted(scene.points)
    colors = np.stack([scene.points[pid].color for pid in point_ids_sorted]).astype(np.uint8)

    ply_path = ROOT / "outputs" / "hand_ba" / "fountain_hand_ba_colored.ply"
    write_xyzrgb_ply(ply_path, points, colors)
    print(f"Wrote colored PLY -> {ply_path}")

    cameras = []
    for cid in sorted(scene.cameras):
        cam = scene.cameras[cid]
        center = cam.camera_center()
        forward = cam.R.T @ np.array([0.0, 0.0, 1.0])  # camera's forward (viewing) direction in world coords
        cameras.append(
            {
                "id": cid,
                "center": center.tolist(),
                "forward": forward.tolist(),
            }
        )

    out = {
        "points": points.flatten().tolist(),
        "colors": colors.flatten().tolist(),
        "n_points": int(points.shape[0]),
        "cameras": cameras,
    }
    out_path = ROOT / "outputs" / "viewer_data.json"
    out_path.write_text(json.dumps(out))
    print(f"Wrote {points.shape[0]} points + {len(cameras)} cameras -> {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
