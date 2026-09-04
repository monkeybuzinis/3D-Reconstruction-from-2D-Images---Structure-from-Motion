"""
General-purpose reconstruction entry point: any folder of photos in, a
colored point cloud (and optionally a dense mesh) out. Unlike the
Fountain-specific verify_*.py scripts, this doesn't hardcode a dataset --
it's what turns the pipeline from "a set of validated algorithms" into
something you can actually point at new photos.

Camera intrinsics (K) are approximated from EXIF (see sfm/io/calibration.py)
unless given explicitly. This is inherently less accurate than a real
calibration, so treat results on new data as a reasonable starting point,
not something to trust the way the Fountain-P11 evaluation numbers were
trusted (that benchmark's K is the real, precisely known value).

Usage:
    .venv/bin/python3 scripts/reconstruct.py path/to/photos
    .venv/bin/python3 scripts/reconstruct.py path/to/photos --dense
    .venv/bin/python3 scripts/reconstruct.py path/to/photos --focal-mm 26 --sensor-width-mm 5.6
    .venv/bin/python3 scripts/reconstruct.py path/to/photos --fov 65 --seed 0 2
"""

from pathlib import Path
import argparse
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfm.ba.lm import hand_bundle_adjust
from sfm.features.tracks import build_tracks
from sfm.io.colorize import colorize_points
from sfm.io.generic import load_scene
from sfm.io.ply import write_xyzrgb_ply
from sfm.recon.incremental import filter_outlier_observations, run_incremental_sfm


def reprojection_errors(scene) -> np.ndarray:
    errs = []
    for obs in scene.observations:
        cam = scene.cameras[obs.camera_id]
        pt = scene.points[obs.point_id]
        proj = cam.projection_matrix() @ np.append(pt.xyz, 1.0)
        uv = proj[:2] / proj[2]
        errs.append(np.linalg.norm(uv - obs.uv))
    return np.array(errs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", type=Path, help="Folder of input photos")
    parser.add_argument("--output", type=Path, default=None, help="Output directory (default: outputs/<folder-name>)")
    parser.add_argument("--focal-mm", type=float, default=None, help="Known focal length in mm (with --sensor-width-mm, skips EXIF guessing)")
    parser.add_argument("--sensor-width-mm", type=float, default=None, help="Known sensor width in mm")
    parser.add_argument("--fov", type=float, default=55.0, help="Fallback horizontal FOV in degrees if EXIF has no usable focal length (default: 55)")
    parser.add_argument("--seed", type=int, nargs=2, default=(0, 1), metavar=("CAM_I", "CAM_J"), help="Two-view seed camera indices (default: 0 1)")
    parser.add_argument("--outlier-px", type=float, default=4.0, help="Reprojection error threshold (px) for the outlier filter pass (default: 4.0)")
    parser.add_argument("--dense", action="store_true", help="Also run dense stereo (MVS) + Poisson meshing")
    args = parser.parse_args()

    out_dir = args.output or (ROOT / "outputs" / args.images.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading images from {args.images} ...")
    images, K = load_scene(args.images, focal_mm=args.focal_mm, sensor_width_mm=args.sensor_width_mm, fallback_fov_deg=args.fov)
    print(f"Loaded {len(images)} images. Estimated K:\n{K}")

    t0 = time.time()
    keypoints, tracks = build_tracks(images)
    print(f"Built {len(tracks)} tracks in {time.time()-t0:.1f}s")

    t0 = time.time()
    try:
        scene, log = run_incremental_sfm(
            images, K, keypoints, tracks, seed=tuple(args.seed),
            ba_max_nfev=30, bundle_adjust_fn=hand_bundle_adjust,
        )
    except RuntimeError as e:
        print(f"\nReconstruction failed: {e}")
        print("Try a different --seed pair (needs enough shared tracks and real parallax),")
        print("or check that photos are roughly in capture order and overlap enough to match.")
        raise SystemExit(1)
    n_registered = 2 + len(log)
    print(f"Registered {n_registered}/{len(images)} cameras in {time.time()-t0:.1f}s")
    if n_registered < len(images):
        print(f"WARNING: {len(images) - n_registered} camera(s) could not be registered "
              f"(insufficient correspondences to the growing reconstruction).")

    hand_bundle_adjust(scene, fixed_camera_ids=(args.seed[0],), max_iters=50)
    n_before = len(scene.points)
    scene = filter_outlier_observations(scene, max_error_px=args.outlier_px)
    hand_bundle_adjust(scene, fixed_camera_ids=(args.seed[0],), max_iters=50)
    print(f"Outlier filter (>{args.outlier_px}px): kept {len(scene.points)}/{n_before} points")

    errs = reprojection_errors(scene)
    print(f"Reprojection error (px): median={np.median(errs):.3f} mean={errs.mean():.3f} max={errs.max():.3f}")

    colorize_points(scene, images)
    xyz = scene.points_xyz()
    point_ids = sorted(scene.points)
    colors = np.stack([scene.points[pid].color for pid in point_ids]).astype(np.uint8)

    sparse_path = out_dir / "sparse_colored.ply"
    write_xyzrgb_ply(sparse_path, xyz, colors)
    print(f"Wrote sparse cloud -> {sparse_path}")

    # Cache camera poses + points (same format as scripts/build_scene_cache.py
    # uses for Fountain) so a later step -- e.g. scripts/export_viewer.py --
    # can build an interactive view without re-running the whole pipeline.
    cam_ids_all = sorted(scene.cameras)
    cache_path = out_dir / "scene_cache.npz"
    np.savez(
        cache_path,
        cam_ids=np.array(cam_ids_all),
        Ks=np.stack([scene.cameras[c].K for c in cam_ids_all]),
        Rs=np.stack([scene.cameras[c].R for c in cam_ids_all]),
        ts=np.stack([scene.cameras[c].t for c in cam_ids_all]),
        widths=np.array([scene.cameras[c].width for c in cam_ids_all]),
        heights=np.array([scene.cameras[c].height for c in cam_ids_all]),
        xyz=xyz,
        colors=colors,
    )
    print(f"Wrote scene cache -> {cache_path}")

    if args.dense:
        from sfm.mvs.fuse import dense_reconstruct
        from sfm.mvs.mesh import clean_dense_cloud, keep_largest_cluster, poisson_mesh
        import open3d as o3d

        cam_ids = sorted(scene.cameras)
        Ks = np.stack([scene.cameras[c].K for c in cam_ids])
        Rs = np.stack([scene.cameras[c].R for c in cam_ids])
        ts = np.stack([scene.cameras[c].t for c in cam_ids])

        min_depth, max_depth = np.percentile(xyz[:, 2], [1, 99])
        t0 = time.time()
        dense_pts, dense_colors = dense_reconstruct(images, cam_ids, Ks, Rs, ts, min_depth, max_depth)
        print(f"Dense stereo: {dense_pts.shape[0]:,} raw points in {time.time()-t0:.1f}s")

        lo, hi = np.percentile(xyz, [1, 99], axis=0)
        voxel_size = np.linalg.norm(hi - lo) / 250
        pcd = clean_dense_cloud(dense_pts, dense_colors, voxel_size=voxel_size)
        pcd = keep_largest_cluster(pcd, eps=voxel_size * 4)
        print(f"Cleaned dense cloud: {len(pcd.points):,} points")

        mesh = poisson_mesh(pcd, depth=9)
        mesh_path = out_dir / "dense_mesh.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        print(f"Wrote dense mesh ({len(mesh.triangles):,} triangles) -> {mesh_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
