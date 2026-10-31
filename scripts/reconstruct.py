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
import json
import platform
import subprocess
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
from sfm.recon.incremental import (filter_centred_points, filter_far_points,
                                   filter_outlier_observations, run_incremental_sfm)


def _git_commit() -> str:
    """Short commit of the working tree, with a marker if it is dirty."""
    try:
        root = str(ROOT)
        sha = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if sha.returncode != 0:
            return "unknown"
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10)
        return sha.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


def _environment() -> dict:
    """Versions of everything whose behaviour could change a result."""
    import cv2
    import scipy
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "scipy": scipy.__version__,
    }
    try:
        import open3d
        env["open3d"] = open3d.__version__
    except Exception:
        pass
    return env


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
    parser.add_argument("--rng-seed", type=int, default=0,
                        help="Seed for the RANSAC random streams (default: 0). Every stage is "
                             "deterministic given this value; vary it to sample run-to-run spread")
    parser.add_argument("--outlier-px", type=float, default=4.0, help="Reprojection error threshold (px) for the outlier filter pass (default: 4.0)")
    parser.add_argument("--max-depth-ratio", type=float, default=10.0,
                        help="Drop points deeper than this multiple of the median scene depth "
                             "(default: 10, generous enough to leave benchmark datasets untouched; "
                             "try 2-3 for a small object shot indoors, 0 to disable)")
    parser.add_argument("--subject-only", action="store_true",
                        help="Keep only structure that stays centred across views -- the object you "
                             "walked around -- and discard surroundings. Use for object captures where "
                             "the background sits at a similar distance to the subject")
    parser.add_argument("--dense", action="store_true", help="Also run dense stereo (MVS) + Poisson meshing")
    args = parser.parse_args()

    out_dir = args.output or (ROOT / "outputs" / args.images.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading images from {args.images} ...")
    images, K = load_scene(args.images, focal_mm=args.focal_mm, sensor_width_mm=args.sensor_width_mm, fallback_fov_deg=args.fov)
    print(f"Loaded {len(images)} images. Camera matrix K:\n{K}")

    timings = {}
    t0 = time.time()
    keypoints, tracks = build_tracks(images, rng_seed=args.rng_seed + 42)
    timings["tracks_s"] = round(time.time() - t0, 2)
    print(f"Built {len(tracks)} tracks in {timings['tracks_s']:.1f}s")

    t0 = time.time()
    try:
        scene, log = run_incremental_sfm(
            images, K, keypoints, tracks, seed=tuple(args.seed),
            ba_max_nfev=30, bundle_adjust_fn=hand_bundle_adjust,
            rng_seed=args.rng_seed,
        )
    except RuntimeError as e:
        print(f"\nReconstruction failed: {e}")
        print("Try a different --seed pair (needs enough shared tracks and real parallax),")
        print("or check that photos are roughly in capture order and overlap enough to match.")
        raise SystemExit(1)
    timings["register_s"] = round(time.time() - t0, 2)
    n_registered = 2 + len(log)

    # A reconstruction can come out of the loop with nothing in it: the seed
    # pair triangulates few points, no further camera shares enough of them to
    # resection against, and filtering removes what little there was. Bundle
    # adjustment on an empty scene raises deep inside the Jacobian assembly,
    # which hides the actual problem, so say what happened instead. The same
    # check runs again after the outlier filter, which is the other way the
    # scene can empty: a badly conditioned solve puts every observation past
    # the residual threshold and the filter then removes all of them.
    def exit_if_empty(stage: str) -> None:
        if scene.observations:
            return
        print(f"\nReconstruction is empty after {stage}: {n_registered}/{len(images)} "
              f"cameras registered and no surviving observations.")
        print("The seed pair did not give enough structure for any other view to register "
              "against. Try a different --seed pair, or check that consecutive photographs "
              "overlap enough to share features.")
        raise SystemExit(1)

    exit_if_empty("registration")
    print(f"Registered {n_registered}/{len(images)} cameras in {timings['register_s']:.1f}s")
    if n_registered < len(images):
        print(f"WARNING: {len(images) - n_registered} camera(s) could not be registered "
              f"(insufficient correspondences to the growing reconstruction).")

    t0 = time.time()
    hand_bundle_adjust(scene, fixed_camera_ids=(args.seed[0],),
                       gauge_pair=tuple(args.seed), max_iters=50)
    n_before = len(scene.points)
    scene = filter_outlier_observations(scene, max_error_px=args.outlier_px)
    exit_if_empty(f"the >{args.outlier_px}px outlier filter")
    hand_bundle_adjust(scene, fixed_camera_ids=(args.seed[0],),
                       gauge_pair=tuple(args.seed), max_iters=50)
    timings["final_ba_s"] = round(time.time() - t0, 2)
    print(f"Outlier filter (>{args.outlier_px}px): kept {len(scene.points)}/{n_before} points")

    if args.subject_only:
        n_before_sub = len(scene.points)
        scene = filter_centred_points(scene)
        print(f"Subject filter (centred across views): kept {len(scene.points)}/{n_before_sub} points")

    if args.max_depth_ratio > 0:
        n_before_far = len(scene.points)
        scene = filter_far_points(scene, max_depth_ratio=args.max_depth_ratio)
        dropped = n_before_far - len(scene.points)
        if dropped:
            print(f"Far-point filter (>{args.max_depth_ratio:g}x median depth): "
                  f"dropped {dropped}, kept {len(scene.points)}/{n_before_far} points")

    errs = reprojection_errors(scene)
    print(f"Reprojection error (px): median={np.median(errs):.3f} mean={errs.mean():.3f} max={errs.max():.3f}")

    # A low reprojection error does not mean the geometry is meaningful. If the
    # cameras collapse onto one point, bundle adjustment has fitted a
    # pure-rotation solution: depth is unconstrained, points sit at infinity,
    # and the residuals still look excellent. Check against the seed's unit
    # baseline, which fixes the gauge.
    centres = np.stack([-c.R.T @ c.t for c in scene.cameras.values()])
    span = float(np.linalg.norm(centres - centres.mean(axis=0), axis=1).max())
    if span < 1e-3:
        print(f"\nWARNING: the recovered cameras span only {span:.3g} against a unit seed "
              f"baseline -- they have collapsed to a single point.")
        print("The solve found a pure-rotation (panorama) solution, so this reconstruction "
              "has no usable depth despite the low reprojection error above.")
        print("Usual cause: the subject gives too few stable features of its own (specular "
              "or textureless), so only distant background was matched, where parallax is "
              "negligible.")

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

    # Everything needed to reproduce this exact run, and the measurements the
    # report quotes as runtime. Written next to the result so a number in the
    # report can always be traced back to the run that produced it.
    errs_px = reprojection_errors(scene)
    manifest = {
        "dataset": str(args.images),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "environment": _environment(),
        "seeds": {
            "rng_seed": args.rng_seed,
            "fundamental_ransac": args.rng_seed + 42,
            "pnp_ransac": args.rng_seed,
            "seed_pair": list(args.seed),
        },
        "intrinsics": {"fx": float(K[0, 0]), "cx": float(K[0, 2]), "cy": float(K[1, 2])},
        "result": {
            "n_images": len(images),
            "n_registered": int(n_registered),
            "n_points": int(len(scene.points)),
            "n_tracks": int(len(tracks)),
            "reproj_median_px": float(np.median(errs_px)),
            "reproj_mean_px": float(errs_px.mean()),
            "camera_span": float(span),
            "seed_baseline": float(np.linalg.norm(centres[1] - centres[0])) if len(centres) > 1 else None,
        },
        "runtime_s": {**timings, "total_s": round(sum(timings.values()), 2)},
    }
    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote run manifest -> {manifest_path}")

    if args.dense:
        from sfm.mvs.fuse import dense_reconstruct
        from sfm.mvs.mesh import clean_dense_cloud, keep_significant_clusters, poisson_mesh
        from sfm.mvs.params import dense_params
        import open3d as o3d

        cam_ids = sorted(scene.cameras)
        Ks = np.stack([scene.cameras[c].K for c in cam_ids])
        Rs = np.stack([scene.cameras[c].R for c in cam_ids])
        ts = np.stack([scene.cameras[c].t for c in cam_ids])

        # Depth window from per-camera depths, voxel size from the sensor's
        # resolution limit -- neither from world coordinates, which is the
        # mistake sfm/mvs/params.py documents.
        params = dense_params(xyz, Rs, ts, Ks, downscale=2)
        print(f"Dense depth window: {params.min_depth:.2f}-{params.max_depth:.2f} "
              f"({params.n_core_points:,}/{params.n_points:,} points seen in range by >=2 cameras)")
        print(f"Voxel size: {params.voxel_size:.4f} (12 pixel footprints; one pixel spans "
              f"{params.pixel_footprint:.5f} at median depth {params.median_depth:.2f})")

        t0 = time.time()
        dense_pts, dense_colors = dense_reconstruct(
            images, cam_ids, Ks, Rs, ts, params.min_depth, params.max_depth, downscale=2
        )
        print(f"Dense stereo: {dense_pts.shape[0]:,} raw points in {time.time()-t0:.1f}s")

        # Pruning the sparse cloud to the subject does not by itself keep the
        # surroundings out of the mesh: dense stereo re-matches whole images
        # and happily rebuilds the fence behind the object. Clip the dense
        # result to the volume the retained sparse points occupy, with a
        # margin, so --subject-only reaches all the way to the mesh.
        if args.subject_only and len(xyz) > 10:
            lo, hi = np.percentile(xyz, [1, 99], axis=0)
            pad = 0.15 * (hi - lo)
            lo, hi = lo - pad, hi + pad
            inside = np.all((dense_pts >= lo) & (dense_pts <= hi), axis=1)
            print(f"Subject volume clip: kept {inside.sum():,}/{len(dense_pts):,} dense points")
            dense_pts, dense_colors = dense_pts[inside], dense_colors[inside]

        voxel_size = params.voxel_size
        pcd = clean_dense_cloud(dense_pts, dense_colors, voxel_size=voxel_size)
        pcd = keep_significant_clusters(pcd, eps=voxel_size * 4)
        print(f"Cleaned dense cloud: {len(pcd.points):,} points")

        # Poisson (and the colour transfer after it) need real geometry to
        # work with. A sparse solve that registered only a few cameras can
        # leave almost nothing here, and crashing on an empty array hides
        # the actual problem, which is upstream.
        if len(pcd.points) < 100:
            print("Too few dense points to mesh -- skipping. The sparse "
                  "reconstruction is too thin; check how many cameras registered.")
            raise SystemExit(0)

        mesh = poisson_mesh(pcd, depth=9)
        mesh_path = out_dir / "dense_mesh.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        print(f"Wrote dense mesh ({len(mesh.triangles):,} triangles) -> {mesh_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
