"""
Dense reconstruction check: run pairwise dense stereo across all 11 Fountain
cameras (using the poses already recovered by incremental SfM + hand BA),
fuse into one dense colored cloud, then Poisson-mesh it.

Uses the cached sparse scene from scripts/build_scene_cache.py -- run that
first if outputs/scene_cache.npz doesn't exist yet.

Run from the repo root:
    .venv/bin/python3 scripts/verify_mvs.py
"""

from pathlib import Path
import sys
import time

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfm.io.fountain import default_fountain_dir, load_images
from sfm.io.ply import write_xyzrgb_ply
from sfm.mvs.fuse import dense_reconstruct
from sfm.mvs.params import dense_params
from sfm.mvs.mesh import clean_dense_cloud, keep_significant_clusters, poisson_mesh, simplify_for_web


def main() -> None:
    cache = np.load(ROOT / "outputs" / "scene_cache.npz")
    cam_ids, Ks, Rs, ts = cache["cam_ids"], cache["Ks"], cache["Rs"], cache["ts"]
    sparse_xyz = cache["xyz"]

    images = load_images(default_fountain_dir())

    # Depth window + voxel size from per-camera depths, NOT from world Z --
    # see sfm/mvs/params.py for why world Z silently stopped working.
    params = dense_params(sparse_xyz, Rs, ts, Ks, downscale=2)
    min_depth, max_depth = params.min_depth, params.max_depth
    print(f"Sparse depth range (guides dense stereo's sanity filter): {min_depth:.2f} - {max_depth:.2f} "
          f"({params.n_core_points:,}/{params.n_points:,} points seen in range by >=2 cameras)")

    t0 = time.time()
    points, colors = dense_reconstruct(
        images, cam_ids, Ks, Rs, ts, min_depth, max_depth, downscale=2, num_disparities=256,
    )
    print(f"Dense stereo across {len(cam_ids)-1} pairs: {points.shape[0]:,} raw points in {time.time()-t0:.1f}s")

    out_dir = ROOT / "outputs" / "mvs"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_ply = out_dir / "fountain_dense_raw.ply"
    write_xyzrgb_ply(raw_ply, points, colors)
    print(f"Wrote raw dense cloud -> {raw_ply}")

    # Voxel size: robust bounding-box diagonal / ~250, so density scales
    # sensibly regardless of the reconstruction's arbitrary SfM scale. The
    # extent is measured over the points at least two cameras see within
    # the depth window above -- the subset dense stereo can actually
    # reconstruct -- so far-field strays can't coarsen the whole pipeline.
    voxel_size = params.voxel_size
    print(f"Voxel size: {voxel_size:.4f} (12 pixel footprints; one pixel spans "
          f"{params.pixel_footprint:.5f} at median depth {params.median_depth:.2f})")

    t0 = time.time()
    pcd = clean_dense_cloud(points, colors, voxel_size=voxel_size)
    print(f"Cleaned cloud (statistical + radius outlier removal): {len(pcd.points):,} points in {time.time()-t0:.1f}s")

    n_before_cluster = len(pcd.points)
    pcd = keep_significant_clusters(pcd, eps=voxel_size * 4)
    print(f"Significant clusters kept: {len(pcd.points):,}/{n_before_cluster:,} points")

    clean_xyz = np.asarray(pcd.points)
    clean_rgb = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    clean_ply = out_dir / "fountain_dense_clean.ply"
    write_xyzrgb_ply(clean_ply, clean_xyz, clean_rgb)
    print(f"Wrote cleaned dense cloud -> {clean_ply}")

    t0 = time.time()
    mesh = poisson_mesh(pcd, depth=9)
    print(f"Poisson mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles in {time.time()-t0:.1f}s")

    mesh_path = out_dir / "fountain_dense_mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    print(f"Wrote mesh -> {mesh_path}")

    web_mesh = simplify_for_web(mesh, target_triangles=40000)
    web_path = out_dir / "fountain_dense_mesh_web.ply"
    o3d.io.write_triangle_mesh(str(web_path), web_mesh)
    print(f"Wrote web-simplified mesh ({len(web_mesh.triangles):,} triangles) -> {web_path}")

    print("\nDense reconstruction OK.")


if __name__ == "__main__":
    main()
