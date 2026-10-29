"""
Run the dense stage on a sparse reconstruction that already exists.

scripts/reconstruct.py --dense does sparse and dense in one invocation, which
means re-solving a reconstruction that is already on disk just to mesh it.
This takes the cached scene instead, so a finished sparse result can be
densified without repeating the twenty minutes that produced it.

Usage:
    .venv/bin/python3 scripts/densify.py outputs/_rev3/woodlog dataset/woodlog
    .venv/bin/python3 scripts/densify.py outputs/_rev3/guitar dataset/guitar --subject-only
"""

from pathlib import Path
import argparse
import sys
import time

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfm.io.generic import load_image_folder
from sfm.mvs.fuse import dense_reconstruct
from sfm.mvs.mesh import clean_dense_cloud, keep_significant_clusters, poisson_mesh
from sfm.mvs.params import dense_params


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path, help="Directory holding scene_cache.npz")
    ap.add_argument("images", type=Path, help="The image folder that produced it")
    ap.add_argument("--downscale", type=int, default=2, help="Stereo downscale factor (default: 2)")
    ap.add_argument("--subject-only", action="store_true",
                    help="Clip the dense cloud to the volume the sparse points occupy")
    args = ap.parse_args()

    cache = np.load(args.scene_dir / "scene_cache.npz")
    images, _paths = load_image_folder(args.images)
    cam_ids = [int(c) for c in cache["cam_ids"]]
    Ks, Rs, ts, xyz = cache["Ks"], cache["Rs"], cache["ts"], cache["xyz"]
    print(f"Densifying {len(cam_ids)} cameras / {len(xyz):,} sparse points from {args.scene_dir}")

    params = dense_params(xyz, Rs, ts, Ks, downscale=args.downscale)
    print(f"Depth window {params.min_depth:.2f}-{params.max_depth:.2f}, voxel {params.voxel_size:.4f}")

    t0 = time.time()
    dense_pts, dense_colors = dense_reconstruct(
        images, cam_ids, Ks, Rs, ts, params.min_depth, params.max_depth, downscale=args.downscale
    )
    print(f"Dense stereo: {dense_pts.shape[0]:,} raw points in {time.time()-t0:.1f}s")

    if args.subject_only and len(xyz) > 10:
        lo, hi = np.percentile(xyz, [1, 99], axis=0)
        pad = 0.15 * (hi - lo)
        inside = np.all((dense_pts >= lo - pad) & (dense_pts <= hi + pad), axis=1)
        print(f"Subject volume clip: kept {inside.sum():,}/{len(dense_pts):,}")
        dense_pts, dense_colors = dense_pts[inside], dense_colors[inside]

    pcd = clean_dense_cloud(dense_pts, dense_colors, voxel_size=params.voxel_size)
    pcd = keep_significant_clusters(pcd, eps=params.voxel_size * 4)
    print(f"Cleaned dense cloud: {len(pcd.points):,} points")
    if len(pcd.points) < 100:
        print("Too few dense points to mesh -- the sparse reconstruction is too thin.")
        raise SystemExit(0)

    mesh = poisson_mesh(pcd, depth=9)
    out = args.scene_dir / "dense_mesh.ply"
    o3d.io.write_triangle_mesh(str(out), mesh)
    print(f"Wrote dense mesh ({len(mesh.triangles):,} triangles) -> {out}")


if __name__ == "__main__":
    main()
