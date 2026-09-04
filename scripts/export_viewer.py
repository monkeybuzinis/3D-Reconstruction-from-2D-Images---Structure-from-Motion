"""
Build interactive-viewer JSON (points + cameras, and a decimated dense mesh
if present) from any reconstruction's scene_cache.npz -- the same cache
scripts/reconstruct.py and scripts/build_scene_cache.py both produce, so
this works for any dataset without re-running the pipeline.

Usage:
    .venv/bin/python3 scripts/export_viewer.py outputs/rathaus
    .venv/bin/python3 scripts/export_viewer.py outputs/rathaus --mesh-triangles 40000
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def export_points(scene_dir: Path) -> dict:
    cache = np.load(scene_dir / "scene_cache.npz")
    cam_ids, Rs, ts = cache["cam_ids"], cache["Rs"], cache["ts"]
    xyz = cache["xyz"].astype(np.float32)
    colors = cache["colors"].astype(np.uint8)

    cameras = []
    for i, cid in enumerate(cam_ids):
        center = -Rs[i].T @ ts[i]
        forward = Rs[i].T @ np.array([0.0, 0.0, 1.0])
        cameras.append({"id": int(cid), "center": center.tolist(), "forward": forward.tolist()})

    return {
        "points": xyz.flatten().tolist(),
        "colors": colors.flatten().tolist(),
        "n_points": int(xyz.shape[0]),
        "cameras": cameras,
    }


def export_mesh(mesh_path: Path, target_triangles: int) -> dict:
    import open3d as o3d
    from sfm.mvs.mesh import simplify_for_web

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh = simplify_for_web(mesh, target_triangles=target_triangles)

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    colors = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
    if len(colors) == 0:
        colors = np.full((len(verts), 3), 180, dtype=np.uint8)  # neutral gray if the mesh has no vertex colors

    return {
        "verts": verts.flatten().tolist(),
        "colors": colors.flatten().tolist(),
        "faces": faces.flatten().tolist(),
        "n_verts": int(len(verts)),
        "n_faces": int(len(faces)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene_dir", type=Path, help="Directory containing scene_cache.npz (and optionally dense_mesh.ply)")
    parser.add_argument("--mesh-triangles", type=int, default=40000, help="Target triangle count for the web-simplified mesh")
    args = parser.parse_args()

    points_data = export_points(args.scene_dir)
    points_path = args.scene_dir / "viewer_data.json"
    points_path.write_text(json.dumps(points_data))
    print(f"Wrote {points_data['n_points']:,} points + {len(points_data['cameras'])} cameras -> {points_path}")

    mesh_path = args.scene_dir / "dense_mesh.ply"
    if mesh_path.is_file():
        mesh_data = export_mesh(mesh_path, args.mesh_triangles)
        mesh_out_path = args.scene_dir / "mesh_viewer_data.json"
        mesh_out_path.write_text(json.dumps(mesh_data))
        print(f"Wrote {mesh_data['n_verts']:,} verts / {mesh_data['n_faces']:,} faces -> {mesh_out_path}")
    else:
        print(f"No {mesh_path.name} found -- skipping mesh export (run reconstruct.py with --dense to get one)")


if __name__ == "__main__":
    main()
