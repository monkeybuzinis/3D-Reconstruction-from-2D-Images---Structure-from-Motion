"""
Export the dense Poisson mesh (vertices + faces + colors) as compact JSON
for the browser-based interactive viewer.

Run from the repo root:
    .venv/bin/python3 scripts/export_mesh_for_viewer.py
"""

from pathlib import Path
import json
import sys

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    mesh = o3d.io.read_triangle_mesh(str(ROOT / "outputs" / "mvs" / "fountain_dense_mesh_web.ply"))
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    colors = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)

    print(f"{len(verts)} vertices, {len(faces)} faces")

    out = {
        "verts": verts.flatten().tolist(),
        "colors": colors.flatten().tolist(),
        "faces": faces.flatten().tolist(),
        "n_verts": int(len(verts)),
        "n_faces": int(len(faces)),
    }
    out_path = ROOT / "outputs" / "mesh_viewer_data.json"
    out_path.write_text(json.dumps(out))
    print(f"Wrote -> {out_path} ({out_path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
