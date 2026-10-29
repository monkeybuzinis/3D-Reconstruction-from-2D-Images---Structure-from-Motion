"""
Write a standalone, self-contained HTML viewer for a finished reconstruction.

scripts/export_viewer.py produces JSON for the hosted viewer; this wraps the
same data in a single HTML file you can open straight from disk -- useful when
no mesh viewer is installed, or when Open3D cannot open a GL window (headless
machines, some Wayland sessions).

Usage:
    .venv/bin/python3 scripts/export_viewer_html.py outputs/buffalo-crop
    .venv/bin/python3 scripts/export_viewer_html.py outputs/herzjesu --title "Herz-Jesu"
    xdg-open outputs/buffalo-crop/viewer.html
"""

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from export_viewer import export_mesh, export_points  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path, help="Directory holding scene_cache.npz (and optionally dense_mesh.ply)")
    ap.add_argument("--title", default=None, help="Heading shown on the page (default: folder name)")
    ap.add_argument("--subtitle", default=None, help="Line under the heading")
    ap.add_argument("--error", default="--", help="Median reprojection error to display, e.g. '0.37 px'")
    ap.add_argument("--mesh-triangles", type=int, default=40000, help="Target triangle count for the web mesh")
    ap.add_argument("--output", type=Path, default=None, help="Output file (default: <scene_dir>/viewer.html)")
    args = ap.parse_args()

    points = export_points(args.scene_dir)
    mesh_path = args.scene_dir / "dense_mesh.ply"
    if mesh_path.is_file():
        mesh = export_mesh(mesh_path, args.mesh_triangles)
    else:
        # The page always expects both modes; an empty mesh keeps it working
        # as a point-cloud-only viewer instead of failing to load.
        mesh = {"verts": [], "colors": [], "faces": [], "n_verts": 0, "n_faces": 0}
        print(f"No {mesh_path.name} -- writing a sparse-only viewer")

    title = args.title or args.scene_dir.name
    subtitle = args.subtitle or (
        f"recovered from <strong>{len(points['cameras'])} camera poses</strong> "
        "via classical Structure-from-Motion"
    )

    html = (ROOT / "scripts" / "viewer_template.html").read_text()
    for token, value in (
        ("__POINTS_DATA__", json.dumps(points)),
        ("__MESH_DATA__", json.dumps(mesh)),
        ("__TITLE__", title),
        ("__SUBTITLE__", subtitle),
        ("__ERROR__", args.error),
    ):
        html = html.replace(token, value)

    out = args.output or (args.scene_dir / "viewer.html")
    out.write_text(html)
    summary = f"{points['n_points']:,} points + {len(points['cameras'])} cameras"
    if mesh["n_faces"]:
        summary += f", mesh {mesh['n_faces']:,} faces"
    print(f"Wrote {summary} -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"Open it with:  xdg-open {out}")


if __name__ == "__main__":
    main()
