"""
Render the reconstructions to static images for the report.

Every 3D result this project produces lives in a .ply file or an interactive
viewer, neither of which can go in a document. This script renders them
offscreen to PNG.

Offscreen is not a stylistic choice: Open3D's interactive GUI does not start
under this machine's Wayland session. OffscreenRenderer goes through EGL
instead and works headless, which also makes figure generation
reproducible -- no manual camera posing, so re-running after a pipeline
change regenerates every figure identically.

Each dataset is rendered as a horizontal strip of views orbiting the object,
on white, which is what a printed report wants.

Run from the repo root:
    .venv/bin/python3 scripts/render_figures.py                  # every dataset, mesh + sparse
    .venv/bin/python3 scripts/render_figures.py woodlog shoe     # a subset
    .venv/bin/python3 scripts/render_figures.py --views 4 --width 900 woodlog
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Elevation is kept shallow: most of these were shot from roughly eye level,
# and a steep view shows mainly the holes where no camera ever looked.
ELEVATION_DEG = 18.0

# Vertical field of view used for every render. Fixed rather than left to
# the renderer's default so that the framing calculation below is exact and
# every dataset is framed identically.
FOV_DEG = 45.0


def _orbit_eye(centre: np.ndarray, radius: float, azimuth_deg: float, up: np.ndarray,
               elevation_deg: float = ELEVATION_DEG) -> np.ndarray:
    """A camera position orbiting `centre` in the plane perpendicular to `up`."""
    up = up / np.linalg.norm(up)
    # Any two axes perpendicular to up give the orbit plane.
    seed = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(up, seed); e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    direction = np.cos(el) * (np.cos(az) * e1 + np.sin(az) * e2) + np.sin(el) * up
    return centre + radius * direction


def camera_arc(centres: np.ndarray, target: np.ndarray, up: np.ndarray):
    """
    The azimuth range the real cameras covered, and their mean elevation.

    A full 360 degree orbit of a reconstruction mostly shows the side no
    camera ever looked at, which is empty by construction and makes a
    reconstruction look far worse than it is. Sampling only within the
    recorded capture arc shows the surface that was actually observed.
    """
    up = up / np.linalg.norm(up)
    seed = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(up, seed); e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)

    v = centres - target
    az = np.degrees(np.arctan2(v @ e2, v @ e1)) % 360.0
    horiz = np.linalg.norm(np.stack([v @ e1, v @ e2], axis=1), axis=1)
    elev = float(np.degrees(np.arctan2(v @ up, np.maximum(horiz, 1e-9))).mean())

    # The arc is the complement of the largest angular gap between cameras.
    order = np.sort(az)
    gaps = np.diff(np.concatenate([order, [order[0] + 360.0]]))
    g = int(np.argmax(gaps))
    start = order[(g + 1) % len(order)]
    span = 360.0 - gaps[g]
    return float(start), float(span), elev


def render_geometry(geom, width: int, height: int, n_views: int, is_cloud: bool,
                    arc=None) -> list[np.ndarray]:
    """Render `geom` from `n_views` positions orbiting it. Returns RGB arrays."""
    renderer = rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background([1.0, 1.0, 1.0, 1.0])

    material = rendering.MaterialRecord()
    if is_cloud:
        material.shader = "defaultUnlit"
        # Scale point size with output resolution so a small figure is not
        # a scatter of invisible single pixels.
        material.point_size = max(1.5, width / 420.0)
    elif geom.has_vertex_colors():
        # Unlit, so the figure shows the photographic colour carried by the
        # vertices. Shading a photogrammetry mesh darkens exactly the
        # recessed detail -- carving, masonry joints -- that the figure is
        # meant to demonstrate, and adds a light direction that is not part
        # of the measurement.
        material.shader = "defaultUnlit"
    else:
        material.shader = "defaultLit"
        material.base_color = [0.85, 0.85, 0.85, 1.0]
    scene.add_geometry("geom", geom, material)
    scene.scene.set_sun_light([-0.3, -0.8, -0.5], [1.0, 1.0, 1.0], 85000)
    scene.scene.enable_sun_light(True)

    bbox = geom.get_axis_aligned_bounding_box()
    centre = bbox.get_center()
    extent = bbox.get_extent()

    # Frame the object properly instead of trusting the default projection:
    # to fit a bounding sphere of radius r in a vertical field of view f,
    # the camera must sit at r / tan(f/2). The 1.15 leaves a small margin.
    sphere_radius = float(np.linalg.norm(extent)) / 2.0
    scene.camera.set_projection(
        FOV_DEG, width / height, max(1e-4, sphere_radius * 0.01), sphere_radius * 100.0,
        rendering.Camera.FovType.Vertical,
    )
    radius = 1.15 * sphere_radius / np.tan(np.radians(FOV_DEG) / 2.0)

    # "Up" is the smallest-extent axis's complement: for a facade or a
    # standing object the tallest axis is up, which keeps orbits level.
    up = np.zeros(3)
    up[int(np.argmax(extent))] = 1.0

    if arc is not None:
        start, span, elevation = arc
        # Pull in from the extremes, where the object is seen most obliquely.
        azimuths = [start + span * (0.12 + 0.76 * i / max(1, n_views - 1)) for i in range(n_views)]
    else:
        elevation = ELEVATION_DEG
        azimuths = [360.0 * i / n_views for i in range(n_views)]

    images = []
    for az in azimuths:
        eye = _orbit_eye(centre, radius, az, up, elevation)
        scene.camera.look_at(centre, eye, up)
        images.append(np.asarray(renderer.render_to_image()))
    return images


def strip(images: list[np.ndarray], gap: int = 14) -> np.ndarray:
    """Join renders side by side with a white gutter."""
    h = max(im.shape[0] for im in images)
    pieces = []
    for i, im in enumerate(images):
        if i:
            pieces.append(np.full((h, gap, 3), 255, np.uint8))
        pieces.append(im)
    return np.hstack(pieces)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", help="dataset names under outputs/ (default: all)")
    ap.add_argument("--views", type=int, default=3)
    ap.add_argument("--width", type=int, default=760)
    ap.add_argument("--height", type=int, default=620)
    ap.add_argument("--out", type=Path, default=ROOT / "evaluation" / "figures")
    args = ap.parse_args()

    targets = args.datasets
    if not targets:
        targets = sorted(
            d.name for d in (ROOT / "outputs").iterdir()
            if d.is_dir() and not d.name.startswith("_")
            and ((d / "dense_mesh.ply").exists() or (d / "sparse_colored.ply").exists())
        )

    args.out.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    for name in targets:
        base = ROOT / "outputs" / name
        for kind, filename, is_cloud in (
            ("mesh", "dense_mesh.ply", False),
            ("sparse", "sparse_colored.ply", True),
        ):
            path = base / filename
            if not path.is_file():
                continue
            if is_cloud:
                geom = o3d.io.read_point_cloud(str(path))
                n = len(geom.points)
            else:
                geom = o3d.io.read_triangle_mesh(str(path))
                geom.compute_vertex_normals()
                n = len(geom.triangles)
            if n == 0:
                print(f"  [skip] {name}/{kind}: empty")
                continue

            arc = None
            cache = base / "scene_cache.npz"
            if cache.is_file():
                z = np.load(cache)
                if len(z["cam_ids"]) >= 2:
                    centres = np.stack([-R.T @ t for R, t in zip(z["Rs"], z["ts"])])
                    bb = geom.get_axis_aligned_bounding_box()
                    up_axis = np.zeros(3)
                    up_axis[int(np.argmax(bb.get_extent()))] = 1.0
                    arc = camera_arc(centres, bb.get_center(), up_axis)

            images = render_geometry(geom, args.width, args.height, args.views, is_cloud, arc)
            out_path = args.out / f"{kind}_{name}.png"
            Image.fromarray(strip(images)).save(out_path)
            unit = "points" if is_cloud else "triangles"
            print(f"  {out_path.relative_to(ROOT)}  ({n:,} {unit}, {args.views} views)")


if __name__ == "__main__":
    main()
