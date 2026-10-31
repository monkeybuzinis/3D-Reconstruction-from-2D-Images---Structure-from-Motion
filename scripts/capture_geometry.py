"""
Capture geometry measured from BOTH solves, side by side.

Why this script replaces the earlier numbers. The arc / angular-step /
baseline-over-depth table was originally computed from this project's own
recovered camera poses. That is fine where the reconstruction succeeded and
circular where it did not: on a dataset this pipeline got wrong, the
"measured capture geometry" is a description of the failure, not an
explanation of it. Running COLMAP over the same photographs (see
scripts/compare_colmap.py) provides an independent set of poses to measure
from, so the two columns below can be compared and the circular entries
identified.

Definitions, chosen so they need no hand-placed "subject" estimate:
  depth        median over every (camera, observed point) pair of that
               point's depth in that camera's frame
  baseline     median distance between cameras adjacent in filename order
  arc          total angle swept about the scene centroid, summed over
               adjacent camera pairs
  step         median angle about the centroid between adjacent cameras

baseline/depth is the scale-free quantity that matters for triangulation:
it is the parallax available per adjacent pair.

Run from the repo root:
    .venv/bin/python3 scripts/capture_geometry.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_colmap import DATASETS, image_names, our_cache_path


def geometry(centres: np.ndarray, points: np.ndarray, depths: np.ndarray) -> dict:
    """centres in capture order; depths = per-(camera,point) depths already computed."""
    if len(centres) < 2 or len(points) == 0:
        return {}
    lo, hi = np.percentile(points, [5, 95], axis=0)
    inside = np.all((points >= lo) & (points <= hi), axis=1)
    centroid = points[inside].mean(axis=0) if inside.any() else points.mean(axis=0)

    baselines = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    median_depth = float(np.median(depths))

    v = centres - centroid
    n = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / np.where(n == 0, 1, n)
    cos = np.clip(np.sum(v[:-1] * v[1:], axis=1), -1, 1)
    steps = np.degrees(np.arccos(cos))

    return {
        "n_cameras": int(len(centres)),
        "median_depth": median_depth,
        "median_baseline": float(np.median(baselines)),
        "baseline_over_depth": float(np.median(baselines) / median_depth) if median_depth else float("nan"),
        "arc_total_deg": float(steps.sum()),
        "median_step_deg": float(np.median(steps)),
    }


def from_colmap(dataset: str) -> dict:
    import pycolmap
    model = ROOT / "colmap" / dataset / "sparse" / "0"
    if not model.exists():
        return {}
    r = pycolmap.Reconstruction(model)
    posed = [im for im in r.images.values() if im.has_pose]
    if len(posed) < 2:
        return {}
    posed.sort(key=lambda im: im.name)
    centres = np.stack([im.projection_center() for im in posed])

    xyz = {pid: np.array(p.xyz) for pid, p in r.points3D.items()}
    depths = []
    for im in posed:
        cfw = im.cam_from_world()
        R, t = np.array(cfw.rotation.matrix()), np.array(cfw.translation)
        seen = [xyz[p2.point3D_id] for p2 in im.points2D if p2.has_point3D()]
        if seen:
            depths.extend(((np.stack(seen) @ R.T) + t)[:, 2])
    pts = np.stack(list(xyz.values()))
    return geometry(centres, pts, np.array(depths))


def from_ours(dataset: str) -> dict:
    path = our_cache_path(dataset)
    if not path.is_file():
        return {}
    c = np.load(path)
    cam_ids, Rs, ts, xyz = c["cam_ids"], c["Rs"], c["ts"], c["xyz"]
    if len(cam_ids) < 2 or len(xyz) == 0:
        return {}
    order = np.argsort(cam_ids)  # cam id is filename order
    centres = np.stack([-Rs[i].T @ ts[i] for i in order])
    depths = np.concatenate([((xyz @ Rs[i].T) + ts[i])[:, 2] for i in order])
    depths = depths[depths > 0]
    return geometry(centres, xyz, depths)


def main() -> None:
    import pycolmap
    pycolmap.logging.minloglevel = 3

    rows = {}
    hdr = (f"{'dataset':16s} | {'ours: b/d':>9s} {'arc':>7s} {'step':>6s} | "
           f"{'COLMAP: b/d':>11s} {'arc':>7s} {'step':>6s} | {'cams o/c':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for name in DATASETS:
        o, c = from_ours(name), from_colmap(name)
        rows[name] = {"ours": o, "colmap": c}
        fmt = lambda g, k, s: f"{g[k]:{s}}" if g else "-"
        print(f"{name:16s} | {fmt(o,'baseline_over_depth','9.3f')} {fmt(o,'arc_total_deg','7.1f')} "
              f"{fmt(o,'median_step_deg','6.1f')} | {fmt(c,'baseline_over_depth','11.3f')} "
              f"{fmt(c,'arc_total_deg','7.1f')} {fmt(c,'median_step_deg','6.1f')} | "
              f"{(str(o.get('n_cameras','-'))+'/'+str(c.get('n_cameras','-'))):>9s}")

    out = ROOT / "colmap" / "capture_geometry.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
