"""
Evaluate a reconstruction against whatever reference geometry its dataset
ships, in one place.

This replaces three near-identical dataset-specific scripts. The evaluation
itself never varied between them -- align with Umeyama, compare rotations,
compare structure -- only the file format of the reference did. So the
format-specific part is three small loaders and everything after is shared.

Supported reference formats, detected automatically:

  strecha   <image>.camera sidecars, optionally with <image>.3Dpoints.
            Convention is x = K[R^T | -R^T t]X, so the stored R is
            camera-to-world and the stored t is the camera CENTRE -- not
            this project's world-to-camera (R, t). Used by: rathaus.
  colmap    cameras.txt / images.txt / points3D.txt, the COLMAP text model
            that ETH3D also distributes. Used by: statue.
  graph     Fountain-P11's solution.graph, an independent SfM solve rather
            than measured geometry. Used by: Fountain.
  pmatrix   <image>_P.txt 3x4 projection matrices. Used by: buddha.

What each reference supports differs, and the script says so rather than
printing metrics that do not mean anything. A sparse reference cloud makes
completeness meaningful but not accuracy: a correct point of ours lying
between two surveyed points has no nearby reference point through no fault
of its own.

Run from the repo root:
    .venv/bin/python3 scripts/evaluate.py rathaus
    .venv/bin/python3 scripts/evaluate.py statue --recon outputs/statue2
    .venv/bin/python3 scripts/evaluate.py Fountain
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfm.eval.metrics import evaluate_reconstruction
from sfm.eval.umeyama import apply_similarity, umeyama_alignment

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".ppm", ".pgm", ".bmp", ".tif", ".tiff"}


# --------------------------------------------------------------- helpers
def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic angle between two rotations, in degrees."""
    cos = (np.trace(R_a @ R_b.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _rel(p: Path) -> str:
    """Repo-relative when possible, absolute otherwise."""
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def image_names(folder: Path) -> list[str]:
    names = sorted(p.name for p in folder.iterdir()
                   if p.suffix.lower() in IMAGE_EXT and "vertex_colors" not in p.name)
    if names:
        return names
    subdirs = [p for p in folder.iterdir() if p.is_dir()]
    return image_names(subdirs[0]) if len(subdirs) == 1 else []


# ------------------------------------------------------- reference loaders
# Each returns (poses, points, note) where poses maps image name ->
# (world-to-camera R, camera centre), points is (N,3) or None, and note
# says what the reference actually is.

def load_strecha(folder: Path):
    poses, pts = {}, []
    for sidecar in sorted(folder.glob("*.camera")):
        v = [float(x) for x in sidecar.read_text().split()]
        R_c2w = np.array(v[12:21]).reshape(3, 3)
        centre = np.array(v[21:24])
        poses[sidecar.name[: -len(".camera")]] = (R_c2w.T, centre)
        p3d = sidecar.with_suffix(".3Dpoints")
        if p3d.is_file():
            lines = p3d.read_text().strip().split("\n")
            n = int(lines[0].split()[1])
            pts.extend([float(x) for x in line.split()] for line in lines[1:1 + n])
    points = None
    if pts:
        # The per-image files re-list the same surveyed points; counting a
        # point once per image that sees it would weight the metrics toward
        # whatever the overlap region happens to be.
        points = np.unique(np.round(np.array(pts), 4), axis=0)
    return poses, points, "measured calibration; reference cloud is surveyed points (sparse)"


def load_colmap(folder: Path):
    import pycolmap
    pycolmap.logging.minloglevel = 3
    rec = pycolmap.Reconstruction(str(folder))
    poses = {
        Path(im.name).name: (np.array(im.cam_from_world().rotation.matrix()),
                             np.array(im.projection_center()))
        for im in rec.images.values() if im.has_pose
    }
    pts = np.stack([np.array(p.xyz) for p in rec.points3D.values()]) if rec.num_points3D() else None
    return poses, pts, "benchmark calibration; points3D.txt is a sparse triangulation, not a laser scan"


def load_pmatrix(folder: Path):
    from scipy.linalg import rq
    poses = {}
    for f in sorted(folder.glob("*_P.txt")):
        P = np.loadtxt(f)
        K, R = rq(P[:, :3])
        S = np.diag(np.sign(np.diag(K)))
        K, R = K @ S, S @ R
        t = np.linalg.inv(K) @ P[:, 3]
        stem = f.name[: -len("_P.txt")]
        poses[stem] = (R, -R.T @ t)
    return poses, None, "true projection matrices; no reference point cloud"


def load_graph(folder: Path):
    from sfm.io.reference import load_reference_camera_centers, load_reference_points
    centres = load_reference_camera_centers(folder)
    names = image_names(folder)
    poses = {names[cid]: (None, c) for cid, c in centres.items() if cid < len(names)}
    return poses, load_reference_points(folder), "an independent SfM solve, NOT measured geometry"


def detect(folder: Path):
    if list(folder.glob("*.camera")):
        return "strecha", load_strecha
    if (folder / "cameras.txt").is_file() and (folder / "images.txt").is_file():
        return "colmap", load_colmap
    if list(folder.glob("*_P.txt")):
        return "pmatrix", load_pmatrix
    if (folder / "solution.graph").is_file():
        return "graph", load_graph
    return None, None


# ------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="folder name under dataset/")
    ap.add_argument("--recon", type=Path, default=None,
                    help="output dir holding scene_cache.npz (default: outputs/<dataset>)")
    ap.add_argument("--csv", action="store_true", help="also write per-camera errors to outputs/eval/")
    ap.add_argument("--figure", action="store_true",
                    help="also write an alignment overlay to outputs/eval/<dataset>_alignment_overlay.png")
    args = ap.parse_args()

    folder = ROOT / "dataset" / args.dataset
    if not folder.is_dir():
        raise SystemExit(f"No dataset at {folder}")

    kind, loader = detect(folder)
    if kind is None:
        raise SystemExit(
            f"{args.dataset} ships no reference geometry this script recognises "
            "(.camera, cameras.txt, *_P.txt or solution.graph).\n"
            "Internal metrics only -- use scripts/compare_colmap.py to compare against COLMAP instead."
        )

    recon = (args.recon.resolve() if args.recon else ROOT / "outputs" / args.dataset)
    cache = recon / "scene_cache.npz"
    if not cache.is_file():
        raise SystemExit(f"No reconstruction at {cache}. Run scripts/reconstruct.py first.")

    ref_poses, ref_pts, note = loader(folder)
    names = image_names(folder)
    c = np.load(cache)
    ours = {names[int(i)]: (c["Rs"][k], -c["Rs"][k].T @ c["ts"][k])
            for k, i in enumerate(c["cam_ids"]) if int(i) < len(names)}
    our_pts = c["xyz"]

    print("=" * 74)
    print(f"{args.dataset}  —  reference format: {kind}")
    print("=" * 74)
    print(f"  reference : {len(ref_poses)} poses"
          + (f", {len(ref_pts):,} points" if ref_pts is not None else ", no point cloud"))
    print(f"              {note}")
    print(f"  ours      : {len(ours)} cameras, {len(our_pts):,} points   ({_rel(cache)})")

    # Reference keys may carry an extension the images do not, or vice versa.
    def match(n):
        for k in (n, Path(n).stem, n + ".camera"):
            if k in ref_poses:
                return k
        return next((k for k in ref_poses if Path(k).stem == Path(n).stem), None)

    pairs = [(n, match(n)) for n in ours]
    pairs = [(a, b) for a, b in pairs if b is not None]
    if len(pairs) < 3:
        raise SystemExit(f"Only {len(pairs)} cameras matched the reference by name — cannot align.")

    A = np.stack([ours[a][1] for a, _ in pairs])
    B = np.stack([ref_poses[b][1] for _, b in pairs])
    scale, R_align, t_align = umeyama_alignment(A, B)
    centre_err = np.linalg.norm(apply_similarity(scale, R_align, t_align, A) - B, axis=1)
    baseline = float(np.median(np.linalg.norm(np.diff(B, axis=0), axis=1)))

    have_rot = all(ref_poses[b][0] is not None for _, b in pairs)
    rot_err = (np.array([rotation_angle_deg(ours[a][0] @ R_align.T, ref_poses[b][0])
                         for a, b in pairs]) if have_rot else None)

    print("\n" + "-" * 74)
    print("CAMERA POSE ACCURACY")
    print("-" * 74)
    print(f"  similarity scale {scale:.4f}   median reference baseline {baseline:.4f}")
    hdr = f"\n  {'image':24s} {'centre err':>12s} {'% baseline':>12s}"
    print(hdr + (f" {'rotation':>12s}" if have_rot else ""))
    print("  " + "-" * (len(hdr) + (13 if have_rot else 0)))
    for i, (a, _) in enumerate(pairs):
        line = f"  {a:24s} {centre_err[i]:12.4f} {100*centre_err[i]/baseline:11.2f}%"
        print(line + (f" {rot_err[i]:11.3f}°" if have_rot else ""))
    print("  " + "-" * (len(hdr) + (13 if have_rot else 0)))
    for label, fn in (("mean", np.mean), ("median", np.median), ("max", np.max)):
        line = f"  {label:24s} {fn(centre_err):12.4f} {100*fn(centre_err)/baseline:11.2f}%"
        print(line + (f" {fn(rot_err):11.3f}°" if have_rot else ""))
    if not have_rot:
        print("\n  (this reference supplies camera centres only, so no rotation comparison)")

    if ref_pts is not None:
        print("\n" + "-" * 74)
        print("STRUCTURE")
        print("-" * 74)
        aligned = apply_similarity(scale, R_align, t_align, our_pts)
        taus = tuple(round(baseline * f, 5) for f in (0.05, 0.1, 0.25, 0.5))
        m = evaluate_reconstruction(aligned, ref_pts, thresholds=taus)
        print(f"  completeness (reference → ours)  mean={m['completeness_mean']:8.4f}  "
              f"median={m['completeness_median']:8.4f}")
        print(f"  accuracy     (ours → reference)  mean={m['accuracy_mean']:8.4f}  "
              f"median={m['accuracy_median']:8.4f}")
        print(f"\n  {'tau':>10s} {'recall':>9s} {'precision':>11s} {'F-score':>9s}")
        for tau in taus:
            print(f"  {tau:10.5f} {m[f'recall@{tau}']:9.3f} {m[f'precision@{tau}']:11.3f} "
                  f"{m[f'f_score@{tau}']:9.3f}")
        print("\n  A sparse reference makes completeness meaningful and accuracy not:")
        print("  a correct point of ours between two reference points is penalised for nothing.")

    if args.figure and ref_pts is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        aligned_c = apply_similarity(scale, R_align, t_align, A)
        aligned_p = apply_similarity(scale, R_align, t_align, our_pts)
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")
        rng = np.random.default_rng(0)
        shown = ref_pts if len(ref_pts) <= 20000 else ref_pts[rng.choice(len(ref_pts), 20000, replace=False)]
        ax.scatter(*shown.T, s=6, alpha=0.9, c="black", label="reference points")
        ax.scatter(*aligned_p.T, s=1, alpha=0.25, c="red", label="ours (aligned)")
        ax.scatter(*aligned_c.T, s=60, c="blue", marker="^", label="our cameras (aligned)")
        ax.scatter(*B.T, s=60, c="green", marker="v", label="reference cameras")
        # Frame the structure and the camera arc on a common cube, so the
        # standoff distance is not distorted by per-axis autoscaling.
        lo = np.minimum(np.percentile(ref_pts, 1, axis=0), B.min(axis=0))
        hi = np.maximum(np.percentile(ref_pts, 99, axis=0), B.max(axis=0))
        mid, half = (lo + hi) / 2, (hi - lo).max() / 2
        ax.set_xlim(mid[0]-half, mid[0]+half); ax.set_ylim(mid[1]-half, mid[1]+half)
        ax.set_zlim(mid[2]-half, mid[2]+half)
        ax.view_init(elev=15, azim=-70)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title(f"{args.dataset}: reconstruction aligned to its reference")
        fig.tight_layout()
        figp = ROOT / "evaluation" / "metrics" / f"{args.dataset}_alignment_overlay.png"
        figp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figp, dpi=120)
        print(f"\nWrote {_rel(figp)}")

    if args.csv:
        out = ROOT / "evaluation" / "metrics" / f"{args.dataset}_pose_accuracy.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            f.write("image,centre_error,centre_error_pct_of_baseline,rotation_error_deg\n")
            for i, (a, _) in enumerate(pairs):
                r = f"{rot_err[i]:.6f}" if have_rot else ""
                f.write(f"{a},{centre_err[i]:.6f},{100*centre_err[i]/baseline:.4f},{r}\n")
        print(f"\nWrote {_rel(out)}")


if __name__ == "__main__":
    main()
