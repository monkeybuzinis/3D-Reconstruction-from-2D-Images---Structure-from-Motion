"""
External baseline: run COLMAP on the same photographs and compare.

Why this exists. Every other evaluation in this project is either against a
reference that ships with a benchmark (Fountain, Rathaus, Buddha) or purely
internal (reprojection error). The self-captured sets had neither, so a
claim like "shoe is unreconstructable because of how it was shot" rested on
measured capture geometry plus this pipeline's own failure -- which cannot
by itself distinguish "the data is inadequate" from "the implementation is
inadequate". Running a mature, widely-used SfM system over the identical
inputs separates those two.

COLMAP is run through pycolmap with its DEFAULT settings (one shared camera
across each dataset, since every set is one camera; SIMPLE_RADIAL, which is
COLMAP's default and slightly more capable than this project's distortion-
free pinhole model). No per-dataset tuning on either side.

What is reported per dataset:
  - cameras registered, points, mean reprojection error, for both systems
  - agreement between the two solves: Umeyama-align our camera centres onto
    COLMAP's over the commonly-registered images, then residual centre
    error and rotation difference
  - where ground truth exists (Rathaus), BOTH systems scored against it,
    so the comparison is to truth rather than to each other

Run from the repo root:
    .venv/bin/python3 scripts/compare_colmap.py                 # all datasets
    .venv/bin/python3 scripts/compare_colmap.py rathaus shoe    # a subset
    .venv/bin/python3 scripts/compare_colmap.py --force rathaus # re-run COLMAP
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfm.eval.umeyama import apply_similarity, umeyama_alignment

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".ppm", ".pgm"}

# dataset name -> (image dir, our scene_cache). Every cache lives at
# outputs/<dataset>/scene_cache.npz. Fountain's folder holds a
# vertex_colors.ppm that is not a photograph (the generic loader would
# otherwise count it as a 12th image -- a known wart, see PLAN.md).
DATASETS = [
    "Fountain", "rathaus", "herzjesu",
    "woodlog", "office", "church", "guitar", "fire-hydrant",
    "library-statue", "buffalo", "buffalo-crop", "shoe", "robotpu", "statue",
    "child-statue", "open-mouth",
]
NOT_PHOTOGRAPHS = {"vertex_colors.ppm"}


def dataset_dir(dataset: str) -> Path:
    """
    The folder actually holding the photographs. Several self-captured sets
    arrived as dataset/<name>/<name>/*.JPEG (the phone export nested them),
    so fall through one level when the top level holds no images at all.
    """
    folder = ROOT / "dataset" / dataset
    if any(p.suffix.lower() in IMAGE_EXTENSIONS for p in folder.iterdir()):
        return folder
    subdirs = [p for p in folder.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        return subdirs[0]
    return folder


def image_names(dataset: str) -> list[str]:
    """The same images, in the same order, that sfm/io/generic.py would load."""
    return sorted(
        p.name for p in dataset_dir(dataset).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.name not in NOT_PHOTOGRAPHS
    )


def our_cache_path(dataset: str) -> Path:
    return ROOT / "outputs" / dataset / "scene_cache.npz"


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    cos = (np.trace(R_a @ R_b.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def true_intrinsics(dataset: str):
    """
    (fx, fy, cx, cy) from a Strecha `.camera` sidecar, if this dataset ships
    one. Only Fountain and Rathaus do.
    """
    sidecars = sorted(dataset_dir(dataset).glob("*.camera"))
    if not sidecars:
        return None
    v = [float(x) for x in sidecars[0].read_text().split()[:9]]
    K = np.array(v).reshape(3, 3)
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def run_colmap(dataset: str, force: bool = False, fix_intrinsics: bool = False) -> dict:
    """
    Run COLMAP into outputs/colmap/<dataset>/, caching the result.

    fix_intrinsics: hand COLMAP the dataset's true calibration and stop it
    refining. This matters more than it looks. Left to itself COLMAP pins
    the principal point at the image centre, which on Rathaus is 127px from
    the true one -- atan(127/3950) = 1.84 degrees of systematic camera tilt,
    which is essentially the whole gap against ground truth. This project's
    pipeline reads the sidecars, so comparing the two without this flag
    compares calibration sources, not reconstruction algorithms.
    """
    import pycolmap

    work = ROOT / "colmap" / (dataset + ("-fixedK" if fix_intrinsics else ""))
    db, sparse = work / "database.db", work / "sparse"
    names = image_names(dataset)

    if force and work.exists():
        shutil.rmtree(work)
    if not (sparse / "0").exists():
        work.mkdir(parents=True, exist_ok=True)
        sparse.mkdir(exist_ok=True)
        t0 = time.time()
        reader = pycolmap.ImageReaderOptions()
        pipeline = pycolmap.IncrementalPipelineOptions()
        if fix_intrinsics:
            fx, fy, cx, cy = true_intrinsics(dataset)
            reader.camera_model = "PINHOLE"
            reader.camera_params = f"{fx},{fy},{cx},{cy}"
            pipeline.ba_refine_focal_length = False
            pipeline.ba_refine_principal_point = False
            pipeline.ba_refine_extra_params = False
        pycolmap.extract_features(
            db, dataset_dir(dataset), image_names=names,
            camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
        )
        pycolmap.match_exhaustive(db)
        pycolmap.incremental_mapping(db, dataset_dir(dataset), sparse, options=pipeline)
        (work / "runtime.txt").write_text(f"{time.time() - t0:.1f}\n")

    # COLMAP splits an unconnectable set into several models; take the
    # largest, and record how many there were -- a fragmented result is
    # itself a finding about the capture.
    models = sorted(p for p in sparse.iterdir() if p.is_dir())
    if not models:
        return {"n_images": len(names), "n_models": 0, "registered": 0,
                "points": 0, "mean_reproj": float("nan"), "images": {}}

    best, best_recon = None, None
    for m in models:
        r = pycolmap.Reconstruction(m)
        if best_recon is None or len(r.images) > len(best_recon.images):
            best, best_recon = m, r

    poses = {
        im.name: (np.array(im.cam_from_world().rotation.matrix()),
                  np.array(im.projection_center()))
        for im in best_recon.images.values() if im.has_pose
    }
    return {
        "n_images": len(names),
        "n_models": len(models),
        "registered": len(poses),
        "points": best_recon.num_points3D(),
        "mean_reproj": float(best_recon.compute_mean_reprojection_error()),
        "runtime_s": float((work / "runtime.txt").read_text().strip()),
        "images": poses,
    }


def our_result(dataset: str) -> dict:
    cache_path = our_cache_path(dataset)
    if not cache_path.is_file():
        return {}
    cache = np.load(cache_path)
    names = image_names(dataset)
    cam_ids, Rs, ts = cache["cam_ids"], cache["Rs"], cache["ts"]
    poses = {}
    for i, cid in enumerate(cam_ids):
        if int(cid) < len(names):
            poses[names[int(cid)]] = (Rs[i], -Rs[i].T @ ts[i])
    return {"n_images": len(names), "registered": len(poses),
            "points": int(len(cache["xyz"])), "images": poses}


def agreement(ours: dict, theirs: dict) -> dict | None:
    """Umeyama our camera centres onto COLMAP's over commonly-registered images."""
    common = sorted(set(ours.get("images", {})) & set(theirs.get("images", {})))
    if len(common) < 3:
        return None
    A = np.stack([ours["images"][n][1] for n in common])
    B = np.stack([theirs["images"][n][1] for n in common])
    scale, R_align, t_align = umeyama_alignment(A, B)
    centre_err = np.linalg.norm(apply_similarity(scale, R_align, t_align, A) - B, axis=1)
    baseline = float(np.median(np.linalg.norm(np.diff(B, axis=0), axis=1)))
    rot_err = np.array([
        rotation_angle_deg(ours["images"][n][0] @ R_align.T, theirs["images"][n][0])
        for n in common
    ])
    return {
        "n_common": len(common),
        "centre_err_pct_baseline": float(100 * centre_err.mean() / baseline) if baseline else float("nan"),
        "centre_err_max_pct_baseline": float(100 * centre_err.max() / baseline) if baseline else float("nan"),
        "rot_err_mean_deg": float(rot_err.mean()),
        "rot_err_max_deg": float(rot_err.max()),
    }


def score_against_rathaus_truth(poses: dict) -> dict | None:
    """Both systems scored against dataset/rathaus's shipped .camera files."""
    from scripts.verify_rathaus_evaluation import load_reference_camera

    cam_paths = sorted((ROOT / "dataset" / "rathaus").glob("*.camera"))
    ref = {p.name.replace(".camera", ""): load_reference_camera(p) for p in cam_paths}
    common = sorted(set(poses) & set(ref))
    if len(common) < 3:
        return None
    A = np.stack([poses[n][1] for n in common])
    B = np.stack([ref[n][2] for n in common])
    scale, R_align, t_align = umeyama_alignment(A, B)
    centre_err = np.linalg.norm(apply_similarity(scale, R_align, t_align, A) - B, axis=1)
    baseline = float(np.median(np.linalg.norm(np.diff(B, axis=0), axis=1)))
    rot_err = np.array([rotation_angle_deg(poses[n][0] @ R_align.T, ref[n][1]) for n in common])
    return {
        "n": len(common),
        "centre_err_pct_baseline": float(100 * centre_err.mean() / baseline),
        "rot_err_mean_deg": float(rot_err.mean()),
        "rot_err_max_deg": float(rot_err.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=None)
    ap.add_argument("--force", action="store_true", help="re-run COLMAP even if cached")
    ap.add_argument("--quiet", action="store_true", default=True)
    args = ap.parse_args()

    if args.quiet:
        import pycolmap
        pycolmap.logging.minloglevel = 3

    targets = args.datasets or DATASETS

    # Seeded from whatever is already on disk, so running a subset tops the
    # file up instead of replacing it with just those few datasets.
    out = ROOT / "colmap" / "comparison.json"
    results = json.loads(out.read_text()) if out.exists() else {}

    for name in targets:
        if not (ROOT / "dataset" / name).is_dir():
            print(f"[skip] dataset/{name} not found")
            continue
        print(f"\n=== {name} ===", flush=True)
        theirs = run_colmap(name, force=args.force)
        ours = our_result(name)

        print(f"  COLMAP: {theirs['registered']}/{theirs['n_images']} registered, "
              f"{theirs['points']:,} points, mean reproj {theirs['mean_reproj']:.3f}px, "
              f"{theirs['n_models']} model(s), {theirs.get('runtime_s', 0):.0f}s")
        if ours:
            print(f"  Ours:   {ours['registered']}/{ours['n_images']} registered, "
                  f"{ours['points']:,} points")
        else:
            print("  Ours:   no cached reconstruction")

        agree = agreement(ours, theirs) if ours else None
        if agree:
            print(f"  Agreement over {agree['n_common']} common cameras: "
                  f"rotation {agree['rot_err_mean_deg']:.3f}° mean "
                  f"({agree['rot_err_max_deg']:.3f}° max), "
                  f"centre {agree['centre_err_pct_baseline']:.2f}% of baseline")

        # Where the dataset ships real calibration, also run COLMAP with it
        # fixed, so the two systems are compared on equal intrinsics.
        if true_intrinsics(name) is not None:
            fixed = run_colmap(name, force=args.force, fix_intrinsics=True)
            print(f"  COLMAP (true K, not refined): {fixed['registered']}/{fixed['n_images']} "
                  f"registered, {fixed['points']:,} points, mean reproj {fixed['mean_reproj']:.3f}px")
            entry_fixed = {k: v for k, v in fixed.items() if k != "images"}
        else:
            fixed, entry_fixed = None, None

        entry = {
            "colmap": {k: v for k, v in theirs.items() if k != "images"},
            "ours": {k: v for k, v in ours.items() if k != "images"},
            "agreement": agree,
            "colmap_true_K": entry_fixed,
        }

        if name == "rathaus":
            # Both pose dicts are keyed by image filename ("rdimage.000.ppm"),
            # which is exactly how the .camera sidecars are named too.
            gt_ours = score_against_rathaus_truth(ours["images"]) if ours else None
            gt_colmap = score_against_rathaus_truth(theirs["images"])
            gt_fixed = score_against_rathaus_truth(fixed["images"]) if fixed else None
            entry["vs_ground_truth"] = {"ours": gt_ours, "colmap": gt_colmap,
                                        "colmap_true_K": gt_fixed}
            print("\n  --- all three vs the dataset's shipped ground truth ---")
            for label, g in (("ours", gt_ours), ("COLMAP", gt_colmap),
                             ("COLMAP+K", gt_fixed)):
                if g:
                    print(f"    {label:8s} rotation {g['rot_err_mean_deg']:.3f}° mean "
                          f"({g['rot_err_max_deg']:.3f}° max), "
                          f"centre {g['centre_err_pct_baseline']:.2f}% of baseline")

        results[name] = entry

        # Written after every dataset, not once at the end: a full sweep is
        # tens of minutes of CPU SIFT and losing all of it to an interrupted
        # run partway through is a needless way to spend an evening.
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 86)
    print(f"{'dataset':16s} {'ours':>12s} {'COLMAP':>12s} {'rot agree':>11s} {'centre agree':>13s}")
    print("=" * 86)
    for name in targets:
        e = results.get(name)
        if e is None:
            continue
        o = f"{e['ours'].get('registered','-')}/{e['ours'].get('n_images','-')}" if e["ours"] else "-"
        c = f"{e['colmap']['registered']}/{e['colmap']['n_images']}"
        a = e["agreement"]
        rot = f"{a['rot_err_mean_deg']:.3f}°" if a else "-"
        ctr = f"{a['centre_err_pct_baseline']:.2f}%" if a else "-"
        print(f"{name:16s} {o:>12s} {c:>12s} {rot:>11s} {ctr:>13s}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
