"""
Both systems against the same survey: the table in Section 5.4.

Every other comparison in this project scores one system against another.
This one scores this pipeline AND COLMAP against surveyed camera poses, over
the same photographs, with the same alignment. That is the only arrangement
that answers "how far is a from-scratch implementation from a mature one",
because it puts a measured reference on both sides instead of treating
COLMAP as the reference.

Why the intrinsics are fixed on both sides. Left to its defaults COLMAP
refines the calibration and starts the principal point at the image centre.
On Rathaus the surveyed principal point is 127 px away from the centre, and
atan(127/3950) is 1.84 degrees of systematic camera tilt -- which is almost
exactly the gap the default run shows against the survey (2.15 deg, falling
to 0.181 when the surveyed calibration is supplied). This pipeline reads the
sidecars, so comparing against a self-calibrating COLMAP would compare
calibration sources rather than reconstruction algorithms.

Why the rotation column is split. The similarity that aligns a
reconstruction to the survey is fitted to camera CENTRES; it never sees the
orientations. Whatever rotation offset the two frames differ by therefore
survives the alignment and is charged to every camera equally. The script
separates that shared offset from the per-camera spread about it, because
the two mean different things: the spread is the reconstruction's internal
orientation consistency, the shared part is how the whole solve sits against
the survey.

Datasets. Rathaus lives in dataset/rathaus. The three EPFL sets are not in
the repository (they are large, and redistributing them is not ours to do);
fetch them from the openMVG mirror into dataset/_benchmarks/<name>/, which
ships each image beside its <image>.camera sidecar:

    github.com/openMVG/SfM_quality_evaluation
        /Benchmarking_Camera_Calibration_2008/{fountain-P11,Herz-Jesus-P8,Herz-Jesus-P25}

Run from the repo root:
    .venv/bin/python3 scripts/benchmark_vs_colmap.py
    .venv/bin/python3 scripts/benchmark_vs_colmap.py --force      # re-run COLMAP
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfm.eval.umeyama import apply_similarity, umeyama_alignment

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm"}

# label -> (image folder, our scene_cache dir, COLMAP work dir)
BENCHMARKS = [
    ("Fountain-P11",  "dataset/_benchmarks/fountain-P11",   "outputs/_final/fountain-P11", "fountain-P11-fixedK"),
    ("Rathaus",       "dataset/rathaus",                    "outputs/_final/rathaus",     "rathaus-fixedK"),
    ("Herz-Jesus-P8", "dataset/_benchmarks/Herz-Jesus-P8",  "outputs/_final/Herz-Jesus-P8", "herzjesu-P8-fixedK"),
    ("Herz-Jesu-P25", "dataset/_benchmarks/Herz-Jesus-P25", "outputs/_final/Herz-Jesus-P25", "herzjesu-P25-fixedK"),
]


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    cos = (np.trace(R_a @ R_b.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def image_names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir()
                  if p.suffix.lower() in IMAGE_EXTENSIONS and "vertex_colors" not in p.name)


def load_survey(folder: Path) -> dict:
    """
    Strecha .camera sidecars. The stored convention is x = K[R^T | -R^T t]X,
    so the 3x3 block is camera-to-world and the 3-vector is the camera
    CENTRE, not this project's world-to-camera translation.
    """
    survey = {}
    for sidecar in sorted(folder.glob("*.camera")):
        v = [float(n) for n in sidecar.read_text().split()]
        survey[sidecar.name[: -len(".camera")]] = (
            np.array(v[12:21]).reshape(3, 3).T, np.array(v[21:24]))
    return survey


def load_ours(folder: Path, recon: Path):
    names = image_names(folder)
    cache = np.load(recon / "scene_cache.npz")
    poses = {names[int(i)]: (cache["Rs"][k], -cache["Rs"][k].T @ cache["ts"][k])
             for k, i in enumerate(cache["cam_ids"]) if int(i) < len(names)}
    return poses, len(cache["xyz"])


def load_colmap(folder: Path, work_name: str, force: bool):
    """COLMAP with the surveyed intrinsics fixed, cached under colmap/."""
    import pycolmap
    pycolmap.logging.minloglevel = 3

    work = ROOT / "colmap" / work_name
    if force and work.exists():
        shutil.rmtree(work)
    if not (work / "sparse" / "0").exists():
        images = work / "images"
        images.mkdir(parents=True, exist_ok=True)
        for name in image_names(folder):
            if not (images / name).exists():
                shutil.copy(folder / name, images / name)
        database = work / "database.db"
        database.unlink(missing_ok=True)

        K = np.array([float(n) for n in
                      sorted(folder.glob("*.camera"))[0].read_text().split()[:9]]).reshape(3, 3)
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = "PINHOLE"
        reader.camera_params = f"{K[0,0]},{K[1,1]},{K[0,2]},{K[1,2]}"
        pipeline = pycolmap.IncrementalPipelineOptions()
        pipeline.ba_refine_focal_length = False
        pipeline.ba_refine_principal_point = False
        pipeline.ba_refine_extra_params = False

        pycolmap.extract_features(database, images, camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=reader)
        pycolmap.match_exhaustive(database)
        pycolmap.incremental_mapping(database, images, work / "sparse", options=pipeline)

    rec = pycolmap.Reconstruction(str(work / "sparse" / "0"))
    poses = {Path(im.name).name: (np.array(im.cam_from_world().rotation.matrix()),
                                  np.array(im.projection_center()))
             for im in rec.images.values() if im.has_pose}
    return poses, rec.num_points3D()


def score(poses: dict, survey: dict) -> dict | None:
    common = sorted(set(poses) & set(survey))
    if len(common) < 3:
        return None
    ours = np.stack([poses[n][1] for n in common])
    truth = np.stack([survey[n][1] for n in common])
    scale, R_align, t_align = umeyama_alignment(ours, truth)
    centre_err = np.linalg.norm(apply_similarity(scale, R_align, t_align, ours) - truth, axis=1)

    # Baseline from the whole surveyed track, not just the matched subset, so
    # a system that registers fewer cameras is not measured on a shorter ruler.
    all_centres = np.stack([survey[n][1] for n in sorted(survey)])
    baseline = float(np.median(np.linalg.norm(np.diff(all_centres, axis=0), axis=1)))

    residual = [(poses[n][0] @ R_align.T).T @ survey[n][0] for n in common]
    angles = np.array([rotation_angle_deg(np.eye(3), Ri) for Ri in residual])
    U, _, Vt = np.linalg.svd(np.mean(residual, axis=0))     # the shared part
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1
    shared = U @ Vt
    spread = np.array([rotation_angle_deg(shared, Ri) for Ri in residual])
    return {
        "n": len(common),
        "rot_mean_deg": float(angles.mean()),
        "rot_shared_deg": rotation_angle_deg(np.eye(3), shared),
        "rot_spread_deg": float(spread.mean()),
        "centre_pct": float(100 * centre_err.mean() / baseline),
        "centre_max_pct": float(100 * centre_err.max() / baseline),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-run COLMAP instead of using the cache")
    args = ap.parse_args()

    header = (f"{'dataset':23s} {'imgs':>5s} {'reg':>7s} {'points':>8s} "
              f"{'rotation':>9s} {'shared':>8s} {'spread':>8s} {'centre':>8s} {'max':>7s}")
    print(header)
    print("-" * len(header))
    for label, folder_rel, recon_rel, work_name in BENCHMARKS:
        folder = ROOT / folder_rel
        if not folder.is_dir():
            print(f"{label:23s} missing: {folder_rel} (see the module docstring)")
            continue
        recon = ROOT / recon_rel
        if not (recon / "scene_cache.npz").is_file():
            print(f"{label:23s} missing: {recon_rel}/scene_cache.npz -- run scripts/reconstruct.py")
            continue

        survey = load_survey(folder)
        n_images = len(image_names(folder))
        for system, (poses, n_points) in (("ours", load_ours(folder, recon)),
                                          ("COLMAP", load_colmap(folder, work_name, args.force))):
            m = score(poses, survey)
            if m is None:
                print(f"{label:23s} {system}: fewer than three cameras matched the survey")
                continue
            name = f"{label} ({system})" if system == "COLMAP" else label
            print(f"{name:23s} {n_images:5d} {m['n']:3d}/{n_images:<3d} {n_points:8,d} "
                  f"{m['rot_mean_deg']:8.3f}° {m['rot_shared_deg']:7.3f}° "
                  f"{m['rot_spread_deg']:7.3f}° {m['centre_pct']:7.2f}% {m['centre_max_pct']:6.2f}%")


if __name__ == "__main__":
    main()
