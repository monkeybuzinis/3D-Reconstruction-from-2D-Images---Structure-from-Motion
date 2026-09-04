# 3D Reconstruction from 2D Images — Structure from Motion

A classical Structure-from-Motion (SfM) pipeline, built mostly from scratch in Python: reconstructs a 3D point cloud and camera poses from a set of overlapping photographs. No deep learning, no Gaussian Splatting, no generative AI — every 3D point is measured, provably, via multi-view geometry.

**Hand-derived**: Fundamental/Essential matrix estimation (Hartley normalization, normalized 8-point algorithm, RANSAC), pose recovery via cheirality, DLT triangulation, calibrated DLT PnP, and a full Levenberg-Marquardt Bundle Adjuster (analytic Jacobian, so(3) rotation parameterization, Schur-complement linear solve).

**Library-assisted, deliberately** (the same "safe library for a solved sub-problem" boundary throughout): SIFT/FLANN for 2D feature matching, `scipy.optimize.least_squares` as a first working Bundle Adjustment before the hand-written LM solver replaced it, `cv2.StereoSGBM` for dense per-pixel stereo matching, and Open3D for Poisson surface reconstruction.

See [PLAN.md](PLAN.md) for the full stage-by-stage build log and results, [ARCHITECTURE.md](ARCHITECTURE.md) for the module dependency graph and data flow, and [BUNDLE_ADJUSTMENT.md](BUNDLE_ADJUSTMENT.md) for a from-scratch walkthrough of the hand-written optimizer's math.

## Results

On the Fountain-P11 benchmark: all 11 cameras correctly recovered, **0.18px median reprojection error**, and (evaluated against an independent SfM solve of the same photos, via hand-derived Umeyama alignment on 11 corresponding camera centers) an **F-score of 0.88** at 10% of the camera baseline and **0.98** at 25%.

The pipeline has also been run successfully on two other real datasets it was never tuned for — Rathaus (7 images) and Herzjesu (23 images, real camera EXIF, no pre-supplied calibration) — which surfaced and fixed two genuine bugs, including a documented degeneracy of linear PnP on near-planar scenes (see PLAN.md's "Evaluation & demo" section for the full story).

## Setup

The system Python is externally-managed (PEP 668), so this project uses a dedicated virtual environment rather than overriding that protection:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run everything with `.venv/bin/python3`, not the system `python3`.

## Usage

### Fountain-P11 walkthrough (one script per pipeline stage)

```bash
.venv/bin/python3 scripts/verify_week1.py          # data model + dataset loading
.venv/bin/python3 scripts/verify_two_view.py        # two-view seed (F/E/pose/triangulation)
.venv/bin/python3 scripts/verify_incremental.py     # full incremental SfM + scipy BA
.venv/bin/python3 scripts/verify_ba_jacobian.py      # finite-difference check of the analytic BA Jacobian
.venv/bin/python3 scripts/verify_hand_ba.py         # full pipeline with the hand-written LM solver
.venv/bin/python3 scripts/verify_evaluation.py      # Umeyama alignment + Accuracy/Completeness/F-score
.venv/bin/python3 scripts/verify_mvs.py             # dense stereo (MVS) + Poisson meshing
```

### General reconstruction tool (any folder of photos)

```bash
.venv/bin/python3 scripts/reconstruct.py path/to/photos/            # sparse point cloud
.venv/bin/python3 scripts/reconstruct.py path/to/photos/ --dense     # + dense mesh
```

Camera intrinsics are determined automatically, in order of trust: an explicit `--focal-mm`/`--sensor-width-mm` override, a Strecha-style `<image>.camera` sidecar file if the dataset ships one, EXIF metadata, or finally an assumed field of view if nothing else is available. Photos should be named/numbered in roughly the order they were taken while moving around the subject.

### Interactive 3D viewer

```bash
.venv/bin/python3 scripts/build_scene_cache.py            # or reconstruct.py, which caches automatically
.venv/bin/python3 scripts/export_viewer.py outputs/<dataset>
```

Produces the JSON an interactive rotate/zoom viewer (published as a Claude Artifact) is built from — sparse measured points and, if `--dense` was used, the real Poisson mesh.

## Project layout

```
sfm/map.py     — shared scene state (Camera, Point3D, Observation, Map)
sfm/io/        — dataset loading (Fountain-P11 + any photo folder), calibration, PLY export
sfm/features/  — SIFT/FLANN matching, multi-view tracks
sfm/geometry/  — F/E estimation, triangulation, PnP (hand-written)
sfm/ba/        — Bundle Adjustment: scipy-backed, and hand-written Schur-complement LM
sfm/recon/     — incremental SfM orchestration
sfm/eval/      — Umeyama alignment, Accuracy/Completeness/F-score
sfm/mvs/       — dense stereo, fusion, Poisson meshing
scripts/       — per-stage checks + the general reconstruct.py/export_viewer.py tools
```

Full dependency graph and data flow: [ARCHITECTURE.md](ARCHITECTURE.md).

## Datasets

`Fountain/`, `rathaus/`, and `herzjesu/` are from the same benchmark family (Strecha et al.). Ground-truth caveat: Fountain-P11's real laser-scan ground truth is no longer hosted anywhere reachable; evaluation instead uses `Fountain/solution.graph`, an independent SfM solve of the same photos — see PLAN.md for what that does and doesn't prove.
