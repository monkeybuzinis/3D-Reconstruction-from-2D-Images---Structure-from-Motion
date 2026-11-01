# 3D Reconstruction from 2D Images — Structure from Motion

A classical Structure-from-Motion (SfM) pipeline, built mostly from scratch in Python: give it a folder of overlapping photographs of one subject and it recovers where each photograph was taken from, plus a 3D point cloud of what was photographed. No depth sensor, no scanner, no deep learning, no Gaussian Splatting, no generative AI — every 3D point is measured via multi-view geometry.

It has been run on **16 datasets**: four published benchmarks that ship externally measured camera poses, and twelve sets photographed on a phone. Both the successes and the failures are reported below.

**Hand-derived**: Fundamental/Essential matrix estimation (Hartley normalization, normalized 8-point algorithm, RANSAC), pose recovery via cheirality, DLT triangulation, calibrated DLT PnP, and a full Levenberg–Marquardt bundle adjuster (analytic Jacobian, so(3) rotation parameterization, Schur-complement linear solve).

**Library-assisted, deliberately** (the same "safe library for a solved sub-problem" boundary throughout): SIFT/FLANN for 2D feature matching, `scipy.optimize.least_squares` as a first working bundle adjustment before the hand-written LM solver replaced it, `cv2.StereoSGBM` for dense per-pixel stereo, and Open3D for Poisson surface reconstruction.

**[→ View the reconstructions in 3D](https://monkeybuzinis.github.io/3D-Reconstruction-from-2D-Images---Structure-from-Motion/)** — every result below, rotatable in your browser, including the failures. Nothing to download.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the module dependency graph and data flow, and [BUNDLE_ADJUSTMENT.md](BUNDLE_ADJUSTMENT.md) for a from-scratch walkthrough of the hand-written optimizer's math.

## Results

Every reconstruction below can be inspected directly at the [results site](https://monkeybuzinis.github.io/3D-Reconstruction-from-2D-Images---Structure-from-Motion/).

### Against externally measured camera poses

Four datasets ship per-image camera poses measured independently of any software. These are the only figures here that compare against a survey rather than against another program. Both systems are given the same surveyed intrinsic matrix, held fixed, so what is compared is the reconstruction algorithm and not the calibration source. Errors are reported after a Umeyama similarity alignment, which removes the seven degrees of freedom a reconstruction cannot determine.

| dataset | registered | rotation error | centre error | COLMAP, same calibration |
|---|---|---|---|---|
| Fountain-P11 | 11/11 | **0.035°** | 0.13% | 11/11, 0.031°, 0.15% |
| Rathaus | 5/7 | **0.102°** | 0.22% | 7/7, 0.181°, 0.35% |
| Herz-Jesu-P25 | 3/25 | **0.634°** | 0.05% | 25/25, 0.043°, 0.17% |
| Herz-Jesus-P8 | 6/8 | **15.683°** | 23.56% | 8/8, 0.102°, 0.12% |

Centre error is the mean camera-centre error as a percentage of the median distance between neighbouring cameras, which makes it independent of scene scale. Reproduce with `scripts/benchmark_vs_colmap.py`.

**The honest reading: where this pipeline converges the poses are accurate, and whether it converges is not guaranteed.** On three of the four it lands within 0.7° of the surveyed orientation, and on Rathaus it is the more accurate of the two systems. Herz-Jesus-P8 is reported as the default random seed produced it — over six seeds, five land between 0.121° and 0.169° and one lands at 15.683°. The pipeline's own reprojection residual separates the two cases without reference to the survey: 0.56 px against 0.15–0.16 px.

**The gap against COLMAP is completeness, not accuracy.** COLMAP registers every available image on all four; this pipeline does on one. Most of that traces to one design choice — COLMAP refines the calibration while it builds and this pipeline holds it fixed, so when the initial estimate is poor COLMAP recovers and this pipeline is stuck with it.

### Against another SfM solve

Reported separately because it is a weaker claim: agreement between two estimators, not accuracy against truth.

- **Fountain-P11**: F-score **0.88** at 10% of the camera baseline, **0.98** at 25%, computed against the `solution.graph` distributed with the dataset — which is itself the output of a prior SfM solve.
- **Rathaus** is the exception, with structure that *was* measured: recall **0.996** at 10% of the baseline, median completeness 0.27% of the scene diagonal.
- **Phone-captured**: `dataset/woodlog`, 12 iPhone photographs of a log pile and the one self-captured set distributed here, registers **12/12** cameras and agrees with COLMAP to **0.35°** of rotation and 1.0% of baseline — with no calibration available to either system beyond EXIF. On the best of the sets not distributed, 25 photographs of a boot, both register all 25 and agree to 1.31° and 6.9%.

### Where it fails

- **Smooth, shiny or textureless subjects** give feature matching almost nothing to work with. These are the datasets where this pipeline places a minority of the cameras.
- **Implementation limits, not capture limits.** On several self-captured sets this pipeline could not reconstruct, COLMAP can — so the photographs were adequate and the code was not. Running a mature system over identical inputs is what separates those two cases, and it corrected an earlier conclusion of this project's.
- **One genuine capture failure**: `dataset/robotpu`, where both systems register 2 of 39 images.
- Development surfaced and fixed several real bugs: a degeneracy of linear PnP on near-planar scenes, a cheirality test performed in the wrong coordinate frame, and three faults in how the dense stage derived its own parameters. Each was measured rather than assumed.

## Setup

The system Python is externally-managed (PEP 668), so this project uses a dedicated virtual environment rather than overriding that protection:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run everything with `.venv/bin/python3`, not the system `python3`.

## Usage

### Reconstruct any folder of photographs

```bash
.venv/bin/python3 scripts/reconstruct.py path/to/photos/          # sparse point cloud
.venv/bin/python3 scripts/reconstruct.py path/to/photos/ --dense  # + dense mesh
```

Writes to `outputs/<folder-name>/` unless `--output` says otherwise: a coloured sparse cloud, a scene cache, a run manifest recording the commit, library versions, random seeds and timings, and with `--dense` a Poisson mesh.

Camera intrinsics are resolved automatically, in order of trust: an explicit `--focal-mm`/`--sensor-width-mm` override, a Strecha-style `<image>.camera` sidecar if the dataset ships one, EXIF metadata, or finally an assumed field of view if nothing else is available. The pipeline prints which of those it used. **The last is a guess and the results reflect it** — on the same 11 photographs, the surveyed matrix gives 11/11 cameras at 0.18 px while the field-of-view fallback gives 9/11 at 0.58 px.

Photographs should be numbered in roughly the order they were taken while moving around the subject.

### Compare against COLMAP

```bash
cp -r path/to/photos dataset/myscene        # the comparison resolves datasets by name
.venv/bin/python3 scripts/reconstruct.py dataset/myscene
.venv/bin/python3 scripts/compare_colmap.py myscene
```

Runs COLMAP over the same photographs through `pycolmap`, which `requirements.txt` installs as a prebuilt wheel — no system COLMAP build is needed. Reports both solves side by side (cameras registered, points, mean reprojection error) and the agreement between them.

Two conventions matter. The photographs must sit in `dataset/<name>/`, and this project's own result must be at `outputs/<name>/` — which is where `reconstruct.py` writes by default, so do not pass `--output` if you intend to compare. COLMAP's result is cached in `colmap/<name>/` and reused on later runs; `--force` deletes that cache and re-solves.

COLMAP runs with default settings and is **not** treated as ground truth. It is an independent second opinion, and on Rathaus it is the less accurate of the two.

### Score both systems against a survey

```bash
.venv/bin/python3 scripts/benchmark_vs_colmap.py   # the four datasets with measured poses
```

Reproduces the first table above, with the surveyed calibration held fixed on both sides.

### Interactive 3D viewer

```bash
.venv/bin/python3 scripts/export_viewer_html.py outputs/<dataset>
xdg-open outputs/<dataset>/viewer.html
```

A single self-contained HTML file that opens straight from disk — sparse measured points and, if `--dense` was used, the real Poisson mesh. `scripts/export_viewer.py` produces the same data as JSON instead.

### Per-stage verification

Each stage has a script that exercises it alone, useful for reading the pipeline or checking a change:

```bash
.venv/bin/python3 scripts/verify_two_view.py         # two-view seed (F/E/pose/triangulation)
.venv/bin/python3 scripts/verify_incremental.py      # full incremental SfM + scipy BA
.venv/bin/python3 scripts/verify_ba_jacobian.py      # finite-difference check of the analytic BA Jacobian
.venv/bin/python3 scripts/verify_hand_ba.py          # full pipeline with the hand-written LM solver
.venv/bin/python3 scripts/verify_evaluation.py       # Umeyama alignment + Accuracy/Completeness/F-score
.venv/bin/python3 scripts/verify_rathaus_evaluation.py   # pose + structure vs Rathaus's measured sidecars
.venv/bin/python3 scripts/verify_mvs.py              # dense stereo (MVS) + Poisson meshing
.venv/bin/python3 scripts/verify_cheirality.py       # cheirality regression test (synthetic, no dataset needed)
.venv/bin/python3 scripts/capture_geometry.py        # capture statistics from both solves, side by side
.venv/bin/python3 scripts/verify_subject_coverage.py <dataset>   # did the points land on the subject?
```

Most expect a benchmark dataset present under `dataset/`; `verify_cheirality.py` is synthetic and needs none.

## Project layout

```
sfm/map.py     — shared scene state (Camera, Point3D, Observation, Map)
sfm/io/        — photo-folder loading, calibration resolution, reference data, PLY export
sfm/features/  — SIFT/FLANN matching, multi-view tracks
sfm/geometry/  — F/E estimation, triangulation, PnP (hand-written)
sfm/ba/        — bundle adjustment: scipy-backed, and hand-written Schur-complement LM
sfm/recon/     — incremental SfM orchestration
sfm/eval/      — Umeyama alignment, Accuracy/Completeness/F-score
sfm/mvs/       — dense stereo, fusion, Poisson meshing, dense-parameter derivation
scripts/       — the reconstruction tool, the COLMAP comparison, per-stage checks
```

Full dependency graph and data flow: [ARCHITECTURE.md](ARCHITECTURE.md).

## Datasets

Four datasets ship with this repository, chosen so that every path through the pipeline is reproducible from a clone.

`dataset/Fountain/`, `dataset/rathaus/` and `dataset/herzjesu/` come from the same benchmark family (Strecha et al.) and carry the measured-pose claims above. Each photograph has a `<image>.camera` sidecar, so these exercise the surveyed-calibration path.

`dataset/woodlog/` is 12 iPhone photographs with no sidecar at all — the calibration comes from EXIF, which is the path any folder of your own photographs will take. It is the smallest honest end-to-end example in the repository, and runs with the commands in the usage section above.

The other eleven phone-captured sets are not distributed. The pipeline works on any folder of photographs, so bring your own — see the usage section above.

Two caveats worth knowing before trusting a number:

- **Fountain-P11's laser-scan ground truth is no longer hosted anywhere reachable.** Structural evaluation falls back to `dataset/Fountain/solution.graph`, an independent SfM solve of the same photographs, so it measures agreement between two estimators rather than accuracy against a survey. The *camera poses* in the `.camera` sidecars are surveyed, and those are what the first table uses.
- **`dataset/rathaus/` is the one set with both.** Its `.camera` and `.3Dpoints` sidecars are externally measured, and `scripts/verify_rathaus_evaluation.py` scores pose and structure against them.
