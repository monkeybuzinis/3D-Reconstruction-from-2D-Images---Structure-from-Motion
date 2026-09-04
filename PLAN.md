# Capstone Plan — 3D Reconstruction from 2D Images (SfM)

Guiding principle from the plan: **build a fully working pipeline with safe libraries first, then replace pieces one at a time with hand-written algorithms.**

Core architecture:
```
sfm/map.py        — Camera, Point3D, Map, Observation (scene graph)
sfm/io/           — dataset loading (Fountain-P11, generic photo folders), calibration, PLY export
sfm/features/     — SIFT extraction, FLANN matching, Lowe's ratio test, multi-view tracks
sfm/geometry/     — Hartley normalization, F/E estimation, RANSAC, triangulation, PnP
sfm/ba/           — Bundle Adjustment (scipy-backed, and hand-written LM: so(3), analytic Jacobian, Schur complement)
sfm/recon/        — incremental SfM orchestration
sfm/eval/         — Umeyama alignment, Accuracy/Completeness/F-score
sfm/mvs/          — dense stereo, fusion, Poisson meshing
```

---

## Data model & image loading — ✅ DONE
- [x] Create the package skeleton (`sfm/io`, `sfm/features`, `sfm/geometry`, `sfm/ba`, `sfm/recon`, `sfm/eval`)
- [x] `sfm/map.py`: `Camera`, `Point3D`, `Map` (+ `Observation` for 2D–3D links)
- [x] Loader for the 11 Fountain-P11 images + shared intrinsic matrix K (`sfm/io/fountain.py`)
- [x] Smoke test (`scripts/verify_week1.py`) — loads all 11 cameras, writes reference cloud to `outputs/week1/`

---

## Two-view seed — ✅ DONE
- [x] OpenCV SIFT extraction + FLANN matching + Lowe's ratio test (library allowed here)
- [x] Hand-write Hartley normalization + normalized 8-point algorithm (SVD) + RANSAC for the Fundamental matrix F
- [x] Compute Essential matrix `E = Kᵀ F K`, decompose to `[R|t]`, resolve the 4-way ambiguity with a cheirality check
- [x] Hand-write DLT triangulation to get the first 3D points, store into `Map`

Implemented in `sfm/features/matching.py`, `sfm/geometry/{fundamental,essential,triangulation}.py`, `sfm/recon/two_view.py`. Verified via `scripts/verify_two_view.py`: 1499/1500 RANSAC inliers pass cheirality, median reprojection error 0.32px.

`check.py` at the repo root prototyped F/E estimation, RANSAC, and Sampson error on the 0000/0001 pair before this stage formalized it into the `sfm/` package and added DLT triangulation.

**Deliverable:** a real 3D point cloud from the first camera pair, saved via `Map`.

---

## Scene expansion + safe Bundle Adjustment — ✅ DONE
- [x] Build observation tracks across all 11 images (which 2D points across views correspond to the same 3D point)
- [x] Hand-write PnP (Perspective-n-Point) to localize each new camera from known 2D–3D correspondences
- [x] Triangulate newly-visible 3D points as each camera is added (incremental SfM)
- [x] Wrap `scipy.optimize.least_squares` as a "safe" Bundle Adjustment to jointly refine all cameras + points

**Deliverable:** first end-to-end pipeline — sparse reconstruction + camera poses for all 11 Fountain images, using scipy for BA.

Implemented in `sfm/features/tracks.py` (all-pairs matching + union-find tracks: 8606 tracks over 11 images), `sfm/geometry/pnp.py` (calibrated DLT resectioning + RANSAC), `sfm/recon/incremental.py` (seed → PnP registration loop → triangulation, with periodic BA between camera additions to stop drift compounding), `sfm/ba/scipy_ba.py` (sparse-Jacobian scipy BA). Verified via `scripts/verify_incremental.py`: all 11 cameras registered, 7382 points after outlier filtering, median reprojection error 0.18px, p95 0.89px, max 7.66px.

**Note:** periodic BA during registration (not just once at the end) turned out to be necessary — without it, early triangulation error compounded silently and PnP started failing outright by camera 8-9. An outlier-filter pass (drop observations >4px, re-run BA) was also needed: a handful of bad tracks were dominating the squared-error cost and masking real convergence.

---

## Hand-written Bundle Adjustment — ✅ DONE
- [x] Analytic Jacobian of reprojection error w.r.t. camera and point parameters
- [x] Parameterize camera rotation as `so(3)` (axis-angle)
- [x] Solve the sparse normal equations efficiently via the Schur complement
- [x] Validate the analytic Jacobian with a finite-difference check before swapping out scipy

**Deliverable:** a self-written LM bundle adjuster that matches or beats the scipy version, with a passing finite-diff check.

Implemented in `sfm/ba/so3.py` (hand-rolled Rodrigues exponential map, verified against `cv2.Rodrigues` to 1e-15), `sfm/ba/jacobian.py` (analytic reprojection Jacobian, vectorized over all observations), `sfm/ba/lm.py` (Schur-complement Levenberg-Marquardt). Verified via `scripts/verify_ba_jacobian.py` (finite-difference check: max error 3e-7 against numerical derivatives, well under the 1e-4 tolerance) and `scripts/verify_hand_ba.py` (full 11-camera pipeline).

**Result:** matches scipy's final quality almost exactly (median reprojection error 0.18px vs scipy's 0.18px, p95 0.88px vs 0.88px) while running about **2x faster end-to-end** (50s vs ~99s) and **8x faster per BA call** on the two-view case (0.3s vs 2.4s) — the Schur complement avoids scipy's generic sparse-Jacobian machinery entirely. `sfm/recon/incremental.py` accepts either optimizer via a `bundle_adjust_fn` parameter, so the same incremental pipeline runs on scipy or the hand-written solver interchangeably.

---

## Evaluation & demo — evaluation math done, demo still open
- [x] Implement Umeyama alignment (scale + rotation + translation) to register the reconstructed cloud to ground truth
- [x] Compute Accuracy, Completeness, and F-score against the Fountain-P11 ground truth
- [x] Generalize the pipeline into a reusable tool (any photo folder, not just Fountain-P11) — the prerequisite for the self-captured demo
- [ ] Run the full pipeline on a self-captured object/scene for the demo
- [ ] Prepare final report and demo materials

**General reconstruction tool** (`scripts/reconstruct.py`): takes any folder of photos, determines camera intrinsics from — in order of trust — an explicit override, a Strecha-style `<image>.camera` sidecar file if the dataset ships one (`sfm/io/camera_file.py`), EXIF (`sfm/io/calibration.py` — prefers `FocalLengthIn35mmFilm`, falls back to `FocalLength`+sensor size), or finally an assumed FOV if none of that is available (e.g. stripped/screenshot images). Then runs the same hand-written incremental SfM + BA pipeline, with `--dense` to also run MVS + Poisson meshing.

Validated on three independent real datasets:
- **Fountain-P11 without its known K** (forcing the FOV fallback, ~7% focal-length error vs. the true calibrated value): all 11 cameras still registered correctly, reprojection error stayed low (0.40px median vs. 0.18px with exact K).
- **Rathaus** (a different Strecha-family building dataset, 7 images, `.ppm` format, real per-image `.camera` calibration): all 7 cameras registered, median reprojection error 0.92px, dense mesh visibly shows the building's facade structure. This also exercised the `.camera` sidecar parser and caught a real bug (`.ppm`/`.pgm` were missing from the supported extension list).
- **Herzjesu** (23 real Canon EOS D60 photos, genuine EXIF-derived K, no pre-supplied calibration): surfaced two real, distinct bugs, both fixed:
  1. A single failed `ransac_pnp()` call (`RuntimeError`) was uncaught anywhere in the incremental loop, crashing the entire reconstruction instead of just skipping that one camera. Fixed in `sfm/recon/incremental.py`: a camera that fails PnP is now parked in a retry set rather than either crashing the run or being permanently abandoned — it gets another shot once triangulating other cameras adds new points to the scene (this can unblock it), and the loop still provably terminates.
  2. A deeper, theoretically-grounded bug in `sfm/geometry/pnp.py`'s `ransac_pnp`: after RANSAC's search phase found a good pose, the function unconditionally re-fit using *every* inlier and returned that — but linear/DLT resectioning is a known-degenerate method on (near-)coplanar 3D points (a flat facade being the textbook case), and it gets *worse*, not better, the more coplanar points you feed it. Church-facade-heavy Herzjesu triggered this hard: some cameras were "succeeding" with 0-4 real inliers out of 700-1500 available. Fixed by only trusting the refit if it's at least as good as what the search phase already validated, otherwise falling back to the search-phase result. Result: registered cameras went from 11/23 (with garbage poses silently included) to 21/23 (correctly posed), and the outlier-filter keep-rate went from 14% to 88% — back in line with Fountain (97%) and Rathaus (99.5%).

This is the tool the self-captured demo will actually run. It's also now been stress-tested past the point where it silently produced wrong answers, which the earlier two datasets hadn't managed to trigger — worth knowing the fix exists before trusting future runs on new, less-planar-friendly data.

**Ground-truth caveat (important for the report):** the real Strecha laser-scan ground truth for Fountain-P11 is unobtainable — both known hosting URLs (`cvlabwww.epfl.ch`, `icwww.epfl.ch`) fail DNS resolution, and a web search independently confirms "its online service is not available anymore." Evaluation instead uses `Fountain/solution.graph`, an independent SfM/BA solve of the same photos (not laser-measured). This is a standard, defensible fallback — it measures "does our from-scratch reconstruction agree with another SfM solve" rather than "how accurate vs. a true scan" — but be upfront about which claim your report is making. (The Rathaus dataset's `.3Dpoints` files may be genuine ground truth rather than another SfM solve — not yet investigated.)

Implemented in `sfm/io/reference.py` (parses `solution.graph`'s explicit `VERTEX_CAM`/`VERTEX_XYZ` tags — more reliable than `solution.txt`'s unlabeled header, which turned out to have a real bug: the Week-1 loader was discarding 9 real points, mistaking them for camera-parameter lines; fixed), `sfm/eval/umeyama.py` (closed-form similarity alignment, validated against a synthetic known transform to <0.001 error), `sfm/eval/metrics.py` (Accuracy/Completeness/F-score via `scipy.spatial.cKDTree`, the standard Tanks & Temples / ETH3D protocol). Verified via `scripts/verify_evaluation.py`.

**Result:** aligning via the 11 corresponding camera centers gives a camera-position residual of just 0.0013 (mean, vs. a 0.20 reference baseline — well under 1%), and F-score against the reference cloud reaches 0.88 at 10% of baseline distance and 0.98 at 25% — strong agreement between this from-scratch reconstruction and an independent SfM solve of the same photos.

**Deliverable:** quantitative evaluation numbers + a self-captured demo reconstruction, ready for the report.

---

## Extension: Dense reconstruction (MVS) — ✅ DONE

Not part of the original plan — added because the sparse point cloud (7,256 points) didn't visually read as "the fountain" to a non-technical viewer, even correctly colored. This stage answers "make it actually look like the object," which the classical sparse-SfM stages above were never trying to do.

- [x] Pairwise dense stereo across consecutive camera pairs, using poses already recovered by SfM (rectification via `cv2.stereoRectify`, dense disparity via `cv2.StereoSGBM` + WLS left-right consistency filtering)
- [x] Fuse into one dense cloud, voxel-downsample + statistical/radius outlier removal + DBSCAN largest-cluster filtering to kill classical stereo's characteristic streak artifacts
- [x] Poisson surface reconstruction (Open3D) with density-based trimming of the "invented" low-confidence surface Poisson always adds to close gaps
- [x] Colored via the same `colorize_points`-style technique already used for the sparse cloud — no new appearance pipeline needed

Implemented in `sfm/mvs/{stereo,fuse,mesh}.py`, verified via `scripts/verify_mvs.py` on Fountain-P11 and via `scripts/reconstruct.py --dense` on Rathaus. **Result on Fountain:** 4.6M raw dense points → 18,655 after cleaning → a 40,000-triangle colored mesh that is visually recognizable as the fountain facade (visible stone/masonry texture, coherent architectural structure), a qualitative step change from the sparse cloud.

**Scope note:** this pushed the project setup itself further than planned — the system Python is externally-managed (PEP 668), so a dedicated project venv (`.venv/`, see `requirements.txt`) was created rather than overriding system protections; Open3D and OpenCV's contrib stereo/WLS modules are now real dependencies of the project, not just the original numpy/opencv-python/matplotlib set. Run MVS scripts with `.venv/bin/python3`, not plain `python3`.

---

## Extension: Interactive 3D viewer — ✅ DONE

Also not part of the original plan — built so reconstruction results are actually visible (rotate/zoom in a browser) rather than only static PNG snapshots or raw `.ply` files. A hand-rolled Canvas 2D viewer (no Three.js/WebGL library — CSP on the hosting artifact platform blocks arbitrary CDN scripts), with two modes: the sparse measured point cloud, and the dense Poisson mesh, sharing one orbit camera.

Two real bugs surfaced and got fixed: the world "up" axis was inverted (this project's world frame follows OpenCV's Y-down camera convention, but the viewer assumed Y-up like standard 3D tools — fixed by negating Y wherever data enters the viewer), and an earlier client-side ad hoc Delaunay-triangulated "grid" overlay was removed entirely at user request once the real Poisson mesh made it redundant and it was visibly distorting the object's appearance.

Generalized beyond Fountain-P11: `scripts/export_viewer.py <output_dir>` builds the viewer JSON from any `scene_cache.npz` (which `scripts/reconstruct.py` and `scripts/build_scene_cache.py` both produce), so any dataset run through the general pipeline gets the same interactive view. Published for both Fountain-P11 and Rathaus.
