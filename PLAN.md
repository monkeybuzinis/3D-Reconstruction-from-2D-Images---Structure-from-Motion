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

---

## Review fix: cheirality in the wrong coordinate frame — ✅ DONE

Found by code review (professor), not by a crash. `sfm/recon/incremental.py::triangulate_new_points`
tested depth as `xyz[2]` — the **world** Z — while `xyz` is only in a camera's
frame for the camera that defines the world origin. The frame is pinned to the
**seed** camera (`seed_two_view` gives it `R=I, t=0`), and `--seed 2 5` would make
camera 2 the origin, so the bug was wider than "not camera 0": it applied whenever
`cam_a` was not the seed camera, which on Fountain-P11 was 7,013 of 7,184
triangulation attempts (97.6%).

- [x] Measure depth in each camera's own frame: `(cam.R @ xyz + cam.t)[2]` for both cameras
- [x] Regression test with triangulation pairs that exclude the seed camera (`scripts/verify_cheirality.py`)
- [x] Audit the three other depth checks — `geometry/essential.py:89`, `incremental.py:76`,
      `recon/two_view.py:61` are all correct, because each triangulates against `P0 = K[I|0]`,
      so world coordinates genuinely are camera-0 coordinates there

**Impact on Fountain-P11:** 1,148 valid points had been wrongly rejected; 0 invalid points
had been wrongly accepted (the capture's geometry hid that direction). Points after
filtering 7,382 → 8,553, outlier keep-rate 97% → 99.4%, median reprojection error
unchanged at 0.18px.

**What the recovered points turned out to be:** not noise. They have 2.0–3.0° parallax
(median 2.69°, versus 10.8° scene median) — well above the ~1.5° degeneracy threshold a
min-parallax filter would use, so no such filter was added. Projecting them back into
camera 9 shows them landing on a **second building** in the Fountain-P11 scene, a pale
palace facade beside the fountain that every one of the old world-Z rejections had
discarded (`outputs/eval/recovered_building_cam9.png`). The fix recovered real structure.

`scripts/verify_cheirality.py` builds a synthetic 4-camera rig and checks two
counterexamples in opposite directions — a valid point at negative world Z (old code
rejected it) and a point behind camera A at positive world Z (old code accepted it) —
plus two controls both versions must agree on, and it fails loudly if no case
distinguishes the two implementations.

---

## Dense-stage parameter derivation — ✅ DONE

The cheirality fix made Fountain's dense mesh **worse** (18,655 → 8,571 cleaned points),
which exposed three latent problems in how the MVS stage chose its parameters. All three
are now in `sfm/mvs/params.py`, shared by `scripts/reconstruct.py` and `scripts/verify_mvs.py`
(which each had their own copy of the derivation before).

1. **Depth window from world Z.** Same mistake as the cheirality bug, one layer down:
   `np.percentile(sparse_xyz[:, 2], [1, 99])` treats world Z as depth. Once points behind
   the seed camera were legitimately accepted, Fountain's window went **negative**
   (-6.21..8.45). Now computed from actual per-camera depths, so it is positive by
   construction, and padded outward (×0.8, ×1.25) because the percentiles come from sparse
   points that thin out exactly at a surface's far edge — unpadded, Rathaus lost an entire
   wing of the building.
2. **Voxel size from the sparse cloud's bounding box.** The sparse cloud spans the whole
   scene the tracks reach; dense stereo only reconstructs the near surface, and the ratio
   between them is dataset-dependent (≈4× on Fountain, ≈1× on Rathaus), so no fixed divisor
   works everywhere. Now derived from the sensor's resolution limit — one pixel at distance
   d covers `d / focal_px`, so the voxel is a fixed **12 pixel footprints**. That constant
   transfers between datasets because it is anchored to the camera.
3. **Depth window capped at 20× the median depth.** Low-parallax points reproject well from
   anywhere along their ray, so nothing upstream removes them, and they can sit effectively
   at infinity: the buffalo capture produced windows of 2,642 and then **5,029,180** against
   a median depth near 4. The cap leaves every verified dataset untouched (Fountain 23.66,
   Rathaus 10.33, Herzjesu 67.57).

**`keep_largest_cluster` → `keep_significant_clusters`** (`sfm/mvs/mesh.py`) was the largest
single improvement. Keeping only the biggest DBSCAN cluster assumes the real surface is one
connected blob; windows, doorways and occlusion gaps break a facade into several. Rathaus was
keeping **5,096 of 14,773 points (34%)** and discarding a 4,466-point wing as though it were a
streak artifact. Now every cluster at least 10% the size of the largest is kept. The old
function remains for experiments that want exactly it.

Every mesh was checked by rendering it, not by triangle count (before/after images in
`outputs/eval/dense_before_after_*.png`):

| dataset  | mesh before | mesh after | verdict |
|---|---|---|---|
| Fountain | 85,981 tris | 202,118 | denser surface, fewer holes |
| Rathaus  | 36,567      | 92,618  | missing wing recovered |
| Statue   | 108,843     | 394,831 | blocky → smooth |
| Herzjesu | 138,378     | 195,152 | arches solid instead of speckled |

Also fixed: `reconstruct.py` crashed with `ValueError: zero-size array` when a thin sparse
solve left an empty dense cloud. It now reports the real problem — too few cameras
registered — and exits cleanly.

---

## Self-captured datasets — what actually determines success

Five self-captured sets were run. The results separate cleanly by **surface** and
**calibration**, not by anything in the code.

| dataset | photos | EXIF | cameras | median error | outcome |
|---|---|---|---|---|---|
| statue | 11 | stripped (Shotwell) | 6/11 | 0.315px | FOV guessed; unusable mesh |
| child-statue | 13 | stripped | 8/13 | 0.537px | rough form only |
| open-mouth | 9 | stripped | 4/9 | 0.683px | failed — views ~40° apart |
| buffalo | 23 | **iPhone 13, intact** | **23/23** | 0.417px | poses perfect, mesh poor |
| buffalo (cropped) | 23 | intact + override | **23/23** | **0.373px** | same |

**Calibration is the first gate.** The three stripped sets fell back to an assumed 55° field
of view (`sfm/io/calibration.py`), which caps achievable quality — a sweep of 40–82° only
ever found a least-bad value. Buffalo's EXIF gave `focal_px = 26/36 × 2048 = 1479.1`
directly, and it is the only self-captured set where every camera registered.

**Surface is the second gate, and it binds harder.** The buffalo is mirror-polished steel:
dense stereo assumes a surface looks the same from two viewpoints, and on a mirror the
reflection moves with the camera. Isolating the sculpture in 3D (ground-plane fit, then
keeping what stands above it near the camera ring) found only **~1,000 of 14,449 dense points**
anywhere on it — the rest is grass and trees. No crop or parameter can recover geometry the
photos never contained.

**Cropping photos requires correcting K.** Centre-cropping buffalo to 1550×1250 kept 23/23
cameras and slightly improved median error, but the EXIF formula `focal_px = (f35/36) × width`
is only valid for the full frame: recomputing it from the narrower width would have given
1119px instead of 1479px, a 24% error. The crop was run with
`--focal-mm 26 --sensor-width-mm 27.2461`, chosen so `26/27.2461 × 1550 = 1479.1`. Crops must
also be centred (the pipeline assumes the principal point is the image centre) and identical
across every photo (one K is shared). Uniform *resizing* is safe; cropping is not.

**Capture guidance that follows from this:** matte textured subject; keep EXIF (copy
originals, don't export through an editor); fill 60–80% of the frame; 40–60 photos at 10–15°
steps; stay at 1× zoom (changing zoom or lens breaks the shared-K assumption); overcast light.

---

## Buddha MVS benchmark: two-view accuracy against ground truth — ✅ DONE

`dataset/buddha/` is a published MVS benchmark: 67 views at 2736×1540 of a matte carved
stone head, shipping `mvs.ini`, per-image `_P.txt` projection matrices and `_seeds.bin`.
Decomposing the projection matrices (RQ) gives one identical K for all 67 views —
`fx = fy = 1860.9`, zero skew, principal point within 5px of the image centre — which matches
this pipeline's assumptions almost exactly.

**This dataset provides what Fountain-P11 cannot: true camera poses.** Fountain's
"ground truth" is another SfM solve (the laser scan is no longer hosted), so it measures
agreement between two solvers. Here the hand-written two-view chain can be scored against
real camera matrices.

Measured over 64 adjacent view pairs — `ransac_fundamental_matrix` → `essential_from_fundamental`
→ `decompose_essential` → `cheirality_check`, i.e. the hand-derived core of the project:

| group | median rotation error | p90 | max | n |
|---|---|---|---|---|
| pairs with ≥100 inliers | **0.137°** | 0.316° | 0.50° | 54 |
| view change ≤ 20° | 0.139° | 0.317° | 0.44° | 22 |
| view change ≤ 30° | 0.138° | 0.318° | 0.49° | 42 |
| all pairs | 0.172° | 0.767° | 140.9° | 64 |

Translation is recoverable only up to scale from two views, so its **direction** is scored:
median 0.578°, p90 1.599° (pairs with ≥100 inliers). Per-pair numbers in
`outputs/eval/buddha_pose_accuracy.csv`.

The 5 failing pairs are self-identifying and need no special pleading: each had 9–40 inliers
against a median of 278 for accurate pairs, and a 49–140° view change — they are not really
adjacent views. An inlier-count threshold separates them cleanly.

**Where the pipeline stops working, measured.** Incremental SfM registered only 7 of 67
cameras here despite that two-view accuracy, and dense stereo produced unusable geometry. The
cause is capture geometry, quantified across every dataset:

| dataset | baseline/depth | angular step | cameras registered |
|---|---|---|---|
| buffalo | 0.063 | 3.4° | 23/23 |
| herzjesu | 0.111 | 5.6° | 21/23 |
| rathaus | 0.204 | 8.5° | 7/7 |
| Fountain | 0.209 | 11.9° | 11/11 |
| **buddha** | **0.325** | **15.8°** | **7/67** |

Buddha is 1.5–3× wider-baseline than anything that works, and it orbits a **convex object** on
a sphere, so each surface patch appears in only a few views — whereas Fountain and Herzjesu are
facades every camera sees at once. Incremental registration needs ≥6 already-triangulated
points visible in the next camera, and that condition keeps failing. The dense stage hits the
same wall from another direction: disparities run 316–927px against SGBM's 256 search range,
and widening it to 512 removes the fragments but cannot fix the distortion from rectifying a
15.8° rotation.

**Published reconstruction.** With pose estimation measured separately, the matching and
triangulation side was exercised on this dataset using the supplied poses: 59 adjacent pairs
matched and RANSAC-verified, triangulated with the project's own `triangulate_dlt`, filtered
by cheirality and a 2px reprojection threshold, giving **18,360 points** that mesh into a
recognizable head (`outputs/buddha/`). The page states plainly that the poses are the
benchmark's and the geometry is ours.

Two further findings from this dataset:

- **Filename order is not capture order here.** Consecutive filenames average 89° apart
  (max 177°). `sfm/io/generic.py` documents that it treats filename order as capture order —
  correct for self-shot sequences, wrong for benchmarks that number views arbitrarily. Sorting
  by the ground-truth camera positions brings the step to 21.7°. A next-best-view seed
  selection would remove the assumption.
- **Image resolution matters for matching.** Halving resolution halved the matches
  (877 → 407 on one pair) at the same 95% inlier rate.

---

## Published interactive viewers

| dataset | link |
|---|---|
| Fountain-P11 | https://claude.ai/artifact/GAUeroF7kvQD8f5R1WUAkz |
| Rathaus | https://claude.ai/artifact/7RHBPZVdvFNNrbqfweFs6a |
| Herzjesu | https://claude.ai/artifact/HWdD1YLspXeTawnmsDAxbf |
| Statue | https://claude.ai/artifact/Um3qju1eeDEDpUKRqjP9hK |
| Buffalo | https://claude.ai/artifact/XFYN9vPwpkG8LPZTusTYQj |
| Child statue | https://claude.ai/artifact/To13rVX4WMrXEz9LxZhMd9 |
| Open mouth | https://claude.ai/artifact/DQF2f18idmdpNLHxxHHWTX |
| Buddha (benchmark poses + our triangulation) | https://claude.ai/artifact/PttdQZWFbAkSH8gJ8srFR5 |
| Code map (module/flow reference) | https://claude.ai/artifact/JYMyv82xFigucZGPgPinhc |

Pre-fix outputs are preserved under `outputs/_pre_cheirality_fix/` for before/after comparison.
