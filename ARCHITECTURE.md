# System Architecture

```
=====================================================================================
  STRUCTURE-FROM-MOTION PIPELINE  -  SYSTEM ARCHITECTURE
  26 modules | 3,310 lines | 22 scripts | 3 dependency tiers | 0 cycles
=====================================================================================


LAYER 1 - ENTRY POINTS   scripts/, 22 files
+-----------------------------------------------------------------------------------+
|  reconstruct.py        evaluate.py            compare_colmap.py  export_viewer.py |
|  any image folder      metrics vs GT          vs COLMAP          web viewer       |
|                                                                                   |
|  verify_*.py (7)       ablate_pnp_guard.py    capture_geometry.py  *_figures.py |
|  stage regression      ablation               capture geometry     report figures|
|                                                                                   |
|  Only this layer knows dataset paths. The library takes images and intrinsics.    |
+-----------------------------------------------------------------------------------+
                                      |
                                      v
LAYER 2 - CORE LIBRARY   sfm/, 26 modules
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  +-----------------+      +-----------------+      +-----------------------+      |
|  | sfm/io/      (7)|      |sfm/features/ (2)|      | sfm/geometry/      (4)|      |
|  |-----------------|      |-----------------|      |-----------------------|      |
|  | generic         |      | matching        |      | fundamental           |      |
|  | calibration     |= = =>| tracks          |= = =>| essential             |      |
|  | camera_file     |      |                 |      | triangulation         |      |
|  | fountain        |      | SIFT            |      | pnp                   |      |
|  | colorize        |      | ratio test      |      |                       |      |
|  | ply             |      | union-find      |      | 8-point, E = K'FK,    |      |
|  | reference       |      |   tracks        |      | DLT, RANSAC           |      |
|  |                 |      |                 |      |                       |      |
|  | [ cv2, PIL ]    |      | [ cv2 ]         |      | [ N U M P Y  O N L Y ]|      |
|  +-----------------+      +-----------------+      +-----------------------+      |
|                                                                 |                 |
|                                                                 v                 |
|  +-----------------------+                       +------------------------+       |
|  | sfm/ba/            (4)|                       | sfm/recon/          (2)|       |
|  |-----------------------|   injected as the     |------------------------|       |
|  | so3      Rodrigues    |<= = = = = = = = = = = | two_view    seed pair  |       |
|  | jacobian analytic     |   bundle_adjust_fn    | incremental 479 lines  |       |
|  | lm       hand-written |   argument, NOT       |                        |       |
|  | scipy_ba baseline     |   imported            | resect -> triangulate  |       |
|  |                       |                       |   -> local BA          |       |
|  | LM + Schur complement |                       |   -> global BA         |       |
|  | [ numpy, scipy, cv2 ] |                       | [ numpy ]              |       |
|  +-----------------------+                       +------------------------+       |
|                                                                 |                 |
|  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -|- - - - - - -    |
|  below the line: these import NOTHING from the rest of sfm/     |                 |
|                                                                 v                 |
|  +-----------------------+                       +------------------------+       |
|  | sfm/mvs/           (4)|                       | sfm/eval/           (2)|       |
|  |-----------------------|                       |------------------------|       |
|  | params   geometry ->  |                       | umeyama   7-DOF        |       |
|  |          SGBM knobs   |                       |           similarity   |       |
|  | stereo   rectify,     |                       |           alignment    |       |
|  |          SGBM, WLS    |                       | metrics   RMSE,        |       |
|  | fuse     Z = fB/d     |                       |           coverage     |       |
|  | mesh     DBSCAN,      |                       |                        |       |
|  |          Poisson      |                       |                        |       |
|  | [ cv2, open3d ]       |                       | [ numpy, scipy ]       |       |
|  +-----------------------+                       +------------------------+       |
|                                                                                   |
+-----------------------------------------------------------------------------------+
                                      |
                                      v
LAYER 3 - SHARED STATE
+-----------------------------------------------------------------------------------+
|  IN PROCESS                             ON DISK                                   |
|  sfm/map.py                             outputs/<dataset>/scene_cache.npz         |
|  +--------------------------+           +------------------------------+          |
|  | Camera (id, K, R, t,     |           | cam_ids, Ks, Rs, ts,         |          |
|  |         width, height)   |= = = = = >| widths, heights,             |          |
|  | Point3D (id, xyz, color) |           | xyz, colors                  |          |
|  | Observation (cam, pt, uv)|           +------------------------------+          |
|  | Map (cameras, points)    |           read by 12 scripts; the only              |
|  +--------------------------+           hand-off from sparse to dense             |
|  imports nothing, imported by 7 modules                                           |
+-----------------------------------------------------------------------------------+
                                      |
                                      v
LAYER 4 - EXTERNAL LIBRARIES   I/O, features and dense stereo only
+-----------------------------------------------------------------------------------+
|  NumPy 1.26      OpenCV 4.10      SciPy 1.13      Open3D 0.18     Pillow 10.3     |
|  arrays          SIFT, SGBM,      sparse solve,   normals,        EXIF read       |
|  linear algebra  rectification    least_squares   DBSCAN, Poisson                 |
|                                                                                   |
|  No OpenCV call estimates pose or structure. findFundamentalMat, recoverPose,     |
|  solvePnP, triangulatePoints and the OpenCV bundle adjuster are all unused.       |
+-----------------------------------------------------------------------------------+


-------------------------------------------------------------------------------------
DEPENDENCY TIERS      computed from the import graph; acyclic, depth 3
-------------------------------------------------------------------------------------

  TIER 0   15 modules, import nothing from sfm/
           map ............... the data model
           geometry/ ......... fundamental, triangulation, pnp
           io/ ............... calibration, camera_file, reference, ply
           features/ ......... matching
           ba/ ............... so3
           mvs/ .............. stereo, mesh, params
           eval/ ............. umeyama, metrics

  TIER 1   8 modules, import only tier 0
           io/ ............... generic, colorize, fountain
           features/ ......... tracks
           geometry/ ......... essential
           ba/ ............... jacobian, scipy_ba
           mvs/ .............. fuse

  TIER 2   3 modules, the deepest the graph goes
           recon/ ............ incremental (479 lines), two_view
           ba/ ............... lm

  A module imports only from a strictly lower tier. No cycles anywhere.


-------------------------------------------------------------------------------------
INVARIANTS
-------------------------------------------------------------------------------------

  1  sfm/map.py imports nothing and is imported by 7 modules. The data model is
     the one thing every stage agrees on.

  2  sfm/geometry/ imports nothing but NumPy. Every estimator is ours.

  3  sfm/mvs/ and sfm/eval/ import nothing from the rest of sfm/. Dense stereo
     and evaluation run from scene_cache.npz alone, so either can be re-run or
     replaced without touching the sparse reconstruction.

  4  recon/incremental.py never imports ba/lm.py. The solver arrives as the
     argument bundle_adjust_fn, which is what let us measure the hand-written
     LM against SciPy on identical input.

  5  Only scripts/ knows dataset paths. The library takes images and intrinsics.
```

A rendered version of this diagram is at [site/architecture.html](site/architecture.html).
