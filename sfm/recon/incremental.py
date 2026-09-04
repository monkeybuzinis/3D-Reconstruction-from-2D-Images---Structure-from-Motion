"""
Incremental SfM: track-based two-view seed, then PnP registration +
triangulation per camera. This is the orchestration layer -- it doesn't do
any of the actual math itself, it just calls sfm/geometry, sfm/features, and
sfm/ba in the right order and keeps the Map consistent as cameras are added
one at a time.

High-level algorithm (run_incremental_sfm, at the bottom, is the entry point):
  1. Seed from one image pair (Essential-matrix pose recovery).
  2. Triangulate whatever else that pair already supports.
  3. Repeat: pick the unregistered camera with the most known 3D points
     visible in it, solve its pose via PnP, triangulate any newly-possible
     points, then re-run Bundle Adjustment before moving on.
"""

from __future__ import annotations

import inspect
from typing import Callable, Dict, List, Tuple

import numpy as np

from sfm.ba.scipy_ba import bundle_adjust as scipy_bundle_adjust
from sfm.features.tracks import Track
from sfm.geometry.essential import cheirality_check, decompose_essential, essential_from_fundamental
from sfm.geometry.fundamental import ransac_fundamental_matrix
from sfm.geometry.pnp import ransac_pnp
from sfm.geometry.triangulation import triangulate_dlt
from sfm.map import Camera, Map, Observation, Point3D


def _add_camera_stub(scene: Map, images: Dict[int, np.ndarray], K: np.ndarray, camera_id: int, R: np.ndarray, t: np.ndarray) -> None:
    """Shared helper: build a posed Camera for `camera_id` and add it to the Map."""
    h, w = images[camera_id].shape[:2]
    scene.add_camera(
        Camera(id=camera_id, K=K.copy(), image_name=f"{camera_id:04d}.png", width=w, height=h, R=R, t=t)
    )


def seed_two_view(
    scene: Map,
    images: Dict[int, np.ndarray],
    K: np.ndarray,
    keypoints: Dict[int, np.ndarray],
    tracks: List[Track],
    cam0: int,
    cam1: int,
    ransac_threshold: float = 1.5,
    ransac_iters: int = 3000,
) -> int:
    """
    Seed the Map from tracks visible in both cam0 and cam1 -- the same
    two-view pose recovery as sfm/recon/two_view.py, but reading its
    correspondences from the pre-built multi-view tracks (sfm/features/tracks.py)
    instead of re-matching cam0/cam1 in isolation.

    Point IDs are set to the originating track's ID, so later stages can
    look up "is this track already triangulated?" via
    `track.id in scene.points` -- track identity IS 3D point identity from
    here on.

    Returns the number of points added.
    """
    seed_tracks = [t for t in tracks if cam0 in t.observations and cam1 in t.observations]
    if len(seed_tracks) < 8:
        raise RuntimeError(f"Only {len(seed_tracks)} shared tracks between cameras {cam0}/{cam1}")

    pts0 = np.array([keypoints[cam0][t.observations[cam0]] for t in seed_tracks])
    pts1 = np.array([keypoints[cam1][t.observations[cam1]] for t in seed_tracks])

    F, inlier_idx = ransac_fundamental_matrix(pts0, pts1, threshold=ransac_threshold, max_iters=ransac_iters)
    E = essential_from_fundamental(F, K)
    candidates = decompose_essential(E)
    R, t, xyz = cheirality_check(K, candidates, pts0[inlier_idx], pts1[inlier_idx])

    depth0 = xyz[:, 2]
    depth1 = (R @ xyz.T + t.reshape(3, 1))[2, :]
    valid = (depth0 > 0) & (depth1 > 0)

    _add_camera_stub(scene, images, K, cam0, np.eye(3), np.zeros(3))
    _add_camera_stub(scene, images, K, cam1, R, t)

    n_added = 0
    for local_i in np.where(valid)[0]:
        track = seed_tracks[inlier_idx[local_i]]
        point_id = track.id
        scene.add_point(Point3D(id=point_id, xyz=xyz[local_i]))
        scene.add_observation(Observation(camera_id=cam0, point_id=point_id, uv=pts0[inlier_idx[local_i]]))
        scene.add_observation(Observation(camera_id=cam1, point_id=point_id, uv=pts1[inlier_idx[local_i]]))
        n_added += 1
    return n_added


def register_camera_pnp(
    scene: Map,
    images: Dict[int, np.ndarray],
    K: np.ndarray,
    keypoints: Dict[int, np.ndarray],
    tracks_by_id: Dict[int, Track],
    camera_id: int,
    ransac_threshold_px: float = 8.0,
    ransac_iters: int = 2000,
) -> int:
    """
    Register one new camera via RANSAC PnP against tracks that are already
    triangulated (`track.id in scene.points`) and visible in this camera --
    i.e. "3D points we already trust, and where they land in the new image".

    Returns the number of PnP inliers, or -1 if this camera couldn't be
    registered this round -- either too few correspondences to attempt PnP
    at all, or RANSAC ran out of iterations without finding a pose good
    enough. Neither is treated as a hard failure by the caller
    (run_incremental_sfm): with many images it's normal for one view's
    correspondences to be too weak or too degenerate the first time it's
    tried, since visibility isn't uniform across a whole capture arc. The
    caller retries a camera that failed here once triangulating other
    cameras adds new points to the scene, rather than giving up on it
    permanently after one attempt.
    """
    correspondences = [
        (track.id, keypoints[camera_id][track.observations[camera_id]])
        for track in tracks_by_id.values()
        if camera_id in track.observations and track.id in scene.points
    ]
    if len(correspondences) < 6:
        return -1

    point_ids = [c[0] for c in correspondences]
    pts2d = np.array([c[1] for c in correspondences])
    pts3d = np.array([scene.points[pid].xyz for pid in point_ids])

    try:
        R, t, inlier_idx = ransac_pnp(K, pts3d, pts2d, threshold_px=ransac_threshold_px, max_iters=ransac_iters)
    except RuntimeError:
        return -1

    _add_camera_stub(scene, images, K, camera_id, R, t)
    for i in inlier_idx:
        scene.add_observation(Observation(camera_id=camera_id, point_id=point_ids[i], uv=pts2d[i]))

    return len(inlier_idx)


def triangulate_new_points(
    scene: Map,
    keypoints: Dict[int, np.ndarray],
    tracks_by_id: Dict[int, Track],
) -> int:
    """
    For every track not yet triangulated but visible in >=2 already-posed
    cameras, DLT-triangulate from the first two such cameras and add
    observations from ALL posed cameras that see it (extra views don't
    change the initial triangulated position here, but they give the later
    Bundle Adjustment more residuals to refine that point against).

    "First two" is a simplification -- picking the pair with the widest
    baseline would triangulate more accurately -- but BA cleans up the
    resulting error anyway, so it isn't worth the extra bookkeeping here.
    """
    added = 0
    for track in tracks_by_id.values():
        if track.id in scene.points:
            continue
        seen_in = [cid for cid in track.observations if cid in scene.cameras]
        if len(seen_in) < 2:
            continue

        cam_a_id, cam_b_id = seen_in[0], seen_in[1]
        cam_a, cam_b = scene.cameras[cam_a_id], scene.cameras[cam_b_id]
        pt_a = keypoints[cam_a_id][track.observations[cam_a_id]]
        pt_b = keypoints[cam_b_id][track.observations[cam_b_id]]

        xyz = triangulate_dlt(
            cam_a.projection_matrix(), cam_b.projection_matrix(), pt_a[None, :], pt_b[None, :]
        )[0]

        depth_a = xyz[2]
        depth_b = (cam_b.R @ xyz + cam_b.t)[2]
        if depth_a <= 0 or depth_b <= 0:
            continue  # cheirality failure for this pair; skip rather than add a bogus point

        scene.add_point(Point3D(id=track.id, xyz=xyz))
        for cid in seen_in:
            scene.add_observation(
                Observation(camera_id=cid, point_id=track.id, uv=keypoints[cid][track.observations[cid]])
            )
        added += 1
    return added


def run_incremental_sfm(
    images: Dict[int, np.ndarray],
    K: np.ndarray,
    keypoints: Dict[int, np.ndarray],
    tracks: List[Track],
    seed: Tuple[int, int] = (0, 1),
    ba_max_nfev: int = 50,
    bundle_adjust_fn: Callable[..., dict] = scipy_bundle_adjust,
) -> Tuple[Map, List[Tuple[int, int, int]]]:
    """
    Full incremental pipeline: seed from `seed`, then repeatedly register the
    remaining camera with the most 2D-3D correspondences via PnP (the
    standard "next-best-view" heuristic -- more correspondences means a
    better-conditioned PnP solve), triangulate any tracks newly spanned by
    >=2 posed cameras, and re-run Bundle Adjustment on the whole scene so far.

    The BA step after every camera matters: without it, triangulation error
    from early (often small-baseline) points compounds silently and later
    PnP calls -- which depend on those points being roughly right -- start
    failing outright once the drift gets large enough. (Discovered the hard
    way: running BA only once at the end let camera 8-10 registration fail
    completely on this dataset.)

    bundle_adjust_fn defaults to the scipy-backed optimizer
    (sfm.ba.scipy_ba.bundle_adjust) but accepts any callable with the same
    (scene, fixed_camera_ids=..., ...) -> dict contract -- in particular
    sfm.ba.lm.hand_bundle_adjust, the hand-written Levenberg-Marquardt
    solver, which is both faster and equally accurate on this dataset (see
    scripts/verify_hand_ba.py). ba_max_nfev is passed through as max_nfev to
    scipy's optimizer, or as max_iters to the hand-written one -- both
    accept it as their iteration-count knob.

    Returns (scene, log) where log is [(camera_id, n_correspondences_available,
    n_pnp_inliers), ...] in registration order.
    """
    # scipy's optimizer takes its iteration cap as max_nfev; the hand-written
    # one takes it as max_iters. Figure out which once, up front.
    iter_param = "max_iters" if "max_iters" in inspect.signature(bundle_adjust_fn).parameters else "max_nfev"

    def run_ba(s: Map) -> None:
        # seed[0] stays fixed at the world origin for every BA call: this is
        # the "gauge fix" that removes the 6-DOF (rotation+translation)
        # ambiguity BA would otherwise have (reprojection error is unchanged
        # by rigidly transforming the whole scene, so without a fixed
        # reference frame the optimizer has nothing to anchor to).
        bundle_adjust_fn(s, fixed_camera_ids=(seed[0],), **{iter_param: ba_max_nfev})

    scene = Map()
    tracks_by_id = {t.id: t for t in tracks}

    seed_two_view(scene, images, K, keypoints, tracks, seed[0], seed[1])
    triangulate_new_points(scene, keypoints, tracks_by_id)
    run_ba(scene)

    remaining = set(images) - set(scene.cameras)
    failed: set = set()  # PnP failed for these on their last attempt -- see note below
    log: List[Tuple[int, int, int]] = []

    while remaining - failed:
        # Next-best-view: register whichever still-viable camera currently
        # has the most already-triangulated points visible in it.
        best_cid, best_count = None, -1
        for cid in remaining - failed:
            count = sum(
                1
                for t in tracks_by_id.values()
                if cid in t.observations and t.id in scene.points
            )
            if count > best_count:
                best_cid, best_count = cid, count

        if best_cid is None or best_count < 6:
            break  # nothing left has enough correspondences for PnP

        n_inliers = register_camera_pnp(scene, images, K, keypoints, tracks_by_id, best_cid)
        if n_inliers < 0:
            # Not a permanent failure: with many images it's normal for a
            # camera's correspondences to be too weak/degenerate the first
            # time it's tried (visibility isn't uniform across a whole
            # capture arc). Park it in `failed` rather than dropping it from
            # `remaining` -- if a later camera's triangulation adds NEW
            # points, `failed` gets cleared below and this one gets another
            # shot with a richer point set. Termination is still guaranteed:
            # each pass either registers a camera (shrinking `remaining`) or
            # grows `failed` by one, and the loop condition `remaining -
            # failed` empties out once every still-unregistered camera has
            # failed at least once since the last new points appeared.
            failed.add(best_cid)
            continue

        remaining.discard(best_cid)
        log.append((best_cid, best_count, n_inliers))
        n_new_points = triangulate_new_points(scene, keypoints, tracks_by_id)
        if n_new_points > 0:
            failed.clear()
        run_ba(scene)

    return scene, log


def filter_outlier_observations(scene: Map, max_error_px: float = 4.0) -> Map:
    """
    Drop observations whose reprojection error exceeds max_error_px, and any
    point left with fewer than 2 observations afterwards.

    A few catastrophic outlier tracks (bad matches chained into a track that
    shouldn't exist) are normal in incremental SfM. Because BA minimizes
    SQUARED reprojection error, even one 600px-error point can dominate the
    total cost and make the optimizer's progress look stalled when it's
    really just being dragged around by that one bad point. This is the
    standard cleanup pass run between BA rounds to fix that.

    Returns a new Map; `scene` is left untouched.
    """
    cleaned = Map()
    for cam in scene.cameras.values():
        cleaned.add_camera(
            Camera(
                id=cam.id,
                K=cam.K.copy(),
                image_name=cam.image_name,
                width=cam.width,
                height=cam.height,
                R=cam.R.copy(),
                t=cam.t.copy(),
            )
        )

    kept_obs_by_point: Dict[int, List[Observation]] = {}
    for obs in scene.observations:
        cam = scene.cameras[obs.camera_id]
        pt = scene.points[obs.point_id]
        proj = cam.projection_matrix() @ np.append(pt.xyz, 1.0)
        uv = proj[:2] / proj[2]
        if np.linalg.norm(uv - obs.uv) <= max_error_px:
            kept_obs_by_point.setdefault(obs.point_id, []).append(obs)

    for point_id, obs_list in kept_obs_by_point.items():
        if len(obs_list) < 2:
            continue  # a point needs >=2 views to be triangulated at all
        cleaned.add_point(Point3D(id=point_id, xyz=scene.points[point_id].xyz.copy()))
        for obs in obs_list:
            cleaned.add_observation(Observation(camera_id=obs.camera_id, point_id=point_id, uv=obs.uv.copy()))

    return cleaned
