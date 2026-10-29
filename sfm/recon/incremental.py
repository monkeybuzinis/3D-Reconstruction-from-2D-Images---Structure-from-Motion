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
    rng_seed: int = 42,
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

    F, inlier_idx = ransac_fundamental_matrix(pts0, pts1, threshold=ransac_threshold,
                                              max_iters=ransac_iters, seed=rng_seed)
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
    rng_seed: int = 0,
    min_inliers: int = 30,
    min_inlier_ratio: float = 0.25,
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
        R, t, inlier_idx = ransac_pnp(K, pts3d, pts2d, threshold_px=ransac_threshold_px,
                                      max_iters=ransac_iters, seed=rng_seed)
    except RuntimeError:
        return -1

    # RANSAC always returns its best pose, however weak the support for it is.
    # Accepting that unconditionally lets a camera join on a handful of
    # inliers, and a wrongly placed camera is worse than a missing one: the
    # next bundle adjustment fits the rest of the scene to it, and every
    # camera resectioned afterwards is matched against the points it moved.
    # On child-statue two cameras registered on inlier ratios of 0.06 and
    # 0.12 while every correctly placed camera scored 0.44 or above.
    # The thresholds are COLMAP's published mapper defaults
    # (abs_pose_min_num_inliers, abs_pose_min_inlier_ratio) rather than
    # values chosen against the datasets used here.
    ratio = len(inlier_idx) / len(correspondences)
    if len(inlier_idx) < min_inliers or ratio < min_inlier_ratio:
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

        # Cheirality: the point must be in FRONT of both cameras. Depth has to
        # be measured in each camera's own frame -- `xyz` is in world
        # coordinates, and world Z is only camera depth for the seed camera
        # (the one seed_two_view pins to identity). cam_a here is just
        # whichever posed camera came first in the track, so transforming it
        # is not optional: skipping the transform both accepts points behind
        # cam_a and rejects valid points whose world Z happens to be negative.
        depth_a = (cam_a.R @ xyz + cam_a.t)[2]
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


def filter_far_points(scene: Map, max_depth_ratio: float = 10.0) -> Map:
    """
    Drop points lying far beyond the scene the cameras were actually looking at.

    A point triangulated from near-parallel rays has essentially unconstrained
    depth: it reprojects well from anywhere along the ray, so the reprojection
    filter keeps it, and it can land thousands of times further away than any
    real structure. Photographing a small object indoors produces these in
    bulk -- the subject is close while the walls and clutter behind it are far
    enough to carry no usable parallax.

    They are not harmless. The dense stage derives its depth window and voxel
    size from the point cloud, and any viewer frames the whole of it, so a
    handful of runaway points can hide the object completely. Measured on a
    shoe captured indoors: the cloud spanned 98,551,799 units before this
    filter and 647 after, with 83% of the points kept.

    Depth is measured per point as the median over the cameras that see it in
    front of them, and compared against the median over all points -- both
    robust, since the whole problem here is a long tail.

    Returns a new Map; `scene` is left untouched. The default ratio is
    deliberately generous: the benchmark datasets reach 1.2-3.0x their median
    depth, so this leaves them untouched. Object-centric captures benefit from
    something tighter, around 2-3.
    """
    cams = list(scene.cameras.values())
    point_depth: Dict[int, float] = {}
    for pid, pt in scene.points.items():
        depths = [float((c.R @ pt.xyz + c.t)[2]) for c in cams if c.has_pose()]
        in_front = [d for d in depths if d > 0]
        # No camera sees it in front: it cannot be real geometry.
        point_depth[pid] = float(np.median(in_front)) if in_front else np.inf

    finite = [d for d in point_depth.values() if np.isfinite(d)]
    if not finite:
        return scene
    threshold = float(np.median(finite)) * max_depth_ratio

    keep = {pid for pid, d in point_depth.items() if d <= threshold}
    if len(keep) == len(scene.points):
        return scene

    cleaned = Map()
    for cam in scene.cameras.values():
        cleaned.add_camera(
            Camera(id=cam.id, K=cam.K.copy(), image_name=cam.image_name, width=cam.width,
                   height=cam.height, R=None if cam.R is None else cam.R.copy(),
                   t=None if cam.t is None else cam.t.copy())
        )
    for pid in keep:
        pt = scene.points[pid]
        cleaned.add_point(Point3D(id=pid, xyz=pt.xyz.copy(),
                                  color=None if pt.color is None else pt.color.copy()))
    for obs in scene.observations:
        if obs.point_id in keep:
            cleaned.add_observation(
                Observation(camera_id=obs.camera_id, point_id=obs.point_id, uv=obs.uv.copy())
            )
    return cleaned


def filter_centred_points(
    scene: Map,
    centre_frac: float = 0.28,
    min_views: int = 5,
    min_fraction: float = 0.7,
) -> Map:
    """
    Keep only the structure the photographer kept in frame: the subject.

    When someone walks around an object they hold it near the centre of every
    shot while the surroundings drift through the edges. That framing is a
    reliable signal about which points are the subject, and it survives cases
    where depth cannot separate them -- photographing a boot in a garden puts
    the boot and the fence behind it at nearly the same distance, so no depth
    threshold can tell them apart, but only the boot stays centred.

    A point is kept when it projects inside the central `centre_frac` of the
    frame (as a fraction of width/height from the centre) in at least
    `min_fraction` of the cameras that see it at all, having been seen by at
    least `min_views`. Measured on a boot shot in a garden: 10,033 points
    including fence and ground reduced to 3,975 forming just the boot.

    This is the automatic equivalent of the reconstruction bounding box that
    photogrammetry packages ask the user to drag around the subject by hand.
    Returns a new Map; `scene` is left untouched.
    """
    cams = [c for c in scene.cameras.values() if c.has_pose()]
    if not cams:
        return scene

    pids = sorted(scene.points)
    if not pids:
        return scene
    xyz = np.stack([scene.points[pid].xyz for pid in pids])
    seen = np.zeros(len(pids), dtype=int)
    centred = np.zeros(len(pids), dtype=int)

    for cam in cams:
        p_cam = (cam.R @ xyz.T + cam.t.reshape(3, 1)).T
        in_front = p_cam[:, 2] > 1e-9
        z = np.where(in_front, p_cam[:, 2], 1.0)
        uv = (cam.K @ (p_cam / z[:, None]).T).T[:, :2]
        in_frame = in_front & (uv[:, 0] >= 0) & (uv[:, 0] < cam.width) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
        inner = (
            in_frame
            & (np.abs(uv[:, 0] - cam.width / 2) < cam.width * centre_frac)
            & (np.abs(uv[:, 1] - cam.height / 2) < cam.height * centre_frac)
        )
        seen += in_frame
        centred += inner

    frac = np.divide(centred, np.maximum(seen, 1))
    keep = {pid for pid, s, f in zip(pids, seen, frac) if s >= min_views and f >= min_fraction}
    if not keep or len(keep) == len(scene.points):
        return scene

    cleaned = Map()
    for cam in scene.cameras.values():
        cleaned.add_camera(
            Camera(id=cam.id, K=cam.K.copy(), image_name=cam.image_name, width=cam.width,
                   height=cam.height, R=None if cam.R is None else cam.R.copy(),
                   t=None if cam.t is None else cam.t.copy())
        )
    for pid in keep:
        pt = scene.points[pid]
        cleaned.add_point(Point3D(id=pid, xyz=pt.xyz.copy(),
                                  color=None if pt.color is None else pt.color.copy()))
    for obs in scene.observations:
        if obs.point_id in keep:
            cleaned.add_observation(
                Observation(camera_id=obs.camera_id, point_id=obs.point_id, uv=obs.uv.copy())
            )
    return cleaned


def run_incremental_sfm(
    images: Dict[int, np.ndarray],
    K: np.ndarray,
    keypoints: Dict[int, np.ndarray],
    tracks: List[Track],
    seed: Tuple[int, int] = (0, 1),
    ba_max_nfev: int = 50,
    bundle_adjust_fn: Callable[..., dict] = scipy_bundle_adjust,
    rng_seed: int = 0,
    min_inliers: int = 30,
    min_inlier_ratio: float = 0.25,
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
    ba_params = inspect.signature(bundle_adjust_fn).parameters
    iter_param = "max_iters" if "max_iters" in ba_params else "max_nfev"
    # Optimizers that understand the scale gauge get told about the seed pair;
    # the scipy-backed one does not take it, so it is omitted there.
    gauge_kw = {"gauge_pair": tuple(seed)} if "gauge_pair" in ba_params else {}

    def run_ba(s: Map) -> None:
        # seed[0] stays fixed at the world origin for every BA call: this is
        # the "gauge fix" that removes the 6-DOF (rotation+translation)
        # ambiguity BA would otherwise have (reprojection error is unchanged
        # by rigidly transforming the whole scene, so without a fixed
        # reference frame the optimizer has nothing to anchor to). Scale is
        # the 7th degree of freedom and a fixed camera does not pin it, so
        # gauge_pair holds the seed baseline at one unit as well.
        bundle_adjust_fn(s, fixed_camera_ids=(seed[0],), **gauge_kw, **{iter_param: ba_max_nfev})

    scene = Map()
    tracks_by_id = {t.id: t for t in tracks}

    seed_two_view(scene, images, K, keypoints, tracks, seed[0], seed[1], rng_seed=rng_seed)
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

        n_inliers = register_camera_pnp(scene, images, K, keypoints, tracks_by_id, best_cid,
                                        rng_seed=rng_seed, min_inliers=min_inliers,
                                        min_inlier_ratio=min_inlier_ratio)
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
