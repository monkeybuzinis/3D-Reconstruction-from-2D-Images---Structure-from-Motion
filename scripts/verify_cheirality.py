"""
Regression test for the cheirality check in
sfm/recon/incremental.py::triangulate_new_points().

The bug this guards against: depth was read straight off `xyz[2]`, the
WORLD Z coordinate, instead of transforming the point into each camera's
own frame first. World Z is camera depth only for the seed camera (the one
seed_two_view pins to R=I, t=0) -- and the triangulation pair here is
whichever two posed cameras come first in the track, which in general
includes neither camera 0 nor the seed camera.

So this builds a synthetic scene where the triangulating pair deliberately
EXCLUDES the seed camera, with two counterexamples the old code gets wrong
in opposite directions:

  Track A -- in front of both cameras, but at negative world Z.
             Valid point. Old code REJECTED it (false negative).
  Track B -- behind camera A, in front of camera B, positive world Z.
             Invalid point. Old code ACCEPTED it (false positive).

Plus two controls that both versions must agree on, so the test proves the
fix is specific rather than just stricter or looser overall:

  Track C -- pair includes the seed camera, point in front of both: accept.
  Track D -- behind both cameras of the pair: reject.

Geometry (all cameras share K; world +Z is "into" camera 0):
  cam 0  center (0, 0, 0)     looks +Z   <- seed camera, identity pose
  cam 1  center (-1, 0, 10)   looks -Z   (back toward the origin)
  cam 2  center (1, 0, 10)    looks -Z
  cam 3  center (2, 0, 12)    looks +Z   (away from the origin)

Run from the repo root:
    .venv/bin/python3 scripts/verify_cheirality.py
"""

from pathlib import Path
from typing import Dict, Tuple
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfm.features.tracks import Track
from sfm.geometry.triangulation import triangulate_dlt
from sfm.map import Camera, Map, Point3D
from sfm.recon.incremental import triangulate_new_points

K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
LOOK_BACK = np.diag([-1.0, 1.0, -1.0])  # 180 degrees about Y: a camera facing -Z

# (camera id, rotation, camera center). t is derived as -R @ center, i.e. the
# world-to-camera translation this project's Camera stores.
CAMERA_RIG = [
    (0, np.eye(3), np.array([0.0, 0.0, 0.0])),
    (1, LOOK_BACK, np.array([-1.0, 0.0, 10.0])),
    (2, LOOK_BACK, np.array([1.0, 0.0, 10.0])),
    (3, np.eye(3), np.array([2.0, 0.0, 12.0])),
]

# (track id, world point, (cam_a, cam_b), should_be_accepted, what it proves).
# cam_a is listed first and is therefore the one triangulate_new_points picks
# as `seen_in[0]` -- the camera whose depth the old code never transformed.
CASES = [
    (0, np.array([0.2, -0.1, -5.0]), (1, 2), True, "valid, negative world Z"),
    (1, np.array([0.3, 0.2, 20.0]), (1, 3), False, "behind cam_a, positive world Z"),
    (2, np.array([0.5, 0.3, 25.0]), (0, 3), True, "control: pair includes the seed camera"),
    (3, np.array([0.0, 0.0, 30.0]), (1, 2), False, "control: behind both cameras"),
]


def project(R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Pixel coordinates of X under (R, t). Also valid behind the camera, where z < 0."""
    cam = R @ X + t
    uv = K @ cam
    return uv[:2] / uv[2]


def depth_in(R: np.ndarray, t: np.ndarray, X: np.ndarray) -> float:
    return float((R @ X + t)[2])


def build_scene() -> Tuple[Map, Dict[int, np.ndarray], Dict[int, Track]]:
    """A Map with all four cameras posed, and one track per case in CASES."""
    scene = Map()
    poses = {}
    for cam_id, R, center in CAMERA_RIG:
        t = -R @ center
        poses[cam_id] = (R, t)
        scene.add_camera(
            Camera(id=cam_id, K=K.copy(), image_name=f"synthetic_{cam_id}.png", width=640, height=480, R=R, t=t)
        )

    # Every camera gets a keypoint slot per track, so track i's observation
    # index is simply i in whichever images see it.
    keypoints = {
        cam_id: np.array([project(*poses[cam_id], X) for _, X, _, _, _ in CASES])
        for cam_id, _, _ in CAMERA_RIG
    }

    # Insertion order matters: dict order is what decides cam_a vs cam_b.
    tracks = {
        track_id: Track(id=track_id, observations={cam_a: track_id, cam_b: track_id})
        for track_id, _, (cam_a, cam_b), _, _ in CASES
    }
    return scene, keypoints, tracks


def triangulate_new_points_old(scene: Map, keypoints, tracks_by_id) -> int:
    """
    The pre-fix implementation, kept here only so this test can show the two
    counterexamples really do flip. Identical to the current function except
    that depth_a is read off world Z instead of camera A's frame.
    """
    added = 0
    for track in tracks_by_id.values():
        if track.id in scene.points:
            continue
        seen_in = [cid for cid in track.observations if cid in scene.cameras]
        if len(seen_in) < 2:
            continue
        cam_a, cam_b = scene.cameras[seen_in[0]], scene.cameras[seen_in[1]]
        pt_a = keypoints[seen_in[0]][track.observations[seen_in[0]]]
        pt_b = keypoints[seen_in[1]][track.observations[seen_in[1]]]
        xyz = triangulate_dlt(
            cam_a.projection_matrix(), cam_b.projection_matrix(), pt_a[None, :], pt_b[None, :]
        )[0]
        depth_a = xyz[2]  # the bug: world Z, not depth in cam_a
        depth_b = (cam_b.R @ xyz + cam_b.t)[2]
        if depth_a <= 0 or depth_b <= 0:
            continue
        scene.add_point(Point3D(id=track.id, xyz=xyz))
        added += 1
    return added


def main() -> None:
    print("=" * 78)
    print("Cheirality check: triangulation pairs that exclude the seed camera")
    print("=" * 78)

    scene, keypoints, tracks = build_scene()
    poses = {cid: (scene.cameras[cid].R, scene.cameras[cid].t) for cid in scene.cameras}

    print("\nCase geometry (depth = Z in that camera's own frame):")
    for track_id, X, (cam_a, cam_b), expected, label in CASES:
        d_a = depth_in(*poses[cam_a], X)
        d_b = depth_in(*poses[cam_b], X)
        print(
            f"  track {track_id}: pair ({cam_a},{cam_b})  world Z {X[2]:+7.2f}  "
            f"depth in cam{cam_a} {d_a:+7.2f}  depth in cam{cam_b} {d_b:+7.2f}  "
            f"-> expect {'accept' if expected else 'reject':6s}  [{label}]"
        )

    # What the fixed implementation does.
    n_added = triangulate_new_points(scene, keypoints, tracks)
    got = {track_id: track_id in scene.points for track_id, *_ in CASES}

    # What the old implementation did, on an identical fresh scene.
    old_scene, old_keypoints, old_tracks = build_scene()
    triangulate_new_points_old(old_scene, old_keypoints, old_tracks)
    got_old = {track_id: track_id in old_scene.points for track_id, *_ in CASES}

    print(f"\nFixed implementation added {n_added} point(s).\n")
    header = f"  {'track':6s} {'expected':9s} {'fixed':9s} {'old':9s} {'verdict':8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    failures = []
    differed = []
    for track_id, X, pair, expected, label in CASES:
        ok = got[track_id] == expected
        if not ok:
            failures.append((track_id, label))
        if got_old[track_id] != expected:
            differed.append((track_id, label))
        print(
            f"  {track_id:<6d} {'accept' if expected else 'reject':9s} "
            f"{'accept' if got[track_id] else 'reject':9s} "
            f"{'accept' if got_old[track_id] else 'reject':9s} "
            f"{'PASS' if ok else 'FAIL':8s}"
        )

    # Accepted points must also be in the right PLACE, not merely accepted:
    # a depth-sign check says nothing about triangulation accuracy.
    print("\nTriangulated position error for accepted points:")
    for track_id, X, _, expected, _ in CASES:
        if not expected:
            continue
        err = float(np.linalg.norm(scene.points[track_id].xyz - X))
        print(f"  track {track_id}: |triangulated - truth| = {err:.2e}")
        if err > 1e-6:
            failures.append((track_id, f"triangulated position off by {err:.2e}"))

    print()
    if not differed:
        failures.append((-1, "no case distinguished old from fixed -- the test proves nothing"))
    else:
        cases = ", ".join(f"track {tid} ({label})" for tid, label in differed)
        print(f"Counterexamples that the old code got wrong: {cases}")

    if failures:
        print("\nFAILED:")
        for track_id, label in failures:
            print(f"  track {track_id}: {label}")
        raise SystemExit(1)

    print("\nAll cheirality cases pass, including both pairs that exclude the seed camera.")


if __name__ == "__main__":
    main()
