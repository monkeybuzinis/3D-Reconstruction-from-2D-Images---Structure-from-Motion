"""
Two-view seed: SIFT match -> RANSAC F -> E -> pose -> DLT triangulation -> Map.

Standalone entry point for reconstructing just a single image pair (used by
scripts/verify_two_view.py). The full 11-image pipeline in
sfm/recon/incremental.py does the same first step but via the track-based
seed_two_view() there instead, so that triangulated points share IDs with
the tracks used to register later cameras.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from sfm.features.matching import sift_match
from sfm.geometry.essential import cheirality_check, decompose_essential, essential_from_fundamental
from sfm.geometry.fundamental import ransac_fundamental_matrix
from sfm.map import Camera, Map, Observation, Point3D


def reconstruct_two_view(
    img0: np.ndarray,
    img1: np.ndarray,
    K: np.ndarray,
    name0: str = "0000.png",
    name1: str = "0001.png",
    ransac_threshold: float = 1.5,
    ransac_iters: int = 3000,
) -> Tuple[Map, np.ndarray]:
    """
    Run the full two-view seed on an image pair.

    Camera 0 is fixed at the world origin (this defines the reconstruction's
    world coordinate frame); camera 1's pose is the cheirality-checked
    solution recovered from the Essential matrix. Returns a Map with both
    cameras posed and one Point3D + two Observations per inlier that ends up
    in front of both cameras, plus the RANSAC inlier mask into the original
    SIFT matches (for diagnostics/plotting).
    """
    h, w = img0.shape[:2]

    # 1. Find raw pixel correspondences (still contains outliers).
    pts0, pts1, _kp0, _kp1, _matches = sift_match(img0, img1)

    # 2. Robustly estimate the epipolar geometry, discarding outliers.
    F, inlier_idx = ransac_fundamental_matrix(
        pts0, pts1, threshold=ransac_threshold, max_iters=ransac_iters
    )
    inlier_pts0, inlier_pts1 = pts0[inlier_idx], pts1[inlier_idx]

    # 3. Lift F to the calibrated Essential matrix and recover the one
    #    physically valid relative pose out of its 4 algebraic candidates.
    E = essential_from_fundamental(F, K)
    candidates = decompose_essential(E)
    R, t, xyz = cheirality_check(K, candidates, inlier_pts0, inlier_pts1)

    # cheirality_check triangulates ALL inliers under the winning pose, but
    # a few can still land behind a camera even then; drop just those.
    depth0 = xyz[:, 2]
    depth1 = (R @ xyz.T + t.reshape(3, 1))[2, :]
    valid = (depth0 > 0) & (depth1 > 0)

    scene = Map()
    scene.add_camera(
        Camera(id=0, K=K.copy(), image_name=name0, width=w, height=h, R=np.eye(3), t=np.zeros(3))
    )
    scene.add_camera(Camera(id=1, K=K.copy(), image_name=name1, width=w, height=h, R=R, t=t))

    for i in np.where(valid)[0]:
        point_id = scene.next_point_id()
        scene.add_point(Point3D(id=point_id, xyz=xyz[i]))
        scene.add_observation(Observation(camera_id=0, point_id=point_id, uv=inlier_pts0[i]))
        scene.add_observation(Observation(camera_id=1, point_id=point_id, uv=inlier_pts1[i]))

    return scene, inlier_idx
