"""
Pairwise dense stereo, using camera poses already recovered by SfM.

This is where OpenCV is used as the "safe library" for the one narrow,
well-established sub-problem (per-pixel disparity from a rectified pair) --
exactly the same role SIFT/FLANN play in sfm/features/matching.py. The
things that are specific to THIS reconstruction -- turning our recovered
(R, t) into a rectification, and turning disparities back into world-frame
3D points -- are ours.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def relative_pose(R_i: np.ndarray, t_i: np.ndarray, R_j: np.ndarray, t_j: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pose of camera j relative to camera i: R_rel, t_rel such that
    x_j = R_rel @ x_i + t_rel, given x_i = R_i @ X + t_i and x_j = R_j @ X + t_j.
    """
    R_rel = R_j @ R_i.T
    t_rel = t_j - R_rel @ t_i
    return R_rel, t_rel


def dense_stereo_pair(
    img_i: np.ndarray,
    img_j: np.ndarray,
    K: np.ndarray,
    R_i: np.ndarray,
    t_i: np.ndarray,
    R_j: np.ndarray,
    t_j: np.ndarray,
    min_depth: float,
    max_depth: float,
    num_disparities: int = 512,
    block_size: int = 7,
    downscale: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dense stereo between an already-posed camera pair.

    Steps: rectify both images using the known relative pose (no guessing --
    this is the exact geometry SfM already solved for), run SGBM for a dense
    disparity map, then map each valid disparity back through the rectified
    camera model AND the original camera i pose to get a world-frame 3D
    point + its color.

    downscale: work on a 1/downscale-resolution copy of the images (faster,
    smaller disparity search range for the same physical depth range) --
    the returned 3D points are still in the same world units as the sparse
    reconstruction, just sparser in image-space sampling.

    Returns (points_world (N,3), colors (N,3) uint8, disparity map (H,W) for diagnostics).
    """
    h, w = img_i.shape[:2]
    if downscale > 1:
        img_i = cv2.resize(img_i, (w // downscale, h // downscale), interpolation=cv2.INTER_AREA)
        img_j = cv2.resize(img_j, (w // downscale, h // downscale), interpolation=cv2.INTER_AREA)
        K = K.copy()
        K[0, 0] /= downscale; K[1, 1] /= downscale
        K[0, 2] /= downscale; K[1, 2] /= downscale
        h, w = img_i.shape[:2]

    R_rel, t_rel = relative_pose(R_i, t_i, R_j, t_j)
    R_rel = np.ascontiguousarray(R_rel, dtype=np.float64)
    t_rel = np.ascontiguousarray(t_rel, dtype=np.float64).reshape(3, 1)
    K = np.ascontiguousarray(K, dtype=np.float64)

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K, None, K, None, (w, h), R_rel, t_rel, flags=0, alpha=0
    )

    map1x, map1y = cv2.initUndistortRectifyMap(K, None, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K, None, R2, P2, (w, h), cv2.CV_32FC1)
    rect_i = cv2.remap(img_i, map1x, map1y, cv2.INTER_LINEAR)
    rect_j = cv2.remap(img_j, map2x, map2y, cv2.INTER_LINEAR)

    gray_i = cv2.cvtColor(rect_i, cv2.COLOR_BGR2GRAY)
    gray_j = cv2.cvtColor(rect_j, cv2.COLOR_BGR2GRAY)

    num_disparities = (num_disparities // 16) * 16  # SGBM requires a multiple of 16
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    # A left-right consistency check (via WLS's right matcher) is what
    # actually kills the textureless-region noise (sky, blank walls): a
    # false match essentially never agrees when matched in both directions,
    # while a true match does. This is the single biggest quality lever for
    # classical block-matching stereo on wide-baseline, real-world photos.
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    disp_left = left_matcher.compute(gray_i, gray_j)
    disp_right = right_matcher.compute(gray_j, gray_i)

    wls = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls.setLambda(8000.0)
    wls.setSigmaColor(1.5)
    disparity = wls.filter(disp_left, gray_i, disparity_map_right=disp_right).astype(np.float32) / 16.0

    points_rect = cv2.reprojectImageTo3D(disparity, Q)  # in the RECTIFIED camera-i frame

    confidence = wls.getConfidenceMap()
    valid = (disparity > 0.5) & (confidence > 150.0)  # WLS confidence: low = unreliable match, discard

    # Depth sanity filter: reject anything wildly outside the sparse
    # reconstruction's known depth range (bad matches, sky/background, etc).
    depth_rect = points_rect[:, :, 2]
    valid &= np.isfinite(depth_rect) & (depth_rect > min_depth * 0.5) & (depth_rect < max_depth * 1.5)

    pts_rect = points_rect[valid]  # (M, 3), in rectified-camera-i frame
    colors = cv2.cvtColor(rect_i, cv2.COLOR_BGR2RGB)[valid]  # sample color from the same rectified image

    # Undo rectification's extra rotation R1, then undo camera i's own pose,
    # to land back in the shared world frame every other point is in.
    pts_cam_i = pts_rect @ R1  # R1 is orthogonal: x_cam_i = R1^T @ x_rect = x_rect @ R1
    pts_world = (pts_cam_i - t_i) @ R_i  # X = R_i^T @ (x_cam_i - t_i) = (x_cam_i - t_i) @ R_i

    return pts_world, colors, disparity
