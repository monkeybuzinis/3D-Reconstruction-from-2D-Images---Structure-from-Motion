"""
2D feature matching between one image pair. This is the only place OpenCV's
detector/matcher gets used -- everything after this (Fundamental/Essential
estimation, triangulation, PnP) is hand-written, working only off the pixel
coordinates this hands back.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def sift_match(
    img0: np.ndarray, img1: np.ndarray, ratio: float = 0.75
) -> Tuple[np.ndarray, np.ndarray, List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch]]:
    """
    OpenCV SIFT + FLANN + Lowe's ratio test.

    SIFT finds distinctive local patches ("keypoints") in each image and a
    128-d descriptor per patch. FLANN then finds each img0 descriptor's two
    nearest neighbors in img1's descriptors. Lowe's ratio test keeps a match
    only if the best neighbor is convincingly closer than the second-best
    (match.distance < ratio * second_best.distance) -- this is what filters
    out ambiguous matches (e.g. repeated texture) before any geometry is
    involved.

    Returns pixel correspondences (pts0, pts1) plus the raw keypoint/match
    objects, in case a caller wants to draw them.
    """
    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp0, desc0 = sift.detectAndCompute(gray0, None)
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    if desc0 is None or desc1 is None:
        raise RuntimeError("SIFT produced no descriptors")

    index_params = dict(algorithm=1, trees=5)  # FLANN KD-tree
    search_params = dict(checks=64)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn = flann.knnMatch(desc0, desc1, k=2)  # k=2: need best + second-best for the ratio test

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < 8:
        # 8 correspondences is the minimum the (uncalibrated) 8-point algorithm needs downstream.
        raise RuntimeError(f"Too few matches after ratio test: {len(good)}")

    pts0 = np.array([kp0[m.queryIdx].pt for m in good], dtype=np.float64)
    pts1 = np.array([kp1[m.trainIdx].pt for m in good], dtype=np.float64)
    return pts0, pts1, kp0, kp1, good
