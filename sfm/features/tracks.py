"""
Multi-view feature tracks: chain pairwise 2-view matches into "this same
physical point was seen in images 0, 3, 5, and 9" tracks spanning many
images. Two-view matching only tells you about one pair at a time; a track
is what lets incremental SfM ask "does the next camera see anything we've
already triangulated?" and is also what a triangulated Point3D's `id` is set
to (see sfm/recon/incremental.py) -- track identity IS 3D point identity
here.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple

import cv2
import numpy as np

from sfm.geometry.fundamental import ransac_fundamental_matrix


@dataclass
class Track:
    """A set of same-point observations: image_id -> which keypoint index in that image is this point."""

    id: int
    observations: Dict[int, int]  # image_id -> keypoint index into that image's keypoint array


class _UnionFind:
    """
    Classic union-find (disjoint-set): starts with every (image, keypoint)
    as its own singleton set, then `union` merges two sets whenever a
    verified match says they're the same physical point. After all pairs
    are processed, everything still connected to the same root is one track
    -- this is what turns a pile of pairwise matches into multi-view chains
    without ever explicitly walking "0-1, 1-3, 3-9 must be the same point".
    """

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def extract_all_keypoints(
    images: Dict[int, np.ndarray]
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    SIFT keypoints (pixel coords) and descriptors, computed once per image
    and reused for every pair below -- unlike sfm/features/matching.py's
    sift_match(), which is convenient for a single pair but would redo SIFT
    detection redundantly if called once per pair here (each image appears
    in 10 pairs out of 11 images).
    """
    sift = cv2.SIFT_create()
    keypoints: Dict[int, np.ndarray] = {}
    descriptors: Dict[int, np.ndarray] = {}
    for image_id, img in images.items():
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = sift.detectAndCompute(gray, None)
        if desc is None:
            raise RuntimeError(f"SIFT produced no descriptors for image {image_id}")
        keypoints[image_id] = np.array([k.pt for k in kp], dtype=np.float64)
        descriptors[image_id] = desc
    return keypoints, descriptors


def _flann_match_descriptors(desc0: np.ndarray, desc1: np.ndarray, ratio: float = 0.75) -> np.ndarray:
    """FLANN KNN + Lowe's ratio test. Returns (M, 2) array of (idx0, idx1) index pairs."""
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=64)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn = flann.knnMatch(desc0, desc1, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append((m.queryIdx, m.trainIdx))
    return np.array(good, dtype=int).reshape(-1, 2)


def build_tracks(
    images: Dict[int, np.ndarray],
    ratio: float = 0.75,
    ransac_threshold: float = 1.5,
    ransac_iters: int = 1000,
    min_track_len: int = 2,
) -> Tuple[Dict[int, np.ndarray], List[Track]]:
    """
    Match every image pair, geometrically verify with RANSAC F (raw
    ratio-test matches still contain outliers; a track built from those
    would chain wrong correspondences together), and chain the verified
    2-view matches into multi-view tracks via union-find.

    All-pairs (55 pairs for 11 images) is the simple, "get it working"
    choice -- fine at this dataset's scale, but doesn't scale to hundreds of
    images without smarter pair selection (vocabulary tree, sequential-only,
    etc).

    Returns (keypoints, tracks): keypoints[image_id] is the (N, 2) pixel
    array SIFT found for that image; tracks[i].observations maps
    image_id -> keypoint index into that array. A track with two entries
    from the same image (from a bad chain merge, e.g. two different real
    points accidentally getting unioned together through a bad match
    somewhere in the chain) has that image's entry dropped rather than
    discarding the whole track.
    """
    keypoints, descriptors = extract_all_keypoints(images)
    image_ids = sorted(images)

    # Union-find needs one flat integer per (image, keypoint); `offsets`
    # is where each image's keypoints start in that flat numbering.
    offsets: Dict[int, int] = {}
    total = 0
    for image_id in image_ids:
        offsets[image_id] = total
        total += len(keypoints[image_id])
    uf = _UnionFind(total)

    def node(image_id: int, kp_idx: int) -> int:
        return offsets[image_id] + kp_idx

    for i, j in combinations(image_ids, 2):
        matches = _flann_match_descriptors(descriptors[i], descriptors[j], ratio=ratio)
        if len(matches) < 8:
            continue
        idx_i, idx_j = matches[:, 0], matches[:, 1]
        pts_i = keypoints[i][idx_i]
        pts_j = keypoints[j][idx_j]

        try:
            _F, inlier_idx = ransac_fundamental_matrix(
                pts_i, pts_j, threshold=ransac_threshold, max_iters=ransac_iters
            )
        except RuntimeError:
            continue  # this pair never agreed on a consistent F; skip it rather than chain garbage
        if len(inlier_idx) < 8:
            continue

        for k in inlier_idx:
            uf.union(node(i, idx_i[k]), node(j, idx_j[k]))

    # Walk every (image, keypoint) node, group by union-find root, and turn
    # each group into a Track -- dropping any image that appears twice
    # within one group (a conflicting merge) rather than the whole group.
    node_to_image_kp = [
        (image_id, kp_idx) for image_id in image_ids for kp_idx in range(len(keypoints[image_id]))
    ]

    groups: Dict[int, Dict[int, int]] = {}
    for n_id, (image_id, kp_idx) in enumerate(node_to_image_kp):
        root = uf.find(n_id)
        obs = groups.setdefault(root, {})
        if image_id in obs:
            obs[image_id] = -1  # conflicting merge; that image's entry gets dropped below
        else:
            obs[image_id] = kp_idx

    tracks: List[Track] = []
    for obs in groups.values():
        obs_clean = {img: idx for img, idx in obs.items() if idx >= 0}
        if len(obs_clean) >= min_track_len:
            tracks.append(Track(id=len(tracks), observations=obs_clean))

    return keypoints, tracks
