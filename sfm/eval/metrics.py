"""
Accuracy / Completeness / F-score against a reference point cloud -- the
standard MVS-benchmark protocol (Tanks & Temples, ETH3D, and the original
Strecha benchmark this dataset comes from all use this same definition).

Nearest-neighbor search itself uses scipy's cKDTree: a solved, well-known
data structure, not the point of this capstone to re-derive (same "safe
library for an established sub-problem" role as OpenCV's SIFT/FLANN/SGBM
elsewhere in this project) -- what's specific to this evaluation is what
the distances mean and how they're combined into Accuracy/Completeness/F-score.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.spatial import cKDTree


def nearest_neighbor_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """For every point in `source`, its distance to the nearest point in `target`."""
    tree = cKDTree(target)
    dists, _ = tree.query(source, k=1)
    return dists


def evaluate_reconstruction(
    reconstructed: np.ndarray,
    reference: np.ndarray,
    thresholds: tuple = (0.01, 0.02, 0.05, 0.1),
) -> Dict[str, float]:
    """
    Both point sets must already be in the SAME frame (i.e. `reconstructed`
    has already been Umeyama-aligned into the reference frame -- see
    sfm/eval/umeyama.py).

    - Accuracy: how close our points are to the reference surface (per-point
      distance from reconstructed -> nearest reference point). Low accuracy
      distance = our points sit right on the true surface.
    - Completeness: how much of the reference surface we captured (per-point
      distance from reference -> nearest reconstructed point). Low
      completeness distance = little of the true surface was missed.
    - Precision(tau) / Recall(tau) / F-score(tau): the same two distance
      sets, thresholded -- Precision is the fraction of our points within
      tau of the reference surface, Recall is the fraction of the reference
      surface within tau of our points, F-score is their harmonic mean.
    """
    acc_dists = nearest_neighbor_distances(reconstructed, reference)
    comp_dists = nearest_neighbor_distances(reference, reconstructed)

    result: Dict[str, float] = {
        "accuracy_mean": float(acc_dists.mean()),
        "accuracy_median": float(np.median(acc_dists)),
        "completeness_mean": float(comp_dists.mean()),
        "completeness_median": float(np.median(comp_dists)),
    }

    for tau in thresholds:
        precision = float(np.mean(acc_dists < tau))
        recall = float(np.mean(comp_dists < tau))
        f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        result[f"precision@{tau}"] = precision
        result[f"recall@{tau}"] = recall
        result[f"f_score@{tau}"] = f_score

    return result
