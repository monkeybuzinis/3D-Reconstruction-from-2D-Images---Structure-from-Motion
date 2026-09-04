"""
Load the reference camera poses + point cloud from Fountain/solution.graph
(a g2o-format sparse bundle adjustment solve, NOT the true Strecha laser-scan
ground truth -- that dataset's hosting service is no longer reachable, see
PLAN.md). This is the reference used for the evaluation stage: honest about
being "someone else's independent SfM solve," not laser-measured geometry.

solution.graph explicitly tags each line's type (VERTEX_CAM / VERTEX_XYZ),
unlike solution.txt, which just concatenates numbers with no labels --
that's why this file parses solution.graph directly rather than trying to
carefully re-guess solution.txt's ambiguous header.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def _quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def load_reference_camera_centers(fountain_dir: Path) -> Dict[int, np.ndarray]:
    """
    Reference camera centers, keyed by image index (0..10) to match this
    project's own camera ids.

    solution.graph's VERTEX_CAM vertex ids (0, 1, 675, 9504, ...) are NOT
    image indices -- they're g2o's incremental-insertion order, which
    matches Fountain/order.txt's processing sequence (0004, 0009, 0008, ...).
    Confirmed empirically: remapping through order.txt makes the camera
    centers' X coordinate vary smoothly and monotonically along the arc,
    exactly as expected for a sequential capture -- the wrong mapping
    produces no such pattern.

    VERTEX_CAM's (tx, ty, tz) is the camera CENTER directly (not a
    world-to-camera translation) -- also confirmed empirically the same way.
    """
    graph_path = fountain_dir / "solution.graph"
    order_path = fountain_dir / "order.txt"
    if not graph_path.is_file():
        raise FileNotFoundError(f"No {graph_path}")
    if not order_path.is_file():
        raise FileNotFoundError(f"No {order_path}")

    vertex_id_to_center: Dict[int, np.ndarray] = {}
    with graph_path.open() as f:
        for line in f:
            if not line.startswith("VERTEX_CAM"):
                continue
            parts = line.split()
            vid = int(parts[1])
            tx, ty, tz = (float(v) for v in parts[2:5])
            vertex_id_to_center[vid] = np.array([tx, ty, tz], dtype=np.float64)

    # order.txt lists image filenames in the order cameras were inserted,
    # which is the same order VERTEX_CAM ids increase in.
    insertion_order_names = [line.strip() for line in order_path.read_text().splitlines() if line.strip()]
    insertion_order_image_ids = [int(name.split(".")[0]) for name in insertion_order_names]

    sorted_vertex_ids = sorted(vertex_id_to_center)
    if len(sorted_vertex_ids) != len(insertion_order_image_ids):
        raise RuntimeError(
            f"VERTEX_CAM count ({len(sorted_vertex_ids)}) != order.txt entry count "
            f"({len(insertion_order_image_ids)})"
        )

    return {
        image_id: vertex_id_to_center[vid]
        for image_id, vid in zip(insertion_order_image_ids, sorted_vertex_ids)
    }


def load_reference_points(fountain_dir: Path) -> np.ndarray:
    """All VERTEX_XYZ points from solution.graph, as an (N, 3) array."""
    graph_path = fountain_dir / "solution.graph"
    if not graph_path.is_file():
        raise FileNotFoundError(f"No {graph_path}")

    points = []
    with graph_path.open() as f:
        for line in f:
            if not line.startswith("VERTEX_XYZ"):
                continue
            parts = line.split()
            points.append([float(v) for v in parts[2:5]])
    if not points:
        raise RuntimeError(f"No VERTEX_XYZ points parsed from {graph_path}")
    return np.array(points, dtype=np.float64)
