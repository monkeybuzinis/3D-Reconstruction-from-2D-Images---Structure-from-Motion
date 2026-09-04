"""
Attach real photo colors to each reconstructed Point3D.

This is standard photogrammetry practice, not experimental: every
Observation already records exactly which pixel (u, v) in which source
image a point came from (that's what made triangulation possible in the
first place), so sampling that pixel's color is free information we already
have -- nothing about it is inferred or generated.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from sfm.map import Map


def colorize_points(scene: Map, images: Dict[int, np.ndarray]) -> None:
    """
    For each point, sample RGB from the pixel of its most accurate
    observation (smallest reprojection error) -- a close-to-perfect pixel
    location gets sampled rather than a noisy one. Mutates scene.points[*].color
    in place; images are OpenCV BGR arrays, converted to RGB here.
    """
    for point_id, point in scene.points.items():
        obs_list = scene.track(point_id)
        if not obs_list:
            continue

        best_obs = None
        best_err = None
        for obs in obs_list:
            cam = scene.cameras[obs.camera_id]
            proj = cam.projection_matrix() @ np.append(point.xyz, 1.0)
            uv = proj[:2] / proj[2]
            err = float(np.linalg.norm(uv - obs.uv))
            if best_err is None or err < best_err:
                best_err = err
                best_obs = obs

        img = images[best_obs.camera_id]
        h, w = img.shape[:2]
        u, v = best_obs.uv
        ui = int(round(np.clip(u, 0, w - 1)))
        vi = int(round(np.clip(v, 0, h - 1)))
        b, g, r = img[vi, ui]  # OpenCV loads BGR
        point.color = np.array([r, g, b], dtype=np.uint8)
