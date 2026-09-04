"""
Reconstruction state: cameras, 3D points, and 2D-3D observation tracks.

This is the "database" every other module reads from and writes into. Feature
matching fills in Observations, geometry (two-view / PnP) fills in Camera
poses, triangulation fills in Point3D, and Bundle Adjustment reads/writes
everything at once. Keeping this one shared data structure is what lets those
independent pieces stay decoupled from each other.

Pose convention (Hartley-Zisserman / OpenCV):
    x_cam = R @ X_world + t
so the camera center in world coordinates is C = -R.T @ t.
Poses are optional in week 1; they are filled in during two-view / PnP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Camera:
    """One image's intrinsics (K) and, once solved for, its pose (R, t)."""

    id: int
    K: np.ndarray  # (3, 3) intrinsic matrix: focal length + principal point
    image_name: str
    width: int
    height: int
    R: Optional[np.ndarray] = None  # (3, 3) world-to-camera rotation
    t: Optional[np.ndarray] = None  # (3,)   world-to-camera translation

    def has_pose(self) -> bool:
        """True once two-view init or PnP has solved for this camera's (R, t)."""
        return self.R is not None and self.t is not None

    def camera_center(self) -> np.ndarray:
        """Where this camera sits in world coordinates (not the same as t, which is world-to-camera)."""
        if not self.has_pose():
            raise ValueError(f"Camera {self.id} has no pose")
        return -self.R.T @ self.t

    def projection_matrix(self) -> np.ndarray:
        """3x4 projection P = K [R | t], mapping a homogeneous world point to a homogeneous pixel."""
        if not self.has_pose():
            raise ValueError(f"Camera {self.id} has no pose")
        rt = np.hstack([self.R, self.t.reshape(3, 1)])
        return self.K @ rt


@dataclass
class Point3D:
    """One reconstructed 3D point. `id` doubles as the originating feature track's id (see sfm/features/tracks.py)."""

    id: int
    xyz: np.ndarray  # (3,)
    color: Optional[np.ndarray] = None  # (3,) RGB in 0-255, optional (not needed for geometry)


@dataclass
class Observation:
    """One pixel measurement: 'this 3D point was seen at pixel uv in this camera'. The atomic unit BA optimizes against."""

    camera_id: int
    point_id: int
    uv: np.ndarray  # (2,)


class Map:
    """
    Scene graph used by incremental SfM and bundle adjustment.

    Observations are stored once in a flat list; `_obs_by_point` and
    `_obs_by_camera` are just index lists into that one list, so looking up
    "every measurement of this point" (a BA residual block) or "every
    measurement in this camera" doesn't require scanning all observations
    every time.
    """

    def __init__(self) -> None:
        self.cameras: Dict[int, Camera] = {}
        self.points: Dict[int, Point3D] = {}
        self.observations: List[Observation] = []
        self._obs_by_point: Dict[int, List[int]] = {}
        self._obs_by_camera: Dict[int, List[int]] = {}
        self._next_camera_id = 0
        self._next_point_id = 0

    def add_camera(self, camera: Camera) -> Camera:
        if camera.id in self.cameras:
            raise ValueError(f"Camera id {camera.id} already exists")
        self.cameras[camera.id] = camera
        self._obs_by_camera.setdefault(camera.id, [])
        self._next_camera_id = max(self._next_camera_id, camera.id + 1)
        return camera

    def add_point(self, point: Point3D) -> Point3D:
        if point.id in self.points:
            raise ValueError(f"Point id {point.id} already exists")
        self.points[point.id] = point
        self._obs_by_point.setdefault(point.id, [])
        self._next_point_id = max(self._next_point_id, point.id + 1)
        return point

    def add_observation(self, obs: Observation) -> Observation:
        if obs.camera_id not in self.cameras:
            raise KeyError(f"Unknown camera {obs.camera_id}")
        if obs.point_id not in self.points:
            raise KeyError(f"Unknown point {obs.point_id}")
        idx = len(self.observations)
        self.observations.append(obs)
        self._obs_by_point.setdefault(obs.point_id, []).append(idx)
        self._obs_by_camera.setdefault(obs.camera_id, []).append(idx)
        return obs

    def next_camera_id(self) -> int:
        return self._next_camera_id

    def next_point_id(self) -> int:
        return self._next_point_id

    def track(self, point_id: int) -> List[Observation]:
        """All 2D observations of one 3D point (the BA residual block for that point)."""
        return [self.observations[i] for i in self._obs_by_point.get(point_id, [])]

    def observations_of_camera(self, camera_id: int) -> List[Observation]:
        return [self.observations[i] for i in self._obs_by_camera.get(camera_id, [])]

    def points_xyz(self) -> np.ndarray:
        """All point coordinates stacked into one (N, 3) array, e.g. for plotting or PLY export."""
        if not self.points:
            return np.zeros((0, 3), dtype=np.float64)
        ids = sorted(self.points)
        return np.stack([self.points[i].xyz for i in ids], axis=0)

    def summary(self) -> str:
        n_posed = sum(1 for c in self.cameras.values() if c.has_pose())
        return (
            f"Map: {len(self.cameras)} cameras ({n_posed} with pose), "
            f"{len(self.points)} points, {len(self.observations)} observations"
        )
