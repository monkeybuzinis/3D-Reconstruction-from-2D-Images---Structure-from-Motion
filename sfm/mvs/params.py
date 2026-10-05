"""
Derive dense-stereo parameters (depth window, voxel size) from a finished
sparse reconstruction.

Why this exists as its own module: both MVS callers used to read these
numbers straight off the sparse cloud's *world* Z percentiles --

    min_depth, max_depth = np.percentile(sparse_xyz[:, 2], [1, 99])
    voxel_size = norm(percentile(sparse_xyz, 99) - percentile(sparse_xyz, 1)) / 250

-- and both halves of that were wrong in ways that stayed hidden for as long
as the Fountain-P11 numbers happened to be benign.

1. World Z is only camera depth for the seed camera (the one
   sfm/recon/incremental.py pins to R=I, t=0), and only while no point sits
   behind it. When that file's cheirality check was corrected to measure
   depth in each camera's own frame, points legitimately in front of two
   OTHER cameras but behind the seed camera's plane started being accepted,
   and Fountain's 1st-percentile world Z went NEGATIVE (-6.21) -- a
   meaningless depth window handed to dense stereo.

2. The bounding box of the SPARSE cloud is not the scale of the thing being
   voxelized. Sparse tracks reach across the whole scene (Fountain: robust
   diagonal 24.3) while dense stereo only reconstructs the near surface
   (6.3). The ratio between the two is dataset-dependent -- about 4x on
   Fountain, about 1x on Rathaus -- so ANY fixed divisor applied to the
   sparse extent is miscalibrated on some dataset. (Measured the hard way:
   re-deriving it from the dense cloud's extent instead fixed Fountain and
   broke Rathaus, whose cleaned cloud fell to 1,655 points.)

So neither parameter is taken from world coordinates any more:

- The depth window comes from actual per-camera depths, so it is positive by
  construction no matter where the world origin sits.
- The voxel size comes from the sensor's resolution limit. Dense stereo
  cannot resolve detail finer than one pixel's footprint at scene depth, so
  the voxel is a fixed number of pixel footprints -- a physically meaningful
  quantity that needs no per-dataset tuning. See voxel_size_for().
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class DenseParams(NamedTuple):
    min_depth: float
    max_depth: float
    voxel_size: float
    median_depth: float
    pixel_footprint: float  # world units covered by one (downscaled) pixel at median_depth
    n_core_points: int  # sparse points at least `min_views` cameras see within the window
    n_points: int


def camera_depths(xyz: np.ndarray, Rs: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """
    Depth of every point in every camera: (n_cameras, n_points).

    Depth is the Z coordinate after the world-to-camera transform
    x_cam = R @ X + t, matching sfm/map.py's Camera convention. Negative
    entries mean the point is behind that camera, which is normal -- no
    camera sees the whole scene.
    """
    return np.stack([(R @ xyz.T + t.reshape(3, 1))[2, :] for R, t in zip(Rs, ts)])


def voxel_size_for(
    median_depth: float,
    focal_px: float,
    downscale: int = 2,
    pixels_per_voxel: float = 12.0,
) -> float:
    """
    Voxel edge length for cleaning a dense cloud, in world units.

    One pixel at distance d subtends d / focal_px in world units, so after
    stereo runs on images reduced by `downscale` the finest detail the
    disparity map can carry is (d / focal_px) * downscale wide. Voxels any
    smaller than that just preserve noise; voxels much larger throw away
    real surface detail. `pixels_per_voxel` is therefore the only knob, and
    it reads in a unit that means something: how many pixels of footprint
    one voxel spans.

    12 comes from rendering before/after meshes for all four datasets and
    comparing them by eye, which pointed the same way every time: finer
    helped (Fountain gained detail over its old 0.061; the statue went from
    blocky to smooth), coarser lost real surface (at 0.035 Rathaus dropped
    an entire wing of the building). 12 puts Rathaus back at the 0.030 its
    inspected run used and leaves every other dataset at or below its old
    value. Unlike a bounding-box fraction, the same constant transfers
    between datasets because it is anchored to the sensor rather than to
    how far the sparse cloud happens to sprawl.
    """
    if focal_px <= 0 or median_depth <= 0 or downscale < 1:
        raise ValueError(f"bad inputs: median_depth={median_depth}, focal_px={focal_px}, downscale={downscale}")
    return float(median_depth / focal_px * downscale * pixels_per_voxel)


def dense_params(
    xyz: np.ndarray,
    Rs: np.ndarray,
    ts: np.ndarray,
    Ks: np.ndarray,
    downscale: int = 2,
    lo_pct: float = 1.0,
    hi_pct: float = 99.0,
    min_views: int = 2,
    pixels_per_voxel: float = 12.0,
    pad_lo: float = 0.8,
    pad_hi: float = 1.25,
    max_depth_median_ratio: float = 20.0,
) -> DenseParams:
    """
    Depth window and voxel size for dense stereo on this reconstruction.

    `downscale` must match what gets passed to sfm.mvs.fuse.dense_reconstruct,
    since it changes the effective focal length and hence the pixel footprint.

    lo_pct/hi_pct trim the depth window against the outlier points that
    survive RANSAC and BA; pad_lo/pad_hi then widen it back out so the
    trimming cannot clip real surface (see the comment on the padding).

    min_views only affects a reported diagnostic: how many points at least
    that many cameras see inside the window -- the part of the sparse cloud
    stereo has any chance of densifying, since stereo needs a pair.
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N, 3), got {xyz.shape}")

    depths = camera_depths(xyz, Rs, ts)
    in_front = depths[depths > 0]
    if in_front.size == 0:
        raise ValueError("No sparse point is in front of any camera -- is the reconstruction valid?")

    lo_depth, hi_depth = (float(v) for v in np.percentile(in_front, [lo_pct, hi_pct]))
    # Pad outwards. This window exists to reject nonsense (negative or absurd
    # depths), not to decide where the object ends -- and the percentiles are
    # taken over SPARSE points, which thin out exactly at a surface's far
    # edge. Unpadded, Rathaus's 99th percentile landed at 8.26 while the
    # building ran to 8.81, and dense stereo clipped off a whole wing.
    min_depth, max_depth = lo_depth * pad_lo, hi_depth * pad_hi
    median_depth = float(np.median(in_front))

    # Percentiles are not robust enough on their own. A cloud that contains
    # low-parallax points -- triangulated from near-parallel rays, so their
    # depth is unconstrained and they land anywhere, including effectively at
    # infinity -- can push even the 99th percentile absurdly far: the buffalo
    # capture produced depth windows of 2,642 and then 5,029,180 against a
    # median depth near 4. Such points pass the reprojection-error filter
    # precisely because low parallax means they reproject well from anywhere
    # along the ray, so nothing upstream removes them. Cap the window at a
    # generous multiple of the median depth: scenes really do span a wide
    # range (Herzjesu runs 11 to 68, about 2.6x its median), but nothing real
    # sits thousands of times further away than the typical point.
    depth_cap = median_depth * max_depth_median_ratio
    if max_depth > depth_cap:
        max_depth = depth_cap

    within = (depths >= min_depth) & (depths <= max_depth)
    n_core = int((within.sum(axis=0) >= min_views).sum())

    focal_px = float(np.mean([K[0, 0] for K in Ks]))
    voxel = voxel_size_for(median_depth, focal_px, downscale, pixels_per_voxel)

    return DenseParams(
        min_depth=min_depth,
        max_depth=max_depth,
        voxel_size=voxel,
        median_depth=median_depth,
        pixel_footprint=median_depth / focal_px * downscale,
        n_core_points=n_core,
        n_points=len(xyz),
    )
