"""
Dense-cloud cleanup and Poisson surface reconstruction, via Open3D.

Poisson is a well-established, off-the-shelf algorithm -- re-deriving it
isn't the point of an SfM capstone, unlike the F/E/PnP/BA math that IS
hand-written throughout this project. This is the same "safe library for a
solved sub-problem" pattern as OpenCV's SIFT/FLANN/StereoSGBM.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import open3d as o3d


def clean_dense_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    radius_factor: float = 3.0,
    radius_min_points: int = 12,
) -> o3d.geometry.PointCloud:
    """
    Voxel-downsample (removes redundant points where overlapping stereo
    pairs both reconstructed the same physical surface), then two outlier
    passes:

    - Statistical: an isolated point far from its neighbors is almost
      always a bad stereo match.
    - Radius/density: classical block-matching stereo on repetitive or
      near-textureless regions produces thin, internally-consistent
      "streak" artifacts along ambiguous epipolar directions -- each point
      on a streak looks fine to its immediate streak-neighbors, so
      statistical removal alone misses them. But a streak is always much
      SPARSER (fewer neighbors per unit volume) than the real, thick
      surface patch, so a minimum local point-count within a radius catches
      what the statistical pass doesn't.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    pcd = pcd.voxel_down_sample(voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd, _ = pcd.remove_radius_outlier(nb_points=radius_min_points, radius=voxel_size * radius_factor)
    return pcd


def keep_largest_cluster(pcd: o3d.geometry.PointCloud, eps: float, min_points: int = 30) -> o3d.geometry.PointCloud:
    """
    DBSCAN clustering, keeping only the largest connected cluster -- a
    second line of defense against any streak/artifact cluster that
    survives density filtering but is still spatially separate from the
    main reconstructed surface.
    """
    labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    if labels.max() < 0:
        return pcd  # nothing formed a cluster (all noise) -- return unchanged rather than emptying the cloud
    counts = np.bincount(labels[labels >= 0])
    largest = int(np.argmax(counts))
    return pcd.select_by_index(np.where(labels == largest)[0])


def poisson_mesh(
    pcd: o3d.geometry.PointCloud,
    depth: int = 9,
    density_trim_quantile: float = 0.05,
) -> o3d.geometry.TriangleMesh:
    """
    Poisson surface reconstruction, with the standard density-based trim:
    Poisson always produces a *closed* surface, which means it invents
    a smooth, low-confidence "guessed" surface to fill gaps where the point
    cloud has no data (e.g. the back of the fountain, never photographed).
    Vertices in those guessed regions have low reconstruction "density";
    dropping the bottom `density_trim_quantile` removes most of that
    invented geometry, leaving mainly the surface actually supported by
    real points.
    """
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(30)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    densities = np.asarray(densities)
    threshold = np.quantile(densities, density_trim_quantile)
    mesh.remove_vertices_by_mask(densities < threshold)
    return mesh


def simplify_for_web(mesh: o3d.geometry.TriangleMesh, target_triangles: int) -> o3d.geometry.TriangleMesh:
    """Decimate a dense mesh down to a triangle count a browser canvas can render interactively."""
    if len(mesh.triangles) <= target_triangles:
        return mesh
    return mesh.simplify_quadric_decimation(target_triangles)
