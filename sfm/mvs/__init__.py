"""Dense reconstruction: pairwise stereo -> fused dense cloud -> Poisson mesh. Extends beyond sparse SfM."""

from sfm.mvs.stereo import dense_stereo_pair, relative_pose

__all__ = ["dense_stereo_pair", "relative_pose"]
