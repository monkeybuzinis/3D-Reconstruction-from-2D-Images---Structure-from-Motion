"""Image, calibration and point-cloud I/O.

The public names here are dataset-neutral: any folder of photos goes through
load_scene(), which resolves K from a .camera sidecar, EXIF, or a fallback FOV.
Dataset-specific readers (sfm.io.fountain, sfm.io.reference) stay out of this
namespace and are imported directly by the scripts that need them.
"""

from sfm.io.calibration import estimate_intrinsics
from sfm.io.camera_file import find_camera_file, load_camera_file
from sfm.io.generic import load_image_folder, load_scene
from sfm.io.ply import write_xyz_ply, write_xyzrgb_ply

__all__ = [
    "estimate_intrinsics",
    "find_camera_file",
    "load_camera_file",
    "load_image_folder",
    "load_scene",
    "write_xyz_ply",
    "write_xyzrgb_ply",
]
