"""
Write a finished reconstruction in COLMAP's text model format.

The point is to view this project's result in the same viewer COLMAP's own
output is viewed in, so a visual comparison is not confounded by two different
renderers. It converts what is already on disk and re-solves nothing.

    .venv/bin/python3 scripts/export_colmap.py outputs/demo/fountain

writes cameras.txt, images.txt and points3D.txt under <scene_dir>/colmap/.

Note on conventions: scene_cache stores world-to-camera R and t, which is what
COLMAP's images.txt also wants, so the rotation is converted to a quaternion
and the translation copied across unchanged.
"""
from pathlib import Path
import argparse
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfm.io.generic import load_image_folder


def rotation_to_quaternion(R: np.ndarray) -> tuple:
    """(qw, qx, qy, qz) from a rotation matrix, via the largest-diagonal branch
    so the division never happens on a near-zero number."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        qw, qx, qy, qz = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw, qx, qy, qz = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw, qx, qy, qz = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw, qx, qy, qz = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([qw, qx, qy, qz])
    return tuple(q / np.linalg.norm(q))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir", type=Path, help="directory holding scene_cache.npz")
    ap.add_argument("images", type=Path, nargs="?", help="image folder, for the file names")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    cache = np.load(args.scene_dir / "scene_cache.npz")
    out = args.output or (args.scene_dir / "colmap")
    out.mkdir(parents=True, exist_ok=True)

    if args.images is not None:
        _imgs, paths = load_image_folder(args.images)
        names = [p.name for p in paths]
    else:
        names = [f"{int(i):04d}.jpg" for i in cache["cam_ids"]]

    cam_ids, Ks = cache["cam_ids"], cache["Ks"]
    Rs, ts = cache["Rs"], cache["ts"]
    widths, heights = cache["widths"], cache["heights"]
    xyz, colors = cache["xyz"], cache["colors"]

    with open(out / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for k in range(len(cam_ids)):
            K = Ks[k]
            f.write(f"{k + 1} PINHOLE {int(widths[k])} {int(heights[k])} "
                    f"{K[0, 0]:.6f} {K[1, 1]:.6f} {K[0, 2]:.6f} {K[1, 2]:.6f}\n")

    with open(out / "images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for k, cid in enumerate(cam_ids):
            qw, qx, qy, qz = rotation_to_quaternion(Rs[k])
            tx, ty, tz = ts[k]
            name = names[int(cid)] if int(cid) < len(names) else f"{int(cid):04d}.jpg"
            f.write(f"{k + 1} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                    f"{tx:.9f} {ty:.9f} {tz:.9f} {k + 1} {name}\n")
            f.write("\n")   # no per-image 2D observations are cached

    with open(out / "points3D.txt", "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for i, (p, c) in enumerate(zip(xyz, colors), start=1):
            f.write(f"{i} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{int(c[0])} {int(c[1])} {int(c[2])} 0.0\n")

    print(f"Wrote COLMAP text model for {len(cam_ids)} cameras and {len(xyz):,} points -> {out}")

    # The COLMAP web viewer reads the binary model, not the text one, so convert
    # through pycolmap: it parses what was just written and re-emits .bin.
    try:
        import pycolmap
        pycolmap.logging.minloglevel = 3
        rec = pycolmap.Reconstruction(str(out))
        rec.write_binary(str(out))
        print(f"Wrote binary model (cameras.bin, images.bin, points3D.bin) -> {out}")
    except Exception as exc:                                  # pragma: no cover
        print(f"Could not write the binary model ({exc}). The text model is still usable "
              f"in the desktop COLMAP GUI.")
    print("Upload this folder to a COLMAP viewer to see this project's result in COLMAP's renderer.")


if __name__ == "__main__":
    main()
