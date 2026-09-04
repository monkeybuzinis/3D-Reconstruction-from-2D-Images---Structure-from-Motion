"""
Week 1 check: load Fountain-P11, print K, hold a Map, dump the reference cloud.

This is the sanity check for the very first project stage (folder skeleton +
Map data model + dataset loader) -- it doesn't reconstruct anything itself,
it just proves the loader and Map wiring both work before any real geometry
gets built on top of them.

Run from the repo root:
    python scripts/verify_week1.py
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")  # headless backend: this script only saves PNGs, never opens a window
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # let `python scripts/verify_week1.py` find the sfm package without installing it

from sfm.io.fountain import FOUNTAIN_K, load_fountain_dataset
from sfm.io.ply import write_xyz_ply
from sfm.map import Camera, Map, Observation, Point3D


def _smoke_test_tracks() -> None:
    """Make sure observation tracks index correctly before BA exists -- a tiny 2-camera/1-point Map, checked by hand."""
    scene = Map()
    scene.add_camera(
        Camera(id=0, K=FOUNTAIN_K.copy(), image_name="dummy0.png", width=10, height=10)
    )
    scene.add_camera(
        Camera(id=1, K=FOUNTAIN_K.copy(), image_name="dummy1.png", width=10, height=10)
    )
    scene.add_point(Point3D(id=0, xyz=np.array([0.0, 0.0, 1.0])))
    scene.add_observation(Observation(camera_id=0, point_id=0, uv=np.array([1.0, 2.0])))
    scene.add_observation(Observation(camera_id=1, point_id=0, uv=np.array([3.0, 4.0])))
    track = scene.track(0)
    assert len(track) == 2
    assert len(scene.observations_of_camera(0)) == 1


def main() -> None:
    _smoke_test_tracks()

    out_dir = ROOT / "outputs" / "week1"
    out_dir.mkdir(parents=True, exist_ok=True)

    # include_reference_points=True pulls in the ~59k-point BA solution-pack
    # cloud from sfm/io/fountain.py, purely so there's something to look at
    # in week 1 -- it's not ground truth and isn't used by later stages.
    scene, images = load_fountain_dataset(include_reference_points=True)

    print(scene.summary())
    print("\nIntrinsic matrix K:")
    print(FOUNTAIN_K)

    for cam_id in sorted(scene.cameras):
        cam = scene.cameras[cam_id]
        img = images[cam_id]
        h, w = img.shape[:2]
        assert (w, h) == (cam.width, cam.height)
        assert np.allclose(cam.K, FOUNTAIN_K)
        print(
            f"  camera {cam.id:2d}  {cam.image_name}  "
            f"{cam.width}x{cam.height}  pose={'yes' if cam.has_pose() else 'no'}"
        )

    xyz = scene.points_xyz()
    ply_path = out_dir / "fountain_reference_sparse.ply"
    write_xyz_ply(ply_path, xyz)
    print(f"\nWrote {xyz.shape[0]} reference points -> {ply_path}")

    # Subsample before plotting -- matplotlib's 3D scatter gets very slow
    # (and the PNG very cluttered) past a few thousand points.
    rng = np.random.default_rng(0)
    sample_n = min(8000, xyz.shape[0])
    sample = xyz[rng.choice(xyz.shape[0], sample_n, replace=False)]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=0.4, c="k", alpha=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Fountain-P11 reference sparse cloud (BA solution pack)")
    ax.view_init(elev=15, azim=-70)
    fig.tight_layout()
    fig_path = out_dir / "fountain_reference_sparse.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"Wrote preview -> {fig_path}")
    print("\nWeek 1 loader OK.")


if __name__ == "__main__":
    main()
