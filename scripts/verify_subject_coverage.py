"""
Where did the reconstructed points actually land in the photograph?

This exists because of a failure no aggregate metric in this project could
see. On `guitar`, COLMAP registers 16/16 cameras at 0.748px mean
reprojection error with 987 points per image; on `shoe` it registers 25/25
at 0.723px with 1,050 points per image. By every number either system
reports, the two datasets are equally successful.

They are not. Projecting the triangulated points back into the images shows
that on `shoe` they cover the boot itself, while on `guitar` they cover the
fence, the gravel and the ground, and the instrument is nearly bare. The
reconstruction succeeded; the SUBJECT of the photographs did not get
reconstructed. Registration count, reprojection error and point total are
all blind to that distinction, because they are computed over whatever was
reconstructed rather than over what was meant to be.

The two datasets make an unintentionally well-controlled pair: same
photographer, same camera, same fence, same afternoon, comparable arc. The
boot is scuffed matte leather; the guitar is gloss lacquer, whose specular
highlights move with the viewpoint and so produce no viewpoint-stable
features to match.

Run from the repo root:
    .venv/bin/python3 scripts/verify_subject_coverage.py guitar shoe
    .venv/bin/python3 scripts/verify_subject_coverage.py --n 4 woodlog
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_colmap import dataset_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--n", type=int, default=3, help="images to show per dataset")
    ap.add_argument("--out", type=Path, default=ROOT / "evaluation" / "metrics")
    args = ap.parse_args()

    import pycolmap
    pycolmap.logging.minloglevel = 3

    fig, axes = plt.subplots(len(args.datasets), args.n,
                             figsize=(5 * args.n, 5.5 * len(args.datasets)), squeeze=False)

    for row, name in enumerate(args.datasets):
        model = ROOT / "colmap" / name / "sparse" / "0"
        if not model.exists():
            raise SystemExit(f"No COLMAP model at {model} -- run scripts/compare_colmap.py {name}")
        rec = pycolmap.Reconstruction(model)
        folder = dataset_dir(name)
        posed = sorted((im for im in rec.images.values() if im.has_pose), key=lambda i: i.name)
        # Spread the picks across the capture rather than taking the first N,
        # so a subject that only fails from certain angles still shows it.
        idx = np.linspace(0, len(posed) - 1, args.n).round().astype(int)
        for col, i in enumerate(idx):
            im = posed[int(i)]
            ax = axes[row][col]
            ax.imshow(Image.open(folder / im.name))
            uv = np.array([p.xy for p in im.points2D if p.has_point3D()])
            if len(uv):
                ax.scatter(uv[:, 0], uv[:, 1], s=1.5, c="lime", alpha=0.55)
            ax.set_title(f"{name} / {im.name}\n{len(uv):,} triangulated observations", fontsize=9)
            ax.axis("off")

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / ("subject_coverage_" + "_".join(args.datasets) + ".png")
    fig.tight_layout()
    fig.savefig(path, dpi=85)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
