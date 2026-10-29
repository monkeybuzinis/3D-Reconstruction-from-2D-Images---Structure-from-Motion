"""
Generate the diagrams and charts that go in the report but are not renders of
reconstructions (those come from scripts/render_figures.py).

These are static figures for print: no hover layer, no dark mode. Series
colours are the validated categorical pair blue/orange, used consistently so
that "this pipeline" is the same colour in every figure.

Run from the repo root:
    .venv/bin/python3 scripts/make_report_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation" / "figures"

OURS = "#2a78d6"     # categorical slot 1
COLMAP = "#eb6834"   # categorical slot 2
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d8d7d2"

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 9,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.titlecolor": INK,
})


def _recessive(ax, xgrid=True):
    """Grid and spines stay behind the data, per the marks spec."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


# --------------------------------------------------------------- diagram
def pipeline_diagram(path: Path) -> None:
    """Block diagram of the pipeline. Not a chart -- no palette rules apply."""
    stages = [
        ("Photographs", "unordered set of images"),
        ("Intrinsics K", "override / sidecar / EXIF / FOV fallback"),
        ("SIFT + FLANN matching", "Lowe ratio test at 0.75"),
        ("Geometric verification", "RANSAC on F, scored by Sampson distance"),
        ("Track construction", "union-find over verified pairs"),
        ("Two-view seed", "F \u2192 E \u2192 four poses \u2192 cheirality"),
        ("Incremental registration", "calibrated DLT PnP inside RANSAC"),
        ("Bundle adjustment", "so(3), analytic Jacobian, Schur complement"),
        ("Filtering", "reprojection, depth, degeneracy checks"),
        ("Sparse cloud + poses", "the output of the sparse pipeline"),
    ]
    dense = [
        ("Rectify", "each adjacent pair"),
        ("SGBM disparity", "with WLS filtering"),
        ("Fuse and clean", "voxel, outliers, DBSCAN"),
        ("Poisson mesh", "density-trimmed"),
    ]

    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    ax.set_xlim(0, 10.4); ax.set_ylim(0, 12.2); ax.axis("off")

    def block(x, y, w, h, title, sub, fill, edge, ts=8.4, ss=6.6):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.10",
            facecolor=fill, edgecolor=edge, linewidth=1.1))
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
                fontsize=ts, color=INK, weight="bold")
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                fontsize=ss, color=INK_2)

    BW, BH, STEP = 5.6, 0.74, 1.15
    y = 11.2
    for i, (t, sub) in enumerate(stages):
        block(0.5, y, BW, BH, t, sub, "#ffffff" if i == 0 else "#eaf1fb", OURS)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch(
                (0.5 + BW / 2, y), (0.5 + BW / 2, y - (STEP - BH)),
                arrowstyle="-|>", mutation_scale=10, color=INK_2, linewidth=1.0))
        y -= STEP
    sparse_bottom_y = y + STEP

    # Dense branch, drawn beside the sparse chain that feeds it.
    DX, DW, DH, DSTEP = 6.7, 3.3, 0.68, 1.00
    yd = 4.5
    ax.text(DX + DW / 2, yd + DH + 0.42, "optional dense stage",
            ha="center", fontsize=7.4, color=INK_2, style="italic")
    for i, (t, sub) in enumerate(dense):
        block(DX, yd, DW, DH, t, sub, "#fdeee7", COLMAP, ts=7.8, ss=6.2)
        if i < len(dense) - 1:
            ax.add_patch(FancyArrowPatch(
                (DX + DW / 2, yd), (DX + DW / 2, yd - (DSTEP - DH)),
                arrowstyle="-|>", mutation_scale=10, color=INK_2, linewidth=1.0))
        yd -= DSTEP

    # Poses feed the dense stage.
    ax.add_patch(FancyArrowPatch(
        (0.5 + BW, sparse_bottom_y + BH / 2), (DX, 4.5 + DH / 2),
        arrowstyle="-|>", mutation_scale=10, color=INK_2, linewidth=1.0,
        connectionstyle="arc3,rad=-0.3", linestyle="--"))

    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # A 3-D axes reserves far more room than its contents occupy, so the
    # saved figure carries a wide band of blank page. Crop to the ink.
    from PIL import Image
    im = Image.open(path).convert("RGB")
    a = np.asarray(im)
    ink = np.where((a < 247).any(axis=2))
    if ink[0].size:
        pad = 14
        im = im.crop((max(0, ink[1].min()-pad), max(0, ink[0].min()-pad),
                      min(a.shape[1], ink[1].max()+pad), min(a.shape[0], ink[0].max()+pad)))
        im.save(path)
    print(f"  {path.relative_to(ROOT)}  ({im.size[0]}x{im.size[1]})")


# ---------------------------------------------------------------- charts
def registration_chart(path: Path, data: dict) -> None:
    """Fraction of images registered, this pipeline against COLMAP."""
    names, ours, theirs = [], [], []
    for k, v in data.items():
        if not v["ours"]:
            continue
        names.append(k)
        ours.append(100 * v["ours"]["registered"] / v["ours"]["n_images"])
        theirs.append(100 * v["colmap"]["registered"] / v["colmap"]["n_images"])
    order = np.argsort(ours)
    names = [names[i] for i in order]
    ours = np.array(ours)[order]; theirs = np.array(theirs)[order]

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    h = 0.36
    ax.barh(y + h / 2 + 0.02, ours, height=h, color=OURS, label="This pipeline")
    ax.barh(y - h / 2 - 0.02, theirs, height=h, color=COLMAP, label="COLMAP")
    for yi, (a, b) in enumerate(zip(ours, theirs)):
        ax.text(a + 1.5, yi + h / 2 + 0.02, f"{a:.0f}%", va="center", fontsize=7, color=INK_2)
        ax.text(b + 1.5, yi - h / 2 - 0.02, f"{b:.0f}%", va="center", fontsize=7, color=INK_2)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 112); ax.set_xlabel("images registered (%)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    _recessive(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # A 3-D axes reserves far more room than its contents occupy, so the
    # saved figure carries a wide band of blank page. Crop to the ink.
    from PIL import Image
    im = Image.open(path).convert("RGB")
    a = np.asarray(im)
    ink = np.where((a < 247).any(axis=2))
    if ink[0].size:
        pad = 14
        im = im.crop((max(0, ink[1].min()-pad), max(0, ink[0].min()-pad),
                      min(a.shape[1], ink[1].max()+pad), min(a.shape[0], ink[0].max()+pad)))
        im.save(path)
    print(f"  {path.relative_to(ROOT)}  ({im.size[0]}x{im.size[1]})")


def capture_arc_chart(path: Path, geom: dict) -> None:
    """
    Capture arc measured from our own poses against COLMAP's.

    This is the circularity figure: where our solve failed, the arc derived
    from it understates the real capture by a factor of two to five.
    """
    names, a_ours, a_col = [], [], []
    for k, v in geom.items():
        if not v["ours"] or not v["colmap"]:
            continue
        # An "arc" needs at least three cameras to mean anything; with two it
        # is a single angle. robotpu registers 2/39 in both systems.
        if min(v["ours"]["n_cameras"], v["colmap"]["n_cameras"]) < 3:
            continue
        names.append(k)
        a_ours.append(v["ours"]["arc_total_deg"])
        a_col.append(v["colmap"]["arc_total_deg"])
    ratio = np.array(a_col) / np.maximum(np.array(a_ours), 1e-6)
    order = np.argsort(ratio)
    names = [names[i] for i in order]
    a_ours = np.array(a_ours)[order]; a_col = np.array(a_col)[order]
    ratio = ratio[order]

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    h = 0.36
    ax.barh(y + h / 2 + 0.02, a_ours, height=h, color=OURS, label="measured from our poses")
    ax.barh(y - h / 2 - 0.02, a_col, height=h, color=COLMAP, label="measured from COLMAP's poses")
    for yi, r in enumerate(ratio):
        if r > 1.6:  # label only where the two disagree materially
            ax.text(max(a_col[yi], a_ours[yi]) + 5, yi, f"×{r:.1f}",
                    va="center", fontsize=7.5, color=INK, weight="bold")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("total capture arc swept about the scene (degrees)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    _recessive(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # A 3-D axes reserves far more room than its contents occupy, so the
    # saved figure carries a wide band of blank page. Crop to the ink.
    from PIL import Image
    im = Image.open(path).convert("RGB")
    a = np.asarray(im)
    ink = np.where((a < 247).any(axis=2))
    if ink[0].size:
        pad = 14
        im = im.crop((max(0, ink[1].min()-pad), max(0, ink[0].min()-pad),
                      min(a.shape[1], ink[1].max()+pad), min(a.shape[0], ink[0].max()+pad)))
        im.save(path)
    print(f"  {path.relative_to(ROOT)}  ({im.size[0]}x{im.size[1]})")


def pose_accuracy_chart(path: Path) -> None:
    """
    Rathaus rotation error.

    Two panels rather than one: the fair comparison (0.111 against 0.181) and
    the calibration-source effect (0.181 against 2.150) differ by an order of
    magnitude, and on a shared axis the comparison that matters is invisible.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    def panel(ax, labels, values, colours, title, top):
        bars = ax.bar(labels, values, width=0.46, color=colours)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + top * 0.03, f"{v:.3f}\u00b0",
                    ha="center", fontsize=8.5, color=INK, weight="bold")
        ax.set_ylim(0, top)
        ax.set_title(title, fontsize=8.5, color=INK, pad=10)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)

    panel(ax1, ["This pipeline", "COLMAP"], [0.111, 0.181], [OURS, COLMAP],
          "(a) Both systems given the true intrinsics", 0.24)
    ax1.set_ylabel("mean rotation error vs\nground truth (degrees)", fontsize=8)

    panel(ax2, ["given the true K", "self-calibrating"], [0.181, 2.150],
          [COLMAP, COLMAP], "(b) COLMAP, by calibration source", 2.7)
    ax2.patches[1].set_alpha(0.45)
    ax2.annotate("principal point pinned at the image centre,\n"
                 "127 px from the true one \u2192 1.84\u00b0 of tilt",
                 xy=(1, 2.15), xytext=(-0.35, 1.30), fontsize=6.6, color=INK_2,
                 arrowprops=dict(arrowstyle="->", color=INK_2, linewidth=0.8))

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # A 3-D axes reserves far more room than its contents occupy, so the
    # saved figure carries a wide band of blank page. Crop to the ink.
    from PIL import Image
    im = Image.open(path).convert("RGB")
    a = np.asarray(im)
    ink = np.where((a < 247).any(axis=2))
    if ink[0].size:
        pad = 14
        im = im.crop((max(0, ink[1].min()-pad), max(0, ink[0].min()-pad),
                      min(a.shape[1], ink[1].max()+pad), min(a.shape[0], ink[0].max()+pad)))
        im.save(path)
    print(f"  {path.relative_to(ROOT)}  ({im.size[0]}x{im.size[1]})")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads((ROOT / "colmap" / "comparison.json").read_text())
    geom = json.loads((ROOT / "colmap" / "capture_geometry.json").read_text())

    pipeline_diagram(OUT / "fig_pipeline.png")
    overview_figure(OUT / "fig_overview.png")
    workflow_figure(OUT / "fig_workflow.png")
    registration_chart(OUT / "fig_registration.png", data)
    capture_arc_chart(OUT / "fig_capture_arc.png", geom)
    pose_accuracy_chart(OUT / "fig_pose_accuracy.png")



# ------------------------------------------------- overview (two panels)
def overview_figure(path: Path, dataset: str = "woodlog") -> None:
    """
    Two-panel overview: (a) the pipeline as a block diagram, (b) the capture
    geometry.

    Both panels use this project's own data rather than a generic
    illustration -- panel (a) shows real input photographs and the real
    reconstruction they produce, and panel (b) plots the actual recovered
    camera poses and sparse points, not a drawn impression of them. A
    reader can therefore check the figure against the results.
    """
    from matplotlib.image import imread
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ds = ROOT / "dataset" / dataset
    photos = sorted(p for p in ds.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"})
    cache = np.load(ROOT / "outputs" / dataset / "scene_cache.npz")
    Rs, ts, xyz = cache["Rs"], cache["ts"], cache["xyz"]
    centres = np.stack([-R.T @ t for R, t in zip(Rs, ts)])

    fig = plt.figure(figsize=(9.2, 7.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=-0.04)

    # ---------------- panel (a): the pipeline ----------------
    ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 100); ax.set_ylim(-25, 44); ax.axis("off")
    ax.text(1, 41, "(a)", fontsize=11, weight="bold")

    def group(x, w, title, items, fill, edge, divider_before=None):
        ax.add_patch(FancyBboxPatch((x, 4), w, 32, boxstyle="round,pad=0.4,rounding_size=0.8",
                                    facecolor=fill, edgecolor=edge, linewidth=1.2))
        ax.text(x + w/2, 32.5, title, ha="center", fontsize=9, weight="bold", color=INK)
        n = len(items); bh = 22.0 / n
        centres_y = []
        for i, it in enumerate(items):
            y = 28 - (i + 1) * bh
            centres_y.append(y + bh*0.49)
            ax.add_patch(FancyBboxPatch((x + 1.2, y + bh*0.14), w - 2.4, bh*0.70,
                                        boxstyle="round,pad=0.2,rounding_size=0.5",
                                        facecolor="white", edgecolor=edge, linewidth=0.9))
            ax.text(x + w/2, y + bh*0.49, it, ha="center", va="center", fontsize=7.2, color=INK)
            if divider_before is not None and i == divider_before:
                yd = y + bh * 1.02
                ax.plot([x + 0.9, x + w - 0.9], [yd, yd], color=INK_2,
                        linewidth=0.9, linestyle=(0, (3, 2)))
        return x + w, centres_y

    # input photographs
    for i, p in enumerate(photos[:3]):
        im = imread(p)
        step = max(1, max(im.shape[:2]) // 180)
        ax.imshow(im[::step, ::step], extent=(1.5, 13.5, 26 - i*10.5, 35.5 - i*10.5), aspect="auto", zorder=3)
    ax.add_patch(FancyBboxPatch((1, 4), 13.5, 32, boxstyle="round,pad=0.4,rounding_size=0.8",
                                facecolor="none", edgecolor=INK_2, linewidth=1.2))
    ax.text(7.7, 37.6, f"Photographs ({len(photos)})", ha="center", fontsize=9, weight="bold", color=INK)

    x, _ = group(19, 21, "Keypoint search", ["Feature extraction (SIFT)",
              "Matching + ratio test", "Geometric verification", "Track construction"], "#eaf1fb", OURS)
    sfm_x = x + 5
    # The registration loop is resectioning -> triangulation -> periodic bundle
    # adjustment -> the next camera. The filters are NOT part of it: they run
    # once after the loop finishes, each followed by a final adjustment.
    x, ys = group(sfm_x, 25, "Sparse SfM", ["Two-view seed", "Resectioning (PnP)",
              "Triangulation", "Bundle adjustment", "Outlier + depth filters",
              "Final bundle adjustment"], "#eaf1fb", OURS, divider_before=4)

    # The principal result of this project is the sparse reconstruction, so
    # the main chain ends there. Dense meshing hangs off it as an optional
    # branch, matching how Sections 4.5 and 7.4 describe it.
    def thumb(src, box, strip_last=False):
        if not src.is_file():
            return
        im = imread(src)
        if strip_last and im.shape[1] > 2 * im.shape[0]:
            im = im[:, -im.shape[1]//3:]
        ax.imshow(im, extent=box, aspect="auto", zorder=3)

    sparse_png = ROOT / "evaluation" / "figures" / "single" / f"sparse_{dataset}.png"
    thumb(sparse_png if sparse_png.is_file() else ROOT/"evaluation"/"figures"/f"sparse_{dataset}.png",
          (x + 5.5, 99, 11, 31), strip_last=True)
    ax.add_patch(FancyBboxPatch((x + 5, 4), 94.5 - x, 32, boxstyle="round,pad=0.4,rounding_size=0.8",
                                facecolor="none", edgecolor=INK_2, linewidth=1.4))
    ax.text((x + 100)/2, 37.6, "Sparse cloud + camera poses", ha="center",
            fontsize=9, weight="bold", color=INK)
    ax.text((x + 100)/2, 7.0, "the principal result", ha="center", fontsize=7.4,
            style="italic", color=INK_2)
    ax.text(50, -17.5,
            "Resectioning, triangulation and bundle adjustment iterate once per added camera "
            "(curved arrow). The filters below the dashed line\nrun once, after the loop has "
            "finished, and are followed by a final adjustment \u2014 they do not feed back into "
            "registration.",
            ha="center", va="top", fontsize=6.9, color=INK_2)

    for ax_x in (15.3, 40.8, 71.0):
        ax.annotate("", xy=(ax_x + 3.2, 20), xytext=(ax_x, 20),
                    arrowprops=dict(arrowstyle="-|>", color=INK_2, linewidth=1.4))
    # bundle adjustment returns to resectioning for the next camera
    ax.annotate("", xy=(sfm_x + 1.0, ys[1]), xytext=(sfm_x + 1.0, ys[3]),
                arrowprops=dict(arrowstyle="-|>", color=INK_2, linewidth=1.1,
                                connectionstyle="arc3,rad=0.75"))


    # optional dense branch, drawn beneath and dashed
    ax.add_patch(FancyBboxPatch((42, -14), 34, 13, boxstyle="round,pad=0.4,rounding_size=0.8",
                                facecolor="#fdeee7", edgecolor=COLMAP, linewidth=1.1,
                                linestyle="--"))
    ax.text(59, -3.8, "Dense MVS  (optional demonstration)", ha="center",
            fontsize=7.8, weight="bold", color=INK)
    ax.text(59, -9.0, "rectify \u2192 stereo \u2192 fuse + clean \u2192 Poisson mesh",
            ha="center", fontsize=6.6, color=INK_2)
    ax.annotate("", xy=(59, -0.6), xytext=(59, 3.4),
                arrowprops=dict(arrowstyle="-|>", color=COLMAP, linewidth=1.0, linestyle="--"))
    thumb(ROOT/"evaluation"/"figures"/"single"/f"mesh_{dataset}.png", (78.5, 97, -14.0, 0.0), strip_last=True)

    # ---------------- panel (b): the capture geometry ----------------
    ax2 = fig.add_subplot(gs[1], projection="3d")
    ax2.text2D(0.01, 0.96, "(b)", transform=ax2.transAxes, fontsize=11, weight="bold")

    # Show the subject, not the whole scene: keep the points closest to the
    # median, which is where the photographed object sits, and drop the
    # background that every outdoor capture also reconstructs.
    med = np.median(xyz, axis=0)
    d = np.linalg.norm(xyz - med, axis=1)
    pts = xyz[d < np.percentile(d, 55)]
    rng = np.random.default_rng(0)
    sample = pts[rng.choice(len(pts), min(7000, len(pts)), replace=False)]
    ax2.scatter(*sample.T, s=1.4, c="#7a7a74", alpha=0.45, linewidths=0)

    span = float(np.linalg.norm(centres - centres.mean(0), axis=1).max())
    fs = span * 0.14                                   # frustum size
    for i, (R, t) in enumerate(zip(Rs, ts)):
        C = -R.T @ t
        corners_cam = np.array([[-1, -0.7, 1.6], [1, -0.7, 1.6], [1, 0.7, 1.6], [-1, 0.7, 1.6]]) * fs
        corners = (R.T @ corners_cam.T).T + C
        faces = [[C, corners[j], corners[(j + 1) % 4]] for j in range(4)]
        ax2.add_collection3d(Poly3DCollection(faces, facecolor=OURS, edgecolor=OURS,
                                              alpha=0.22, linewidths=0.7))
        ax2.add_collection3d(Poly3DCollection([corners], facecolor="none",
                                              edgecolor=OURS, linewidths=0.9))
    ax2.plot(*centres.T, color=INK_2, linewidth=1.0, linestyle="--", zorder=1)

    # lines of sight from three cameras to two shared points
    # Two points near the centre of the subject, so the sight lines read as
    # rays onto the object rather than at stray background structure.
    near = sample[np.argsort(np.linalg.norm(sample - med, axis=1))[: max(200, len(sample)//20)]]
    # Pick the two most separated of those, so the sight lines read as two
    # distinct bundles rather than overlapping into one.
    k = near[rng.choice(len(near), min(60, len(near)), replace=False)]
    dm = np.linalg.norm(k[:, None] - k[None], axis=2)
    i, j = np.unravel_index(np.argmax(dm), dm.shape)
    target = k[[i, j]]
    for ci in (0, len(centres)//2, len(centres)-1):
        for T in target:
            ax2.plot(*np.stack([centres[ci], T]).T, color=COLMAP, linewidth=0.8, alpha=0.8)
    ax2.scatter(*target.T, s=34, c=COLMAP, depthshade=False, zorder=5)

    allp = np.vstack([sample, centres])  # frame the subject together with the camera arc
    mid, half = (allp.min(0) + allp.max(0)) / 2, (allp.max(0) - allp.min(0)).max() / 2
    for setter, m in ((ax2.set_xlim, mid[0]), (ax2.set_ylim, mid[1]), (ax2.set_zlim, mid[2])):
        setter(m - half, m + half)
    ax2.view_init(elev=26, azim=-72)
    ax2.set_axis_off()
    ax2.text2D(0.5, 0.06,
               f"{len(centres)} recovered camera poses (R, t) around the subject; "
               f"grey = reconstructed points, orange = lines of sight to two shared points",
               transform=ax2.transAxes, ha="center", fontsize=7.6, color=INK_2)

    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # A 3-D axes reserves far more room than its contents occupy, so the
    # saved figure carries a wide band of blank page. Crop to the ink.
    from PIL import Image
    im = Image.open(path).convert("RGB")
    a = np.asarray(im)
    ink = np.where((a < 247).any(axis=2))
    if ink[0].size:
        pad = 14
        im = im.crop((max(0, ink[1].min()-pad), max(0, ink[0].min()-pad),
                      min(a.shape[1], ink[1].max()+pad), min(a.shape[0], ink[0].max()+pad)))
        im.save(path)
    print(f"  {path.relative_to(ROOT)}  ({im.size[0]}x{im.size[1]})")


# ------------------------------------------------- three-phase workflow
def workflow_figure(path: Path) -> None:
    """
    The project workflow as three phases: image preparation, point-cloud
    generation, evaluation and comparison.

    Laid out after the convention used in the photogrammetry literature --
    numbered phase labels down the left, the flow down the centre, outputs
    noted on the right. Two departures from that convention are deliberate
    and are drawn rather than omitted: there is no ground-control branch,
    because no control points were surveyed, and the evaluation phase has
    two tracks rather than one, because this project compares against both
    measured geometry and an external system.

    Arial is requested first and Liberation Sans, which is metrically
    identical, is the fallback on machines without it.
    """
    ARIAL = ["Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]
    BAND = {"prep": "#eef2f7", "cloud": "#e8e8e6", "eval": "#eef2f7"}

    fig, ax = plt.subplots(figsize=(8.6, 14.8))
    ax.set_xlim(0, 100); ax.set_ylim(-2, 148); ax.axis("off")
    ax.invert_yaxis()

    def band(y0, y1, fill):
        ax.add_patch(FancyBboxPatch((24, y0), 75, y1 - y0,
                                    boxstyle="square,pad=0", facecolor=fill,
                                    edgecolor="none", zorder=0))

    def phase(y, n, title):
        ax.add_patch(FancyBboxPatch((1, y - 3.2), 20, 6.4, boxstyle="square,pad=0",
                                    facecolor="#d9d9d9", edgecolor="none"))
        ax.text(2.5, y, f"{n}.  {title}", va="center", ha="left",
                fontsize=10, weight="bold", family=ARIAL, color=INK)
        ax.annotate("", xy=(25.2, y), xytext=(21.5, y),
                    arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.4))

    BX, BW = 27, 39                      # flow column
    def box(y, text, sub=None, h=4.6, bold=False, fill="white", edge=INK):
        ax.add_patch(FancyBboxPatch((BX, y - h/2), BW, h, boxstyle="square,pad=0",
                                    facecolor=fill, edgecolor=edge, linewidth=1.1))
        ax.text(BX + BW/2, y - (0.8 if sub else 0), text, ha="center", va="center",
                fontsize=8.8 if not bold else 9.2, weight="bold" if bold else "normal",
                family=ARIAL, color=INK)
        if sub:
            ax.text(BX + BW/2, y + 1.3, sub, ha="center", va="center",
                    fontsize=6.9, family=ARIAL, color=INK_2)

    def down(y0, y1):
        ax.annotate("", xy=(BX + BW/2, y1), xytext=(BX + BW/2, y0),
                    arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.2))

    def note(y, text, h=4.6):
        ax.add_patch(FancyBboxPatch((68, y - h/2), 30, h, boxstyle="square,pad=0",
                                    facecolor="white", edgecolor=INK_2, linewidth=0.9))
        ax.text(83, y, text, ha="center", va="center", fontsize=7.1, family=ARIAL, color=INK)
        ax.annotate("", xy=(67.5, y), xytext=(BX + BW, y),
                    arrowprops=dict(arrowstyle="-|>", color=INK_2, linewidth=0.9))

    # ---------------- phase 1 ----------------
    band(2, 24, BAND["prep"]); phase(12, 1, "Image\npreparation")
    box(6, "Acquire photographs", "phone camera, or benchmark dataset", h=5.4)
    down(8.7, 11.3)
    box(14, "Resolve intrinsics K",
        "explicit flags → sidecar → EXIF → 55° fallback", h=5.4)
    ax.add_patch(FancyBboxPatch((BX, 18.5), BW, 4.4, boxstyle="square,pad=0",
                                facecolor="none", edgecolor=INK_2, linewidth=1.0,
                                linestyle=(0, (4, 2.5))))
    ax.text(BX + BW/2, 20.7, "no ground control points surveyed:\nscale remains arbitrary",
            ha="center", va="center", fontsize=7.0, style="italic", family=ARIAL, color=INK_2)
    down(16.7, 18.3); down(22.9, 27.7)

    # ---------------- phase 2 ----------------
    band(26, 115, BAND["cloud"]); phase(66, 2, "Point cloud\ngeneration")
    steps = [
        (30, "Feature extraction (SIFT)", None, "keypoints + descriptors"),
        (36, "Matching + ratio test", None, None),
        (42, "Geometric verification", "RANSAC on the fundamental matrix", None),
        (48, "Track construction", "union-find over verified pairs", None),
        (54, "Two-view seed", None, "fixes gauge: R = I, t = 0,\nunit baseline"),
    ]
    for y, t, sub, nt in steps:
        box(y, t, sub, h=5.0 if sub else 4.6)
        if nt: note(y, nt)
    for a, b in zip([32.5, 38.4, 44.6, 50.6], [35.6, 39.6, 45.6, 51.6]):
        down(a, b)

    down(56.4, 59.4)
    loop = [(62, "Resectioning (PnP)"), (68, "Triangulation"), (74, "Bundle adjustment")]
    for y, t in loop:
        box(y, t)
    down(64.4, 65.6); down(70.4, 71.6)
    xl = BX - 4.5
    ax.plot([BX, xl], [74, 74], color=INK, linewidth=1.2)
    ax.plot([xl, xl], [74, 62], color=INK, linewidth=1.2)
    ax.annotate("", xy=(BX, 62), xytext=(xl, 62),
                arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.2))
    ax.text(xl - 1.2, 68, "next\ncamera", ha="right", va="center",
            fontsize=7.2, style="italic", family=ARIAL, color=INK_2)

    ax.plot([BX - 2, BX + BW + 2], [78.4, 78.4], color=INK_2, linewidth=1.0,
            linestyle=(0, (4, 2.5)))
    ax.text(BX + BW + 3, 78.4, "once, after the loop ends", va="center", ha="left",
            fontsize=7.0, style="italic", family=ARIAL, color=INK_2)
    down(76.4, 78.0)

    box(82, "Outlier + depth filters"); down(84.4, 85.6)
    box(88, "Final bundle adjustment"); down(90.4, 91.6)
    box(94, "Degeneracy check", None, h=4.6)
    note(94, "camera-centre span\nvs unit baseline")
    down(96.4, 99.2)

    # outputs
    box(102, "SPARSE CLOUD + CAMERA POSES", None, h=5.4, bold=True, fill="#dce9f9", edge=OURS)
    ax.text(BX + BW + 2.0, 102, "the principal\nresult", ha="left", va="center",
            fontsize=7.4, style="italic", family=ARIAL, color=INK_2)
    down(104.7, 108.0)
    box(111, "Dense MVS", "rectify → stereo → fuse + clean → Poisson",
        h=5.4, fill="#fdeee7", edge=COLMAP)
    note(111, "dense cloud, mesh")
    down(113.7, 117.0)

    # ---------------- phase 3 ----------------
    band(117, 145, BAND["eval"]); phase(129, 3, "Evaluation and\ncomparison")
    box(121, "Umeyama similarity alignment", "camera centres \u2192 reference frame", h=5.4)
    down(123.7, 125.4)
    half = BW / 2 - 0.8
    for x0, title, body in (
        (BX, "vs GROUND TRUTH",
         "4 datasets with ground truth\nrotation \u00b7 centre error\ncompleteness \u00b7 F-score"),
        (BX + BW / 2 + 0.8, "vs COLMAP",
         "13 datasets, same inputs\nregistration \u00b7 agreement\ncapture geometry"),
    ):
        ax.add_patch(FancyBboxPatch((x0, 125.8), half, 8.4, boxstyle="square,pad=0",
                                    facecolor="white", edgecolor=INK, linewidth=1.1))
        ax.text(x0 + half / 2, 127.6, title, ha="center", va="center",
                fontsize=7.6, weight="bold", family=ARIAL, color=INK)
        ax.text(x0 + half / 2, 131.2, body, ha="center", va="center",
                fontsize=6.3, family=ARIAL, color=INK_2)
    ax.annotate("", xy=(BX + BW / 2, 137.8), xytext=(BX + BW / 2, 134.6),
                arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.2))
    ax.add_patch(FancyBboxPatch((BX, 137.8), BW, 5.2, boxstyle="square,pad=0",
                                facecolor="white", edgecolor=INK, linewidth=1.1))
    ax.text(BX + BW / 2, 140.4, "Subject-coverage check\nreproject points into the photographs",
            ha="center", va="center", fontsize=6.9, family=ARIAL, color=INK)

    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
