"""
Per-stage figures for Section 3.4 (point cloud generation).

Each figure is produced from a real run of the pipeline on Fountain-P11, not
from a schematic: the epipolar panel draws the correspondences RANSAC actually
kept, the cheirality panel counts the points each of the four candidate poses
actually places in front of both cameras, and so on.

Run from the repo root:
    .venv/bin/python3 scripts/make_stage_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json

from sfm.ba.jacobian import scene_residuals_and_jacobians
from sfm.ba.lm import hand_bundle_adjust
from sfm.features.tracks import build_tracks
from sfm.geometry.essential import (cheirality_check, decompose_essential,
                                    essential_from_fundamental)
from sfm.geometry.fundamental import ransac_fundamental_matrix
from sfm.geometry.triangulation import triangulate_dlt
from sfm.io.fountain import FOUNTAIN_K, default_fountain_dir, load_images
from sfm.recon.incremental import run_incremental_sfm

OUT = ROOT / "evaluation" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

OURS = "#2a78d6"      # categorical slot 1  (kept / after)
OTHER = "#eb6834"     # categorical slot 2  (rejected / before)
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


def _recessive(ax, xgrid=False, ygrid=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")
    return path


# ------------------------------------------------------------ shared state

def two_view_matches(img0, img1):
    """Ratio-test matches between two images, as the pipeline computes them."""
    sift = cv2.SIFT_create()
    g0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) if img0.ndim == 3 else img0
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    kp0, d0 = sift.detectAndCompute(g0, None)
    kp1, d1 = sift.detectAndCompute(g1, None)
    matcher = cv2.BFMatcher()
    pairs = matcher.knnMatch(d0, d1, k=2)
    good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
    pts0 = np.float64([kp0[m.queryIdx].pt for m in good])
    pts1 = np.float64([kp1[m.trainIdx].pt for m in good])
    return pts0, pts1


# ------------------------------------------------------- 3.4.2  epipolar

def epipolar_figure(img0, img1, pts0, pts1, F, inliers):
    h, w = img0.shape[:2]
    canvas = np.hstack([cv2.cvtColor(img0, cv2.COLOR_BGR2RGB),
                        cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)])

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4))

    # (a) every ratio-test match, split by what RANSAC decided
    ax = axes[0]
    ax.imshow(canvas)
    # one uniform sample of ALL matches, so the picture keeps the true
    # kept/rejected proportion instead of over-drawing the smaller class
    rng = np.random.default_rng(0)
    sel = rng.choice(len(pts0), size=min(160, len(pts0)), replace=False)
    for k in sel:
        col = OURS if inliers[k] else OTHER
        ax.plot([pts0[k, 0], pts1[k, 0] + w], [pts0[k, 1], pts1[k, 1]],
                color=col, lw=0.7, alpha=0.55 if inliers[k] else 0.95,
                zorder=2 if inliers[k] else 3)
    n_out = int((~inliers).sum())
    ax.set_title(f"(a)  a random 160 of the {len(pts0)} ratio-test matches; "
                 f"RANSAC kept {int(inliers.sum())} and rejected {n_out}",
                 fontsize=9, loc="left")
    ax.axis("off")

    # (b) epipolar lines for a sample of the surviving correspondences
    ax = axes[1]
    ax.imshow(canvas)
    idx = np.where(inliers)[0]
    sample = idx[np.linspace(0, len(idx) - 1, 14).astype(int)]
    for k in sample:
        x = np.array([pts0[k, 0], pts0[k, 1], 1.0])
        line = F @ x                       # epipolar line in the right image
        a, b, c = line
        xs = np.array([0, w])
        ys = -(a * xs + c) / b
        ax.plot(xs + w, ys, color=OURS, lw=0.7, alpha=0.9)
        ax.plot(pts0[k, 0], pts0[k, 1], "o", ms=3, mfc="none", mec=OURS, mew=0.9)
        ax.plot(pts1[k, 0] + w, pts1[k, 1], "o", ms=3, mfc="none", mec=OURS, mew=0.9)
    ax.set_xlim(0, 2 * w)
    ax.set_ylim(h, 0)
    ax.set_title("(b)  epipolar lines F x for 14 kept correspondences; "
                 "each match lies on its own line", fontsize=9, loc="left")
    ax.axis("off")

    fig.tight_layout()
    return _save(fig, "fig_epipolar.png")


# ------------------------------------------------------ 3.4.3  cheirality

def cheirality_figure(K, pts0, pts1, E):
    cands = decompose_essential(E)
    counts = []
    for R, t in cands:
        P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = K @ np.hstack([R, t.reshape(3, 1)])
        X = triangulate_dlt(P0, P1, pts0, pts1)
        d0 = X[:, 2]
        Xc = (R @ X.T).T + t.reshape(1, 3)
        d1 = Xc[:, 2]
        counts.append([int(np.sum(d0 > 0)), int(np.sum(d1 > 0)),
                       int(np.sum((d0 > 0) & (d1 > 0)))])
    counts = np.array(counts)

    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    x = np.arange(4)
    width = 0.26
    labels = ["in front of camera 1", "in front of camera 2", "in front of both"]
    cols = [GRID, "#9aa3ad", OURS]
    for j in range(3):
        bars = ax.bar(x + (j - 1) * width, counts[:, j], width * 0.92,
                      color=cols[j], label=labels[j], zorder=3)
        if j == 2:
            for b, v in zip(bars, counts[:, 2]):
                ax.annotate(str(v), (b.get_x() + b.get_width() / 2, v),
                            ha="center", va="bottom", fontsize=8, color=INK,
                            xytext=(0, 2), textcoords="offset points")
    best = int(np.argmax(counts[:, 2]))
    ax.set_xticks(x)
    ax.set_xticklabels([f"candidate {i+1}" + ("\n(selected)" if i == best else "")
                        for i in range(4)])
    ax.set_ylabel("correspondences")
    ax.set_title("The four decompositions of E, scored by cheirality",
                 fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left",
              bbox_to_anchor=(0.0, 1.0), ncol=3, columnspacing=1.4,
              handlelength=1.2)
    ax.set_ylim(0, counts.max() * 1.22)
    _recessive(ax)
    fig.tight_layout()
    return _save(fig, "fig_cheirality.png"), counts, best


# --------------------------------------------------------- 3.4.5  tracks

def tracks_figure(tracks):
    lengths = np.array([len(t.observations) for t in tracks])
    bins = np.arange(2, lengths.max() + 1)
    hist, _ = np.histogram(lengths, bins=np.append(bins, bins[-1] + 1))

    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    ax.bar(bins, hist, width=0.72, color=OURS, zorder=3)
    for b, v in zip(bins, hist):
        if v:
            ax.annotate(f"{v:,}", (b, v), ha="center", va="bottom", fontsize=7.5,
                        color=INK_2, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(bins)
    ax.set_xlabel("number of images the track appears in")
    ax.set_ylabel("tracks")
    ax.set_title(f"Track length over the 11 Fountain-P11 images "
                 f"({len(tracks):,} tracks)", fontsize=9.5, loc="left")
    _recessive(ax)
    fig.tight_layout()
    return _save(fig, "fig_tracks.png"), lengths


# ------------------------------------------------------------ 3.4.7  PnP

def pnp_guard_figure():
    """The measured effect of the refit guard in ransac_pnp, from the ablation."""
    data = json.loads((ROOT / "evaluation" / "metrics"
                       / "ablation_pnp_guard_herzjesu.json").read_text())
    by = {d["label"].split(" (")[0]: d for d in data}
    off, on = by["without the guard"], by["with the guard"]

    panels = [
        ("cameras registered\n(of 23)", off["registered"], on["registered"], "{:.0f}"),
        ("observations kept by the\noutlier filter (%)",
         off["keep_rate_pct"], on["keep_rate_pct"], "{:.1f}"),
        ("median reprojection\nerror (px)",
         off["median_reproj_px"], on["median_reproj_px"], "{:.3f}"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5))
    for ax, (title, a, b, fmt) in zip(axes, panels):
        bars = ax.bar([0, 1], [a, b], width=0.56, color=[OTHER, OURS], zorder=3)
        for bar, v in zip(bars, (a, b)):
            ax.annotate(fmt.format(v), (bar.get_x() + bar.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=8.5, color=INK,
                        xytext=(0, 2), textcoords="offset points")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["without", "with"], fontsize=8.5)
        ax.set_title(title, fontsize=8.5, loc="center", color=INK_2)
        ax.set_ylim(0, max(a, b) * 1.3)
        _recessive(ax)
    fig.suptitle("The refit guard in ransac_pnp, measured on Herz-Jesu-P25",
                 fontsize=9.5, x=0.012, ha="left", y=1.04)
    fig.tight_layout()
    return _save(fig, "fig_pnp_guard.png")


# ------------------------------------------------------------- 3.4.8  BA

def _pixel_errors(scene):
    res, _, _, _, _ = scene_residuals_and_jacobians(scene)
    return np.linalg.norm(res, axis=1)


def ba_figure(images, keypoints, tracks):
    """
    The claim in the text is that bundle adjustment inside the loop is what
    keeps later resectioning alive. Test it: run the same pipeline twice on
    the same tracks, once with the adjuster and once with a no-op in its
    place, and compare what registers and how well it reprojects.
    """
    def no_ba(scene, **kwargs):
        return {"cost_before": 0.0, "cost_after": 0.0,
                "n_observations": len(scene.observations)}

    runs = {}
    for label, fn in (("without BA in the loop", no_ba),
                      ("with BA in the loop", hand_bundle_adjust)):
        try:
            scene, log = run_incremental_sfm(
                images, FOUNTAIN_K, keypoints, tracks, seed=(0, 1),
                ba_max_nfev=30, bundle_adjust_fn=fn,
            )
            runs[label] = (len(scene.cameras), _pixel_errors(scene))
        except Exception as exc:                      # registration collapsed
            print(f"  {label}: run failed -- {exc}")
            runs[label] = (0, np.array([np.nan]))
        print(f"  {label}: {runs[label][0]} cameras, "
              f"median {np.nanmedian(runs[label][1]):.3f} px")

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7),
                             gridspec_kw={"width_ratios": [1, 1.7]})

    ax = axes[0]
    off, on = runs["without BA in the loop"][1], runs["with BA in the loop"][1]
    stats = [(np.nanmedian(off), np.nanmedian(on)),
             (np.nanpercentile(off, 95), np.nanpercentile(on, 95))]
    x = np.arange(2)
    for j, (col, lab) in enumerate(((OTHER, "no BA"), (OURS, "BA"))):
        vals = [stats[0][j], stats[1][j]]
        bars = ax.bar(x + (j - 0.5) * 0.36, vals, 0.33, color=col, zorder=3)
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=8, color=INK,
                        xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x); ax.set_xticklabels(["median", "95th pct"], fontsize=8.5)
    ax.set_ylabel("reprojection error (px)")
    ax.set_ylim(0, stats[1][0] * 1.25)
    n_on = runs["with BA in the loop"][0]
    n_off = runs["without BA in the loop"][0]
    ax.set_title(f"both runs registered {n_off if n_off == n_on else n_off}"
                 f"{'' if n_off == n_on else ' vs ' + str(n_on)} of "
                 f"{len(images)} cameras", fontsize=8, loc="left", color=INK_2)
    _recessive(ax)

    ax = axes[1]
    pair = [runs["without BA in the loop"][1], runs["with BA in the loop"][1]]
    hi = float(np.nanpercentile(np.concatenate(pair), 98))
    bins = np.linspace(0, hi, 55)
    ax.hist(pair[0], bins=bins, color=OTHER, alpha=0.95,
            label="without BA in the loop", zorder=3)
    ax.hist(pair[1], bins=bins, color=OURS, alpha=0.75,
            label="with BA in the loop", zorder=4)
    ax.set_xlabel("reprojection error (pixels)")
    ax.set_ylabel("observations")
    ax.legend(frameon=False, fontsize=8)
    _recessive(ax)

    fig.suptitle("Bundle adjustment inside the registration loop, Fountain-P11",
                 fontsize=9.5, x=0.012, ha="left", y=1.04)
    fig.tight_layout()
    _save(fig, "fig_bundle_adjustment.png")
    return runs


if __name__ == "__main__":
    print("loading Fountain-P11 ...")
    images = load_images(default_fountain_dir())
    img0, img1 = images[0], images[1]

    print("matching the seed pair ...")
    pts0, pts1 = two_view_matches(img0, img1)
    F, inlier_idx = ransac_fundamental_matrix(pts0, pts1)
    inliers = np.zeros(len(pts0), dtype=bool)
    inliers[inlier_idx] = True          # RANSAC returns indices, not a mask
    print(f"  {len(pts0)} matches, {int(inliers.sum())} inliers")

    epipolar_figure(img0, img1, pts0, pts1, F, inliers)

    E = essential_from_fundamental(F, FOUNTAIN_K)
    _, counts, best = cheirality_figure(FOUNTAIN_K, pts0[inliers], pts1[inliers], E)
    print(f"  cheirality counts (both cameras): {counts[:, 2].tolist()}, "
          f"selected candidate {best + 1}")

    print("building tracks over all 11 images ...")
    keypoints, tracks = build_tracks(images)
    _, lengths = tracks_figure(tracks)
    print(f"  {len(tracks):,} tracks, mean length {lengths.mean():.2f}, "
          f"max {lengths.max()}")

    pnp_guard_figure()

    print("bundle-adjustment ablation (two full runs) ...")
    ba_figure(images, keypoints, tracks)
