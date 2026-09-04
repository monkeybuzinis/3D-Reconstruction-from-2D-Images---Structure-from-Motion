"""
Week-2 two-view geometry check on Fountain-P11.

OpenCV is used only for I/O, SIFT, and matching (allowed library boundary).
Hartley normalization, 8-point F, RANSAC, and E = K^T F K are implemented here.

This script is a sanity check, not the final sfm/ package. It uses Fountain/0000.png
and Fountain/0001.png — the usual seed pair for this dataset.

SUPERSEDED: this was the original prototype for the two-view seed stage.
Its logic now lives properly split up in the sfm/ package --
sfm/features/matching.py, sfm/geometry/fundamental.py, sfm/geometry/essential.py,
sfm/geometry/triangulation.py, sfm/recon/two_view.py -- with DLT triangulation
added (this file stops short of that). Kept here only as a historical record
of the first working version; scripts/verify_two_view.py is the current
equivalent to run.
"""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FOUNTAIN_DIR = Path(__file__).resolve().parent / "Fountain"

# Strecha Fountain-P11 intrinsics (same K for all 11 views; also on VERTEX_CAM in solution.graph)
K = np.array(
    [
        [2759.48, 0.0, 1520.69],
        [0.0, 2759.48, 1006.81],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def normalize_points(pts: np.ndarray):
    """
    Hartley Normalization: Translates centroid to origin and scales average distance to sqrt(2).
    Returns normalized points and 3x3 transformation matrix T.
    """
    centroid = np.mean(pts, axis=0)
    shifted = pts - centroid
    mean_dist = np.mean(np.linalg.norm(shifted, axis=1))
    scale = np.sqrt(2) / mean_dist

    T = np.array(
        [
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    pts_homo = np.column_stack([pts, np.ones(len(pts))])
    pts_norm = (T @ pts_homo.T).T

    return pts_norm[:, :2], T


def compute_fundamental_8point(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Computes Fundamental Matrix F using normalized 8-point algorithm with SVD."""
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)

    x1, y1 = pts1_norm[:, 0], pts1_norm[:, 1]
    x2, y2 = pts2_norm[:, 0], pts2_norm[:, 1]

    A = np.column_stack(
        [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, np.ones(len(pts1))]
    )

    _, _, Vt = np.linalg.svd(A)
    F_norm = Vt[-1].reshape(3, 3)

    U, S, Vt_f = np.linalg.svd(F_norm)
    S[2] = 0.0
    F_norm = U @ np.diag(S) @ Vt_f

    F = T2.T @ F_norm @ T1
    return F / F[2, 2]


def sampson_distance(pts1: np.ndarray, pts2: np.ndarray, F: np.ndarray) -> np.ndarray:
    """First-order geometric error (Sampson distance) for point pairs."""
    N = len(pts1)
    p1 = np.column_stack([pts1, np.ones(N)])
    p2 = np.column_stack([pts2, np.ones(N)])

    F_p1 = (F @ p1.T).T
    FT_p2 = (F.T @ p2.T).T

    p2_F_p1 = np.sum(p2 * F_p1, axis=1)

    denom = F_p1[:, 0] ** 2 + F_p1[:, 1] ** 2 + FT_p2[:, 0] ** 2 + FT_p2[:, 1] ** 2
    return (p2_F_p1**2) / (denom + 1e-8)


def ransac_fundamental_matrix(pts1: np.ndarray, pts2: np.ndarray, threshold=1.0, max_iters=2000):
    """Estimates F and an inlier mask using RANSAC + Sampson distance."""
    best_inliers = np.array([], dtype=int)
    best_F = None
    N = len(pts1)

    rng = np.random.default_rng(42)
    for _ in range(max_iters):
        indices = rng.choice(N, 8, replace=False)
        sample1, sample2 = pts1[indices], pts2[indices]

        try:
            F_cand = compute_fundamental_8point(sample1, sample2)
            errors = sampson_distance(pts1, pts2, F_cand)
            inliers = np.where(errors < threshold)[0]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_F = F_cand
        except np.linalg.LinAlgError:
            continue

    if len(best_inliers) >= 8:
        best_F = compute_fundamental_8point(pts1[best_inliers], pts2[best_inliers])

    return best_F, best_inliers


def load_fountain_pair(name0: str = "0000.png", name1: str = "0001.png"):
    img0_path = FOUNTAIN_DIR / name0
    img1_path = FOUNTAIN_DIR / name1
    if not img0_path.is_file() or not img1_path.is_file():
        raise FileNotFoundError(
            f"Expected {name0} and {name1} in {FOUNTAIN_DIR}"
        )

    img0 = cv2.imread(str(img0_path), cv2.IMREAD_COLOR)
    img1 = cv2.imread(str(img1_path), cv2.IMREAD_COLOR)
    if img0 is None or img1 is None:
        raise RuntimeError("OpenCV failed to read Fountain images")
    return img0, img1


def sift_match(img0: np.ndarray, img1: np.ndarray, ratio: float = 0.75):
    """OpenCV SIFT + FLANN + Lowe ratio test. Returns pixel correspondences and match objects."""
    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp0, desc0 = sift.detectAndCompute(gray0, None)
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    if desc0 is None or desc1 is None:
        raise RuntimeError("SIFT produced no descriptors")

    index_params = dict(algorithm=1, trees=5)  # FLANN KD-tree
    search_params = dict(checks=64)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn = flann.knnMatch(desc0, desc1, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < 8:
        raise RuntimeError(f"Too few matches after ratio test: {len(good)}")

    pts0 = np.array([kp0[m.queryIdx].pt for m in good], dtype=np.float64)
    pts1 = np.array([kp1[m.trainIdx].pt for m in good], dtype=np.float64)
    return pts0, pts1, kp0, kp1, good


if __name__ == "__main__":
    img0, img1 = load_fountain_pair()
    pts0, pts1, kp0, kp1, matches = sift_match(img0, img1)

    F_est, inlier_idx = ransac_fundamental_matrix(pts0, pts1, threshold=1.5, max_iters=3000)
    if F_est is None:
        raise RuntimeError("RANSAC failed to estimate F")

    E_est = K.T @ F_est @ K
    U_e, S_e, Vt_e = np.linalg.svd(E_est)
    m = (S_e[0] + S_e[1]) / 2.0
    E_est = U_e @ np.diag([m, m, 0.0]) @ Vt_e

    inlier_errors = sampson_distance(pts0[inlier_idx], pts1[inlier_idx], F_est)

    print("Fountain pair: 0000.png / 0001.png")
    print(f"Image size: {img0.shape[1]} x {img0.shape[0]}")
    print("\n--- Phase 1 Matrix Estimation ---")
    print(f"Total Matches: {len(pts0)}")
    print(
        f"RANSAC Inliers: {len(inlier_idx)} "
        f"({len(inlier_idx) / len(pts0) * 100:.1f}%)"
    )
    print(f"Median Sampson error (inliers): {np.median(inlier_errors):.4f} px^2")
    print("\nEstimated Fundamental Matrix F:\n", np.round(F_est, 6))
    print("\nEstimated Essential Matrix E:\n", np.round(E_est, 6))
    print("\nE singular values (should be ~[s, s, 0]):", np.round(S_e, 6))

    inlier_matches = [matches[i] for i in inlier_idx]
    draw_n = min(50, len(inlier_matches))
    inlier_img = cv2.drawMatches(
        img0,
        kp0,
        img1,
        kp1,
        inlier_matches[:draw_n],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    out_path = Path(__file__).resolve().parent / "fountain_0000_0001_inliers.png"
    plt.figure(figsize=(15, 7))
    plt.imshow(cv2.cvtColor(inlier_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Fountain 0000–0001 RANSAC inliers ({len(inlier_idx)} total, showing {draw_n})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved match preview: {out_path}")
