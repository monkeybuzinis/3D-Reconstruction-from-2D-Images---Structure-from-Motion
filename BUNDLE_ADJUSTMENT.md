# Bundle Adjustment, in detail

A from-scratch walkthrough of `sfm/ba/so3.py`, `sfm/ba/jacobian.py`, and `sfm/ba/lm.py` — the math behind each piece, why it's structured that way, and how it maps back to the code. Companion to [ARCHITECTURE.md](ARCHITECTURE.md) (where this fits in the pipeline) and [PLAN.md](PLAN.md) (when it was built).

## 1. What problem BA is actually solving

Every camera pose and every 3D point up to this point was estimated from small local pieces of evidence (a pair for two-view, a PnP solve using whatever points existed so far). Each of those steps has some error, and errors compound. Bundle Adjustment is the step that looks at **every observation across every camera at once** and asks: "if I nudge every camera pose and every 3D point simultaneously, can I make the total squared reprojection error smaller?" It's a large nonlinear least-squares problem:

```
minimize over {R_i, t_i for each camera i} and {X_j for each point j}:
    sum over all observations (i,j) of || project(K_i, R_i, t_i, X_j) - uv_observed ||^2
```

With 11 cameras and ~7,500 points, that's roughly `11*6 + 7500*3 ~= 22,566` unknowns and `~25,000*2 = 50,000` residual equations. Too large to solve in closed form — it's solved iteratively: linearize around the current estimate, solve a linear system for a small update, apply it, repeat.

## 2. Parameterizing rotation: why axis-angle, not the 3x3 matrix

A rotation matrix has 9 numbers but only 3 degrees of freedom (it must stay orthogonal, `R^T R = I`, `det(R) = 1`). If an optimizer freely adjusts all 9 entries, it immediately breaks that constraint. The fix, in [sfm/ba/so3.py](sfm/ba/so3.py), is the **axis-angle (so(3)) representation**: a 3-vector `omega` where the *direction* is the rotation axis and the *magnitude* `theta = |omega|` is the rotation angle. Exactly 3 numbers for 3 true degrees of freedom — nothing to violate.

Converting `omega` to a matrix is the **Rodrigues formula**:

```
R = I + sin(theta)*K + (1 - cos(theta))*K^2      where K = skew(omega / theta)
```

`skew(v)` is the 3x3 matrix such that `skew(v) @ w` equals the cross product `v x w`. This formula is the matrix exponential `R = exp([omega]_x)` — rotating by angle `theta` around an axis is exactly what you get from exponentiating the skew matrix of that axis, which is why this is called "the so(3) exponential map."

This was validated against `cv2.Rodrigues` on 2,100 random rotations before anything was built on top of it — max difference was `8.9e-16` (floating-point noise, i.e. identical). See `scripts/verify_ba_jacobian.py`'s companion check in the conversation history, or re-run:

```python
import numpy as np, cv2
from sfm.ba.so3 import rodrigues
omega = np.random.default_rng(0).normal(size=3)
np.allclose(rodrigues(omega), cv2.Rodrigues(omega)[0])
```

**Important subtlety — how updates get composed.** Each LM iteration computes a small rotation *change* `delta_omega`, not a full new rotation. That update is applied as:

```
R_new = Rodrigues(delta_omega) @ R_old        (left-multiply)
```

This is a "left perturbation" — the correction is expressed in world coordinates and applied *before* the existing rotation. This isn't arbitrary: it's exactly what the Jacobian derivation below assumes. Applying it the other way (`R_old @ Rodrigues(delta_omega)`, a "right"/local perturbation) would need a different derivative formula. Convention consistency between "how you differentiate" and "how you apply the update" is the easiest place to introduce a silent bug in hand-written BA — get it wrong and the optimizer doesn't crash, it just quietly converges to nonsense.

## 3. Deriving the analytic Jacobian

For one observation, the pipeline from a 3D point to a predicted pixel is:

```
Y  = R @ X              (rotate world point into camera orientation, no shift yet)
Xc = Y + t              (shift into camera coordinates)
u  = fx*Xc.x/Xc.z + cx
v  = fy*Xc.y/Xc.z + cy
```

We need the derivative of `(u,v)` with respect to 9 things: 3 rotation params, 3 translation params, 3 point coordinates.

**Point derivative** is the easy one: `X` only enters through `Y = R@X`, so `dY/dX = R` directly.

**Translation derivative** is even easier: `t` enters `Xc` by pure addition, so `dXc/dt = I` (nudging `t` by `delta_t` nudges `Xc` by exactly `delta_t`).

**Rotation derivative** is the interesting one. Apply a small perturbation `delta_omega` the way agreed above: `R' = Rodrigues(delta_omega) @ R ~= (I + skew(delta_omega)) @ R` for small `delta_omega` (the first-order Taylor expansion of Rodrigues near zero). Then:

```
Y' = R'X = Y + skew(delta_omega) @ Y
```

Using the identity `skew(a)@b = a x b = -b x a = -skew(b)@a`, that becomes:

```
Y' = Y - skew(Y) @ delta_omega     =>     dY/d(delta_omega) = -skew(Y)
```

Then the pinhole projection derivative (`d(u,v)/dXc`) is standard calculus on `u = fx*x/z + cx`:

```
du/dx = fx/z,  du/dy = 0,     du/dz = -fx*x/z^2
dv/dx = 0,     dv/dy = fy/z,  dv/dz = -fy*y/z^2
```

Chaining these via the chain rule gives exactly what's in `batch_project_and_jacobian` ([sfm/ba/jacobian.py](sfm/ba/jacobian.py)):

```
J_cam   = d(u,v)/dXc @ [ -skew(Y)  |  I ]      (2x6: 3 rotation cols + 3 translation cols)
J_point = d(u,v)/dXc @ R                        (2x3)
```

That's the entire derivation — no numerical approximation anywhere, every entry is an exact closed-form expression. The code computes this for **all ~25,000 observations simultaneously** using `einsum`, rather than looping in Python, which is what keeps a full Jacobian assembly fast.

**Why the finite-difference check mattered before trusting this.** It's very easy to get a sign wrong (`+skew(Y)` instead of `-skew(Y)`), or to mix up left vs. right perturbation, or differentiate w.r.t. the wrong intermediate variable. None of those bugs crash — they just make the optimizer converge to nonsense, slowly, with no obvious error message. `scripts/verify_ba_jacobian.py` perturbs each of the 9 parameters by `+-1e-6` on 30 real triangulated points, recomputes the residual by hand each time, and compares `(f(x+eps) - f(x-eps)) / 2eps` against what the formula above predicts. Getting `3e-7` agreement (right at the floating-point noise floor for this step size) is about as strong a confirmation as is possible that the derivation is correct.

## 4. Gauss-Newton normal equations, and why they're expensive

Once residuals `r` and Jacobian `J` are known, the standard nonlinear least-squares step solves:

```
(J^T J) delta = -J^T r
```

The problem: with ~22,566 unknowns, `J^T J` is a 22,566 x 22,566 matrix. Even though it's sparse, naively solving a system that size every iteration is what makes generic solvers slow — this is exactly what scipy's `least_squares` has to grapple with, even with a supplied sparsity pattern.

## 5. The Schur complement — the trick that makes BA fast

The key structural fact about `J^T J` in bundle adjustment: **a single observation only ever touches one camera's 6 parameters and one point's 3 parameters.** No residual depends on two different cameras, or two different points, at once. Written as a block matrix ordered as [camera params | point params]:

```
J^T J = [ B   E ]        B = camera-camera block  -> block-diagonal (one 6x6 block per camera)
        [ E^T C ]        C = point-point block    -> block-diagonal (one 3x3 block per point)
                          E = camera-point coupling -> sparse
```

`B` and `C` being block-diagonal is the whole trick: each point's 3x3 block `C_p` can be inverted **completely independently of every other point**, trivially and cheaply. The Schur complement uses that to eliminate every point-parameter unknown algebraically, collapsing the system down to one that only involves camera parameters:

```
S = B - E * C^-1 * E^T                    (a dense but tiny ~66x66 matrix for 11 cameras)
S @ delta_cam = b - E * C^-1 * b_point
```

Solve that small dense system (trivial — `np.linalg.solve` on a 66x66 matrix is instant), then **back-substitute** to recover each point's own update using only its own `C_p^-1` and the now-known `delta_cam`:

```
delta_point[p] = C_p^-1 * (b_point[p] - sum over cameras seeing p of E_block.T @ delta_cam[camera])
```

This is implemented in [sfm/ba/lm.py](sfm/ba/lm.py)'s `_build_normal_equations` (assembling `B`, `C`, `E` via vectorized `einsum` + `np.add.at` scatter-accumulation, since different observations belong to different cameras/points) and `_solve_and_apply` (the actual `np.linalg.solve` + back-substitution). The one part that can't be fully vectorized is grouping "which free cameras observe this point" per point — a small Python loop over observations, but cheap relative to the linear algebra.

This is *the* reason the hand-written version ended up faster than scipy: scipy's `trf` method does generic sparse-matrix bookkeeping for an arbitrary sparsity pattern; here, the exact block structure is known in advance and exploited directly.

## 6. Levenberg-Marquardt: why not just Gauss-Newton

Plain Gauss-Newton (solving `J^T J delta = -J^T r` directly) can overshoot badly if the current estimate is far from a good linearization point — the linear model is only an approximation, and a big step based on a bad approximation can make things worse, not better. **Levenberg-Marquardt** fixes this by adding a damping term to the diagonal before solving:

```
(J^T J + lambda * diag(J^T J)) delta = -J^T r
```

Implemented as `B[i] += damping * diag(diag(B[i]))` and the same for `C[p]` in `_build_normal_equations`. Two limits make the intuition clear:

- **lambda -> 0**: reduces to plain Gauss-Newton — full, fast, "trust the linear model completely" steps.
- **lambda -> infinity**: the diagonal term dominates, and the step direction converges to a tiny step along `-J^T r` (plain gradient descent) — slow but far more robust when the current model is untrustworthy.

The loop in `hand_bundle_adjust` is the standard **accept/reject** LM strategy: build the system, tentatively take the step, actually recompute the true (nonlinear) cost with the new parameters, and:

- If cost went **down**: accept the step, and *loosen* damping (`lambda /= 10`) — the linear model predicted well, so be more aggressive next time.
- If cost went **up**: reject the step (snapshot/restore the exact previous state), and *tighten* damping (`lambda *= 10`) — the model was untrustworthy at this scale, so shrink toward safer gradient-descent-like steps and retry.

This is why a real run shows damping shrinking smoothly (`1e-4 -> 1e-5 -> 1e-6 -> 1e-7 -> 1e-8`) across 5 accepted iterations on the two-view scene — each step worked well, so LM kept getting more confident and taking bigger, cheaper-to-verify steps, converging in well under a second.

## 7. How it all threads together

`sfm/recon/incremental.py`'s `run_incremental_sfm` takes a `bundle_adjust_fn` parameter — either `sfm.ba.scipy_ba.bundle_adjust` or `sfm.ba.lm.hand_bundle_adjust` — and calls it identically after every camera registration. Both fix camera 0's pose (the **gauge fix**: since reprojection error is unchanged by rigidly rotating/translating the *entire* scene at once, without pinning one camera the optimizer has infinitely many equally-good answers to choose between and no way to pick one). Swapping the function is the only difference between `scripts/verify_incremental.py` (scipy) and `scripts/verify_hand_ba.py` (hand-written) — and they land on essentially the same final answer (median reprojection error 0.18px either way), which is the strongest evidence that the from-scratch math is correct, not just "runs without crashing."

## Results recap

| | scipy BA | hand-written LM BA |
|---|---|---|
| Two-view BA time | 2.4s | 0.3s (8x faster) |
| Full 11-camera pipeline (tracks excluded) | ~99s | ~50s (~2x faster) |
| Final median reprojection error | 0.18px | 0.18px |
| Final p95 reprojection error | 0.89px | 0.88px |

The speed gain comes entirely from exploiting the block-sparse structure via the Schur complement instead of relying on scipy's generic sparse least-squares machinery — the accuracy is (as expected) essentially identical, since both are solving the same optimization problem to convergence.
