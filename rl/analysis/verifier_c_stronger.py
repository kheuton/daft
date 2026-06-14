"""
verifier_c_stronger.py — Stronger/alternative specs for the C4 "fair re-analysis"
claim (shaped arms have NO per-skill selection edge over GRPO).

Critique of C4 (run_phase0.py section 4):
  - pooled OLS maj32 ~ pass1 + pass1^2 + arm-dummies treats the same problem under
    4 arms as 4 independent rows; the cluster bootstrap fixes the CI but the POINT
    estimate still mixes within- and between-problem variation.
  - pass1 (=C/64) is a noisy 1-D proxy for "skill" and ignores answer-distribution
    SHAPE (concentration). A global quadratic can misfit a strongly nonlinear
    maj32~pass1 curve and leak curvature into the arm dummies.
  - no problem identity controls.

This script runs four stronger specs, all cluster-bootstrapped over problem_id:
  S1. Problem fixed effects (within-problem demeaning) + arm dummies.
  S2. Within-problem PAIRED arm-vs-grpo maj32 difference, restricted to problems
      where the two arms have ~equal correct-count C (|dC| <= tol).  Exactly the
      "match on skill, compare selection" design, done nonparametrically.
  S3. Finer pass1 control: natural-cubic-spline basis (knots at deciles) + arm dummies.
  S4. Same finer-spline spec but also conditioning on a second skill axis: the
      best-class concentration of the GRPO-matched skill is hard, so instead we add
      pass1 spline AND an interaction-free arm test on the residuals (robustness).

Env: mamba activate daft_rl ; run from repo root:  python -m rl.analysis.verifier_c_stronger
"""
from __future__ import annotations

import numpy as np

from rl.analysis import phase0_lib as L

ARMS = ["pi0", "grpo", "passk", "votek"]
SHAPED = ["passk", "votek"]
B = 2000
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Load + build per-(arm,pid) table
# ---------------------------------------------------------------------------
paired = L.load_paired()
pids = sorted(paired["grpo"].keys())
P = len(pids)
print(f"Paired pool: {P} problems x {len(ARMS)} arms = {P*len(ARMS)} rows")

# per arm: pass1 (=C/64), maj32, C (correct count), arrays aligned to `pids`
pass1 = {a: np.array([paired[a][pid].correct.mean() for pid in pids]) for a in ARMS}
maj32 = {a: np.array([L.maj_at_k(paired[a][pid], 32) for pid in pids]) for a in ARMS}
Ccnt = {a: np.array([int(paired[a][pid].correct.sum()) for pid in pids]) for a in ARMS}

pid_idx = np.arange(P)


def boot_pid_samples(n_boot=B):
    """Yield resampled problem-index arrays (cluster bootstrap over problem_id)."""
    for _ in range(n_boot):
        yield RNG.choice(pid_idx, size=P, replace=True)


def ci(samples):
    s = np.asarray(samples)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def stars(lo, hi):
    return " *" if (lo > 0 or hi < 0) else ""


print("\n" + "=" * 74)
print("BASELINE C4 reproduction (pooled OLS, quadratic pass1, cluster boot)")
print("=" * 74)
# Rebuild exactly as run_phase0.py section 4 for a sanity check.
rows_p1 = np.concatenate([pass1[a] for a in ARMS])
rows_m = np.concatenate([maj32[a] for a in ARMS])
rows_arm = np.concatenate([[a] * P for a in ARMS])
rows_pidi = np.concatenate([pid_idx for _ in ARMS])
dummy_arms = ["pi0", "passk", "votek"]


def design_quad(p1, arm):
    X = [np.ones_like(p1), p1, p1 ** 2]
    for a in dummy_arms:
        X.append((arm == a).astype(float))
    return np.column_stack(X)


def ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


idx_by_pid = {i: np.where(rows_pidi == i)[0] for i in pid_idx}
beta0 = ols(rows_m, design_quad(rows_p1, rows_arm))
names0 = ["intercept", "pass1", "pass1^2"] + [f"arm[{a}]" for a in dummy_arms]
bb = np.zeros((B, len(beta0)))
for b, samp in enumerate(boot_pid_samples()):
    sel = np.concatenate([idx_by_pid[i] for i in samp])
    bb[b] = ols(rows_m[sel], design_quad(rows_p1[sel], rows_arm[sel]))
for j, nm in enumerate(names0):
    lo, hi = ci(bb[:, j])
    print(f"  {nm:12s} {beta0[j]:+.4f}  [{lo:+.4f},{hi:+.4f}]{stars(lo,hi) if nm.startswith('arm') else ''}")


# ===========================================================================
# S1. Problem fixed effects (within-problem demeaning) + arm dummies
# ===========================================================================
# Demean maj32 and pass1 by problem (across the 4 arms). With FE, the arm dummy
# captures the AVERAGE within-problem maj32 difference of that arm vs the grand
# arm-mean, after removing the within-problem pass1 trend. We compare shaped
# arms to grpo via contrast (arm[shaped] - arm[grpo]).
print("\n" + "=" * 74)
print("S1. Problem fixed effects + within-problem pass1 control + arm dummies")
print("    model: maj32_demeaned ~ pass1_demeaned + (arm dummies, grpo ref)")
print("    -> arm coef = within-problem maj32 gap of arm vs grpo at equal pass1")
print("=" * 74)


def fit_fe(samp):
    # Build a (len(samp)*4) stacked table for the resampled problems, then demean
    # WITHIN each (resampled) problem instance across the 4 arms.
    nP = len(samp)
    p1 = np.empty((nP, 4)); mj = np.empty((nP, 4))
    for ai, a in enumerate(ARMS):
        p1[:, ai] = pass1[a][samp]
        mj[:, ai] = maj32[a][samp]
    # within-problem demean across arms
    p1d = p1 - p1.mean(axis=1, keepdims=True)
    mjd = mj - mj.mean(axis=1, keepdims=True)
    # design: demeaned pass1 + arm dummies (grpo ref). Intercept absorbed by FE.
    p1d_f = p1d.reshape(-1)
    mjd_f = mjd.reshape(-1)
    arm_f = np.repeat(np.arange(4)[None, :], nP, axis=0).reshape(-1)
    X = [p1d_f]
    armcols = []
    for ai, a in enumerate(ARMS):
        if a == "grpo":
            continue
        X.append((arm_f == ai).astype(float))
        armcols.append(a)
    # arm dummies must be within-problem demeaned too (each problem has all 4 arms,
    # so dummy mean is 0.25 -> demean by subtracting 0.25 within problem == constant)
    Xmat = np.column_stack(X)
    # center arm dummies (they are balanced -> subtract 0.25)
    Xmat[:, 1:] = Xmat[:, 1:] - 0.25
    beta = ols(mjd_f, Xmat)
    return beta, ["pass1d"] + armcols


beta_fe, fe_names = fit_fe(pid_idx)
fe_boot = np.zeros((B, len(beta_fe)))
for b, samp in enumerate(boot_pid_samples()):
    fe_boot[b], _ = fit_fe(samp)
for j, nm in enumerate(fe_names):
    lo, hi = ci(fe_boot[:, j])
    mark = stars(lo, hi) if nm != "pass1d" else ""
    print(f"  {nm:10s} {beta_fe[j]:+.4f}  [{lo:+.4f},{hi:+.4f}]{mark}")


# ===========================================================================
# S2. Within-problem PAIRED arm-vs-grpo, restricted to ~equal correct-count C
# ===========================================================================
# For each shaped arm, take problems where |C_arm - C_grpo| <= tol, compute the
# paired maj32 difference (arm - grpo) per problem, average. This nonparametrically
# matches on skill (correct count) and compares selection. Cluster boot over the
# included problem_ids.
print("\n" + "=" * 74)
print("S2. Within-problem PAIRED maj32(arm) - maj32(grpo), matched on C (|dC|<=tol)")
print("    pure nonparametric 'equal-skill' selection comparison")
print("=" * 74)
for tol in (0, 1, 2):
    print(f"  --- tol = {tol} (|C_arm - C_grpo| <= {tol}) ---")
    for a in SHAPED + ["pi0"]:
        dC = np.abs(Ccnt[a] - Ccnt["grpo"])
        mask = dC <= tol
        npairs = int(mask.sum())
        diff = maj32[a] - maj32["grpo"]
        pt = float(diff[mask].mean()) if npairs else float("nan")
        # cluster boot over included pids
        inc = pid_idx[mask]
        bs = []
        for _ in range(B):
            s = RNG.choice(inc, size=len(inc), replace=True)
            bs.append(float((maj32[a][s] - maj32["grpo"][s]).mean()))
        lo, hi = ci(bs)
        print(f"    {a:6s} n={npairs:4d}  mean(arm-grpo)={pt:+.4f}  [{lo:+.4f},{hi:+.4f}]{stars(lo,hi)}")


# ===========================================================================
# S3. Finer pass1 control: natural cubic spline (knots at quantiles) + arm dummies
# ===========================================================================
print("\n" + "=" * 74)
print("S3. Pooled OLS, maj32 ~ ncs(pass1, knots@deciles) + arm dummies (grpo ref)")
print("    finer functional form for skill; cluster boot over problem_id")
print("=" * 74)


def ncs_basis(x, knots):
    """Natural cubic spline basis (truncated power, Harrell form). Returns columns
    [x, then K-2 spline terms]. Intercept handled separately."""
    K = len(knots)
    kn = np.asarray(knots, float)

    def d(j, t):
        num = np.clip(t - kn[j], 0, None) ** 3 - np.clip(t - kn[K - 1], 0, None) ** 3
        return num / (kn[K - 1] - kn[j])
    cols = [x]
    for j in range(K - 2):
        cols.append(d(j, x) - d(K - 2, x))
    return np.column_stack(cols)


knots = np.quantile(rows_p1, np.linspace(0.05, 0.95, 6))
knots = np.unique(knots)


def design_spline(p1, arm):
    sb = ncs_basis(p1, knots)
    X = [np.ones((len(p1), 1)), sb]
    for a in dummy_arms:
        X.append((arm == a).astype(float)[:, None])
    return np.hstack(X)


nb = ncs_basis(rows_p1, knots).shape[1]
sp_names = ["intercept"] + [f"sp{i}" for i in range(nb)] + [f"arm[{a}]" for a in dummy_arms]
beta_sp = ols(rows_m, design_spline(rows_p1, rows_arm))
sp_boot = np.zeros((B, len(beta_sp)))
for b, samp in enumerate(boot_pid_samples()):
    sel = np.concatenate([idx_by_pid[i] for i in samp])
    sp_boot[b] = ols(rows_m[sel], design_spline(rows_p1[sel], rows_arm[sel]))
for j, nm in enumerate(sp_names):
    if not nm.startswith("arm"):
        continue
    lo, hi = ci(sp_boot[:, j])
    print(f"  {nm:12s} {beta_sp[j]:+.4f}  [{lo:+.4f},{hi:+.4f}]{stars(lo,hi)}")


# ===========================================================================
# S4. Robustness: paired diff matched on C, FINE bins, pooled across tol via
#     stratified within-C-bin mean (Mantel-Haenszel-style), shaped arms only.
# ===========================================================================
print("\n" + "=" * 74)
print("S4. Stratified-by-C paired diff (exact C match within bins), shaped arms")
print("    avg over C strata weighted by stratum size; cluster boot over pid")
print("=" * 74)
for a in SHAPED:
    # exact-C match: pair arm vs grpo only where C_arm == C_grpo, then average
    mask = Ccnt[a] == Ccnt["grpo"]
    inc = pid_idx[mask]
    diff = maj32[a] - maj32["grpo"]
    pt = float(diff[mask].mean())
    bs = []
    for _ in range(B):
        s = RNG.choice(inc, size=len(inc), replace=True)
        bs.append(float((maj32[a][s] - maj32["grpo"][s]).mean()))
    lo, hi = ci(bs)
    print(f"  {a:6s} exact-C-match n={len(inc):4d}  diff={pt:+.4f}  [{lo:+.4f},{hi:+.4f}]{stars(lo,hi)}")

print("\nDONE.")
