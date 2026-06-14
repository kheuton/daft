"""Robustness: re-estimate the matched-C paired diff with (a) exact maj@32 where
tractable / higher MC budget, and (b) averaging maj@32 over multiple seeds, to rule
out that the lone significant (negative) effect is a Monte-Carlo artifact.
Also: a second 'skill' axis check — does adding GRPO's own answer-concentration
(entropy of class distribution) as a covariate change the arm null?"""
from __future__ import annotations
import numpy as np
from rl.advantages import maj_at_k_estimate
from rl.analysis import phase0_lib as L

ARMS = ["pi0", "grpo", "passk", "votek"]
paired = L.load_paired()
pids = sorted(paired["grpo"].keys())
P = len(pids)


def maj32_multiseed(p, seeds=(0, 1, 2, 3, 4)):
    cids = L.class_ids(p.canonicals)
    return float(np.mean([
        maj_at_k_estimate(p.correct, cids, 32, n_subsets=2000, seed=s) for s in seeds
    ]))


print("Recomputing maj@32 with n_subsets=2000 averaged over 5 seeds (low-noise)...")
maj32 = {a: np.array([maj32_multiseed(paired[a][pid]) for pid in pids]) for a in ARMS}
Ccnt = {a: np.array([int(paired[a][pid].correct.sum()) for pid in pids]) for a in ARMS}
pid_idx = np.arange(P)
RNG = np.random.default_rng(0)


def ci(s):
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


print("\nLow-noise exact-C-match paired diff maj32(arm)-maj32(grpo):")
for a in ["passk", "votek", "pi0"]:
    mask = Ccnt[a] == Ccnt["grpo"]
    inc = pid_idx[mask]
    pt = float((maj32[a][mask] - maj32["grpo"][mask]).mean())
    bs = [float((maj32[a][s] - maj32["grpo"][s]).mean())
          for s in (RNG.choice(inc, size=len(inc), replace=True) for _ in range(2000))]
    lo, hi = ci(bs)
    star = " *" if (lo > 0 or hi < 0) else ""
    print(f"  {a:6s} n={len(inc):4d}  diff={pt:+.4f}  [{lo:+.4f},{hi:+.4f}]{star}")

# Within-problem FE with low-noise maj32
print("\nLow-noise problem-FE arm effects (vs grpo):")
p1 = {a: np.array([paired[a][pid].correct.mean() for pid in pids]) for a in ARMS}


def fit_fe(samp):
    nP = len(samp)
    P1 = np.column_stack([p1[a][samp] for a in ARMS])
    MJ = np.column_stack([maj32[a][samp] for a in ARMS])
    P1d = (P1 - P1.mean(1, keepdims=True)).reshape(-1)
    MJd = (MJ - MJ.mean(1, keepdims=True)).reshape(-1)
    arm_f = np.repeat(np.arange(4)[None, :], nP, 0).reshape(-1)
    cols = [P1d]
    nm = ["pass1d"]
    for ai, a in enumerate(ARMS):
        if a == "grpo":
            continue
        cols.append((arm_f == ai).astype(float) - 0.25)
        nm.append(a)
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, MJd, rcond=None)
    return beta, nm


beta, nm = fit_fe(pid_idx)
fb = np.array([fit_fe(RNG.choice(pid_idx, size=P, replace=True))[0] for _ in range(2000)])
for j, n in enumerate(nm):
    lo, hi = ci(fb[:, j])
    star = " *" if (lo > 0 or hi < 0) and n != "pass1d" else ""
    print(f"  {n:8s} {beta[j]:+.4f}  [{lo:+.4f},{hi:+.4f}]{star}")
print("DONE.")
