"""
calibrate_prm.py — (1) CPU correctness unit-tests for the Phase-2 verifier-aware
advantages, and (2) scale-constant calibration through the REAL trainer code path
(rl.train_grpo.daft_group_advantages -> rl.advantages.compute_advantages).

Scale methodology mirrors rl/calibrate_scale.py: scale = RMS over ALL entries
(incl zeros). We anchor the new prm_* constants to the canonical grpo constant via
a matched-G=16 RATIO measured on the PRM-scored cache:
    scale_prm = grpo_scale * (rms_prm_cache / rms_grpo_cache)
(absolute RMS differs by data distribution; the ratio is the transferable part;
the trainer's daft/adv_rms_prescale is the in-flight backstop).

Run:  python -m rl.analysis.calibrate_prm
"""
from __future__ import annotations
import glob, json, math, os
import numpy as np

from rl.advantages import compute_advantages
from rl.train_grpo import daft_group_advantages

REPO = "/cluster/tufts/hugheslab/kheuto01/code/daft"
SC = os.path.join(REPO, "rl", "analysis", "outputs", "prm_scored_steps")
SCALE_JSON = os.path.join(REPO, "rl", "configs", "scale_constants.json")
GRPO_SCALE = 0.3065682636713648
G, N_SUB, SEED = 16, 4, 0
AGG, P = "last", 4   # the chosen deployed rule for the launch arm (prm_margin)


def agg_of(steps, strat):
    a = np.asarray(steps, dtype=np.float64)
    if a.size == 0:
        return 0.0
    return {"last": a[-1], "min": a.min(), "prod": np.prod(a), "mean": a.mean()}[strat]


# --------------------------------------------------------------------------- #
# (1) Synthetic correctness unit-tests
# --------------------------------------------------------------------------- #
def unit_tests():
    print("## (1) synthetic correctness unit-tests")
    # group: classes by canonical; correct mask; verifier scores
    # 16 samples: class A (correct) x4 score 0.8; class B (wrong, plurality) x10 score 0.7;
    #             class C (wrong) x2 score 0.2
    correct = np.array([[1,1,1,1, 0,0,0,0,0,0,0,0,0,0, 0,0]], dtype=np.int64)
    cls     = np.array([[0,0,0,0, 1,1,1,1,1,1,1,1,1,1, 2,2]], dtype=np.int64)
    vs      = np.array([[.8,.8,.8,.8, .7,.7,.7,.7,.7,.7,.7,.7,.7,.7, .2,.2]], dtype=np.float64)

    # D3 prm_margin: +w to top-correct class (A), -w to top-wrong class (B), 0 to C
    a3 = compute_advantages(correct, cls, "prm_margin", k=0, verifier_scores=vs, prm_exponent=P)[0]
    w = np.clip(vs[0], 0, None) ** P
    # A members should be positive, B members negative, C members: base 0 -> after
    # centering equal to -mean(base)
    assert (a3[:4] > 0).all(), f"D3 correct-class adv not positive: {a3[:4]}"
    assert (a3[4:14] < 0).all(), f"D3 wrong-plurality adv not negative: {a3[4:14]}"
    # C members (idx 14,15) have base 0; after centering they share one value, and
    # |a3_C| < |a3_A| and < |a3_B| (they're untouched up to the small mean shift)
    assert abs(a3[14] - a3[15]) < 1e-12, "D3 untouched-class members must be equal"
    # zero-mean after centering
    assert abs(a3.mean()) < 1e-9, f"D3 advantage not zero-mean: {a3.mean()}"
    print("  [D3] +correct-class / -wrong-plurality / ~0 elsewhere, zero-mean — OK")

    # D3 mixed-class (review fix): a WRONG (e.g. truncated) member sharing C*'s
    # canonical must NOT receive +w.
    correct_m = np.array([[1,0, 0,0,0,0,0,0,0,0,0,0,0,0, 0,0]], dtype=np.int64)
    cls_m     = np.array([[0,0, 1,1,1,1,1,1,1,1,1,1,1,1, 2,2]], dtype=np.int64)
    vs_m      = np.array([[.8,.9, .7,.7,.7,.7,.7,.7,.7,.7,.7,.7,.7,.7, .2,.2]], dtype=np.float64)
    a3m = compute_advantages(correct_m, cls_m, "prm_margin", k=0, verifier_scores=vs_m, prm_exponent=P)[0]
    # truncated-wrong idx1 (shares C*'s canonical) must get NO +w -> it falls to the
    # untouched baseline (-mean), same as a non-C*/non-W* sample (idx14), and is
    # outranked by the genuinely-correct C* member idx0.
    assert a3m[0] > a3m[1], f"mixed: correct member must outrank truncated-wrong: {a3m[0]} vs {a3m[1]}"
    assert abs(a3m[1] - a3m[14]) < 1e-9, (
        f"truncated-wrong member of C* not treated as untouched: {a3m[1]} vs baseline {a3m[14]}"
    )
    print("  [D3 mixed] truncated-wrong member of correct class gets NO +w (baseline) — OK")

    # D3 drop-empty (review fix): empty-canonical class with top wrong mass must NOT
    # be selected as W*; the real wrong plurality is penalized instead.
    correct_e = np.array([[1, 0,0, 0,0,0]], dtype=np.int64)
    cls_e     = np.array([[0, 1,1, 2,2,2]], dtype=np.int64)     # class2 = empties (high mass)
    vs_e      = np.array([[.6, .4,.4, .9,.9,.9]], dtype=np.float64)
    em_e      = np.array([[False, False,False, True,True,True]], dtype=bool)
    a3e = compute_advantages(correct_e, cls_e, "prm_margin", k=0, verifier_scores=vs_e,
                             prm_exponent=P, empty_mask=em_e)[0]
    assert a3e[1] < 0 and a3e[2] < 0, f"real wrong plurality not penalized as W*: {a3e[1:3]}"
    assert a3e[3] > a3e[1], f"empty class wrongly chosen as W* (over-penalized): {a3e[3]} vs {a3e[1]}"
    print("  [D3 drop-empty] empty class excluded from W* selection — OK")

    # D2 prm_weighted: reward = w*correct; wrong -> reward 0 -> adv = -mean(reward)
    a2 = compute_advantages(correct, cls, "prm_weighted", k=0, verifier_scores=vs, prm_exponent=P)[0]
    rmean = (w * correct[0]).mean()
    assert np.allclose(a2[4:], -rmean), "D2 wrong-sample adv must equal -mean(reward)"
    assert (a2[:4] > 0).all(), "D2 correct adv must be positive (w>mean)"
    print("  [D2] reward==0 for wrong (adv=-mean), correct reweighted by w — OK")

    # D2pm prm_weighted_pm: reward = w*(2c-1)
    a2pm = compute_advantages(correct, cls, "prm_weighted_pm", k=0, verifier_scores=vs, prm_exponent=P)[0]
    assert (a2pm[4:14] < 0).all(), "D2pm wrong-plurality must be negative"
    print("  [D2pm] symmetric +w correct / -w wrong — OK")

    # all-correct and all-wrong -> zeros for all prm modes (grpo parity)
    for cc in (np.ones((1, G), np.int64), np.zeros((1, G), np.int64)):
        for m in ("prm_margin", "prm_weighted", "prm_weighted_pm"):
            a = compute_advantages(cc, np.zeros((1, G), np.int64), m, k=0,
                                   verifier_scores=np.full((1, G), 0.5), prm_exponent=P)
            assert np.allclose(a, 0.0), f"{m} not zero on degenerate group"
    print("  [all] all-correct / all-wrong groups -> zero advantage (grpo parity) — OK")

    # guards: prm mode requires verifier_scores; shape mismatch caught
    try:
        compute_advantages(correct, cls, "prm_margin", k=0)
        raise SystemExit("FAIL: missing verifier_scores not caught")
    except AssertionError:
        pass
    try:
        compute_advantages(correct, cls, "prm_margin", k=0,
                           verifier_scores=np.zeros((1, G - 1)), prm_exponent=P)
        raise SystemExit("FAIL: shape mismatch not caught")
    except AssertionError:
        pass
    print("  [guards] missing verifier_scores + shape mismatch raise — OK")

    # end-to-end via the trainer entry (daft_group_advantages): canonical strings
    canon = [["A","A","A","A","B","B","B","B","B","B","B","B","B","B","C","C"]]
    adv_t = daft_group_advantages(correct, canon, "prm_margin", k=0, scale=1.0,
                                  verifier_scores=vs, prm_exponent=P).numpy()
    assert np.allclose(adv_t, a3, atol=1e-6), "daft_group_advantages != compute_advantages"
    print("  [integration] daft_group_advantages == compute_advantages — OK\n")


# --------------------------------------------------------------------------- #
# (2) Scale calibration on the PRM-scored cache via the real code path
# --------------------------------------------------------------------------- #
def calibrate():
    print("## (2) scale calibration (matched G=16 ratio to grpo, via real code path)")
    rows = []
    for f in sorted(glob.glob(os.path.join(SC, "scored_shard*.jsonl"))):
        for line in open(f):
            if line.strip():
                rows.append(json.loads(line))
    rng = np.random.default_rng(SEED)
    # collect advantages over many simulated G=16 groups, per mode
    modes = {"grpo": dict(), "prm_margin": dict(), "prm_weighted": dict(), "prm_weighted_pm": dict()}
    sq = {m: [] for m in modes}
    nz = {m: [] for m in modes}
    cos_pm = []  # cosine(prm_margin, grpo) sanity
    for r in rows:
        ss = r["step_scores"]; cm = np.asarray(r["correct_mask"], np.int64); canon = r["canonicals"]
        n = len(cm)
        if n < G:
            continue
        agg_scores = np.array([agg_of(s, AGG) for s in ss], dtype=np.float64)
        for _ in range(N_SUB):
            idx = rng.choice(n, size=G, replace=False)
            c = cm[idx].reshape(1, G)
            gc = [[canon[i] for i in idx]]
            vs = agg_scores[idx].reshape(1, G)
            for m in modes:
                if m == "grpo":
                    a = daft_group_advantages(c, gc, "grpo", k=0, scale=1.0).numpy()
                else:
                    a = daft_group_advantages(c, gc, m, k=0, scale=1.0,
                                              verifier_scores=vs, prm_exponent=P).numpy()
                sq[m].append(a ** 2)
                nz[m].append((np.abs(a) > 1e-12).mean())
            # cosine sanity prm_margin vs grpo on non-degenerate groups
            if 0 < c.sum() < G:
                a0 = daft_group_advantages(c, gc, "grpo", k=0, scale=1.0).numpy()
                am = daft_group_advantages(c, gc, "prm_margin", k=0, scale=1.0,
                                           verifier_scores=vs, prm_exponent=P).numpy()
                n0, nm = np.linalg.norm(a0), np.linalg.norm(am)
                if n0 > 1e-12 and nm > 1e-12:
                    cos_pm.append(float(np.dot(a0, am) / (n0 * nm)))
    rms = {m: math.sqrt(np.concatenate(sq[m]).mean()) for m in modes}
    fnz = {m: float(np.mean(nz[m])) for m in modes}
    print(f"  agg={AGG} p={P}  (RMS over ALL entries incl zeros, matched G={G})")
    print("  | mode | rms_cache | frac_nz | ratio→grpo | scale = 0.30657*ratio |")
    out = {}
    for m in modes:
        ratio = rms[m] / rms["grpo"]
        scale = GRPO_SCALE * ratio
        if m != "grpo":
            out[m] = scale
        print(f"  | {m} | {rms[m]:.5f} | {fnz[m]:.3f} | {ratio:.4f} | {scale:.5f} |")
    print(f"\n  cosine(prm_margin, grpo) over non-degenerate groups: "
          f"mean={np.mean(cos_pm):.3f} (verify_designs reported ~0.72; sanity that "
          f"the real code path matches) — {'OK' if np.mean(cos_pm) < 0.85 else 'WARN: not distinct!'}")
    print(f"\n  PROPOSED scale_constants.json additions: {json.dumps(out, indent=2)}")
    return out


if __name__ == "__main__":
    unit_tests()
    calibrate()
