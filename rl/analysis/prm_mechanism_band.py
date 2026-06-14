"""
PRM-SCORE MECHANISM analysis on the BAND (misselected) partition.
Compares arm_a (cached GRPO control rollouts) vs D3 (prm_margin policy, fresh).

Q1: Pooled distribution of LAST-agg PRM score for CORRECT vs WRONG completions
    (mean, median, p10, p90, gap = mean_correct - mean_wrong).
Q2: Per-problem correct-class MASS SHARE under wvote(last, p=4):
    sum(max(s,0)**4 over correct-class, non-empty completions)
    / sum(max(s,0)**4 over all non-empty completions), mean across band problems with coverage.
Q3: Did D3 make the PRM more discriminative on the band?
"""
import statistics as st
from rl.analysis import banddata as B

BAND = "misselected"
P = 4


def pctile(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    # simple linear-interpolation percentile
    if len(xs) == 1:
        return xs[0]
    pos = q / 100.0 * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def desc(xs):
    if not xs:
        return dict(n=0, mean=float("nan"), median=float("nan"),
                    p10=float("nan"), p90=float("nan"))
    return dict(n=len(xs), mean=st.mean(xs), median=st.median(xs),
                p10=pctile(xs, 10), p90=pctile(xs, 90))


def pooled_correct_wrong(rows):
    """Pool LAST-agg scores across ALL completions in band, split by correct_mask.
    Two variants: ALL completions, and DROP-EMPTY (canonical=='') to match deployed rule."""
    cor_all, wro_all = [], []
    cor_ne, wro_ne = [], []  # non-empty
    for r in rows:
        last = r["last"]
        cm = r["correct_mask"]
        canon = r["canonicals"]
        for i in range(len(last)):
            s = float(last[i])
            if cm[i] > 0.5:
                cor_all.append(s)
                if canon[i] != "":
                    cor_ne.append(s)
            else:
                wro_all.append(s)
                if canon[i] != "":
                    wro_ne.append(s)
    return (cor_all, wro_all), (cor_ne, wro_ne)


def correct_class_mass_share(rows, p=P):
    """Per-problem: mass on correct-class completions / mass on all non-empty completions.
    mass(i) = max(last_i, 0)**p. Only count problems WITH coverage (>=1 correct, non-empty).
    Returns list of shares + count of covered problems + count with any non-empty mass."""
    shares = []
    n_covered = 0
    for r in rows:
        last = r["last"]
        cm = r["correct_mask"]
        canon = r["canonicals"]
        cc = B.correct_classes(canon, cm)  # set of correct canonical strings
        # coverage = at least one correct, non-empty completion
        has_cov = any(cm[i] > 0.5 and canon[i] != "" for i in range(len(last)))
        if not has_cov:
            continue
        n_covered += 1
        tot = 0.0
        cor = 0.0
        for i in range(len(last)):
            if canon[i] == "":
                continue
            m = max(float(last[i]), 0.0) ** p
            tot += m
            if canon[i] in cc:
                cor += m
        if tot > 0:
            shares.append(cor / tot)
        else:
            shares.append(0.0)  # all-non-empty mass is zero -> share 0
    return shares, n_covered


def main():
    bm = B.bandmap()
    arma = [r for r in B.load(B.ARMA_CACHE_DIR, bm) if r["band"] == BAND]
    d3 = [r for r in B.load(B.D3_DIR, bm) if r["band"] == BAND]
    print(f"BAND='{BAND}'  arm_a n={len(arma)}  D3 n={len(d3)}")
    print()

    print("=" * 78)
    print("Q1: POOLED LAST-agg PRM score, CORRECT vs WRONG completions (band-wide)")
    print("=" * 78)
    for name, rows in [("arm_a", arma), ("D3", d3)]:
        (ca, wa), (cne, wne) = pooled_correct_wrong(rows)
        for variant, (c, w) in [("ALL completions", (ca, wa)),
                                  ("DROP-EMPTY (non-empty only)", (cne, wne))]:
            dc, dw = desc(c), desc(w)
            gap = dc["mean"] - dw["mean"]
            print(f"\n[{name}] {variant}")
            print(f"  CORRECT n={dc['n']:6d}  mean={dc['mean']:.4f}  med={dc['median']:.4f}  "
                  f"p10={dc['p10']:.4f}  p90={dc['p90']:.4f}")
            print(f"  WRONG   n={dw['n']:6d}  mean={dw['mean']:.4f}  med={dw['median']:.4f}  "
                  f"p10={dw['p10']:.4f}  p90={dw['p90']:.4f}")
            print(f"  GAP (mean_correct - mean_wrong) = {gap:+.4f}")

    print()
    print("=" * 78)
    print(f"Q2: Per-problem CORRECT-CLASS MASS SHARE under wvote(last, p={P})")
    print("    share = mass(correct-class) / mass(all non-empty), mean over covered problems")
    print("=" * 78)
    res = {}
    for name, rows in [("arm_a", arma), ("D3", d3)]:
        shares, ncov = correct_class_mass_share(rows, P)
        res[name] = (shares, ncov)
        mean_share = st.mean(shares) if shares else float("nan")
        med_share = st.median(shares) if shares else float("nan")
        print(f"\n[{name}] covered problems={ncov}  (shares computed on {len(shares)})")
        print(f"  mean correct-class mass share = {mean_share:.4f}")
        print(f"  median correct-class mass share = {med_share:.4f}")
        print(f"  p10={pctile(shares,10):.4f}  p90={pctile(shares,90):.4f}")
        # fraction of covered problems where correct class is the TOP-mass class
        top_correct = sum(1 for s in shares if s > 0.5)
        print(f"  problems where correct-class mass > 50% of total = {top_correct}/{len(shares)} "
              f"({top_correct/len(shares):.3f})")

    print()
    print("=" * 78)
    print("Q3 SUMMARY: did D3 make the PRM more discriminative on the band?")
    print("=" * 78)
    # recompute gaps (non-empty variant = deployed-rule relevant) for the verdict
    def gap_ne(rows):
        _, (cne, wne) = pooled_correct_wrong(rows)
        return st.mean(cne) - st.mean(wne), desc(cne)["mean"], desc(wne)["mean"]
    ga, gca, gwa = gap_ne(arma)
    gd, gcd, gwd = gap_ne(d3)
    print(f"  correct-vs-wrong GAP (non-empty): arm_a={ga:+.4f}  D3={gd:+.4f}  delta={gd-ga:+.4f}")
    print(f"    (correct-mean: arm_a={gca:.4f} D3={gcd:.4f} ; wrong-mean: arm_a={gwa:.4f} D3={gwd:.4f})")
    msa = st.mean(res["arm_a"][0])
    msd = st.mean(res["D3"][0])
    print(f"  correct-class MASS SHARE (mean over covered): arm_a={msa:.4f}  D3={msd:.4f}  delta={msd-msa:+.4f}")


if __name__ == "__main__":
    main()
