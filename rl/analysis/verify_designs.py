"""
verify_designs.py — head-to-head of candidate Phase-2 advantage designs on the
fixed per-step PRM cache, to pick the ONE to launch. Decides between:

  D0    control:        a = c - mean(c)                       (the bar; already won R1)
  D2    correct-reweight: r = w*c;          a = r - mean(r)   (synth pick w/ prod)
  D2pm  symmetric ±:    r = w*(2c-1);        a = r - mean(r)  (adds wrong-side pressure)
  D3    class-margin:   +w to top-correct-class members, -w to top-wrong-class
                        members, 0 else; a = base - mean(base) (vote-aware, dense-ish)

where w_i = clip(agg(step_scores_i),0)**p is the SAME per-completion PRM weight the
deployed PRM-weighted-vote uses. Reported per (design, agg, p):
  - mean cosine to D0 over simulated G=16 groups (distinguishability; <~0.85 good)
  - frac_nonzero advantage (density; R1 lesson: dense beats sparse)
  - wrong-side spread = mean std of advantage among WRONG samples (D0=0 by constr.;
    >0 means the design differentiates the confidently-wrong plurality)
  - correct-side spread = mean std among CORRECT samples
  - rmsD/rmsD0 ratio -> scale constant = 0.3066 * ratio

Goal: a design that is (i) dense, (ii) distinct from control, (iii) has wrong-side
pressure, ideally (iv) usable with the control-SAFE last-agg deployed rule (+0.072).

Run:  python -m rl.analysis.verify_designs
"""
from __future__ import annotations
import glob, json, math, os
from collections import defaultdict
import numpy as np

REPO = "/cluster/tufts/hugheslab/kheuto01/code/daft"
SC = os.path.join(REPO, "rl", "analysis", "outputs", "prm_scored_steps")
GRPO_SCALE, G, N_SUB, SEED = 0.3066, 16, 4, 0
AGGS = ["last", "prod"]
PS = [2, 4]


def agg_of(steps, strat):
    if not steps:
        return 0.0
    a = np.asarray(steps, dtype=np.float64)
    return {"last": a[-1], "min": a.min(), "prod": np.prod(a), "mean": a.mean()}[strat]


def class_ids(canon):
    m, out = {}, []
    for c in canon:
        out.append(m.setdefault(c, len(m)))
    return np.asarray(out, dtype=np.int64)


def adv_D0(c, w, cid):
    return c - c.mean()


def adv_D2(c, w, cid):
    r = w * c
    return r - r.mean()


def adv_D2pm(c, w, cid):
    r = w * (2 * c - 1.0)
    return r - r.mean()


def adv_D3(c, w, cid):
    # class masses under weight w
    mass = defaultdict(float)
    for j, wi in zip(cid, w):
        mass[int(j)] += wi
    correct_cls = {int(cid[i]) for i in range(len(c)) if c[i] > 0.5}
    wrong_cls = {int(j) for j in cid.tolist()} - correct_cls
    # top correct class and top wrong class by mass
    Cstar = max(correct_cls, key=lambda j: mass[j]) if correct_cls else None
    Wstar = max(wrong_cls, key=lambda j: mass[j]) if wrong_cls else None
    base = np.zeros_like(w)
    for i in range(len(c)):
        j = int(cid[i])
        if j == Cstar:
            base[i] = w[i]
        elif j == Wstar:
            base[i] = -w[i]
    return base - base.mean()


DESIGNS = {"D2": adv_D2, "D2pm": adv_D2pm, "D3": adv_D3}


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(SC, "scored_shard*.jsonl"))):
        for line in open(f):
            if line.strip():
                rows.append(json.loads(line))
    for r in rows:
        ss = r["step_scores"]
        r["_agg"] = {a: np.array([agg_of(s, a) for s in ss], dtype=np.float64) for a in AGGS}
        r["_c"] = np.asarray(r["correct_mask"], dtype=np.float64)
        r["_cid"] = class_ids(r["canonicals"])
    rng = np.random.default_rng(SEED)
    subs = []
    for r in rows:
        n = len(r["_c"])
        if n < G:
            continue
        for _ in range(N_SUB):
            idx = rng.choice(n, size=G, replace=False)
            cc = r["_c"][idx]
            if cc.sum() == 0 or cc.sum() == G:
                continue
            subs.append((r, idx))
    print(f"# verify_designs — {len(rows)} problems, {len(subs)} non-degenerate G={G} groups\n")

    # D0 rms
    d0sq = []
    for r, idx in subs:
        c = r["_c"][idx]
        a0 = c - c.mean()
        d0sq.append(a0 ** 2)
    rms0 = math.sqrt(np.concatenate(d0sq).mean())

    print("| design | agg | p | mean cos→D0 | frac_nz | wrong-side spread | correct-side spread | rmsD/rms0 | scale |")
    print("|" + "---|" * 9)
    for name, fn in DESIGNS.items():
        for a in AGGS:
            for p in PS:
                cosines, fnz, wspread, cspread, dsq = [], [], [], [], []
                for r, idx in subs:
                    c = r["_c"][idx]
                    w = np.clip(r["_agg"][a][idx], 0.0, None) ** p
                    cid = class_ids([r["canonicals"][i] for i in idx])
                    adv = fn(c, w, cid)
                    a0 = c - c.mean()
                    dsq.append(adv ** 2)
                    fnz.append((np.abs(adv) > 1e-12).mean())
                    if (c <= 0.5).sum() > 1:
                        wspread.append(adv[c <= 0.5].std())
                    if (c > 0.5).sum() > 1:
                        cspread.append(adv[c > 0.5].std())
                    n1, n0 = np.linalg.norm(adv), np.linalg.norm(a0)
                    if n1 > 1e-12 and n0 > 1e-12:
                        cosines.append(float(np.dot(adv, a0) / (n1 * n0)))
                ratio = math.sqrt(np.concatenate(dsq).mean()) / rms0
                print(f"| {name} | {a} | {p} | {np.mean(cosines):.3f} | {np.mean(fnz):.3f} | "
                      f"{np.mean(wspread):.4f} | {np.mean(cspread):.4f} | {ratio:.3f} | {GRPO_SCALE*ratio:.4f} |")
    print(f"\n(D0 rms={rms0:.4f}; D0 wrong-side spread=0 and correct-side spread=0 by construction.)")
    print("Decision target: dense (frac_nz high), distinct (cos<~0.85), wrong-side spread>0 "
          "(differentiates the confidently-wrong plurality), ideally works at agg=last (control-safe deploy).")


if __name__ == "__main__":
    main()
