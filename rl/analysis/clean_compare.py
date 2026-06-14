"""
clean_compare.py — the definitive, confound-free Phase 2 comparison.

Three arms, all measured the SAME (RTM-confounded) way on the same 878 band+control
problems, so the confound CANCELS in the right diff:
  cached_arm_a  : arm_a rollouts that DEFINED the band  (B.ARMA_CACHE_DIR)
  fresh_arm_a   : arm_a re-rolled fresh                  (B.ARMA_FRESH_DIR)
  D3            : prm_margin policy, fresh               (B.D3_DIR)

Diffs:
  RTM   = fresh_arm_a - cached_arm_a   (same policy re-rolled => pure regression-to-mean / sampling)
  CLEAN = D3 - fresh_arm_a             (confound-free Phase 2 effect: D3 vs control, both fresh)
  raw   = D3 - cached_arm_a            (the original, RTM-confounded headline)

For each arm/band: coverage(pass@64), ncorr(mean #correct/64), plain_maj,
wvote(last,p4,drop-empty), selection|covered (wvote over covered problems),
and the pool projection (DEAD 349 -> 0, WON 1064 * control, BAND 578 * band)/1991.
"""
from __future__ import annotations
import argparse
import numpy as np
from rl.analysis import banddata as B


def arm_stats(rows):
    bb = B.by_band(rows)
    out = {}
    for band in ("misselected", "control"):
        rs = bb.get(band, [])
        if not rs:
            out[band] = None
            continue
        cover, ncorr, maj, wv, wv_cov = [], [], [], [], []
        for r in rs:
            cm = r["correct_mask"]
            nc = sum(1 for x in cm if x > 0.5)
            cover.append(nc > 0)
            ncorr.append(nc / len(cm))
            maj.append(B.plain_maj(r["canonicals"], cm))
            w = B.wvote(r["canonicals"], r["last"], cm, p=4)
            wv.append(w)
            if nc > 0:
                wv_cov.append(w)
        out[band] = dict(
            n=len(rs), cover=float(np.mean(cover)), ncorr=float(np.mean(ncorr)),
            maj=float(np.mean(maj)), wv=float(np.mean(wv)),
            wv_cov=float(np.mean(wv_cov)) if wv_cov else float("nan"),
        )
    b, c = out.get("misselected"), out.get("control")
    if b and c:
        out["pool"] = dict(
            maj=(B.N_WON * c["maj"] + B.N_BAND * b["maj"]) / B.N_POOL,
            wv=(B.N_WON * c["wv"] + B.N_BAND * b["wv"]) / B.N_POOL,
        )
    return out


def show(name, st):
    print(f"\n=== {name} ===")
    print(f"{'band':12s} {'n':>4s} {'cover':>7s} {'ncorr':>7s} {'maj':>7s} {'wvote':>7s} {'wv|cov':>7s}")
    for band in ("misselected", "control"):
        o = st.get(band)
        if o:
            print(f"{band:12s} {o['n']:4d} {o['cover']:7.3f} {o['ncorr']:7.3f} {o['maj']:7.3f} {o['wv']:7.3f} {o['wv_cov']:7.3f}")
    if "pool" in st:
        print(f"{'pool proj':12s} {B.N_POOL:4d} {'':7s} {'':7s} {st['pool']['maj']:7.3f} {st['pool']['wv']:7.3f}")


def diff(name, a, b):
    """a - b for the comparable scalar fields."""
    print(f"\n--- {name} ---")
    for band in ("misselected", "control"):
        oa, ob = a.get(band), b.get(band)
        if oa and ob:
            print(f"  {band:12s} cover {oa['cover']-ob['cover']:+.3f}  maj {oa['maj']-ob['maj']:+.3f}  "
                  f"wvote {oa['wv']-ob['wv']:+.3f}  wv|cov {oa['wv_cov']-ob['wv_cov']:+.3f}")
    if "pool" in a and "pool" in b:
        print(f"  {'pool':12s} maj {a['pool']['maj']-b['pool']['maj']:+.3f}  wvote {a['pool']['wv']-b['pool']['wv']:+.3f}")


def _covered(r):
    return sum(1 for x in r["correct_mask"] if x > 0.5) > 0


def _deployed_correct(r):
    """deployed-rule selected-correct flag: wvote(last,p4) >= 0.5."""
    return B.wvote(r["canonicals"], r["last"], r["correct_mask"], p=4) >= 0.5


def paired(rows_a, rows_b, name, band="misselected"):
    """LIKE-FOR-LIKE paired comparison on the SAME problem_ids (band only).
    Removes the subset-change confound: compares both arms on the intersection of
    problems each covers, and counts deployed-rule flips per problem."""
    a = {r["problem_id"]: r for r in rows_a if r["band"] == band}
    b = {r["problem_id"]: r for r in rows_b if r["band"] == band}
    pids = sorted(set(a) & set(b))
    print(f"\n--- PAIRED {name}  (band={band}, joined {len(pids)} problems) ---")
    # common-covered: problems BOTH arms cover -> like-for-like selection
    common_cov = [p for p in pids if _covered(a[p]) and _covered(b[p])]
    if common_cov:
        wa = float(np.mean([B.wvote(a[p]["canonicals"], a[p]["last"], a[p]["correct_mask"], p=4) for p in common_cov]))
        wb = float(np.mean([B.wvote(b[p]["canonicals"], b[p]["last"], b[p]["correct_mask"], p=4) for p in common_cov]))
        ma = float(np.mean([B.plain_maj(a[p]["canonicals"], a[p]["correct_mask"]) for p in common_cov]))
        mb = float(np.mean([B.plain_maj(b[p]["canonicals"], b[p]["correct_mask"]) for p in common_cov]))
        print(f"  common-covered n={len(common_cov)}:  wvote A {wa:.3f} -> B {wb:.3f} (B-A {wb-wa:+.3f}) | "
              f"plain_maj A {ma:.3f} -> B {mb:.3f} (B-A {mb-ma:+.3f})")
    # coverage: who covers what
    only_a = [p for p in pids if _covered(a[p]) and not _covered(b[p])]
    only_b = [p for p in pids if _covered(b[p]) and not _covered(a[p])]
    print(f"  coverage: A-only {len(only_a)}  B-only {len(only_b)}  both {len(common_cov)}  "
          f"(net B-A {len(only_b)-len(only_a):+d})")
    # deployed-rule flips (B vs A) over ALL joined band problems
    rec = sum(1 for p in pids if (not _deployed_correct(a[p])) and _deployed_correct(b[p]))
    broke = sum(1 for p in pids if _deployed_correct(a[p]) and (not _deployed_correct(b[p])))
    sr = sum(1 for p in pids if _deployed_correct(a[p]) and _deployed_correct(b[p]))
    sw = sum(1 for p in pids if (not _deployed_correct(a[p])) and (not _deployed_correct(b[p])))
    print(f"  deployed flips (B vs A): RECOVERED {rec}  BROKE {broke}  net {rec-broke:+d}  "
          f"(stay-right {sr}, stay-wrong {sw})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", default=B.ARMA_CACHE_DIR)
    ap.add_argument("--fresh", default=B.ARMA_FRESH_DIR)
    ap.add_argument("--d3", default=B.D3_DIR)
    args = ap.parse_args()
    bm = B.bandmap()
    cached_rows = B.load(args.cached, bm)
    fresh_rows = B.load(args.fresh, bm)
    d3_rows = B.load(args.d3, bm)
    cached = arm_stats(cached_rows)
    fresh = arm_stats(fresh_rows)
    d3 = arm_stats(d3_rows)

    show("cached_arm_a (defined the band)", cached)
    show("fresh_arm_a (re-rolled)", fresh)
    show("D3 prm_margin (fresh)", d3)

    diff("RTM = fresh_arm_a - cached_arm_a  (pure re-roll; expect cover<0 on band, wv|cov~0)", fresh, cached)
    diff("CLEAN = D3 - fresh_arm_a  (CONFOUND-FREE Phase 2 effect)", d3, fresh)
    diff("raw = D3 - cached_arm_a  (original confounded headline)", d3, cached)

    # like-for-like paired comparisons (removes subset-change confound)
    paired(cached_rows, fresh_rows, "RTM (A=cached_arm_a, B=fresh_arm_a)")
    paired(fresh_rows, d3_rows, "CLEAN (A=fresh_arm_a, B=D3)")
    paired(cached_rows, d3_rows, "raw (A=cached_arm_a, B=D3)")

    print("\n=== BOTTOM LINE ===")
    if fresh.get("pool") and d3.get("pool") and cached.get("pool"):
        print(f"  pool wvote:  cached_arm_a {cached['pool']['wv']:.3f} | fresh_arm_a {fresh['pool']['wv']:.3f} | D3 {d3['pool']['wv']:.3f}")
        print(f"  CLEAN pool wvote (D3 - fresh_arm_a): {d3['pool']['wv']-fresh['pool']['wv']:+.3f}")
    if fresh.get("misselected") and d3.get("misselected"):
        fb, db = fresh["misselected"], d3["misselected"]
        print(f"  CLEAN band selection|covered (D3 - fresh_arm_a): {db['wv_cov']-fb['wv_cov']:+.3f}  "
              f"(arm_a {fb['wv_cov']:.3f} -> D3 {db['wv_cov']:.3f})")
        print(f"  CLEAN band coverage (D3 - fresh_arm_a): {db['cover']-fb['cover']:+.3f}  "
              f"(arm_a {fb['cover']:.3f} -> D3 {db['cover']:.3f})")


if __name__ == "__main__":
    main()
