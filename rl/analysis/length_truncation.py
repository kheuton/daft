"""
length_truncation.py — COMPLETION LENGTH and TRUNCATION dimension.

Compares arm_a (cached rollouts that DEFINED the band) vs D3 (prm_margin policy)
on completion length (chars), truncation rate, empty-canonical rate, and asks
whether truncation explains D3's band coverage loss.

CAVEAT baked in: arm_a cached shards have NO num_truncated field (None for every
row). So:
  - completion length (chars) is computed directly for both (apples-to-apples).
  - truncation RATE from the native num_truncated field is available for D3 ONLY.
    For arm_a we report it as unavailable and instead use a text-derived proxy
    (does the completion end with an EOS/boxed-answer terminator) for BOTH arms,
    so at least one truncation-ish proxy is comparable across arms.
  - empty-canonical rate is computed directly for both.

The coverage-loss set (84 band problems where D3 has 0 correct but arm_a-cache
had >=1) is RTM-suspect: arm_a band coverage = 1.0 is CONSTRUCTION-GUARANTEED
(the band is *defined* as problems where arm_a had a correct completion the rule
mis-selected). State this where relevant.
"""
from __future__ import annotations
import glob, os, re
from collections import defaultdict
import numpy as np

from rl.analysis import banddata as B

REPO = B.REPO


def load_rows(d):
    return B.load(d)


def end_terminated(text):
    """Text-derived 'completion finished' proxy, comparable across arms.
    A completion that hit the token cap is cut off mid-stream; a finished one
    typically ends near a boxed answer / EOS punctuation. We flag as 'truncated
    proxy' = NOT terminated."""
    if not text:
        return False  # empty = not a clean termination
    tail = text.rstrip()[-200:]
    # finished answers usually contain a final \boxed{...} near the end,
    # or end with sentence punctuation.
    if "\\boxed" in tail:
        return True
    if tail.endswith((".", "$", "}", ")", "]")):
        return True
    return False


def per_row_metrics(r):
    comps = r["completions"]
    canon = r["canonicals"]
    cm = r["correct_mask"]
    n = len(comps)
    char_lens = [len(c) for c in comps]
    n_empty = sum(1 for c in canon if c == "")
    n_unterm = sum(1 for c in comps if not end_terminated(c))
    n_correct = sum(1 for x in cm if x > 0.5)
    return {
        "char_lens": char_lens,
        "mean_char": float(np.mean(char_lens)),
        "n_empty": n_empty,
        "frac_empty": n_empty / n,
        "n_unterm": n_unterm,
        "frac_unterm": n_unterm / n,
        "num_truncated": r.get("num_truncated"),  # None for arm_a
        "frac_truncated_native": (r["num_truncated"] / n) if r.get("num_truncated") is not None else None,
        "n_correct": n_correct,
        "covered": n_correct > 0,
        "n": n,
    }


def main():
    print("Loading rows...")
    arma = load_rows(B.ARMA_CACHE_DIR)
    d3 = load_rows(B.D3_DIR)
    arma_by = B.by_band(arma)
    d3_by = B.by_band(d3)

    print("\n" + "=" * 78)
    print("1) MEAN COMPLETION LENGTH (chars) and TRUNCATION / EMPTY RATES per band/arm")
    print("=" * 78)
    hdr = f"{'arm':6s} {'band':12s} {'n':>4s} {'mean_char':>10s} {'med_char':>9s} {'p90_char':>9s} {'trunc_native':>13s} {'unterm_proxy':>13s} {'empty':>7s}"
    print(hdr)
    pmap = {}
    for arm, by in [("arm_a", arma_by), ("D3", d3_by)]:
        for band in ("misselected", "control"):
            rows = by.get(band, [])
            if not rows:
                continue
            mets = [per_row_metrics(r) for r in rows]
            all_chars = [c for m in mets for c in m["char_lens"]]
            mean_char = float(np.mean(all_chars))
            med_char = float(np.median(all_chars))
            p90_char = float(np.percentile(all_chars, 90))
            # truncation rate = mean over problems of (num_truncated/64)
            tn = [m["frac_truncated_native"] for m in mets if m["frac_truncated_native"] is not None]
            trunc_native = float(np.mean(tn)) if tn else float("nan")
            unterm = float(np.mean([m["frac_unterm"] for m in mets]))
            empty = float(np.mean([m["frac_empty"] for m in mets]))
            pmap[(arm, band)] = {
                "n": len(rows), "mean_char": mean_char, "med_char": med_char,
                "p90_char": p90_char, "trunc_native": trunc_native,
                "unterm": unterm, "empty": empty, "mets": mets, "rows": rows,
            }
            tstr = f"{trunc_native:13.3f}" if not np.isnan(trunc_native) else f"{'N/A(arm_a)':>13s}"
            print(f"{arm:6s} {band:12s} {len(rows):4d} {mean_char:10.0f} {med_char:9.0f} {p90_char:9.0f} {tstr} {unterm:13.3f} {empty:7.4f}")

    print("\n  NOTE: trunc_native (num_truncated/64) is AVAILABLE FOR D3 ONLY; arm_a cache has no field.")
    print("        unterm_proxy = frac of completions NOT ending in boxed/EOS-punct (comparable across arms).")

    # length delta
    print("\n  Length delta (D3 - arm_a), mean chars:")
    for band in ("misselected", "control"):
        a = pmap.get(("arm_a", band)); d = pmap.get(("D3", band))
        if a and d:
            print(f"    {band:12s}: arm_a {a['mean_char']:.0f} -> D3 {d['mean_char']:.0f}  (delta {d['mean_char']-a['mean_char']:+.0f})")

    print("\n" + "=" * 78)
    print("2) D3 native truncation vs empty-canonical (does truncation -> empty/invalid?)")
    print("=" * 78)
    for band in ("misselected", "control"):
        d = pmap.get(("D3", band))
        if not d:
            continue
        nt = np.array([m["num_truncated"] for m in d["mets"]], dtype=float)
        ne = np.array([m["n_empty"] for m in d["mets"]], dtype=float)
        nu = np.array([m["n_unterm"] for m in d["mets"]], dtype=float)
        corr_te = np.corrcoef(nt, ne)[0, 1] if nt.std() > 0 and ne.std() > 0 else float("nan")
        corr_tu = np.corrcoef(nt, nu)[0, 1] if nt.std() > 0 and nu.std() > 0 else float("nan")
        print(f"  D3 {band:12s}: mean trunc/64={nt.mean()/64:.3f}  mean empty/64={ne.mean()/64:.4f}  "
              f"corr(trunc,empty)={corr_te:.3f}  corr(trunc,unterm_proxy)={corr_tu:.3f}")
    print("  -> if empty/64 ~ 0 and corr(trunc,empty) ~ 0, truncated completions STILL yield a")
    print("     non-empty canonical, so truncation does NOT mechanically kill coverage via empties.")

    print("\n" + "=" * 78)
    print("3) COVERAGE-LOSS SET: band problems where D3 covered=False but arm_a-cache covered=True")
    print("=" * 78)
    print("   (RTM-suspect: arm_a band coverage = 1.0 is CONSTRUCTION-GUARANTEED.)")
    a_band = pmap[("arm_a", "misselected")]
    d_band = pmap[("D3", "misselected")]
    a_cov = {r["problem_id"] for r, m in zip(a_band["rows"], a_band["mets"]) if m["covered"]}
    d_cov = {r["problem_id"] for r, m in zip(d_band["rows"], d_band["mets"]) if m["covered"]}
    a_band_pids = {r["problem_id"] for r in a_band["rows"]}
    lost = a_cov - d_cov  # arm_a covered, D3 not
    print(f"   arm_a band covered: {len(a_cov)} / {len(a_band_pids)}  (should be all = construction)")
    print(f"   D3    band covered: {len(d_cov)} / {len(d_band['rows'])}")
    print(f"   COVERAGE-LOST set (arm_a cov, D3 not): {len(lost)} problems")

    # D3 truncation/length on lost set vs rest of band
    d_lost_mets = [m for r, m in zip(d_band["rows"], d_band["mets"]) if r["problem_id"] in lost]
    d_rest_mets = [m for r, m in zip(d_band["rows"], d_band["mets"]) if r["problem_id"] not in lost]

    def summ(mets, label):
        if not mets:
            print(f"     {label}: (empty)")
            return
        tvals = [m["frac_truncated_native"] for m in mets if m["frac_truncated_native"] is not None]
        trstr = f"{np.mean(tvals):.3f}" if tvals else "N/A"
        un = np.mean([m["frac_unterm"] for m in mets])
        em = np.mean([m["frac_empty"] for m in mets])
        mc = np.mean([m["mean_char"] for m in mets])
        ncorr = np.mean([m["n_correct"] for m in mets])
        print(f"     {label:28s} n={len(mets):3d}  trunc/64={trstr:>5s}  unterm/64={un:.3f}  "
              f"empty/64={em:.4f}  mean_char={mc:.0f}  mean#correct={ncorr:.3f}")

    print("\n   D3 metrics on coverage-LOST set vs REST of band:")
    summ(d_lost_mets, "D3 coverage-LOST")
    summ(d_rest_mets, "D3 rest-of-band")

    # also arm_a length on those same problems (for length comparison on the lost set)
    a_lost_mets = [m for r, m in zip(a_band["rows"], a_band["mets"]) if r["problem_id"] in lost]
    a_rest_mets = [m for r, m in zip(a_band["rows"], a_band["mets"]) if r["problem_id"] not in lost]
    print("\n   arm_a metrics on the SAME coverage-lost problems vs rest:")
    summ(a_lost_mets, "arm_a on D3-lost set")
    summ(a_rest_mets, "arm_a on rest")

    print("\n" + "=" * 78)
    print("4) EMPTY-CANONICAL fraction overall on the band (arm_a vs D3)")
    print("=" * 78)
    for arm in ("arm_a", "D3"):
        d = pmap[(arm, "misselected")]
        print(f"   {arm:6s} band empty-canonical rate (mean over problems of empty/64): {d['empty']:.4f}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
