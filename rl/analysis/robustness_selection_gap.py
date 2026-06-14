"""
Robustness of D3's selection|covered gain across verifier hyperparameters.

For arm_a (ARMA_CACHE_DIR) and D3 (D3_DIR), on bands {misselected, control}:
  selection|covered = mean wvote over problems that HAVE >=1 correct completion (covered),
  for agg in {last,min,prod} x p in {1,2,3,4}  (12 cells).
  Also plain_maj (no verifier) for reference.

Then report D3-minus-arm_a selection|covered GAP on misselected band for all 12 cells,
plus min/median/max of the gap.
"""
import statistics
from rl.analysis import banddata as B

AGGS = ["last", "min", "prod"]
PS = [1, 2, 3, 4]
BANDS = ["misselected", "control"]


def is_covered(row):
    cm = row["correct_mask"]
    return sum(1 for x in cm if x > 0.5) >= 1


def sel_covered_wvote(rows, how, p):
    """mean wvote over covered problems, aggregating step_scores by `how`."""
    vals = []
    for r in rows:
        if not is_covered(r):
            continue
        scores = [B.agg(s, how) for s in r["step_scores"]]
        vals.append(B.wvote(r["canonicals"], scores, r["correct_mask"], p=p, drop_empty=True))
    return (sum(vals) / len(vals), len(vals)) if vals else (float("nan"), 0)


def sel_covered_majority(rows):
    vals = []
    for r in rows:
        if not is_covered(r):
            continue
        vals.append(B.plain_maj(r["canonicals"], r["correct_mask"], drop_empty=True))
    return (sum(vals) / len(vals), len(vals)) if vals else (float("nan"), 0)


def main():
    bm = B.bandmap()
    arma = B.by_band(B.load(B.ARMA_CACHE_DIR, bm))
    d3 = B.by_band(B.load(B.D3_DIR, bm))

    # sanity: counts per band
    print("== Loaded row counts per band ==")
    for name, d in [("arm_a", arma), ("D3", d3)]:
        for b in BANDS:
            n = len(d.get(b, []))
            ncov = sum(1 for r in d.get(b, []) if is_covered(r))
            print(f"  {name:6s} {b:12s}: rows={n:4d}  covered={ncov:4d}")

    results = {}  # (system, band, how, p) -> (mean, ncov)
    maj = {}      # (system, band) -> (mean, ncov)
    for sysname, d in [("arm_a", arma), ("D3", d3)]:
        for b in BANDS:
            rows = d.get(b, [])
            maj[(sysname, b)] = sel_covered_majority(rows)
            for how in AGGS:
                for p in PS:
                    results[(sysname, b, how, p)] = sel_covered_wvote(rows, how, p)

    # ---- Full selection|covered tables per band ----
    for b in BANDS:
        print(f"\n== selection|covered  (band={b}) ==")
        print(f"{'agg':6s} {'p':>2s} | {'arm_a':>8s} {'D3':>8s} | {'gap(D3-arm_a)':>14s}")
        for how in AGGS:
            for p in PS:
                a, _ = results[("arm_a", b, how, p)]
                t, _ = results[("D3", b, how, p)]
                print(f"{how:6s} {p:>2d} | {a:8.4f} {t:8.4f} | {t-a:>14.4f}")
        am, anc = maj[("arm_a", b)]
        tm, tnc = maj[("D3", b)]
        print(f"{'maj':6s} {'-':>2s} | {am:8.4f} {tm:8.4f} | {tm-am:>14.4f}   (ncov arm_a={anc}, D3={tnc})")

    # ---- 12-cell GAP table on misselected band ----
    b = "misselected"
    gaps = []
    print(f"\n== D3-minus-arm_a selection|covered GAP, band={b} (12 cells) ==")
    print(f"{'agg':6s} | " + " ".join(f"p={p:>1d}".rjust(8) for p in PS))
    cell_gaps = {}
    for how in AGGS:
        row = []
        for p in PS:
            a, _ = results[("arm_a", b, how, p)]
            t, _ = results[("D3", b, how, p)]
            g = t - a
            cell_gaps[(how, p)] = g
            gaps.append(g)
            row.append(f"{g:>+8.4f}")
        print(f"{how:6s} | " + " ".join(row))

    gaps_sorted = sorted(gaps)
    print("\n== GAP summary (12 cells, band=misselected) ==")
    print(f"  min    = {min(gaps):+.4f}")
    print(f"  median = {statistics.median(gaps):+.4f}")
    print(f"  max    = {max(gaps):+.4f}")
    print(f"  mean   = {statistics.mean(gaps):+.4f}")
    n_pos = sum(1 for g in gaps if g > 0)
    print(f"  positive cells = {n_pos}/12")
    # deployed cell for reference
    print(f"  deployed cell (last,p=4) gap = {cell_gaps[('last',4)]:+.4f}")
    # majority gap on misselected
    am, _ = maj[("arm_a", b)]
    tm, _ = maj[("D3", b)]
    print(f"  plain_maj gap (no verifier) = {tm-am:+.4f}  (arm_a {am:.4f} -> D3 {tm:.4f})")


if __name__ == "__main__":
    main()
