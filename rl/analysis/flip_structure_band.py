"""
Per-problem DEPLOYED-RULE FLIP STRUCTURE on the misselected band.

Join arm_a-cache (B.ARMA_CACHE_DIR) and D3 (B.D3_DIR) by problem_id on the
'misselected' band. For each problem:
  outcome = wvote(last, p=4, drop_empty=True)  (fraction; tie => fraction correct)
Classify arm_a -> D3 transition under two thresholds (>=0.5 and ==1.0):
  stay-right, RECOVERED, BROKE, stay-wrong.
Plus coverage checks (>=1 correct completion).
"""
from __future__ import annotations
from rl.analysis import banddata as B


def deployed(row):
    """Deployed-rule outcome fraction: wvote last-agg, p=4, drop-empty."""
    return B.wvote(row["canonicals"], row["last"], row["correct_mask"], p=4, drop_empty=True)


def coverage(row):
    """>=1 correct completion among the 64."""
    return sum(row["correct_mask"]) >= 1


def main():
    bm = B.bandmap()
    arma = B.load(B.ARMA_CACHE_DIR, bm)
    d3 = B.load(B.D3_DIR, bm)

    arma_band = {r["problem_id"]: r for r in arma if r["band"] == "misselected"}
    d3_band = {r["problem_id"]: r for r in d3 if r["band"] == "misselected"}

    pids = sorted(set(arma_band) & set(d3_band))
    print(f"arm_a band rows: {len(arma_band)}  D3 band rows: {len(d3_band)}  joined: {len(pids)}")

    results = {}
    for thr_name, is_correct in (("ge0.5", lambda x: x >= 0.5), ("eq1.0", lambda x: x == 1.0)):
        counts = {"stay-right": 0, "RECOVERED": 0, "BROKE": 0, "stay-wrong": 0}
        recovered_pids, staywrong_pids, broke_pids = [], [], []
        for pid in pids:
            a = is_correct(deployed(arma_band[pid]))
            d = is_correct(deployed(d3_band[pid]))
            if a and d:
                counts["stay-right"] += 1
            elif (not a) and d:
                counts["RECOVERED"] += 1
                recovered_pids.append(pid)
            elif a and (not d):
                counts["BROKE"] += 1
                broke_pids.append(pid)
            else:
                counts["stay-wrong"] += 1
                staywrong_pids.append(pid)
        net = counts["RECOVERED"] - counts["BROKE"]

        # Among RECOVERED: how many had D3 coverage (>=1 correct)?
        rec_cov = sum(1 for pid in recovered_pids if coverage(d3_band[pid]))
        # Among stay-wrong: how many did D3 LOSE coverage on (arm_a had it, D3 doesn't)?
        # arm_a coverage on the band is construction-guaranteed (all 578 had >=1 correct),
        # so "D3 lost coverage" = D3 has 0 correct.
        sw_d3_lostcov = sum(1 for pid in staywrong_pids if not coverage(d3_band[pid]))
        sw_arma_cov = sum(1 for pid in staywrong_pids if coverage(arma_band[pid]))

        print(f"\n=== threshold {thr_name} ===")
        print(f"  stay-right : {counts['stay-right']}")
        print(f"  RECOVERED  : {counts['RECOVERED']}")
        print(f"  BROKE      : {counts['BROKE']}")
        print(f"  stay-wrong : {counts['stay-wrong']}")
        print(f"  NET (rec-broke): {net}")
        print(f"  RECOVERED with D3 coverage(>=1 correct): {rec_cov}/{counts['RECOVERED']}")
        print(f"  stay-wrong total: {counts['stay-wrong']}; arm_a had coverage: {sw_arma_cov}; D3 LOST coverage (D3=0 correct): {sw_d3_lostcov}")
        results[thr_name] = dict(counts=counts, net=net, rec_cov=rec_cov,
                                 sw_d3_lostcov=sw_d3_lostcov, sw_arma_cov=sw_arma_cov,
                                 n_recovered=counts["RECOVERED"], n_staywrong=counts["stay-wrong"])

    # Sanity: arm_a band coverage should be ~all (construction-guaranteed)
    arma_cov_all = sum(1 for pid in pids if coverage(arma_band[pid]))
    d3_cov_all = sum(1 for pid in pids if coverage(d3_band[pid]))
    print(f"\nSanity: arm_a coverage on joined band: {arma_cov_all}/{len(pids)}  "
          f"D3 coverage: {d3_cov_all}/{len(pids)}")

    return results


if __name__ == "__main__":
    main()
