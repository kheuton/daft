"""
banddata.py — shared, VERIFIED loaders + deployed-rule metrics for the band eval.
Import this from analysis scripts so every metric is computed identically.

Scored-shard row schema (rl/eval_outputs/*/scored/scored_shard*.jsonl):
  problem_id   str
  problem      str  (raw problem text)
  answer       str  (gold answer)
  completions  list[64] str   (full generated text)
  canonicals   list[64] str   (extracted canonical answer; "" = no/empty answer)
  correct_mask list[64] 0/1   (1 = canonical matches gold)
  num_truncated int            (# of the 64 that hit the token cap)
  agg_scores   list[64] float  (PRM score, last-agg, as written by score_prm)
  step_scores  list[64] list[float]  (per-step PRM scores; agg yourself for last/min/prod/mean)

Deployed rule (validated in de-risk): PRM-weighted-vote, agg=last, exponent p=4,
drop-empty (canonical=="" excluded), tie => fraction of tied-top classes correct.
"""
from __future__ import annotations
import glob, json, os
from collections import Counter, defaultdict

REPO = "/cluster/tufts/hugheslab/kheuto01/code/daft"
BAND_DATA = os.path.join(REPO, "rl", "analysis", "outputs", "band_eval_data.jsonl")

D3_DIR = os.path.join(REPO, "rl/eval_outputs/arm_d_prmmargin_band/scored")
ARMA_CACHE_DIR = os.path.join(REPO, "rl/analysis/outputs/prm_scored_steps")
ARMA_FRESH_DIR = os.path.join(REPO, "rl/eval_outputs/arm_a_fresh_band/scored")

# pool-projection constants (arm_a-defined partition)
N_DEAD, N_WON, N_BAND, N_POOL, BASE = 349, 1064, 578, 1991, 0.540


def bandmap():
    m = {}
    for line in open(BAND_DATA):
        line = line.strip()
        if line:
            r = json.loads(line)
            m[r["problem_id"]] = r["band"]
    return m


def agg(steps, how):
    if not steps:
        return 0.0
    xs = [float(x) for x in steps]
    if how == "last":
        return xs[-1]
    if how == "min":
        return min(xs)
    if how == "mean":
        return sum(xs) / len(xs)
    if how == "prod":
        p = 1.0
        for x in xs:
            p *= x
        return p
    raise ValueError(how)


def correct_classes(canon, cm):
    return {canon[i] for i in range(len(canon)) if cm[i] > 0.5}


def plain_maj(canon, cm, drop_empty=True):
    votes = Counter(c for c in canon if not (drop_empty and c == ""))
    if not votes:
        return 0.0
    top = max(votes.values())
    tied = [c for c, v in votes.items() if v == top]
    cc = correct_classes(canon, cm)
    return sum(1 for c in tied if c in cc) / len(tied)


def wvote(canon, scores, cm, p=4, drop_empty=True):
    """scores must already be aggregated to one float per completion."""
    mass = defaultdict(float)
    for c, s in zip(canon, scores):
        if drop_empty and c == "":
            continue
        mass[c] += max(float(s), 0.0) ** p
    if not mass:
        return 0.0
    top = max(mass.values())
    tied = [c for c, v in mass.items() if v == top]
    cc = correct_classes(canon, cm)
    return sum(1 for c in tied if c in cc) / len(tied)


def load(scored_dir, bm=None):
    """Return list of dict rows with band attached. Raises if a dir has no shards."""
    if bm is None:
        bm = bandmap()
    files = sorted(set(glob.glob(os.path.join(scored_dir, "scored_shard*.jsonl")) +
                       glob.glob(os.path.join(scored_dir, "*.jsonl"))))
    if not files:
        raise FileNotFoundError(f"no scored shards in {scored_dir}")
    rows = []
    for f in files:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "step_scores" not in r:
                continue
            pid = r.get("problem_id") or r.get("unique_id")
            r["problem_id"] = pid
            r["band"] = bm.get(pid, "?")
            r["last"] = [agg(s, "last") for s in r["step_scores"]]
            rows.append(r)
    return rows


def by_band(rows):
    d = defaultdict(list)
    for r in rows:
        d[r["band"]].append(r)
    return d
