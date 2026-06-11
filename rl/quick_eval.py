#!/usr/bin/env python3
"""
quick_eval.py — GPU offline eval using vLLM.

Generates n samples/problem, grades them, writes completions.jsonl and metrics.json.

Prompt-template decision (pinned):
  quick_eval uses the Qwen tokenizer's built-in chat template with
  add_generation_prompt=True, matching the training side (filter_sweep.py and
  TRL's maybe_apply_chat_template).  The historical sal PRM best_of_n / wmaj@n
  runs (pi_0 baseline numbers in DESIGN.md section 1) used sal's DEFAULT
  custom_chat_template (a Llama-3 override) WITHOUT add_generation_prompt.
  Those runs are NOT prompt-comparable to quick_eval metrics.

  Decision: quick_eval is the canonical eval harness for all RL arm comparisons
  (the pre-registered endpoints in DESIGN.md section 5).  When running the sal
  PRM reference eval (DESIGN.md section 5), set custom_chat_template=null and
  align add_generation_prompt handling so both harnesses see identical prompts —
  or document explicitly that the two metric sets are not prompt-comparable.

Must be invoked as a module from the repo root so that `rl` is on sys.path:
  cd /cluster/tufts/hugheslab/kheuto01/code/daft && python -m rl.quick_eval ...
Invoking as a plain script (`python /path/to/quick_eval.py`) puts rl/ at
sys.path[0] and causes `from rl.rewards import ...` to raise ImportError.

Stop tokens: [151645, 151643] for Qwen2 models.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(path: str):
    """Load a JSONL data file, returning a list of dicts.

    Supports math500.jsonl (fields: problem, answer, unique_id, subject, level,
    solution) and rl/data/math_test_1500.jsonl (fields: problem, answer,
    unique_id / problem_id, level, type).
    """
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_problem_id(record: dict) -> str:
    """Return a stable problem identifier."""
    return record.get("unique_id") or record.get("problem_id") or record["problem"][:80]


def get_system_prompt() -> str:
    """Return the system prompt from sal.config (single source of truth)."""
    try:
        from sal.config import Config
        return Config().system_prompt
    except Exception:
        # Fallback identical string if sal not importable in this context
        return (
            "Solve the following math problem efficiently and clearly:\n\n"
            "- For simple problems (2 steps or fewer):\n"
            "Provide a concise solution with minimal explanation.\n\n"
            "- For complex problems (3 steps or more):\n"
            "Use this step-by-step format:\n\n"
            "## Step 1: [Concise description]\n"
            "[Brief explanation and calculations]\n\n"
            "## Step 2: [Concise description]\n"
            "[Brief explanation and calculations]\n\n"
            "...\n\n"
            "Regardless of the approach, always conclude with:\n\n"
            "Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\n"
            "Where [answer] is just the final number or expression that solves the problem."
        )


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(all_results: list[dict], n_samples: int) -> dict:
    """Compute aggregate and per-problem metrics from graded completions.

    Parameters
    ----------
    all_results : list of per-problem dicts with keys:
        problem_id, problem, answer, completions, canonicals,
        correct_mask, num_truncated, token_counts
    n_samples : total samples per problem (G)

    Returns
    -------
    dict with per_problem list and aggregate dict.
    """
    # Import after we know we're in the right environment
    try:
        from rl.advantages import pass_at_n_unbiased, maj_at_k_estimate as maj_at_k_exact
    except ImportError:
        pass_at_n_unbiased = _pass_at_n_unbiased_fallback
        maj_at_k_exact = _maj_at_k_exact_fallback

    try:
        from rl.rewards import class_ids_from_canonicals
    except ImportError:
        class_ids_from_canonicals = _class_ids_from_canonicals_fallback

    k_values = [1, 2, 4, 8, 16, 32]
    per_problem = []

    agg_pass = {k: [] for k in k_values}
    agg_maj = {k: [] for k in k_values}
    agg_pass64_raw = []
    all_lengths = []
    all_truncated = []
    distinct_canonical_counts = []

    for res in all_results:
        correct_mask = np.array(res["correct_mask"], dtype=int)
        canonicals = res["canonicals"]
        n_total = len(correct_mask)
        num_correct = int(correct_mask.sum())
        token_counts = res.get("token_counts", [])

        class_ids = class_ids_from_canonicals(canonicals)

        pp = {"problem_id": res["problem_id"]}

        # unbiased pass@k
        for k in k_values:
            if k <= n_total:
                pp[f"pass_at_{k}"] = pass_at_n_unbiased(num_correct, n_total, k)
                agg_pass[k].append(pp[f"pass_at_{k}"])
            else:
                pp[f"pass_at_{k}"] = None

        # raw pass@64 (= any correct)
        pp["pass_at_64_raw"] = float(num_correct > 0)
        agg_pass64_raw.append(pp["pass_at_64_raw"])

        # exact maj@k — contract requires k <= G-1 (majority vote needs at least
        # two members to be meaningful; k=G is a degenerate single subset)
        for k in k_values:
            if 1 <= k <= n_total - 1:
                pp[f"maj_at_{k}"] = maj_at_k_exact(correct_mask, class_ids, k)
                agg_maj[k].append(pp[f"maj_at_{k}"])
            else:
                pp[f"maj_at_{k}"] = None

        # raw maj@64 — single subset, flagged as raw
        pp["maj_at_64_raw"] = _maj_at_64_raw(correct_mask, class_ids)

        # length / truncation
        if token_counts:
            mean_len = float(np.mean(token_counts))
            trunc_frac = res["num_truncated"] / len(token_counts)
        else:
            mean_len = None
            trunc_frac = None
        pp["mean_completion_length"] = mean_len
        pp["truncation_rate"] = trunc_frac
        if mean_len is not None:
            all_lengths.extend(token_counts)
        all_truncated.append(res["num_truncated"])

        # distinct canonical count
        distinct = len(set(c for c in canonicals if c != ""))
        pp["num_distinct_canonicals"] = distinct
        distinct_canonical_counts.append(distinct)

        per_problem.append(pp)

    # Aggregate
    agg = {}
    for k in k_values:
        vals = [v for v in agg_pass[k] if v is not None]
        agg[f"pass_at_{k}_mean"] = float(np.mean(vals)) if vals else None
    agg["pass_at_64_raw_mean"] = float(np.mean(agg_pass64_raw))

    for k in k_values:
        vals = [v for v in agg_maj[k] if v is not None]
        agg[f"maj_at_{k}_mean"] = float(np.mean(vals)) if vals else None

    # Note: maj@64 is a single subset (all n), flagged accordingly
    agg["maj_at_64_raw_note"] = "single full subset, not subset-averaged"

    if all_lengths:
        agg["mean_completion_length"] = float(np.mean(all_lengths))
        agg["truncation_rate"] = float(sum(all_truncated) / (len(all_truncated) * n_samples))
    else:
        agg["mean_completion_length"] = None
        agg["truncation_rate"] = None

    # Distinct canonical distribution
    dist_arr = np.array(distinct_canonical_counts)
    agg["distinct_canonical_distribution"] = {
        "mean": float(dist_arr.mean()),
        "median": float(np.median(dist_arr)),
        "min": int(dist_arr.min()),
        "max": int(dist_arr.max()),
        "histogram": {
            str(v): int((dist_arr == v).sum())
            for v in sorted(set(dist_arr.tolist()))
        },
    }

    return {"per_problem": per_problem, "aggregate": agg}


def _maj_at_64_raw(correct_mask: np.ndarray, class_ids: np.ndarray) -> float:
    """Majority vote on the full set (single subset, raw)."""
    from collections import Counter
    counts = Counter(class_ids.tolist())
    if not counts:
        return 0.0
    max_count = max(counts.values())
    argmax_ids = [cid for cid, cnt in counts.items() if cnt == max_count]
    # Determine which class ids are "correct" (those where correct_mask == 1)
    correct_ids = set(class_ids[correct_mask == 1].tolist())
    correct_in_argmax = [cid for cid in argmax_ids if cid in correct_ids]
    return float(len(correct_in_argmax)) / float(len(argmax_ids))


# ---------------------------------------------------------------------------
# Fallback implementations (used if rl.advantages / rl.rewards not importable)
# ---------------------------------------------------------------------------

def _pass_at_n_unbiased_fallback(num_correct: int, n_total: int, k: int) -> float:
    if n_total - num_correct < k:
        return 1.0
    result = 1.0
    for i in range(k):
        result *= (n_total - num_correct - i) / (n_total - i)
    return 1.0 - result


def _maj_at_k_exact_fallback(correct_mask: np.ndarray, class_ids: np.ndarray, k: int) -> float:
    """Brute-force fallback: enumerate C(G,k) subsets."""
    from itertools import combinations
    G = len(correct_mask)
    total = 0.0
    count = 0
    indices = list(range(G))
    for subset in combinations(indices, k):
        sub_ids = class_ids[list(subset)]
        sub_correct = correct_mask[list(subset)]
        from collections import Counter
        cnts = Counter(sub_ids.tolist())
        max_cnt = max(cnts.values())
        argmax = [cid for cid, c in cnts.items() if c == max_cnt]
        correct_ids = set(sub_ids[sub_correct == 1].tolist())
        correct_in_argmax = [cid for cid in argmax if cid in correct_ids]
        total += len(correct_in_argmax) / len(argmax)
        count += 1
    return total / count if count > 0 else 0.0


def _class_ids_from_canonicals_fallback(canonicals: list) -> np.ndarray:
    unique = {}
    ids = []
    for c in canonicals:
        if c not in unique:
            unique[c] = len(unique)
        ids.append(unique[c])
    return np.array(ids, dtype=np.int64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Quick eval: generate and grade n samples/problem")
    p.add_argument("--model_path", required=True)
    p.add_argument("--data", required=True, help="Path to JSONL data file")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n", type=int, default=64, help="Samples per problem")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    records = load_data(args.data)
    print(f"Loaded {len(records)} records from {args.data}")

    # Shard
    if args.num_shards > 1:
        records = [r for i, r in enumerate(records) if i % args.num_shards == args.shard_index]
        print(f"Shard {args.shard_index}/{args.num_shards}: {len(records)} records")

    assert len(records) > 0, "No records after sharding"

    # Import grading — hard-fail if rl.rewards is unavailable.
    # A silent fallback that marks every completion incorrect is worse than
    # failing loudly: it would produce all-zero metrics with exit code 0 and
    # trigger the brute-force maj@k fallback which hangs at n=64 (C(64,16)~5e14
    # subsets).  Run with `python -m rl.quick_eval` from the repo root so that
    # `rl` is on sys.path (as quick_eval.sbatch now does).
    try:
        from rl.rewards import grade_batch, SYSTEM_PROMPT
    except ImportError as e:
        raise ImportError(
            f"Could not import rl.rewards: {e}\n"
            "Run quick_eval as a module from the repo root:\n"
            "  cd /cluster/tufts/hugheslab/kheuto01/code/daft && "
            "python -m rl.quick_eval ...\n"
            "or set PYTHONPATH=/cluster/tufts/hugheslab/kheuto01/code/daft before "
            "invoking the script."
        ) from e

    # Build prompts — mirror sal best_of_n exactly
    from vllm import LLM, SamplingParams

    system_prompt = SYSTEM_PROMPT

    llm = LLM(
        model=args.model_path,
        seed=args.seed,
        # Let vLLM auto-select memory utilization
    )
    tokenizer = llm.get_tokenizer()

    stop_token_ids = (
        [151645, 151643]
        if "qwen2" in args.model_path.lower()
        else None
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop_token_ids=stop_token_ids,
        n=1,  # We duplicate prompts ourselves (same pattern as sal)
    )

    # Build conversations: each problem repeated n times
    problems = [r["problem"] for r in records]
    answers = [r.get("answer", "") for r in records]
    problem_ids = [get_problem_id(r) for r in records]

    # Duplicate: problem_i appears at indices [i*n .. (i+1)*n)
    convs = []
    for prob in problems:
        for _ in range(args.n):
            convs.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prob},
            ])

    templated = tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
    print(f"Example prompt:\n{templated[0][:500]}\n...\n")
    print(f"Generating {len(templated)} completions for {len(problems)} problems ...")

    responses = llm.generate(templated, sampling_params=sampling_params, use_tqdm=True)
    assert len(responses) == len(problems) * args.n, (
        f"Expected {len(problems) * args.n} responses, got {len(responses)}"
    )

    # Grade and write completions.jsonl
    completions_path = out_dir / "completions.jsonl"
    all_results = []

    print("Grading ...")
    with open(completions_path, "w") as fout:
        for i, (problem_id, problem, answer) in enumerate(zip(problem_ids, problems, answers)):
            start = i * args.n
            end = (i + 1) * args.n
            resp_slice = responses[start:end]

            comps = [r.outputs[0].text for r in resp_slice]
            token_counts = [len(r.outputs[0].token_ids) for r in resp_slice]
            num_truncated = sum(
                1 for r in resp_slice
                if r.outputs[0].finish_reason == "length"
            )

            graded = grade_batch(comps, answer)
            correct_mask = [int(g["correct"]) for g in graded]
            canonicals = [g["canonical"] for g in graded]

            row = {
                "problem_id": problem_id,
                "problem": problem,
                "answer": answer,
                "completions": comps,
                "canonicals": canonicals,
                "correct_mask": correct_mask,
                "num_truncated": num_truncated,
                "token_counts": token_counts,
            }
            all_results.append(row)
            # Write without token_counts for downstream consumers (keep file smaller)
            out_row = {k: v for k, v in row.items() if k != "token_counts"}
            fout.write(json.dumps(out_row) + "\n")

    print(f"Wrote {len(all_results)} rows to {completions_path}")

    # Compute and write metrics.json
    print("Computing metrics ...")
    metrics = compute_metrics(all_results, args.n)
    metrics["meta"] = {
        "model_path": args.model_path,
        "data": args.data,
        "n": args.n,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "num_problems": len(records),
    }

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote metrics to {metrics_path}")

    # Print summary
    agg = metrics["aggregate"]
    print("\n=== Aggregate metrics ===")
    for k in [1, 2, 4, 8, 16, 32]:
        print(f"  pass@{k}: {agg.get(f'pass_at_{k}_mean', 'N/A'):.4f}  "
              f"maj@{k}: {agg.get(f'maj_at_{k}_mean', 'N/A'):.4f}")
    print(f"  pass@64 (raw): {agg.get('pass_at_64_raw_mean', 'N/A'):.4f}")
    print(f"  mean length: {agg.get('mean_completion_length', 'N/A')}")
    print(f"  truncation rate: {agg.get('truncation_rate', 'N/A'):.4f}" if agg.get('truncation_rate') is not None else "  truncation rate: N/A")


if __name__ == "__main__":
    main()
