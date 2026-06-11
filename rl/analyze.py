#!/usr/bin/env python3
"""
analyze.py — CPU analysis: paired bootstrap, McNemar, scaling plots, power analysis.

Usage:
    python analyze.py \
        --runs baseline=path/to/completions.jsonl fine_tuned=path/to/completions.jsonl \
        --baseline baseline \
        --out_dir results/

    # Power mode
    python analyze.py --runs baseline=path/to/completions.jsonl \
        --baseline baseline --out_dir results/ --power
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers: import contract modules with graceful fallback
# ---------------------------------------------------------------------------

def _get_advantages_fns():
    try:
        from rl.advantages import pass_at_n_unbiased, maj_at_k_exact
        return pass_at_n_unbiased, maj_at_k_exact
    except ImportError:
        return _pass_at_n_unbiased_fallback, _maj_at_k_exact_fallback


def _get_class_ids_fn():
    try:
        from rl.rewards import class_ids_from_canonicals
        return class_ids_from_canonicals
    except ImportError:
        return _class_ids_from_canonicals_fallback


def _pass_at_n_unbiased_fallback(num_correct: int, n_total: int, k: int) -> float:
    if n_total - num_correct < k:
        return 1.0
    result = 1.0
    for i in range(k):
        result *= (n_total - num_correct - i) / (n_total - i)
    return 1.0 - result


def _maj_at_k_exact_fallback(correct_mask, class_ids, k: int) -> float:
    from itertools import combinations
    G = len(correct_mask)
    total = 0.0
    count = 0
    for subset in combinations(range(G), k):
        sub_ids = class_ids[list(subset)]
        sub_correct = correct_mask[list(subset)]
        cnts = Counter(sub_ids.tolist())
        max_cnt = max(cnts.values())
        argmax = [cid for cid, c in cnts.items() if c == max_cnt]
        correct_ids = set(sub_ids[sub_correct == 1].tolist())
        correct_in_argmax = [cid for cid in argmax if cid in correct_ids]
        total += len(correct_in_argmax) / len(argmax)
        count += 1
    return total / count if count > 0 else 0.0


def _class_ids_from_canonicals_fallback(canonicals):
    unique = {}
    ids = []
    for c in canonicals:
        if c not in unique:
            unique[c] = len(unique)
        ids.append(unique[c])
    return np.array(ids, dtype=np.int64)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_completions(path: str) -> dict:
    """Load completions.jsonl, returns {problem_id: row_dict}."""
    rows = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line.strip())
            pid = row.get("problem_id") or row.get("unique_id") or row["problem"][:80]
            row["problem_id"] = pid
            rows[pid] = row
    return rows


# ---------------------------------------------------------------------------
# Per-problem metric computation
# ---------------------------------------------------------------------------

def compute_per_problem_metrics(rows: dict, k_values=(1, 2, 4, 8, 16, 32)) -> dict:
    """Compute per-problem pass@k and maj@k metrics.

    Returns {problem_id: {metric_name: value}}.
    """
    pass_at_n_unbiased, maj_at_k_exact = _get_advantages_fns()
    class_ids_from_canonicals = _get_class_ids_fn()

    results = {}
    for pid, row in rows.items():
        correct_mask = np.array(row["correct_mask"], dtype=int)
        canonicals = row["canonicals"]
        n_total = len(correct_mask)
        num_correct = int(correct_mask.sum())

        class_ids = class_ids_from_canonicals(canonicals)

        pp = {}
        for k in k_values:
            if k <= n_total:
                pp[f"pass_at_{k}"] = pass_at_n_unbiased(num_correct, n_total, k)
            else:
                pp[f"pass_at_{k}"] = None

        pp["pass_at_64_raw"] = float(num_correct > 0)
        pp["num_correct"] = num_correct
        pp["n_total"] = n_total

        for k in k_values:
            # Contract (rl/advantages.py) requires 1 <= k <= G-1; k=G is degenerate
            # (a single subset of all items) and is excluded.  This matches the guard
            # in quick_eval.py and avoids AssertionError when n_total==k (e.g. n=16).
            if 1 <= k <= n_total - 1:
                pp[f"maj_at_{k}"] = maj_at_k_exact(correct_mask, class_ids, k)
            else:
                pp[f"maj_at_{k}"] = None

        results[pid] = pp
    return results


# ---------------------------------------------------------------------------
# Paired bootstrap CI
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(
    diffs: np.ndarray,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Bootstrap CI for the mean of per-problem differences.

    Primary machinery for all comparisons (see DESIGN.md section 5).

    Parameters
    ----------
    diffs : array of per-problem (arm - baseline) metric values
    Returns dict with mean_diff, ci_lower, ci_upper, p_value (two-sided bootstrap).
    """
    rng = np.random.RandomState(seed)
    n = len(diffs)
    mean_diff = float(np.mean(diffs))

    boot_means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = diffs[rng.randint(0, n, size=n)]
        boot_means[i] = sample.mean()

    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    # Bootstrap p-value (two-sided): shift distribution to null and count extremes
    shifted = boot_means - mean_diff
    p_value = float(2 * min(
        np.mean(shifted >= abs(mean_diff)),
        np.mean(shifted <= -abs(mean_diff)),
    ))
    p_value = max(p_value, 1.0 / n_resamples)  # minimum resolution

    return {
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_excludes_zero": bool(ci_lower > 0 or ci_upper < 0),
        "p_value_bootstrap": p_value,
    }


# ---------------------------------------------------------------------------
# McNemar exact test
# ---------------------------------------------------------------------------

def mcnemar_exact(b01: int, b10: int) -> float:
    """Exact McNemar p-value (two-sided) using binomial distribution.

    b01: cases where baseline=0, arm=1
    b10: cases where baseline=1, arm=0
    """
    from math import comb
    n = b01 + b10
    if n == 0:
        return 1.0
    # Two-sided: P(|X - n/2| >= |b01 - n/2|) where X ~ Bin(n, 0.5)
    k_obs = b01
    # Compute one-sided, then mirror
    p_one = sum(comb(n, k) * (0.5 ** n) for k in range(min(k_obs, n - k_obs) + 1))
    return min(1.0, 2 * p_one)


def mcnemar_on_binary(
    baseline_binary: np.ndarray,
    arm_binary: np.ndarray,
) -> dict:
    """Apply McNemar exact test on paired 0/1 per-problem outcomes."""
    assert len(baseline_binary) == len(arm_binary)
    b01 = int(((baseline_binary == 0) & (arm_binary == 1)).sum())
    b10 = int(((baseline_binary == 1) & (arm_binary == 0)).sum())
    p = mcnemar_exact(b01, b10)
    return {
        "b01": b01,
        "b10": b10,
        "p_value_mcnemar": p,
        "discordant_pairs": b01 + b10,
    }


# ---------------------------------------------------------------------------
# Power analysis
# ---------------------------------------------------------------------------

def power_analysis(
    baseline_correct_counts: np.ndarray,
    n_total: int,
    n_sims: int = 1000,
    alpha: float = 0.05,
    target_power: float = 0.80,
    seed: int = 0,
) -> dict:
    """Simulate MDE for maj@8 and pass@8 paired deltas.

    For a grid of effect sizes (shift in per-problem pass rate), simulate
    n_sims paired experiments and compute the power (fraction where paired
    bootstrap CI excludes zero).

    Parameters
    ----------
    baseline_correct_counts : array of shape (n_problems,) with integer correct counts
    n_total : samples per problem
    """
    pass_at_n_unbiased, maj_at_k_exact = _get_advantages_fns()
    class_ids_from_canonicals = _get_class_ids_fn()

    rng = np.random.RandomState(seed)
    n_problems = len(baseline_correct_counts)

    # Compute baseline per-problem pass@8 and maj@8
    def compute_pass8_maj8(correct_counts, n_total):
        pass8 = np.array([
            pass_at_n_unbiased(int(c), n_total, 8) for c in correct_counts
        ])
        # For maj@8, approximate using correct fraction (exact DP too slow in sim)
        # Build synthetic class_ids: all wrong are class 0, correct ones are class 1
        maj8 = []
        for c in correct_counts:
            c = int(np.clip(c, 0, n_total))
            mask = np.array([1] * c + [0] * (n_total - c))
            cids = np.array([1] * c + [0] * (n_total - c))
            maj8.append(maj_at_k_exact(mask, cids, 8))
        return pass8, np.array(maj8)

    baseline_pass8, baseline_maj8 = compute_pass8_maj8(baseline_correct_counts, n_total)

    # Grid of additive MDEs to test
    effect_grid = np.arange(0.01, 0.15, 0.005)
    powers_pass8 = []
    powers_maj8 = []

    for effect in effect_grid:
        reject_pass8 = 0
        reject_maj8 = 0
        for _ in range(n_sims):
            # Simulate arm: shift correct fraction by effect, re-draw counts
            arm_probs = np.clip(baseline_correct_counts / n_total + effect, 0.0, 1.0)
            arm_counts = rng.binomial(n_total, arm_probs)

            arm_pass8, arm_maj8 = compute_pass8_maj8(arm_counts, n_total)

            diffs_pass = arm_pass8 - baseline_pass8
            diffs_maj = arm_maj8 - baseline_maj8

            # Quick CI check: use simple t-based 95% CI (cheaper than bootstrap in sim)
            se_pass = np.std(diffs_pass, ddof=1) / math.sqrt(n_problems)
            se_maj = np.std(diffs_maj, ddof=1) / math.sqrt(n_problems)
            t_crit = 1.96

            if np.mean(diffs_pass) - t_crit * se_pass > 0:
                reject_pass8 += 1
            if np.mean(diffs_maj) - t_crit * se_maj > 0:
                reject_maj8 += 1

        powers_pass8.append(reject_pass8 / n_sims)
        powers_maj8.append(reject_maj8 / n_sims)

    # Find MDE at target power
    powers_pass8 = np.array(powers_pass8)
    powers_maj8 = np.array(powers_maj8)

    def find_mde(powers, effect_grid, target):
        indices = np.where(powers >= target)[0]
        if len(indices) == 0:
            return float("inf")
        return float(effect_grid[indices[0]])

    mde_pass8 = find_mde(powers_pass8, effect_grid, target_power)
    mde_maj8 = find_mde(powers_maj8, effect_grid, target_power)

    return {
        "n_problems": n_problems,
        "n_total_per_problem": n_total,
        "alpha": alpha,
        "target_power": target_power,
        "mde_pass8": mde_pass8,
        "mde_maj8": mde_maj8,
        "effect_grid": effect_grid.tolist(),
        "powers_pass8": powers_pass8.tolist(),
        "powers_maj8": powers_maj8.tolist(),
        "note": (
            "Parametric simulation, 1000 sims per effect size. "
            "CI check uses t-distribution (faster than bootstrap). "
            "Paired bootstrap is primary machinery for the actual analysis."
        ),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_scaling_curves(
    run_metrics: dict,  # {run_name: {metric_name: value}}
    k_values: list,
    metric_prefix: str,  # "pass_at_" or "maj_at_"
    title: str,
    out_path: Path,
    baseline_name: str = None,
):
    """Plot pass@n or maj@n vs n (log2 x-axis)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(run_metrics), 1)))

    for idx, (run_name, agg) in enumerate(run_metrics.items()):
        xs = []
        ys = []
        for k in k_values:
            val = agg.get(f"{metric_prefix}{k}_mean")
            if val is not None:
                xs.append(k)
                ys.append(val)

        style = dict(
            color=colors[idx],
            marker="o",
            linewidth=2.5 if run_name == baseline_name else 1.5,
            linestyle="--" if run_name == baseline_name else "-",
        )
        ax.plot(xs, ys, label=run_name, **style)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("n (log₂ scale)")
    ax.set_ylabel(metric_prefix.rstrip("_"))
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_diff_curves(
    diff_results: dict,  # {run_name: {metric: {mean_diff, ci_lower, ci_upper}}}
    k_values: list,
    metric_prefix: str,
    title: str,
    out_path: Path,
):
    """Plot per-run differences vs baseline with CI bands."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(diff_results), 1)))

    for idx, (run_name, metrics_dict) in enumerate(diff_results.items()):
        xs = []
        ys = []
        lowers = []
        uppers = []
        for k in k_values:
            key = f"{metric_prefix}{k}"
            r = metrics_dict.get(key)
            if r is not None:
                xs.append(k)
                ys.append(r["mean_diff"])
                lowers.append(r["ci_lower"])
                uppers.append(r["ci_upper"])

        color = colors[idx]
        ax.plot(xs, ys, marker="o", color=color, label=run_name, linewidth=1.5)
        ax.fill_between(xs, lowers, uppers, alpha=0.2, color=color)

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("n (log₂ scale)")
    ax.set_ylabel("Δ (arm − baseline)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_power_curve(power_result: dict, out_path: Path):
    """Plot power vs effect size for pass@8 and maj@8."""
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = power_result["effect_grid"]
    ax.plot(xs, power_result["powers_pass8"], label="pass@8", marker="o", markersize=3)
    ax.plot(xs, power_result["powers_maj8"], label="maj@8", marker="s", markersize=3)
    ax.axhline(0.8, color="gray", linestyle="--", linewidth=0.8, label="80% power")
    ax.set_xlabel("Additive effect size")
    ax.set_ylabel("Power")
    ax.set_title("Power analysis: paired delta at alpha=0.05")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Results markdown table
# ---------------------------------------------------------------------------

def write_results_md(
    run_names: list,
    agg_metrics: dict,  # {run_name: {metric: value}}
    diff_results: dict,  # {run_name: {metric: bootstrap_result}}
    k_values: list,
    out_path: Path,
    baseline_name: str,
):
    """Write results.md table: rows=metrics, cols=arms."""
    metric_keys_pass = [f"pass_at_{k}" for k in k_values] + ["pass_at_64_raw"]
    metric_keys_maj = [f"maj_at_{k}" for k in k_values]
    all_metrics = metric_keys_pass + metric_keys_maj

    lines = ["# Results", "", "**Primary machinery**: paired bootstrap (10 000 resamples over problems, seed 0).", "McNemar exact test applied to maj@k>0.5 indicator and pass@64 raw binary outcomes.", "CI excludes zero → bold.", ""]

    # Header
    header = "| Metric | " + " | ".join(run_names) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(run_names)) + " |"
    lines.append(header)
    lines.append(sep)

    for metric in all_metrics:
        row_parts = [metric]
        for run_name in run_names:
            agg = agg_metrics.get(run_name, {})
            val = agg.get(f"{metric}_mean", agg.get(metric))
            if val is None:
                row_parts.append("—")
                continue

            cell = f"{val:.4f}"
            if run_name != baseline_name:
                dr = diff_results.get(run_name, {}).get(metric)
                if dr is not None:
                    ci_lo = dr["ci_lower"]
                    ci_hi = dr["ci_upper"]
                    diff_str = f"{dr['mean_diff']:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}]"
                    cell = f"{val:.4f} ({diff_str})"
                    if dr["ci_excludes_zero"]:
                        cell = f"**{cell}**"
            row_parts.append(cell)

        lines.append("| " + " | ".join(row_parts) + " |")

    lines.append("")
    lines.append(f"*Baseline: {baseline_name}*")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Analyze paired eval runs")
    p.add_argument(
        "--runs",
        nargs="+",
        required=True,
        metavar="name=path/to/completions.jsonl",
        help="Repeatable name=path pairs",
    )
    p.add_argument("--baseline", required=True, help="Name of the baseline run")
    p.add_argument("--out_dir", required=True)
    p.add_argument(
        "--power",
        action="store_true",
        help="Run power analysis using baseline per-problem correct counts",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse runs
    runs = {}
    for spec in args.runs:
        if "=" not in spec:
            raise ValueError(f"Expected name=path, got: {spec!r}")
        name, path = spec.split("=", 1)
        runs[name] = path

    assert args.baseline in runs, (
        f"Baseline '{args.baseline}' not in runs: {list(runs.keys())}"
    )

    # Load all completions
    print("Loading completions ...")
    all_rows = {}
    for name, path in runs.items():
        all_rows[name] = load_completions(path)
        print(f"  {name}: {len(all_rows[name])} problems")

    # Assert same problem sets
    baseline_pids = set(all_rows[args.baseline].keys())
    for name, rows in all_rows.items():
        pids = set(rows.keys())
        missing = baseline_pids - pids
        extra = pids - baseline_pids
        assert not missing and not extra, (
            f"Run '{name}' has different problems: "
            f"{len(missing)} missing, {len(extra)} extra"
        )
    print(f"All runs share the same {len(baseline_pids)} problems.")

    k_values = [1, 2, 4, 8, 16, 32]

    # Compute per-problem metrics for all runs
    print("Computing per-problem metrics ...")
    all_pp_metrics = {}
    all_agg_metrics = {}
    for name, rows in all_rows.items():
        pp = compute_per_problem_metrics(rows, k_values=k_values)
        all_pp_metrics[name] = pp

        # Aggregate
        agg = {}
        for metric_key in [f"pass_at_{k}" for k in k_values] + ["pass_at_64_raw"] + [f"maj_at_{k}" for k in k_values]:
            vals = [v for v in (pp[pid].get(metric_key) for pid in pp) if v is not None]
            agg[f"{metric_key}_mean"] = float(np.mean(vals)) if vals else None
        all_agg_metrics[name] = agg

    # Paired analysis: each non-baseline run vs baseline
    run_names = list(runs.keys())
    problem_ids = sorted(baseline_pids)
    baseline_pp = all_pp_metrics[args.baseline]

    diff_results = {}  # {run_name: {metric: bootstrap_result}}
    mcnemar_results = {}  # {run_name: {metric: mcnemar_result}}

    for name in run_names:
        if name == args.baseline:
            continue
        arm_pp = all_pp_metrics[name]
        diff_results[name] = {}
        mcnemar_results[name] = {}

        print(f"\nPaired analysis: {name} vs {args.baseline}")

        for metric_key in [f"pass_at_{k}" for k in k_values] + ["pass_at_64_raw"] + [f"maj_at_{k}" for k in k_values]:
            # Build paired arrays — skip any problem where either run has None
            # for this metric (e.g. k==n_total for maj@k).  Coercing None to 0.0
            # would silently fabricate all-zero diffs and bogus paired_stats.json
            # entries for metrics that are legitimately absent at the run's n.
            paired = [
                (baseline_pp[pid].get(metric_key), arm_pp[pid].get(metric_key))
                for pid in problem_ids
            ]
            valid = [(b, a) for b, a in paired if b is not None and a is not None]
            if not valid:
                diff_results[name][metric_key] = None
                print(f"  {metric_key}: skipped (no valid pairs at this n)")
                continue

            baseline_vals = np.array([b for b, _ in valid])
            arm_vals = np.array([a for _, a in valid])
            diffs = arm_vals - baseline_vals

            boot = paired_bootstrap_ci(diffs, n_resamples=10000, seed=0)
            diff_results[name][metric_key] = boot
            sign = "+" if boot["mean_diff"] > 0 else ""
            ci_flag = " ** CI excl 0 **" if boot["ci_excludes_zero"] else ""
            n_skipped = len(paired) - len(valid)
            skip_note = f" ({n_skipped} problems skipped, metric absent)" if n_skipped else ""
            print(
                f"  {metric_key}: diff={sign}{boot['mean_diff']:.4f} "
                f"[{boot['ci_lower']:+.4f}, {boot['ci_upper']:+.4f}]{ci_flag}{skip_note}"
            )

        # McNemar on binary outcomes:
        # (1) maj@k > 0.5 indicator (majority correct) for k in [1, 2, 4, 8, 16, 32]
        # (2) pass@64 raw (any correct)
        # Note: pass@k unbiased estimator is fractional — bootstrap is primary.
        # McNemar is applied only to binary (0/1) per-problem outcomes.
        # Skip the test when the metric is None for any problem (e.g. k==n_total
        # for maj@k) — coercing None to 0.0 would fabricate bogus discordant pairs.
        for k in k_values:
            metric_key = f"maj_at_{k}"
            paired_bin = [
                (baseline_pp[pid].get(metric_key), arm_pp[pid].get(metric_key))
                for pid in problem_ids
            ]
            valid_bin = [(b, a) for b, a in paired_bin if b is not None and a is not None]
            if not valid_bin:
                mcnemar_results[name][f"maj_at_{k}_binary"] = None
                continue
            b_bin = np.array([int(b > 0.5) for b, _ in valid_bin])
            a_bin = np.array([int(a > 0.5) for _, a in valid_bin])
            mc = mcnemar_on_binary(b_bin, a_bin)
            mcnemar_results[name][f"maj_at_{k}_binary"] = mc

        # McNemar on pass@64 raw (any-correct binary)
        b_bin = np.array([int(baseline_pp[pid].get("pass_at_64_raw", 0)) for pid in problem_ids])
        a_bin = np.array([int(arm_pp[pid].get("pass_at_64_raw", 0)) for pid in problem_ids])
        mcnemar_results[name]["pass_at_64_raw"] = mcnemar_on_binary(b_bin, a_bin)

    # Write paired stats JSON
    stats_path = out_dir / "paired_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "diff_results": diff_results,
            "mcnemar_results": mcnemar_results,
            "aggregate_metrics": all_agg_metrics,
            "problem_ids": problem_ids,
            "note": (
                "Primary machinery: paired bootstrap (10000 resamples over problems, seed 0). "
                "McNemar applied to maj@k>0.5 indicator and pass@64 raw binary outcomes."
            ),
        }, f, indent=2)
    print(f"\nWrote {stats_path}")

    # Plots
    print("\nGenerating plots ...")

    # pass@n scaling curves
    plot_scaling_curves(
        run_metrics=all_agg_metrics,
        k_values=k_values,
        metric_prefix="pass_at_",
        title="pass@n vs n",
        out_path=out_dir / "pass_at_n.png",
        baseline_name=args.baseline,
    )

    # maj@n scaling curves
    plot_scaling_curves(
        run_metrics=all_agg_metrics,
        k_values=k_values,
        metric_prefix="maj_at_",
        title="maj@n vs n",
        out_path=out_dir / "maj_at_n.png",
        baseline_name=args.baseline,
    )

    # Diff plots
    if diff_results:
        plot_diff_curves(
            diff_results=diff_results,
            k_values=k_values,
            metric_prefix="pass_at_",
            title="pass@n diff (arm − baseline) with 95% CI",
            out_path=out_dir / "pass_diff.png",
        )
        plot_diff_curves(
            diff_results=diff_results,
            k_values=k_values,
            metric_prefix="maj_at_",
            title="maj@n diff (arm − baseline) with 95% CI",
            out_path=out_dir / "maj_diff.png",
        )

    # Results markdown table
    write_results_md(
        run_names=run_names,
        agg_metrics=all_agg_metrics,
        diff_results=diff_results,
        k_values=k_values,
        out_path=out_dir / "results.md",
        baseline_name=args.baseline,
    )

    # Power analysis
    if args.power:
        print("\nRunning power analysis ...")
        baseline_rows = all_rows[args.baseline]
        # Collect per-problem correct counts and infer n_total
        correct_counts = []
        n_total_values = []
        for pid in problem_ids:
            row = baseline_rows[pid]
            mask = np.array(row["correct_mask"], dtype=int)
            correct_counts.append(int(mask.sum()))
            n_total_values.append(len(mask))

        correct_counts = np.array(correct_counts)
        n_total = int(np.median(n_total_values))

        power_result = power_analysis(
            baseline_correct_counts=correct_counts,
            n_total=n_total,
            n_sims=1000,
            alpha=0.05,
            target_power=0.80,
            seed=0,
        )

        power_path = out_dir / "power_analysis.json"
        with open(power_path, "w") as f:
            json.dump(power_result, f, indent=2)
        print(f"  Wrote {power_path}")
        print(f"  MDE pass@8: {power_result['mde_pass8']:.4f}")
        print(f"  MDE maj@8:  {power_result['mde_maj8']:.4f}")

        plot_power_curve(power_result, out_dir / "power_curve.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
