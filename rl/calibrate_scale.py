"""
calibrate_scale.py — Compute per-arm static advantage-scale constants from sweep data.

For each arm mode, compute advantages on every group in sweep.jsonl and output
rl/configs/scale_constants.json {mode: rms} where rms is the global RMS over
ALL entries (including zeros).

Choice note: RMS is computed over ALL entries (including zeros) rather than
nonzero-only, because this equalizes total gradient power per batch across arms
(the gradient contribution from zero-advantage samples is zero, so only the
nonzero entries contribute; dividing by their per-arm scale constant makes the
magnitude of those contributions equal across arms regardless of sparsity).

All arms are scaled to advantage-RMS = 1.0 at the start of training.

Also prints pilot gate diagnostics:
  - frac zero-adv groups (all advantages in the group are zero)
  - frac nonzero samples (fraction of individual samples with nonzero advantage)
  - percentile distribution of nonzero advantages

These feed the DESIGN §3 pilot gates:
  vote_k hybrid decision at <15% nonzero samples.

Usage:
    python calibrate_scale.py [--sweep rl/data/sweep.jsonl] [--k 8]
    python calibrate_scale.py --sweep rl/data/sweep*.jsonl  (glob)
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RL_DIR = Path(__file__).parent
DATA_DIR = RL_DIR / "data"
CONFIGS_DIR = RL_DIR / "configs"

MODES = ["grpo", "pass_at_k", "vote_k", "vote_passk_hybrid"]


# ---------------------------------------------------------------------------
# Core calibration
# ---------------------------------------------------------------------------

def load_sweep_rows(sweep_glob: str) -> list[dict]:
    """Load all rows from sweep JSONL shard(s)."""
    paths = sorted(glob.glob(sweep_glob))
    if not paths:
        print(f"ERROR: No files found matching {sweep_glob!r}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    print(f"Loaded {len(rows)} sweep rows from {len(paths)} file(s).")
    return rows


def calibrate(
    rows: list[dict],
    k: int = 8,
    hybrid_lambda: float = 0.25,
) -> dict[str, float]:
    """Compute per-arm RMS constants from sweep rows.

    Parameters
    ----------
    rows : sweep JSONL rows (each with correct_mask, canonicals, C, G).
    k : subset size for pass_at_k and vote_k modes.
    hybrid_lambda : weight on pass@k in hybrid mode.

    Returns
    -------
    dict mapping mode -> global RMS of advantages (all entries, including zeros).
    """
    from rl.advantages import compute_advantages
    from rl.rewards import class_ids_from_canonicals

    # Build batch arrays: shape (B, G)
    B = len(rows)
    G = rows[0]["G"]

    correct_all = np.zeros((B, G), dtype=np.float64)
    class_ids_all = np.zeros((B, G), dtype=np.int64)

    for b, row in enumerate(rows):
        correct_all[b] = np.array(row["correct_mask"], dtype=np.float64)
        class_ids_all[b] = class_ids_from_canonicals(row["canonicals"])

    constants: dict[str, float] = {}
    diagnostics: dict[str, dict] = {}

    for mode in MODES:
        advs = compute_advantages(
            correct_all,
            class_ids_all,
            mode=mode,
            k=k,
            hybrid_lambda=hybrid_lambda,
        )  # shape (B, G)

        # Global RMS over ALL entries (including zeros)
        rms = float(np.sqrt(np.mean(advs ** 2)))
        constants[mode] = rms

        # Diagnostics
        group_all_zero = np.all(advs == 0, axis=1)  # (B,)
        frac_zero_groups = float(group_all_zero.mean())
        frac_nonzero_samples = float((advs != 0).mean())

        nonzero_advs = advs[advs != 0]
        if len(nonzero_advs) > 0:
            pcts = np.percentile(np.abs(nonzero_advs), [10, 25, 50, 75, 90, 95, 99])
        else:
            pcts = np.zeros(7)

        diagnostics[mode] = {
            "rms": rms,
            "frac_zero_groups": frac_zero_groups,
            "frac_nonzero_samples": frac_nonzero_samples,
            "n_nonzero_samples": int((advs != 0).sum()),
            "n_total_samples": int(advs.size),
            "abs_nonzero_pct": pcts.tolist(),  # [p10, p25, p50, p75, p90, p95, p99]
        }

    return constants, diagnostics


def print_diagnostics(constants: dict[str, float], diagnostics: dict[str, dict]) -> None:
    """Print per-mode calibration diagnostics."""
    print("\n" + "=" * 70)
    print("Advantage-scale calibration diagnostics")
    print("=" * 70)
    print(
        f"{'Mode':<25} {'RMS':>8} {'ZeroGrp%':>9} {'NzSamp%':>8} "
        f"{'p50|adv|':>9} {'p95|adv|':>9}"
    )
    print("-" * 70)
    for mode in MODES:
        d = diagnostics[mode]
        pcts = d["abs_nonzero_pct"]
        p50 = pcts[3] if len(pcts) > 3 else float("nan")  # index 3 = p75... actually p50 is index 2
        # pcts order: [p10, p25, p50, p75, p90, p95, p99]
        p50 = pcts[2] if len(pcts) > 2 else float("nan")
        p95 = pcts[5] if len(pcts) > 5 else float("nan")
        print(
            f"{mode:<25} {d['rms']:>8.5f} {d['frac_zero_groups']:>8.1%} "
            f"{d['frac_nonzero_samples']:>8.1%} {p50:>9.5f} {p95:>9.5f}"
        )
    print("=" * 70)

    print("\nPilot gate checks (DESIGN §3):")
    for mode in MODES:
        d = diagnostics[mode]
        frac_nz = d["frac_nonzero_samples"]
        if frac_nz < 0.15:
            print(
                f"  [{mode}] WARNING: nonzero-sample fraction {frac_nz:.1%} < 15% "
                "threshold. Consider switching to vote_passk_hybrid mode."
            )
        else:
            print(f"  [{mode}] nonzero-sample fraction {frac_nz:.1%} — OK")

    print("\nScale constants (divide advantages by these to achieve RMS=1.0):")
    for mode, rms in constants.items():
        if rms > 0:
            print(f"  {mode}: {rms:.6f}  (scale factor = {1.0/rms:.4f})")
        else:
            print(f"  {mode}: {rms:.6f}  (WARNING: zero RMS — no gradient signal!)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate per-arm advantage scale constants from sweep data."
    )
    parser.add_argument(
        "--sweep",
        default=str(DATA_DIR / "sweep*.jsonl"),
        help="Path or glob pattern for sweep JSONL file(s).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Subset size k for pass_at_k and vote_k modes.",
    )
    parser.add_argument(
        "--hybrid_lambda",
        type=float,
        default=0.25,
        help="Lambda weight for pass@k term in vote_passk_hybrid.",
    )
    parser.add_argument(
        "--out",
        default=str(CONFIGS_DIR / "scale_constants.json"),
        help="Output path for scale constants JSON.",
    )
    args = parser.parse_args()

    rows = load_sweep_rows(args.sweep)
    if not rows:
        print("ERROR: No rows loaded.", file=sys.stderr)
        sys.exit(1)

    print(f"Computing advantages for {len(rows)} groups, k={args.k}...")
    constants, diagnostics = calibrate(rows, k=args.k, hybrid_lambda=args.hybrid_lambda)

    print_diagnostics(constants, diagnostics)

    # Write output
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump(constants, f, indent=2)
    print(f"\nWrote scale constants to {out_path}")

    # Sanity check: all modes present including grpo
    for mode in MODES:
        assert mode in constants, f"Missing mode in output: {mode}"
    print("All modes present in output. Done.")


if __name__ == "__main__":
    main()
