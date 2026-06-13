"""
filter_sweep.py — π₀ rollouts on MATH train (vLLM offline).

Generates G=16 completions per problem, grades them, and writes per-problem
sweep statistics to JSONL.  Runs on a GPU node via SLURM array jobs.

Usage:
    python filter_sweep.py \
        --model_path /path/to/model \
        [--data rl/data/math_train_decontam.jsonl] \
        [--out rl/data/sweep.jsonl] \
        [--G 16] [--temperature 1.0] [--max_tokens 2048] \
        [--shard_index 0] [--num_shards 4]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RL_DIR = Path(__file__).parent
DATA_DIR = RL_DIR / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_flippable_maj8(correct_mask: list[int], canonicals: list[str]) -> bool:
    """True if there exists a size-8 subset whose majority answer is the correct class.

    Uses the advantages.maj_at_k_exact function: if maj@8 > 0 (i.e., there is
    at least one size-8 subset whose majority is correct), then flippable_maj8=True.

    This is equivalent to asking: does a size-8 subset whose majority is correct exist?
    With the tie-break convention (u = 1/|argmax|), maj_at_k_exact > 0 iff the correct
    class can win a plurality in *some* subset of size 8.
    """
    from rl.advantages import maj_at_k_exact
    from rl.rewards import class_ids_from_canonicals

    correct_arr = np.array(correct_mask, dtype=np.float64)
    # Check if any correct at all
    if correct_arr.sum() == 0:
        return False

    class_ids = class_ids_from_canonicals(canonicals)
    val = maj_at_k_exact(correct_arr, class_ids, k=8)
    return val > 0.0


def build_chat_messages(problem: str, system_prompt: str) -> list[dict]:
    """Build chat messages for a single problem."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]


def apply_chat_template(tokenizer, messages: list[dict]) -> str:
    """Apply the model's chat template to get the prompt string."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Main sweep logic
# ---------------------------------------------------------------------------

def run_sweep(args: argparse.Namespace) -> None:
    # Import here so the module is importable on CPU (for tests)
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # Import rl modules (concurrent agents write these)
    try:
        from rl.rewards import SYSTEM_PROMPT, grade_batch
    except ImportError as e:
        print(f"ERROR: Could not import rl.rewards: {e}", file=sys.stderr)
        sys.exit(1)

    # -- Load data --
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    all_problems: list[dict] = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                all_problems.append(json.loads(line))

    # -- Shard --
    shard_index = args.shard_index
    num_shards = args.num_shards
    problems = [
        (i, p) for i, p in enumerate(all_problems)
        if i % num_shards == shard_index
    ]
    print(
        f"Shard {shard_index}/{num_shards}: {len(problems)} problems "
        f"(of {len(all_problems)} total)"
    )

    # -- Output path --
    out_path = Path(args.out)
    if num_shards > 1:
        # Shard-specific output
        stem = out_path.stem
        out_path = out_path.parent / f"{stem}_shard{shard_index:02d}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # -- Load tokenizer for chat template --
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # -- Build prompts --
    print("Building chat prompts...")
    prompts = []
    for orig_idx, prob in problems:
        messages = build_chat_messages(prob["problem"], SYSTEM_PROMPT)
        prompt_str = apply_chat_template(tokenizer, messages)
        prompts.append(prompt_str)

    # -- Load model --
    print(f"Loading vLLM model from {args.model_path}...")
    llm = LLM(
        model=args.model_path,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        n=args.G,
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
    )

    # -- Generate --
    print(f"Generating {args.G} completions per problem for {len(prompts)} problems...")
    outputs = llm.generate(prompts, sampling_params)
    print("Generation complete.")

    # -- Grade and write --
    print(f"Grading and writing to {out_path}...")
    written = 0
    with open(out_path, "w") as fout:
        for req_idx, (orig_idx, prob) in enumerate(problems):
            output = outputs[req_idx]
            completions = [o.text for o in output.outputs]
            # Check for truncation
            finish_reasons = [o.finish_reason for o in output.outputs]

            # Grade. A malformed ground truth (e.g. empty answer field) is a
            # data bug for that row, not a reason to kill the shard — skip it;
            # absent rows are excluded from the RL set downstream anyway.
            try:
                results = grade_batch(completions, prob["answer"])
            except ValueError as e:
                print(f"  SKIP problem {orig_idx}: {e}")
                continue
            correct_mask = [int(r["correct"]) for r in results]
            canonicals = [r["canonical"] for r in results]

            # Mirror train_grpo.py semantics (C3 lines 554-555):
            # truncated completions are forced incorrect while their canonical
            # is kept, so that C / flippable_maj8 / calibration statistics are
            # computed under the same correctness semantics as the training reward.
            for i, fr in enumerate(finish_reasons):
                if fr == "length":
                    correct_mask[i] = 0

            C = sum(correct_mask)
            G = len(correct_mask)
            trunc_count = sum(1 for fr in finish_reasons if fr == "length")
            trunc_frac = trunc_count / G

            # Mean completion length (in characters)
            mean_len = float(np.mean([len(c) for c in completions]))

            # Flippability (can a majority-8 subset be correct?)
            flippable_maj8 = compute_flippable_maj8(correct_mask, canonicals)

            row = {
                "problem_id": orig_idx,
                "problem": prob["problem"],
                "answer": prob["answer"],
                "level": prob["level"],
                "type": prob["type"],
                "C": C,
                "G": G,
                "correct_mask": correct_mask,
                "canonicals": canonicals,
                "flippable_maj8": flippable_maj8,
                "mean_len": mean_len,
                "trunc_frac": trunc_frac,
            }
            fout.write(json.dumps(row) + "\n")
            written += 1

            if written % 100 == 0:
                print(f"  Progress: {written}/{len(problems)} problems written.")

    print(f"Done. Wrote {written} rows to {out_path}")

    # Tripwire
    if len(problems) > 0:
        all_rows: list[dict] = []
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_rows.append(json.loads(line))
        trunc_fracs = [r["trunc_frac"] for r in all_rows]
        mean_trunc = float(np.mean(trunc_fracs)) if trunc_fracs else 0.0
        print(f"Mean truncation fraction: {mean_trunc:.3f}")
        if mean_trunc > 0.30:
            print(
                f"WARNING: Truncation rate {mean_trunc:.1%} > 30% tripwire! "
                "Consider raising max_tokens or reducing generation length.",
                file=sys.stderr,
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run π₀ rollouts on MATH train for filter sweep."
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the model (HF format directory).",
    )
    parser.add_argument(
        "--data",
        default=str(DATA_DIR / "math_train_decontam.jsonl"),
        help="Path to the decontaminated train JSONL.",
    )
    parser.add_argument(
        "--out",
        default=str(DATA_DIR / "sweep.jsonl"),
        help="Output path for sweep JSONL (shard suffix added if num_shards > 1).",
    )
    parser.add_argument("--G", type=int, default=16, help="Completions per problem.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="0-based shard index for SLURM array.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=4,
        help="Total number of shards.",
    )
    args = parser.parse_args()

    assert 0 <= args.shard_index < args.num_shards, (
        f"shard_index {args.shard_index} out of range [0, {args.num_shards})"
    )

    run_sweep(args)


if __name__ == "__main__":
    main()
