#!/usr/bin/env python3
"""
score_prm.py — Offline PRM scoring for quick_eval completions.

For each row in a completions.jsonl produced by quick_eval.py, scores every
completion with the Qwen2.5-Math-PRM-7B (or another PRM) and writes the row
augmented with "agg_scores": [float per completion].

Aggregation strategy: "last" (sal default) — take the final step score.
Empty / truncated completions are scored normally; they produce a real PRM
score and participate in downstream voting.

Usage (from repo root so rl/ is on sys.path):
    python -m rl.score_prm \
        --completions path/to/completions.jsonl \
        --out path/to/scored.jsonl \
        --prm_path Qwen/Qwen2.5-Math-PRM-7B \
        --batch_size 4

Sharding (for SLURM array jobs):
    python -m rl.score_prm ... --shard_index 0 --num_shards 4

Must be invoked as a module from repo root:
    cd /cluster/tufts/hugheslab/kheuto01/code/daft && python -m rl.score_prm ...
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Aggregation (mirrors sal.utils.math.aggregate_scores / sal.utils.score)
# ---------------------------------------------------------------------------

def aggregate_scores(step_scores: list[float], agg_strategy: str = "last") -> float:
    """Aggregate a list of per-step PRM scores into a single scalar.

    Mirrors sal.utils.math.aggregate_scores with strategy "last" (sal default).
    Returns 0.0 when step_scores is empty (truncated / unparseable completion
    with no step-separator tokens produces an empty list; treat it as worst-case
    score so it loses best-of-n selection but still participates in voting).
    """
    if not step_scores:
        return 0.0
    if agg_strategy == "last":
        return float(step_scores[-1])
    elif agg_strategy == "min":
        return float(min(step_scores))
    elif agg_strategy == "prod":
        return float(math.prod(step_scores))
    else:
        raise ValueError(f"Unknown agg_strategy: {agg_strategy!r}")


# ---------------------------------------------------------------------------
# PRM loading — mirrors QWEN_PRM in sal/src/sal/models/reward_models.py
# ---------------------------------------------------------------------------

def load_qwen_prm(prm_path: str):
    """Load Qwen2.5-Math-PRM-7B (or compatible) in bfloat16.

    Returns (model, tokenizer).  Mirrors QWEN_PRM.load_model_and_tokenizer()
    but uses the caller-supplied prm_path and bfloat16 (daft_rl convention).
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading PRM from {prm_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(prm_path)
    model = AutoModel.from_pretrained(
        prm_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    print(f"  Model device: {next(model.parameters()).device}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Scoring: mirrors QWEN_PRM.score() exactly
# ---------------------------------------------------------------------------

QWEN_PRM_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def score_one_completion(
    question: str,
    completion: str,
    model,
    tokenizer,
) -> list[float]:
    """Score a single (question, completion) pair with the Qwen PRM.

    Mirrors QWEN_PRM.score() step-splitting logic:
      1. Collapse consecutive newlines to a single newline.
      2. Split on "\n\n" to get steps.
      3. Build a chat with the QWEN_PRM_SYSTEM_PROMPT system turn, user=question,
         assistant="<extra_0>".join(steps) + "<extra_0>".
      4. Tokenize and run the model.
      5. Extract logits at positions of <extra_0> (step_sep_id).
      6. Softmax over 2 classes, take probability of class index 1 (positive).

    Returns a list of per-step scores (may be empty if no step separator token
    appears in the tokenized sequence — e.g. for empty completions).
    """
    import torch
    import torch.nn.functional as F

    def make_step_rewards(logits, token_masks):
        """Mirror QWEN_PRM.score() make_step_rewards helper."""
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1)
        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]  # seq_len, num_labels
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        return all_scores_res

    # Step-split: collapse consecutive newlines, then split on double-newline
    comp = re.sub(r'\n+', '\n', completion)
    steps_list = comp.split("\n\n")

    messages = [
        {"role": "system", "content": QWEN_PRM_SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": "<extra_0>".join(steps_list) + "<extra_0>",
        },
    ]

    conversation = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    input_ids = tokenizer.encode(
        conversation,
        return_tensors="pt",
    ).to(next(model.parameters()).device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    step_sep_id = tokenizer.encode("<extra_0>")[0]
    token_masks = (input_ids == step_sep_id)
    step_scores_list = make_step_rewards(outputs[0], token_masks)

    # step_scores_list has one entry per batch element (we have batch size 1)
    if step_scores_list:
        return step_scores_list[0]
    return []


def score_row(
    row: dict,
    model,
    tokenizer,
    batch_size: int = 4,
) -> list[float]:
    """Score all completions in a row, returning agg_scores list.

    Processes completions in chunks of batch_size to bound GPU memory.
    Each completion is scored individually (the Qwen PRM API processes
    one completion at a time inside a loop; batching across completions
    requires padding different step-count sequences, which is not what
    sal does).  We call score_one_completion per completion and aggregate.
    """
    question = row["problem"]
    completions = row["completions"]
    agg_scores = []
    for i in range(0, len(completions), batch_size):
        chunk = completions[i : i + batch_size]
        for comp in chunk:
            step_scores = score_one_completion(question, comp, model, tokenizer)
            agg = aggregate_scores(step_scores, agg_strategy="last")
            agg_scores.append(agg)
    return agg_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Offline PRM scoring for completions.jsonl"
    )
    p.add_argument(
        "--completions",
        required=True,
        help="Path to completions.jsonl (output of quick_eval.py)",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output path for scored JSONL (each row + agg_scores field)",
    )
    p.add_argument(
        "--prm_path",
        default="Qwen/Qwen2.5-Math-PRM-7B",
        help="HuggingFace model id or local path for the PRM",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of completions to process per GPU batch (memory control)",
    )
    p.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Index of this shard (0-based)",
    )
    p.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of shards",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Load completions
    rows = []
    with open(args.completions) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} rows from {args.completions}")

    # Shard
    if args.num_shards > 1:
        rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_index]
        print(
            f"Shard {args.shard_index}/{args.num_shards}: {len(rows)} rows"
        )

    if not rows:
        print("WARNING: No rows after sharding — writing empty output.")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("")
        return

    # Load PRM
    model, tokenizer = load_qwen_prm(args.prm_path)

    # Score and write
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as fout:
        for row_idx, row in enumerate(rows):
            pid = row.get("problem_id") or row.get("unique_id") or row["problem"][:80]
            n_comps = len(row.get("completions", []))
            print(
                f"  [{row_idx + 1}/{len(rows)}] problem_id={pid!r}  "
                f"n_completions={n_comps}"
            )

            agg_scores = score_row(row, model, tokenizer, batch_size=args.batch_size)

            assert len(agg_scores) == len(row["completions"]), (
                f"agg_scores length mismatch: {len(agg_scores)} vs "
                f"{len(row['completions'])} completions for pid={pid!r}"
            )

            out_row = dict(row)
            out_row["agg_scores"] = agg_scores
            fout.write(json.dumps(out_row) + "\n")

    print(f"Wrote {len(rows)} scored rows to {out_path}")


if __name__ == "__main__":
    main()
