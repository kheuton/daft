"""
redecode_logprobs.py — re-decode target problems WITH token logprobs (Phase 1a).

For each target problem (rl/analysis/outputs/redecode_targets.jsonl), generate
n samples at T=1.0 with the SAME prompt/stop/template as rl.quick_eval, capture
per-sample confidence features from vLLM logprobs, grade with rl.rewards, and
write per-problem rows. Downstream rl.analysis.analyze_confidence then asks the
decisive question: can the model's own confidence pick the correct minority over
its confidently-wrong plurality?

Prompt parity with quick_eval (CRITICAL): system=SYSTEM_PROMPT, Qwen chat
template with add_generation_prompt=True, stop_token_ids=[151645,151643].

Run (module, from repo root):
  python -m rl.analysis.redecode_logprobs --model_path .../arm_a_grpo/final \
      --targets rl/analysis/outputs/redecode_targets.jsonl \
      --out_dir rl/analysis/outputs/redecode --num_shards 4 --shard_index 0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def load_targets(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def confidence_features(out):
    """Per-completion confidence from a vLLM CompletionOutput.

    Returns dict of scalars:
      cum_lp   : sum of sampled-token logprobs (vLLM cumulative_logprob)
      n_tokens : number of generated tokens
      mean_lp  : length-normalized confidence  = cum_lp / n_tokens
      tail_lp  : mean logprob of the last <=50 tokens (answer region proxy)
      min_lp   : least-confident single token
    """
    n_tokens = len(out.token_ids)
    cum_lp = float(out.cumulative_logprob) if out.cumulative_logprob is not None else float("nan")
    per_tok = None
    if out.logprobs is not None:
        per_tok = []
        for t, tid in enumerate(out.token_ids):
            lpdict = out.logprobs[t]
            lp = lpdict.get(tid)
            per_tok.append(lp.logprob if lp is not None else float("nan"))
    if per_tok:
        arr = np.array(per_tok, dtype=float)
        mean_lp = float(np.nanmean(arr))
        tail_lp = float(np.nanmean(arr[-50:]))
        min_lp = float(np.nanmin(arr))
    else:
        mean_lp = cum_lp / n_tokens if n_tokens else float("nan")
        tail_lp = mean_lp
        min_lp = float("nan")
    return {"cum_lp": cum_lp, "n_tokens": n_tokens, "mean_lp": mean_lp,
            "tail_lp": tail_lp, "min_lp": min_lp}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--targets", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = load_targets(args.targets)
    if args.num_shards > 1:
        targets = [t for i, t in enumerate(targets) if i % args.num_shards == args.shard_index]
    assert targets, "no targets after sharding"
    print(f"shard {args.shard_index}/{args.num_shards}: {len(targets)} problems")

    from rl.rewards import grade_batch, SYSTEM_PROMPT
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model_path, seed=args.seed)
    tokenizer = llm.get_tokenizer()
    stop_token_ids = [151645, 151643] if "qwen2" in args.model_path.lower() else None

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop_token_ids=stop_token_ids,
        n=1,
        logprobs=1,  # populate sampled-token logprobs + cumulative_logprob
    )

    convs = []
    for t in targets:
        for _ in range(args.n):
            convs.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": t["problem"]},
            ])
    templated = tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
    print(f"Example prompt:\n{templated[0][:400]}\n...")
    print(f"Generating {len(templated)} completions ...")
    responses = llm.generate(templated, sampling_params=sampling_params, use_tqdm=True)
    assert len(responses) == len(targets) * args.n

    out_path = out_dir / f"shard{args.shard_index}.jsonl"
    with open(out_path, "w") as fout:
        for i, t in enumerate(targets):
            sl = responses[i * args.n:(i + 1) * args.n]
            comps = [r.outputs[0].text for r in sl]
            feats = [confidence_features(r.outputs[0]) for r in sl]
            num_trunc = sum(1 for r in sl if r.outputs[0].finish_reason == "length")
            graded = grade_batch(comps, t["answer"])
            row = {
                "problem_id": t["problem_id"],
                "band": t["band"],
                "answer": t["answer"],
                "canonicals": [g["canonical"] for g in graded],
                "correct_mask": [int(g["correct"]) for g in graded],
                "lengths": [len(c) for c in comps],
                "num_truncated": num_trunc,
                "cum_lp":   [f["cum_lp"] for f in feats],
                "n_tokens": [f["n_tokens"] for f in feats],
                "mean_lp":  [f["mean_lp"] for f in feats],
                "tail_lp":  [f["tail_lp"] for f in feats],
                "min_lp":   [f["min_lp"] for f in feats],
            }
            fout.write(json.dumps(row) + "\n")
    print(f"wrote {out_path}  ({len(targets)} problems)")


if __name__ == "__main__":
    main()
