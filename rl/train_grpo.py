"""DAFT GRPO trainer.

Subclasses TRL's GRPOTrainer (pinned to 0.18.2) and overrides
``_generate_and_score_completions`` so that the per-sample *advantage* is computed
by ``rl.advantages.compute_advantages`` (decision-aware credit assignment) instead
of the canonical GRPO ``c_i - mean(c)`` reward normalization.

Design spec: rl/DESIGN.md sections 3-8. The override is a near-verbatim copy of
the upstream method body with the advantage block surgically replaced; a source
hash drift test (tests/test_trainer_hook.py) guards against silent TRL bumps.

Key facts established by reading the installed TRL 0.18.2 source
(``trl/trainer/grpo_trainer.py``):

* ``remove_unused_columns`` defaults to False in GRPOConfig, so extra dataset
  columns (notably ``answer``) survive into ``inputs`` and are exposed to reward
  functions via ``reward_kwargs``. We grade against the local ``answer`` column
  directly and gather it across processes -- this is the answer side-channel the
  source actually supports, so no problem_id dict is needed.
* The RepeatSampler lays out the generation batch as
  ``[p0 x G, p1 x G, ...]`` and accelerate shards it contiguously, so with our
  pinned geometry each group of ``num_generations`` lives entirely on one
  process. After ``gather()`` groups are contiguous, matching upstream's
  ``.view(-1, num_generations)``. We *assert* this by gathering per-completion
  prompt hashes and checking intra-group identity (catches silent scrambling).
* In colocate vLLM mode the completion token-level logprobs are discarded
  (only ``output.token_ids`` is kept) and ``old_per_token_logps`` is None when
  ``num_iterations == 1``. So per-token entropy is NOT cheaply available inside
  the copied body. Per DESIGN sec 6 the entropy tripwire therefore uses the
  ``distinct_classes_per_group`` halving proxy -- SAID SO LOUDLY in the callback.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import inspect
import json
import os
import warnings
from typing import Any, Union

import numpy as np
import torch
import yaml
from accelerate.utils import gather, gather_object

import trl
from trl import GRPOConfig, GRPOTrainer
from trl.trainer.grpo_trainer import (  # noqa: F401  (re-exported helpers used by copied body)
    nanstd,
    pad,
)
from trl.data_utils import is_conversational, maybe_apply_chat_template, apply_chat_template
from trl.extras.profiling import profiling_context

from datasets import Dataset
from transformers import TrainerCallback

import rl.rewards as rewards
import rl.advantages as advantages_mod


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

REQUIRED_TRL_VERSION = "0.18.2"

# sha256 of inspect.getsource(GRPOTrainer._generate_and_score_completions) for
# trl==0.18.2. Recomputed/checked by tests/test_trainer_hook.py. If TRL is bumped
# this MUST be re-reviewed -- the override copies that method body verbatim.
UPSTREAM_GASC_SHA256 = "db1c46e6e219e4089b7e3e28ed3b907d4475479f93992797a21b40a023b1f7b3"

ALLOWED_MODES = {"grpo", "pass_at_k", "vote_k", "vote_passk_hybrid",
                 "prm_margin", "prm_weighted", "prm_weighted_pm"}
VOTE_MODES = {"vote_k", "vote_passk_hybrid"}
# Phase-2 verifier-aware arms: require an in-loop PRM and a [prm] config block.
PRM_MODES = {"prm_margin", "prm_weighted", "prm_weighted_pm"}
# The scale_constants.json prm_* values were calibrated at this exponent
# (rl/analysis/calibrate_prm.py). w = clip(score,0)^p depends strongly on p, so the
# config exponent MUST match the calibration or cross-arm magnitude parity breaks.
PRM_CALIBRATED_EXPONENT = 4.0

SCALE_CONSTANTS_PATH = os.path.join(os.path.dirname(__file__), "configs", "scale_constants.json")


# --------------------------------------------------------------------------- #
# Pure advantage replacement (factored out for unit testing -- no torch dist)
# --------------------------------------------------------------------------- #

def daft_group_advantages(correct, canonicals, mode, k, scale, hybrid_lambda=0.25,
                          verifier_scores=None, prm_exponent=2.0):
    """Pure function: map (B,G) correctness + canonical strings -> (B*G,) advantages.

    This is the exact logic the override applies after gather. Factored out so it
    can be unit-tested on CPU with no torch.distributed / TRL machinery.

    Args:
        correct:    (B, G) array-like of {0,1} ints (truncated -> already 0).
        canonicals: list of B lists of G canonical-answer strings, OR a (B,G)
                    array of strings. Equivalence classes are computed per group
                    via rl.rewards.class_ids_from_canonicals.
        mode:       one of ALLOWED_MODES.
        k:          subset size for pass_at_k / vote_k.
        scale:      static per-arm scale constant (advantages are divided by it).
        hybrid_lambda: weight on pass@k term for vote_passk_hybrid (passed
                    through to rl.advantages.compute_advantages).
        verifier_scores: (B, G) per-completion aggregated PRM score in [0,1], or
                    None. Required for the prm_* modes (the deployed weighted-vote
                    weight is w = clip(score,0)^prm_exponent).
        prm_exponent: p for the prm_* modes' weight w = clip(score,0)^p.

    Returns:
        torch.FloatTensor of shape (B*G,), row-major (group 0 then group 1 ...),
        matching upstream's flattened (B,G).view(-1) ordering.
    """
    correct = np.asarray(correct, dtype=np.int64)
    B, G = correct.shape

    # Build per-group equivalence-class ids from canonical strings, using the
    # SAME canonicalization-derived class assignment as eval (rl.rewards).
    class_ids = np.zeros((B, G), dtype=np.int64)
    for b in range(B):
        group_canon = list(canonicals[b])
        assert len(group_canon) == G, f"group {b} has {len(group_canon)} canonicals, expected {G}"
        class_ids[b] = rewards.class_ids_from_canonicals(group_canon)

    vs = None
    empty_mask = None
    if verifier_scores is not None:
        vs = np.asarray(verifier_scores, dtype=np.float64)
        assert vs.shape == correct.shape, (
            f"verifier_scores shape {vs.shape} != correct shape {correct.shape}"
        )
        # drop-empty parity: mark unparseable-canonical completions (deployed rule
        # drops them from the weighted vote).
        empty_mask = np.array(
            [[(str(canonicals[b][g]) == "") for g in range(G)] for b in range(B)],
            dtype=bool,
        )

    adv = advantages_mod.compute_advantages(
        correct, class_ids, mode=mode, k=k, hybrid_lambda=hybrid_lambda,
        verifier_scores=vs, prm_exponent=prm_exponent, empty_mask=empty_mask,
    )  # (B,G) float64, NO normalization inside
    adv = np.asarray(adv, dtype=np.float64) / float(scale)
    return torch.tensor(adv.reshape(-1), dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Config validation (factored out + tested)
# --------------------------------------------------------------------------- #

def validate_config(cfg: dict) -> None:
    """Fail-fast review-mandated startup assertions on a loaded YAML config dict.

    Raises AssertionError with a clear message on any violation. Does NOT touch
    the runtime trainer -- pure on the config dict so tests can exercise it.
    """
    # --- TRL version + source-hash drift guard ---
    assert trl.__version__ == REQUIRED_TRL_VERSION, (
        f"trl version mismatch: installed {trl.__version__}, required "
        f"{REQUIRED_TRL_VERSION}. The override copies the 0.18.2 method body."
    )
    upstream_src = inspect.getsource(GRPOTrainer._generate_and_score_completions)
    got = hashlib.sha256(upstream_src.encode()).hexdigest()
    assert got == UPSTREAM_GASC_SHA256, (
        "TRL _generate_and_score_completions source hash drift!\n"
        f"  expected {UPSTREAM_GASC_SHA256}\n  got      {got}\n"
        "The DAFT override copied that method body verbatim -- re-review the "
        "upstream changes and update UPSTREAM_GASC_SHA256 before training."
    )

    mode = cfg.get("mode")
    assert mode in ALLOWED_MODES, f"mode {mode!r} not in {sorted(ALLOWED_MODES)}"

    grpo = cfg.get("grpo", {})
    G = grpo["num_generations"]
    pdtb = grpo["per_device_train_batch_size"]
    ga = grpo["gradient_accumulation_steps"]
    world = int(cfg.get("world_size", 1))
    k = cfg.get("k")

    assert grpo.get("scale_rewards") is False, (
        "args.scale_rewards must be False (Dr.GRPO-style; DESIGN sec 3). "
        f"got {grpo.get('scale_rewards')!r}"
    )
    assert grpo.get("num_iterations", 1) == 1, (
        f"num_iterations must be 1 (DESIGN sec 4); got {grpo.get('num_iterations')}"
    )

    global_completions = pdtb * world * ga
    assert global_completions % G == 0, (
        f"global completions (pdtb*world*grad_accum = {pdtb}*{world}*{ga} = "
        f"{global_completions}) must be divisible by num_generations ({G})."
    )
    assert global_completions >= 2 * G, (
        f"global completions ({global_completions}) must be >= 2*num_generations "
        f"({2 * G}); need at least 2 groups/step. Bump per_device_train_batch_size."
    )

    if mode in VOTE_MODES:
        assert k is not None and k <= G - 1, (
            f"vote modes require k <= num_generations-1; got k={k}, G={G}."
        )
    if mode in {"pass_at_k", "vote_passk_hybrid"} or mode in VOTE_MODES:
        assert k is not None and 1 <= k, f"k must be >= 1 for mode {mode}; got {k}."

    if mode in PRM_MODES:
        prm = cfg.get("prm", {})
        assert prm.get("path"), (
            f"mode {mode!r} (verifier-aware) requires a [prm] block with 'path' "
            "(the PRM model dir/id) in the config."
        )
        assert prm.get("agg", "last") in {"last", "min", "prod"}, (
            f"prm.agg must be one of last/min/prod (score_prm.aggregate_scores); "
            f"got {prm.get('agg')!r}."
        )
        assert "exponent" in prm, (
            f"verifier mode {mode!r} requires prm.exponent EXPLICITLY (the scale "
            f"constant was calibrated at p={PRM_CALIBRATED_EXPONENT}); no silent default."
        )
        assert float(prm["exponent"]) == PRM_CALIBRATED_EXPONENT, (
            f"prm.exponent={prm['exponent']} != calibration p={PRM_CALIBRATED_EXPONENT}; "
            "the scale_constants.json prm value would be miscalibrated. Recalibrate "
            "(rl/analysis/calibrate_prm.py) if you really want a different exponent."
        )
        # scale constant MUST be explicit for a production verifier arm (no silent
        # 1.0 fallback, which would reproduce the Round-1 magnitude confound).
        assert os.path.exists(SCALE_CONSTANTS_PATH), (
            f"scale_constants.json missing; verifier arm {mode!r} needs a calibrated "
            "scale (run rl/calibrate_scale.py). Refusing to fall back to 1.0."
        )
        with open(SCALE_CONSTANTS_PATH) as _f:
            _consts = json.load(_f)
        assert mode in _consts, (
            f"scale_constants.json has no calibrated entry for verifier mode {mode!r} "
            f"(keys: {list(_consts)}). Calibrate it (rl/calibrate_scale.py) before "
            "training -- a silent 1.0 scale would under/over-power the arm."
        )


# --------------------------------------------------------------------------- #
# Scale constants
# --------------------------------------------------------------------------- #

def load_scale_constant(mode: str) -> float:
    """Load the static per-arm advantage scale constant from scale_constants.json.

    Missing file -> scale 1.0 with a loud warning (DESIGN sec 4). Missing key for
    the mode -> also 1.0 + warning.
    """
    if not os.path.exists(SCALE_CONSTANTS_PATH):
        warnings.warn(
            "\n" + "=" * 70 + "\n"
            f"  scale_constants.json NOT FOUND at {SCALE_CONSTANTS_PATH}\n"
            "  Falling back to advantage scale = 1.0 for ALL arms.\n"
            "  Cross-arm scale parity (DESIGN sec 3) is NOT enforced -- run\n"
            "  rl/calibrate_scale.py on the pilot rollouts before full runs.\n"
            + "=" * 70,
            stacklevel=2,
        )
        return 1.0
    with open(SCALE_CONSTANTS_PATH) as f:
        consts = json.load(f)
    if mode not in consts:
        warnings.warn(
            f"scale_constants.json has no entry for mode {mode!r} "
            f"(keys: {list(consts)}); using scale = 1.0.",
            stacklevel=2,
        )
        return 1.0
    return float(consts[mode])


# --------------------------------------------------------------------------- #
# Entropy / diversity collapse tripwire
# --------------------------------------------------------------------------- #

class EntropyTripwireCallback(TrainerCallback):
    """Diversity-collapse tripwire (DESIGN sec 6).

    NOTE ON THE METRIC: in colocate vLLM mode the per-token completion logprobs
    are discarded by the copied method body, and with num_iterations==1 the
    per-token logps are not recomputed at generation time. True mean per-token
    entropy is therefore NOT cheaply available inside the step. As permitted by
    the task spec, we use ``daft/distinct_classes_per_group`` as the diversity
    proxy and SAY SO: if the proxy drops below 0.5x its step-10 value for 20
    consecutive steps we checkpoint, log loudly, and stop training.
    """

    PROXY_METRIC = "daft/distinct_classes_per_group"
    BASELINE_STEP = 10
    DROP_FACTOR = 0.5
    PATIENCE = 20

    def __init__(self):
        self.baseline = None          # value at step BASELINE_STEP
        self.consec_below = 0
        self._last_value = None

    def record(self, step: int, value: float):
        """Called from the trainer each step with the current proxy value."""
        self._last_value = value
        if step <= self.BASELINE_STEP:
            # keep refreshing until we pass the baseline step
            self.baseline = value
        # consecutive-below counting only after baseline is established
        if self.baseline is not None and step > self.BASELINE_STEP:
            if value < self.DROP_FACTOR * self.baseline:
                self.consec_below += 1
            else:
                self.consec_below = 0

    def on_step_end(self, args, state, control, **kwargs):
        # record() is fed identical (gathered) values on every process, so this
        # condition evaluates identically on all ranks -> consistent stop (no DDP
        # deadlock). Only the loud print is rank-0.
        if self.baseline is not None and self.consec_below >= self.PATIENCE:
            if state.is_world_process_zero:
                print(
                    "\n" + "!" * 72 + "\n"
                    f"  ENTROPY/DIVERSITY TRIPWIRE FIRED at step {state.global_step}.\n"
                    f"  Proxy metric '{self.PROXY_METRIC}' (distinct classes per group;\n"
                    f"  see callback docstring -- true token entropy unavailable in\n"
                    f"  colocate vLLM path) has been below {self.DROP_FACTOR}x its\n"
                    f"  step-{self.BASELINE_STEP} value ({self.baseline:.4f}) for "
                    f"{self.consec_below} consecutive steps.\n"
                    f"  Last value: {self._last_value!r}. Saving checkpoint and stopping.\n"
                    + "!" * 72,
                    flush=True,
                )
            control.should_save = True
            control.should_training_stop = True
        return control


# --------------------------------------------------------------------------- #
# The trainer
# --------------------------------------------------------------------------- #

class DAFTGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with decision-aware advantage computation.

    Pass a single dummy reward function (returns 0.0) so TRL's reward machinery
    runs but cannot influence anything -- we grade once ourselves inside the
    override and overwrite the advantages entirely.
    """

    def __init__(self, *args, daft_mode: str, daft_k: int, daft_scale: float,
                 daft_hybrid_lambda: float = 0.25, entropy_cb: EntropyTripwireCallback | None = None,
                 trunc_frac_warn: float = 0.30,
                 daft_prm_path: str | None = None, daft_prm_agg: str = "last",
                 daft_prm_exponent: float = 2.0, **kwargs):
        self.daft_mode = daft_mode
        self.daft_k = daft_k
        self.daft_scale = daft_scale
        self.daft_hybrid_lambda = daft_hybrid_lambda
        self.entropy_cb = entropy_cb
        self.trunc_frac_warn = trunc_frac_warn
        self.daft_prm_path = daft_prm_path
        self.daft_prm_agg = daft_prm_agg
        self.daft_prm_exponent = float(daft_prm_exponent)
        self.prm_model = None
        self.prm_tokenizer = None
        super().__init__(*args, **kwargs)
        # Runtime guards that need the constructed args
        assert self.args.scale_rewards is False, "scale_rewards must be False"
        assert self.num_iterations == 1, "num_iterations must be 1"
        assert self.daft_mode in ALLOWED_MODES, f"bad mode {self.daft_mode}"
        if self.daft_mode in VOTE_MODES:
            assert self.daft_k <= self.num_generations - 1, "vote modes need k <= G-1"
        # Phase-2 verifier-aware arms: load the in-loop PRM ON THIS RANK'S DEVICE,
        # AFTER super().__init__() so vLLM colocate has already reserved its
        # gpu_memory_utilization fraction (we take the remainder). beta=0 means no
        # ref model competes for that memory.
        if self.daft_mode in PRM_MODES:
            assert self.daft_prm_path, f"mode {self.daft_mode} requires daft_prm_path"
            self._load_prm()

    # -- in-loop PRM (verifier-aware arms only) ------------------------------ #
    def _load_prm(self):
        """Load the Qwen PRM in bf16, pinned to this rank's device (NOT
        device_map='auto', which would shard/collide with vLLM+DDP)."""
        from transformers import AutoModel, AutoTokenizer
        dev = self.accelerator.device
        rank = self.accelerator.process_index
        print(f"[DAFT][rank{rank}] loading in-loop PRM {self.daft_prm_path} -> {dev} "
              f"(agg={self.daft_prm_agg}, p={self.daft_prm_exponent})", flush=True)
        self.prm_tokenizer = AutoTokenizer.from_pretrained(
            self.daft_prm_path, trust_remote_code=True
        )
        self.prm_model = AutoModel.from_pretrained(
            self.daft_prm_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(dev).eval()
        for p in self.prm_model.parameters():
            p.requires_grad_(False)
        got = next(self.prm_model.parameters()).device
        assert got.type == "cuda", f"PRM not on cuda (got {got})"
        print(f"[DAFT][rank{rank}] PRM ready on {got}", flush=True)

    def _score_prm(self, question: str, completion_text: str) -> float:
        """Aggregated PRM score for one (question, completion). Reuses
        rl.score_prm.score_one_completion VERBATIM (has the per-step split fix) so
        in-loop scoring is IDENTICAL to the deployed offline eval.

        GUARDED: a single-rank PRM failure (OOM on a pathological completion, a
        transient CUDA error) must NOT raise -- otherwise that rank exits before the
        gather_object collective sequence and the other ranks deadlock until the
        NCCL watchdog timeout, burning the run. Fall back to 0.0 (the same worst-case
        value aggregate_scores([]) returns: the completion loses its vote weight).
        """
        from rl.score_prm import score_one_completion, aggregate_scores
        try:
            step_scores = score_one_completion(
                question, completion_text, self.prm_model, self.prm_tokenizer
            )
            return float(aggregate_scores(step_scores, agg_strategy=self.daft_prm_agg))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            warnings.warn(
                f"PRM OOM on one completion (rank {self.accelerator.process_index}); "
                "scoring 0.0 to keep the DDP collective in lockstep.", stacklevel=1
            )
            return 0.0
        except Exception as e:  # noqa: BLE001 - never let one score kill the 4-GPU job
            warnings.warn(
                f"PRM scoring failed (rank {self.accelerator.process_index}): {e!r}; "
                "scoring 0.0.", stacklevel=1
            )
            return 0.0

    # -- override: copied 0.18.2 body, advantage block replaced -------------- #
    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # ============================================================== #
        # BEGIN verbatim copy of GRPOTrainer._generate_and_score_completions
        # (trl 0.18.2). Diff vs upstream is confined to the clearly-marked
        # "DAFT ADVANTAGE BLOCK" below; everything else is unchanged so the
        # source-hash drift test stays meaningful.
        # ============================================================== #
        from contextlib import nullcontext
        from accelerate.utils import broadcast_object_list
        from trl.trainer.grpo_trainer import unwrap_model_for_generation, FSDP
        try:
            from vllm import SamplingParams
            from vllm.sampling_params import GuidedDecodingParams
        except Exception:  # pragma: no cover - login node has no vllm runtime
            SamplingParams = None
            GuidedDecodingParams = None

        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if self.accelerator.is_main_process:
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                    with profiling_context(self, "vLLM.generate"):
                        completion_ids = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                        )
                else:
                    completion_ids = [None] * len(all_prompts_text)
                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]

            elif self.vllm_mode == "colocate":
                if self.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(backend="outlines", regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None
                sampling_params = SamplingParams(
                    n=1,
                    repetition_penalty=self.repetition_penalty,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=-1 if self.top_k is None else self.top_k,
                    min_p=0.0 if self.min_p is None else self.min_p,
                    max_tokens=self.max_completion_length,
                    guided_decoding=guided_decoding,
                )

                if self.vllm_tensor_parallel_size > 1:
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
                else:
                    all_prompts_text = prompts_text

                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(all_prompts_text, sampling_params=sampling_params, use_tqdm=False)

                completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

                if self.vllm_tensor_parallel_size > 1:
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    completion_ids = completion_ids[tp_slice]

            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model:
                with (
                    FSDP.summon_full_params(self.model_wrapped, recurse=False)
                    if self.is_fsdp_enabled
                    else nullcontext()
                ):
                    prompt_completion_ids = unwrapped_model.generate(
                        prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                    )

            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
        ]

        completion_lengths = completion_mask.sum(1)

        # mask_truncated_completions is explicitly False (DESIGN sec 6): truncated
        # completions are kept and counted incorrect in BOTH reward and group
        # stats. We do NOT zero their completion_mask.
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            if self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                old_per_token_logps = None

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # We still run TRL's (dummy) reward funcs to keep the body faithful and to
        # populate the reward logging plumbing; their output is neutralized below.
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                import torch.nn as nn
                if isinstance(reward_func, nn.Module):
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super(GRPOTrainer, self)._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
                else:
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # (upstream NaN-row warning omitted: dummy reward always returns 0.0)

        rewards_per_func = gather(rewards_per_func)

        # ============================================================== #
        # ===================  DAFT ADVANTAGE BLOCK  =================== #
        # Replaces upstream lines computing:
        #     rewards = (rewards_per_func * reward_weights).nansum(dim=1)
        #     mean_grouped = rewards.view(-1,G).mean(1) ; advantages = rewards-mean
        #     if scale_rewards: advantages /= std+1e-4
        #     advantages = advantages[process_slice]
        #
        # Instead: grade ONCE here against the dataset 'answer' column, gather
        # correctness + canonicals + prompt-hashes across processes, reconstruct
        # (B,G) groups (asserting contiguity), compute decision-aware advantages,
        # divide by the static per-arm scale, slice back to the local process.
        # The dummy reward's `rewards_per_func` is intentionally discarded.
        # ============================================================== #

        # Grade local completions against the local 'answer' column. (answer
        # survives because GRPOConfig.remove_unused_columns defaults to False.)
        assert "answer" in reward_kwargs, (
            "dataset must carry an 'answer' column reaching the trainer; "
            f"available extra columns: {list(reward_kwargs)}"
        )
        local_answers = reward_kwargs["answer"]
        local_is_truncated = (~is_eos.any(dim=1)).tolist()  # python bools, len = local rows
        n_local = len(completions_text)
        G = self.num_generations

        # Raw problem text per local row (for the in-loop PRM, which applies its
        # OWN chat template). This is the dataset user-turn, NOT the chat-templated
        # prompts_text. Our prompts are [system, user] (last role 'user'), so the
        # upstream conversational pop() at the assistant-bootstrap branch above does
        # not fire and inputs[i]['prompt'][-1] is still the user turn.
        score_prm_inloop = self.daft_mode in PRM_MODES
        local_questions = None
        local_prm = []
        if score_prm_inloop:
            local_questions = [x["prompt"][-1]["content"] for x in inputs]
            assert all(("<|im_start|>" not in q) for q in local_questions), (
                "in-loop PRM question carries chat markup; expected raw problem text."
            )

        # Local rows are laid out as contiguous groups of G (RepeatSampler +
        # contiguous accelerate sharding -- verified in DESIGN/source notes), and
        # every completion in a group shares one ground-truth 'answer'. Grade one
        # group per grade_batch() call (the contract takes a single gt_answer);
        # cheaper than per-completion and matches the rewards.py API exactly.
        assert n_local % G == 0, (
            f"local rows {n_local} not divisible by num_generations {G}; "
            "group layout assumption violated."
        )
        local_correct = []
        local_canonical = []
        for gstart in range(0, n_local, G):
            group_texts = completions_text[gstart:gstart + G]
            group_answers = local_answers[gstart:gstart + G]
            # ground truth is constant within a group; assert it before batching.
            gt = group_answers[0]
            assert all(a == gt for a in group_answers), (
                "answers vary within a reconstructed group -- group layout broken."
            )
            graded = rewards.grade_batch(group_texts, gt)
            # PRM question is constant within a group (same problem).
            group_question = local_questions[gstart] if score_prm_inloop else None
            for j, g in enumerate(graded):
                is_corr = bool(g["correct"])
                if local_is_truncated[gstart + j]:
                    # truncated -> counts incorrect, but its canonical is KEPT so
                    # it forms a real voting bloc (DESIGN sec 3 fidelity).
                    is_corr = False
                local_correct.append(1 if is_corr else 0)
                local_canonical.append(g["canonical"])
                if score_prm_inloop:
                    # Score the SAME decoded completion text in the SAME order, so
                    # gather_object(local_prm) aligns with local_correct/canonical.
                    local_prm.append(self._score_prm(group_question, group_texts[j]))
            if score_prm_inloop:
                # Bound caching-allocator fragmentation across the G sequential PRM
                # forwards (cheap once per group; per-completion would be slow).
                torch.cuda.empty_cache()

        # Prompt-identity hashes for the silent-scramble assertion.
        local_prompt_hashes = [
            hashlib.sha256(pt.encode()).hexdigest()[:16] for pt in prompts_text
        ]

        # Gather everything across processes, aligned with the gathered
        # completion order. gather() on rewards_per_func already used the same
        # ordering; gather_object preserves [proc0..., proc1..., ...].
        all_correct = gather_object(local_correct)
        all_canonical = gather_object(local_canonical)
        all_prompt_hashes = gather_object(local_prompt_hashes)
        all_is_truncated = gather_object(local_is_truncated)
        all_completion_lengths = self.accelerator.gather(completion_lengths).tolist()
        all_prm = gather_object(local_prm) if score_prm_inloop else None

        n_total = rewards_per_func.size(0)  # gathered B*G
        assert len(all_correct) == n_total, (
            f"gathered correctness length {len(all_correct)} != gathered rewards "
            f"{n_total}; gather order misaligned."
        )
        assert len(all_canonical) == n_total and len(all_prompt_hashes) == n_total

        assert n_total % G == 0, f"gathered {n_total} not divisible by G={G}"
        B = n_total // G

        correct_arr = np.asarray(all_correct, dtype=np.int64).reshape(B, G)
        canon_grouped = [all_canonical[b * G:(b + 1) * G] for b in range(B)]
        hash_arr = np.asarray(all_prompt_hashes).reshape(B, G)
        trunc_arr = np.asarray([1 if t else 0 for t in all_is_truncated], dtype=np.int64).reshape(B, G)

        prm_arr = None
        if score_prm_inloop:
            assert len(all_prm) == n_total, (
                f"gathered PRM length {len(all_prm)} != gathered rewards {n_total}; "
                "PRM gather order misaligned with correctness."
            )
            prm_arr = np.asarray(all_prm, dtype=np.float64).reshape(B, G)

        # Group-identity guard: every completion in a reconstructed group must
        # share the same prompt hash. A failure means TRL's contiguity
        # assumption (the .view(-1, G) that we rely on) has been broken.
        for b in range(B):
            uniq = set(hash_arr[b].tolist())
            assert len(uniq) == 1, (
                f"group {b} is not prompt-homogeneous after gather (hashes={uniq}); "
                "the num_generations contiguity assumption is violated -- "
                "advantages would be computed across mismatched prompts."
            )

        # Decision-aware advantages (no normalization inside; static scale here).
        advantages_full = daft_group_advantages(
            correct_arr, canon_grouped, self.daft_mode, self.daft_k,
            self.daft_scale, hybrid_lambda=self.daft_hybrid_lambda,
            verifier_scores=prm_arr, prm_exponent=self.daft_prm_exponent,
        ).to(device)  # (B*G,)

        # Slice back to local process exactly as upstream does.
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages_full.clone()  # for logging
        advantages = advantages_full[process_slice]

        # ----------------------- DAFT diagnostics ----------------------- #
        # These quantities are computed from GATHERED arrays, so they are
        # identical on every process. We compute them on ALL ranks (cheap) and
        # feed the tripwire on ALL ranks, so the should_training_stop decision is
        # identical everywhere -- transformers does NOT broadcast TrainerControl
        # across processes, so a main-process-only stop would deadlock DDP.
        if mode == "train":
            adv_np = all_process_advantages.detach().cpu().numpy().reshape(B, G)
            group_has_nonzero = (np.abs(adv_np).sum(axis=1) > 1e-12)
            frac_zero_adv_groups = float(1.0 - group_has_nonzero.mean())
            frac_nonzero_adv_samples = float((np.abs(adv_np) > 1e-12).mean())
            adv_rms_prescale = float(np.sqrt(np.mean((adv_np * self.daft_scale) ** 2)))
            mean_correct = float(correct_arr.mean())
            distinct = []
            for b in range(B):
                cids = rewards.class_ids_from_canonicals(canon_grouped[b])
                distinct.append(len(set(np.asarray(cids).tolist())))
            distinct_classes_per_group = float(np.mean(distinct))
            trunc_frac = float(trunc_arr.mean())
            mean_completion_len = float(np.mean(all_completion_lengths))

            # PRM (verifier-aware arm) diagnostics: distinguishability from control,
            # PRM-score health, and the DEPLOYED decision quantity. Computed from
            # gathered arrays (identical on all ranks); logged on main only.
            prm_metrics = {}
            if score_prm_inloop and prm_arr is not None:
                w = np.clip(prm_arr, 0.0, None) ** self.daft_prm_exponent
                cmask = correct_arr > 0.5
                if cmask.any():
                    prm_metrics["daft/prm_mean_correct"] = float(prm_arr[cmask].mean())
                    prm_metrics["daft/prm_std_correct"] = float(prm_arr[cmask].std())
                    prm_metrics["daft/prm_frac_correct_hi"] = float((prm_arr[cmask] > 0.9).mean())
                if (~cmask).any():
                    prm_metrics["daft/prm_mean_wrong"] = float(prm_arr[~cmask].mean())
                coss, wins, ndeg = [], 0, 0
                for b in range(B):
                    c = correct_arr[b].astype(np.float64)
                    nc = int((c > 0.5).sum())
                    if nc == 0 or nc == G:
                        continue
                    ndeg += 1
                    a0 = c - c.mean()                       # implied control advantage
                    ad = adv_np[b]
                    n0, na = np.linalg.norm(a0), np.linalg.norm(ad)
                    if n0 > 1e-12 and na > 1e-12:
                        coss.append(float(np.dot(a0, ad) / (n0 * na)))
                    # deployed decision: top correct-class PRM-mass vs top wrong-class
                    # (drop-empty parity: exclude the unparseable-canonical class).
                    cids = rewards.class_ids_from_canonicals(canon_grouped[b])
                    mass, iscorr, isempty = {}, {}, {}
                    for i in range(G):
                        j = int(cids[i]); mass[j] = mass.get(j, 0.0) + float(w[b, i])
                        if c[i] > 0.5:
                            iscorr[j] = True
                        if str(canon_grouped[b][i]) == "":
                            isempty[j] = True
                    cm = max((mass[j] for j in mass if iscorr.get(j, False) and not isempty.get(j, False)), default=0.0)
                    wm = max((mass[j] for j in mass if not iscorr.get(j, False) and not isempty.get(j, False)), default=0.0)
                    if cm > wm:
                        wins += 1
                if coss:
                    prm_metrics["daft/cosine_to_control"] = float(np.mean(coss))
                if ndeg:
                    prm_metrics["daft/correct_class_mass_wins"] = wins / ndeg

            # Feed the diversity-collapse tripwire on EVERY process (identical
            # input -> identical state -> consistent stop decision).
            if self.entropy_cb is not None:
                self.entropy_cb.record(self.state.global_step, distinct_classes_per_group)

            # Metric logging + warnings only on main (TRL flushes _metrics there).
            if self.accelerator.is_main_process:
                daft_metrics = {
                    "daft/frac_zero_adv_groups": frac_zero_adv_groups,
                    "daft/frac_nonzero_adv_samples": frac_nonzero_adv_samples,
                    "daft/adv_rms_prescale": adv_rms_prescale,
                    "daft/mean_correct": mean_correct,
                    "daft/distinct_classes_per_group": distinct_classes_per_group,
                    "daft/trunc_frac": trunc_frac,
                    "daft/mean_completion_len": mean_completion_len,
                }
                daft_metrics.update(prm_metrics)
                for kname, v in daft_metrics.items():
                    self._metrics[mode][kname].append(v)

                if trunc_frac > self.trunc_frac_warn:
                    warnings.warn(
                        f"truncation fraction {trunc_frac:.2%} exceeds "
                        f"{self.trunc_frac_warn:.0%} (DESIGN sec 6 tripwire); "
                        "consider raising max_completion_length.",
                        stacklevel=1,
                    )

        # ------- end DAFT block; resume upstream (logging) verbatim ------ #
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._textual_logs["advantages"].extend(all_process_advantages.tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
        }


# --------------------------------------------------------------------------- #
# Dummy reward (neutralized): always 0.0
# --------------------------------------------------------------------------- #

def _dummy_reward(prompts, completions, completion_ids, **kwargs):
    """TRL reward stub. Returns 0.0 for every completion; its output is gathered
    by the copied body but DISCARDED in the DAFT advantage block. Present only so
    TRL's reward plumbing (logging, NaN checks) runs unchanged."""
    return [0.0] * len(completions)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def build_dataset(jsonl_path: str, stratify_band=None) -> Dataset:
    """Load RL train JSONL into an HF Dataset with conversational 'prompt' and an
    'answer' extra column.

    JSONL rows: {"problem","answer","level","type","C16":int,"weight":float}.
    'prompt' is built with the SAME system prompt as the sal eval
    (rl.rewards.SYSTEM_PROMPT, which is sal.config.Config().system_prompt).

    Arm-b stratification: if stratify_band=[lo,hi] is given, rows are upweighted
    toward C16 in [lo,hi] by *duplicating* rows proportional to their 'weight'
    (integerized). This is the simplest mechanism the Trainer supports without a
    custom sampler -- the RepeatSampler shuffles the (expanded) dataset, so
    duplication yields weighted-without-replacement-ish sampling. Documented as
    such; rows outside the band keep weight 1.0 (single copy). If 'weight' is
    absent we fall back to a band membership indicator (in-band x2).
    """
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    def to_example(r):
        return {
            "prompt": [
                {"role": "system", "content": rewards.SYSTEM_PROMPT},
                {"role": "user", "content": r["problem"]},
            ],
            "answer": r["answer"],
            "level": r.get("level", ""),
            "type": r.get("type", ""),
        }

    if stratify_band is not None:
        lo, hi = stratify_band
        expanded = []
        for r in rows:
            c16 = r.get("C16", None)
            in_band = (c16 is not None) and (lo <= c16 <= hi)
            if not in_band:
                expanded.append(to_example(r))
                continue
            w = r.get("weight", None)
            reps = max(1, int(round(w))) if w is not None else 2
            for _ in range(reps):
                expanded.append(to_example(r))
        examples = expanded
        print(
            f"[build_dataset] stratify_band={stratify_band}: expanded "
            f"{len(rows)} rows -> {len(examples)} (in-band rows duplicated by "
            f"round(weight)).",
            flush=True,
        )
    else:
        examples = [to_example(r) for r in rows]

    return Dataset.from_list(examples)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_grpo_config(cfg: dict, output_dir: str) -> GRPOConfig:
    """Construct GRPOConfig from the YAML 'grpo' block + fixed DAFT defaults."""
    g = dict(cfg["grpo"])  # copy
    g.setdefault("use_vllm", True)
    g.setdefault("vllm_mode", "colocate")
    g.setdefault("vllm_gpu_memory_utilization", 0.35)
    g.setdefault("vllm_tensor_parallel_size", 1)
    g.setdefault("temperature", 1.0)
    g.setdefault("top_p", 1.0)
    g.setdefault("beta", 0.001)
    g.setdefault("num_iterations", 1)
    g.setdefault("learning_rate", 1e-6)
    g.setdefault("lr_scheduler_type", "constant_with_warmup")
    g.setdefault("warmup_steps", 10)
    g.setdefault("bf16", True)
    g.setdefault("logging_steps", 1)
    g.setdefault("save_strategy", "steps")
    g.setdefault("save_total_limit", 2)
    g.setdefault("report_to", ["wandb"])
    g.setdefault("max_prompt_length", 1024)
    # hard pins
    g["scale_rewards"] = False
    g["num_iterations"] = 1
    g["mask_truncated_completions"] = False
    g.setdefault("seed", cfg.get("seed", 0))

    return GRPOConfig(output_dir=output_dir, **g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", default=None,
                    help="overrides YAML output_dir; falls back to YAML then ./outputs/<name>")
    ap.add_argument("--resume", action="store_true",
                    help="resume from latest checkpoint-* in output_dir if present")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    # world size for geometry validation: the LIVE world size (set by accelerate
    # launch) is authoritative; the YAML 'world_size' is only a documented
    # default for when WORLD_SIZE is unset (e.g. CPU import-time tests).
    if "WORLD_SIZE" in os.environ:
        cfg["world_size"] = int(os.environ["WORLD_SIZE"])
    else:
        cfg.setdefault("world_size", 1)

    # Fail-fast validation (TRL version, source hash, geometry, k, mode).
    validate_config(cfg)

    mode = cfg["mode"]
    k = cfg.get("k", 0)
    hybrid_lambda = cfg.get("hybrid_lambda", 0.25)
    scale = load_scale_constant(mode)
    prm_cfg = cfg.get("prm", {}) or {}

    output_dir = (
        args.output_dir
        or cfg.get("output_dir")
        or os.path.join("outputs", os.path.splitext(os.path.basename(args.config))[0])
    )
    os.makedirs(output_dir, exist_ok=True)

    grpo_config = build_grpo_config(cfg, output_dir)

    dataset = build_dataset(cfg["data"]["train_jsonl"], stratify_band=cfg.get("stratify_band"))

    entropy_cb = EntropyTripwireCallback()

    trainer = DAFTGRPOTrainer(
        model=cfg["model"],
        reward_funcs=_dummy_reward,
        args=grpo_config,
        train_dataset=dataset,
        daft_mode=mode,
        daft_k=k,
        daft_scale=scale,
        daft_hybrid_lambda=hybrid_lambda,
        entropy_cb=entropy_cb,
        trunc_frac_warn=cfg.get("trunc_frac_warn", 0.30),
        daft_prm_path=prm_cfg.get("path"),
        daft_prm_agg=prm_cfg.get("agg", "last"),
        daft_prm_exponent=float(prm_cfg.get("exponent", PRM_CALIBRATED_EXPONENT)),
        callbacks=[entropy_cb],
    )

    resume = None
    if args.resume:
        ckpts = sorted(
            glob.glob(os.path.join(output_dir, "checkpoint-*")),
            key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else -1,
        )
        if ckpts:
            resume = ckpts[-1]
            print(f"[resume] resuming from {resume}", flush=True)

    trainer.train(resume_from_checkpoint=resume)

    # Final HF-format dir loadable directly by vLLM (no Hub round-trip).
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    trainer.processing_class.save_pretrained(final_dir)
    print(f"[done] final model saved to {final_dir}", flush=True)


if __name__ == "__main__":
    main()
