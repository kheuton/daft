"""
rl/rewards.py  — correctness grader for DAFT RL training (owner: C2)

Public API (shared contract):
    canonical_answer(completion_text: str) -> str
    grade_batch(completions, gt_answer, timeout_s=8.0) -> list[dict]
    class_ids_from_canonicals(canonicals: list[str]) -> np.ndarray
    gt_canonical(answer: str) -> str
    SYSTEM_PROMPT: str

Design notes:
    - canonical_answer: extract via extract_answer(text, 'math'), then run
      memoized_canonical_form — the exact same function compute_maj_pred uses
      for vote grouping. This guarantees training/eval canonicalization parity
      (DESIGN.md §3 A3 fidelity, §6 grader bullet).
    - grade_batch: uses pebble.ProcessPool with per-future timeout so that
      sympy hangs cannot wedge training (signal.SIGALRM does not fire off the
      main thread inside TRL's trainer). Per DESIGN.md §6 grader bullet.
    - Pool is a lazily-created module singleton; workers are safe to spawn from
      non-main threads (pebble uses futures, not signals).
    - Extraction (extract_answer + memoized_canonical_form for the raw string
      part) is cheap and done outside the pool; only math_equal/canonical_form
      for sympy go into the pool so that the canonical string is always
      returned even on timeout.
    - Thread safety (Bug C2-1 fix): sal's memoized_canonical_form calls
      signal.signal(SIGALRM) which raises ValueError off the main thread and
      falls back to strip_string, poisoning the shared Manager cache.  When
      canonical_answer is called from a non-main thread, memoized_canonical_form
      is routed through the pebble pool via _canonical_in_worker so that SIGALRM
      runs safely inside a child process and the correct canonical is written to
      the shared cache.
    - Empty ground-truth guard (Bug C2-2 fix): grade_batch raises ValueError
      when gt_answer is empty or does not canonicalize to a non-empty string,
      matching sal's compute_pass_at_k behaviour.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import List

import numpy as np
import pebble

# ---------------------------------------------------------------------------
# Imports from sal (single source of truth for canonicalization and prompts)
# ---------------------------------------------------------------------------
from sal.config import Config
from sal.utils.grader import math_equal
from sal.utils.math import memoized_canonical_form
from sal.utils.qwen_math_parser import extract_answer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — single source of truth is sal.config.Config.system_prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT: str = Config().system_prompt

# ---------------------------------------------------------------------------
# Pool singleton — created lazily, protected by a lock so multiple threads
# calling grade_batch concurrently don't race on initialisation.
# ---------------------------------------------------------------------------
_pool: pebble.ProcessPool | None = None
_pool_lock = threading.Lock()
_POOL_MAX_WORKERS = 8


def _get_pool() -> pebble.ProcessPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = pebble.ProcessPool(max_workers=_POOL_MAX_WORKERS)
    return _pool


# ---------------------------------------------------------------------------
# Worker functions — run in child processes (must be module-level, picklable)
# ---------------------------------------------------------------------------
def _grade_one_worker(gt_canon: str, completion_canon: str) -> bool:
    """Run math_equal in a child process (sympy-safe, may be slow/hang)."""
    from sal.utils.grader import math_equal  # re-import in worker process
    return bool(math_equal(gt_canon, completion_canon))


def _canonical_in_worker(expression: str) -> str:
    """Run memoized_canonical_form in a child process (SIGALRM-safe).

    sal's memoized_canonical_form uses signal.SIGALRM which raises ValueError
    when called off the main thread.  Routing through the pebble pool ensures
    the call always executes in the main thread of a fresh child process, and
    the result is written to the shared Manager cache so subsequent calls
    (from any thread or process) get the correct canonical form.
    """
    from sal.utils.math import memoized_canonical_form  # re-import in worker
    return memoized_canonical_form(expression)


# ---------------------------------------------------------------------------
# Public: canonical_answer
# ---------------------------------------------------------------------------
def _memoized_canonical_form_safe(expression: str, timeout_s: float = 8.0) -> str:
    """Call memoized_canonical_form, routing off-main-thread calls through the pool.

    sal's memoized_canonical_form uses signal.SIGALRM which is only valid from
    the main thread.  When called from a non-main thread the signal.signal() call
    raises ValueError, which is swallowed by the blanket except and the
    strip_string fallback is then written to the shared Manager cache, poisoning
    subsequent main-thread lookups.

    Fix: when we are not on the main thread, schedule _canonical_in_worker in a
    pebble child process where SIGALRM is safe.  The worker writes the correct
    canonical to the shared cache, so any later call (main or non-main) hits the
    cache with the right value.
    """
    if threading.current_thread() is threading.main_thread():
        return memoized_canonical_form(expression)

    # Off main thread: route through the process pool so SIGALRM is safe.
    pool = _get_pool()
    try:
        future = pool.schedule(_canonical_in_worker, args=(expression,), timeout=timeout_s)
        return future.result()
    except (TimeoutError, pebble.ProcessExpired):
        # On timeout the worker did not write to cache; return strip_string only
        # for this call but do NOT write to the shared cache here (the worker
        # may still be running and will write the right value when it finishes,
        # or the next main-thread call will compute it correctly).
        from sal.utils.qwen_math_parser import strip_string
        return strip_string(expression)
    except Exception:
        from sal.utils.qwen_math_parser import strip_string
        return strip_string(expression)


@lru_cache(maxsize=4096)
def canonical_answer(completion_text: str) -> str:
    """Extract and canonicalize the answer from a model completion.

    Uses the exact same pipeline as eval (extract_answer -> memoized_canonical_form),
    matching what compute_maj_pred uses for vote grouping.

    When called from a non-main thread, canonicalization is routed through the
    pebble process pool to avoid SIGALRM/ValueError poisoning the shared cache.

    Returns '' when extraction fails or the extracted string is empty.
    """
    try:
        extracted = extract_answer(completion_text, "math")
    except Exception:
        return ""
    if not extracted:
        return ""
    try:
        return _memoized_canonical_form_safe(extracted)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Public: gt_canonical — for data prep use
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4096)
def gt_canonical(answer: str) -> str:
    """Canonicalize a ground-truth answer string (no extraction step).

    Used during data preparation when the answer is already extracted.
    Mirrors what orchestrate_experiment.py does:
        memoized_canonical_form(gt_sample['answer'])

    When called from a non-main thread, canonicalization is routed through the
    pebble process pool to avoid SIGALRM/ValueError poisoning the shared cache.
    """
    if not answer:
        return ""
    try:
        return _memoized_canonical_form_safe(answer)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Public: grade_batch
# ---------------------------------------------------------------------------
def grade_batch(
    completions: List[str],
    gt_answer: str,
    timeout_s: float = 8.0,
) -> List[dict]:
    """Grade a batch of completions against a ground-truth answer.

    Each element of the returned list is:
        {"correct": bool, "canonical": str}

    The canonical string is always the extracted+canonicalized form of the
    completion (cheap extraction done in the main process). Only math_equal
    (which may invoke sympy) runs in the subprocess pool with a timeout.
    On timeout or exception, correct=False but canonical is still returned.

    Args:
        completions: list of raw model completion strings.
        gt_answer: ground-truth answer (will be canonicalized via memoized_canonical_form,
                   matching orchestrate_experiment.py line 575).
        timeout_s: per-completion wall-clock timeout for math_equal. Default 8.0 s.

    Returns:
        list of dicts with keys "correct" (bool) and "canonical" (str).
    """
    # Guard: empty gt_answer must never silently reward garbage completions.
    # math_equal('', '') is True, so an empty gt would mark every empty/failed
    # completion as correct.  Match sal's compute_pass_at_k behaviour: raise.
    if not gt_answer:
        raise ValueError(
            "grade_batch: gt_answer is empty. "
            "A missing or unparseable ground-truth answer must not silently reward completions."
        )

    # Canonicalize ground truth — matches orchestrate_experiment.py line 575:
    #   memoized_canonical_form(gt_sample['answer'])
    try:
        gt_canon = _memoized_canonical_form_safe(gt_answer)
    except Exception:
        gt_canon = ""

    if not gt_canon:
        raise ValueError(
            f"grade_batch: gt_answer {gt_answer!r} canonicalized to an empty string. "
            "A missing or unparseable ground-truth answer must not silently reward completions."
        )

    # Extract canonicals for all completions (cheap, in the main process)
    completion_canons = [canonical_answer(c) for c in completions]

    pool = _get_pool()
    futures = []
    for comp_canon in completion_canons:
        try:
            future = pool.schedule(
                _grade_one_worker,
                args=(gt_canon, comp_canon),
                timeout=timeout_s,
            )
        except Exception as exc:
            # Pool might be shutting down or similar; treat as failure
            logger.warning("grade_batch: failed to schedule task: %s", exc)
            # Use a dummy future-like object
            future = None
        futures.append(future)

    results = []
    for comp_canon, future in zip(completion_canons, futures):
        if future is None:
            results.append({"correct": False, "canonical": comp_canon})
            continue
        try:
            correct = future.result()  # blocks until done or TimeoutError
        except pebble.ProcessExpired:
            # Worker process was killed (e.g. memory)
            logger.debug("grade_batch: worker process expired")
            correct = False
        except TimeoutError:
            logger.debug("grade_batch: math_equal timed out for canonical=%r", comp_canon)
            correct = False
        except Exception as exc:
            logger.debug("grade_batch: math_equal raised %s", exc)
            correct = False
        results.append({"correct": bool(correct), "canonical": comp_canon})

    return results


# ---------------------------------------------------------------------------
# Public: class_ids_from_canonicals
# ---------------------------------------------------------------------------
def class_ids_from_canonicals(canonicals: List[str]) -> np.ndarray:
    """Assign integer class IDs by exact canonical string equality.

    Unparseable completions (canonical == '') form their own class (like any
    other canonical string). IDs are assigned in first-occurrence order.

    Args:
        canonicals: list of canonical strings (output of canonical_answer).

    Returns:
        np.ndarray of shape (len(canonicals),), dtype int64.
        Each value is a non-negative integer; equal strings get equal IDs.
    """
    id_map: dict[str, int] = {}
    ids = np.empty(len(canonicals), dtype=np.int64)
    for i, c in enumerate(canonicals):
        if c not in id_map:
            id_map[c] = len(id_map)
        ids[i] = id_map[c]
    return ids
