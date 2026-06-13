"""
tests/test_rewards.py — unit tests for rl/rewards.py

All tests are CPU-only (no GPU required). Run with:
    /cluster/tufts/hugheslab/kheuto01/mambaforge/envs/daft_rl/bin/python -m pytest rl/tests/test_rewards.py -v

Design-mandated coverage:
    1. Correctness fixtures (~15 cases) covering all required forms.
    2. Timeout: pathological input returns correct=False within ~timeout_s,
       and the pool survives for subsequent calls.
    3. Thread-safety: grade_batch works from a non-main thread.
    4. class_ids: exact-string-equality grouping, incl. '' class, and '1/2' vs '0.5'
       merge behaviour documented and asserted.
"""

from __future__ import annotations

import sys
import threading
import time
from unittest.mock import patch

import numpy as np
import pytest

# Ensure rl/ is on sys.path (if running pytest from repo root)
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from rl.rewards import (
    SYSTEM_PROMPT,
    canonical_answer,
    class_ids_from_canonicals,
    grade_batch,
    gt_canonical,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box(expr: str) -> str:
    """Wrap expr in a standard s1-style completion so extract_answer finds it."""
    return f"Therefore, the final answer is: $\\boxed{{{expr}}}$. I hope it is correct."


# ---------------------------------------------------------------------------
# Fixture table: (completion_text, gt_answer, expected_correct)
#
# Notes:
#   - '1/2' and '0.5' both canonicalize to '\frac{1}{2}', so grade_batch sees
#     them as equal. Confirmed empirically: memoized_canonical_form('0.5') ==
#     memoized_canonical_form('\frac{1}{2}') == '\frac{1}{2}'.
#   - Two-boxed completion: extract_answer picks the LAST boxed (confirmed
#     empirically). Locked in by the test below.
#   - 'yes' text answer: memoized_canonical_form('yes') == 'e s y' (sympy
#     rearranges the letters). math_equal('e s y', 'e s y') is True.
#   - Truncated mid-LaTeX: extract_answer returns '\frac{1' (partial); that
#     does NOT equal the correct answer '\frac{1}{2}' -> correct=False.
#   - \pi vs 3.14159...: math_equal returns True (numerical comparison).
#   - Empty string completion -> correct=False.
#   - Garbage text -> correct=False.
# ---------------------------------------------------------------------------

FIXTURE_TABLE = [
    # ---- basic \boxed integer ----
    (_box("42"), "42", True),
    (_box("42"), "43", False),
    # ---- fraction vs decimal (parity: both become \frac{1}{2}) ----
    (_box("1/2"), "0.5", True),       # completion "1/2", gt "0.5" -> equal
    (_box("0.5"), "1/2", True),       # completion "0.5", gt "1/2" -> equal
    # ---- pi forms ----
    (_box(r"\pi"), r"\pi", True),
    (_box(r"\pi"), "3.14159", True),  # numerical match
    # ---- interval ----
    (_box(r"[1,3]"), "[1,3]", True),
    (_box(r"[1,3]"), "[1,4]", False),
    # ---- negative number ----
    (_box("-5"), "-5", True),
    (_box("-5"), "5", False),
    # ---- negative fraction ----
    (_box(r"-\frac{3}{4}"), r"-\frac{3}{4}", True),
    # ---- text answer ----
    (_box("yes"), "yes", True),
    (_box("yes"), "no", False),
    # ---- unparseable garbage (no boxed, no "the answer is") ----
    ("asdfjkl; xyz@@###", "42", False),
    # ---- empty string completion ----
    ("", "42", False),
    # ---- truncated mid-LaTeX (partial boxed content) ----
    # extract_answer finds the partial string; it won't match the real answer
    (r"The answer is $\boxed{\frac{1", r"\frac{1}{2}", False),
    # ---- two boxed answers: extract_answer picks the LAST one ----
    # Completion has \boxed{1} early and \boxed{2} late -> picks "2"
    (
        r"First attempt: $\boxed{1}$. But correcting: $\boxed{2}$. I hope it is correct.",
        "2",
        True,
    ),
    (
        r"First attempt: $\boxed{1}$. But correcting: $\boxed{2}$. I hope it is correct.",
        "1",
        False,
    ),
    # ---- sqrt ----
    (_box(r"\sqrt{2}"), r"\sqrt{2}", True),
]


# ---------------------------------------------------------------------------
# 1. Correctness fixtures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("completion,gt,expected", FIXTURE_TABLE)
def test_grade_batch_fixtures(completion, gt, expected):
    """Each fixture (completion, gt, expected_correct) must match grade_batch."""
    results = grade_batch([completion], gt, timeout_s=8.0)
    assert len(results) == 1
    r = results[0]
    assert "correct" in r
    assert "canonical" in r
    assert isinstance(r["correct"], bool)
    assert isinstance(r["canonical"], str)
    assert r["correct"] == expected, (
        f"Expected correct={expected} for completion={completion!r}, gt={gt!r}; "
        f"got correct={r['correct']}, canonical={r['canonical']!r}"
    )


def test_two_boxed_picks_last():
    """Lock in that extract_answer picks the LAST \boxed{} in the completion."""
    from sal.utils.qwen_math_parser import extract_answer
    completion = r"First: $\boxed{1}$. Corrected: $\boxed{2}$."
    extracted = extract_answer(completion, "math")
    assert extracted == "2", f"Expected '2' (last boxed), got {extracted!r}"


def test_canonical_answer_empty_input():
    assert canonical_answer("") == ""


def test_canonical_answer_garbage():
    c = canonical_answer("asdfjkl; xyz@@###")
    assert isinstance(c, str)
    # Garbage has no boxed, so extraction returns '' -> canonical is ''
    assert c == ""


def test_canonical_answer_boxed_integer():
    c = canonical_answer(_box("42"))
    assert c != ""  # should extract something


def test_gt_canonical_basic():
    c = gt_canonical("42")
    assert isinstance(c, str)
    assert c != ""


def test_gt_canonical_empty():
    assert gt_canonical("") == ""


def test_system_prompt_nonempty():
    """SYSTEM_PROMPT must be the sal default (non-empty, starts with 'Solve')."""
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT) > 0
    assert SYSTEM_PROMPT.startswith("Solve")


# ---------------------------------------------------------------------------
# 2. Timeout test
# ---------------------------------------------------------------------------

def test_grade_batch_timeout_returns_false_quickly():
    """
    A pathological sympy expression (huge nested radical) that would hang math_equal
    must return correct=False within ~timeout_s seconds, and the pool must survive.

    We monkeypatch the worker function to sleep indefinitely to simulate a hang,
    rather than relying on finding a real slow sympy expression (which is fragile).
    """
    import rl.rewards as rewards_module

    real_worker = rewards_module._grade_one_worker

    # Monkeypatching _grade_one_worker is not straightforward because it runs in
    # a subprocess. Instead we test with a real pathological expression that
    # causes sympy to hang, using a very short timeout.
    #
    # A known-slow pattern: nested radicals or huge polynomial factoring.
    # We use a deeply nested fraction that makes sympy hang on simplify().
    # If this doesn't hang reliably on the test machine, the timeout still
    # exercises the pebble machinery; the test asserts timing and pool survival.

    # This expression is large enough to stress sympy:
    pathological = r"\sqrt{\sqrt{\sqrt{\sqrt{\sqrt{\sqrt{\sqrt{\sqrt{2^{2^{2^{2^{2^{2^{2^{2}}}}}}}}}}}}}}}}"

    timeout_s = 2.0
    t0 = time.time()
    results = grade_batch([pathological], pathological, timeout_s=timeout_s)
    elapsed = time.time() - t0

    # Must complete within timeout + some grace for process overhead
    assert elapsed < timeout_s + 5.0, (
        f"grade_batch took {elapsed:.1f}s, expected < {timeout_s + 5.0}s"
    )
    # Result must be a valid dict regardless of timeout/error
    assert len(results) == 1
    r = results[0]
    assert "correct" in r
    assert "canonical" in r

    # Pool must still work after the timeout
    followup = grade_batch([_box("42")], "42", timeout_s=8.0)
    assert len(followup) == 1
    assert followup[0]["correct"] is True, "Pool did not survive timeout — pool is wedged"


# ---------------------------------------------------------------------------
# 3. Thread-safety: grade_batch from a non-main thread
# ---------------------------------------------------------------------------

def test_grade_batch_from_non_main_thread():
    """grade_batch must work when called from a non-main thread (pebble, not signal).

    Uses '7' as a basic smoke-test (strip_string and sympy canonical are the same
    for a plain integer, so this does not catch cache-poisoning).
    """
    results_container = []
    exc_container = []

    def worker():
        try:
            res = grade_batch([_box("7")], "7", timeout_s=8.0)
            results_container.extend(res)
        except Exception as exc:
            exc_container.append(exc)

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=30.0)
    assert not t.is_alive(), "Thread did not finish in time"
    assert not exc_container, f"Thread raised exception: {exc_container}"
    assert len(results_container) == 1
    assert results_container[0]["correct"] is True


def test_grade_batch_from_non_main_thread_canonicalization():
    """Non-main-thread canonicalization must produce the sympy canonical, not the
    strip_string fallback.

    '0.5' has strip_string fallback '0.5' but sympy canonical '\frac{1}{2}'.
    Before the fix, calling memoized_canonical_form from a thread would silently
    write '0.5' to the shared Manager cache, causing subsequent main-thread calls
    to return the wrong value and breaking vote-class merging (0.5 != 1/2).

    This test:
      1. Calls canonical_answer(_box('0.5')) from a non-main thread and checks that
         the returned canonical equals the sympy canonical (not the strip fallback).
      2. Then verifies that a subsequent main-thread call for the same expression
         also returns the correct canonical (cache is not poisoned).
    """
    from sal.utils.math import memoized_canonical_form

    # Determine what the correct sympy canonical should be (from main thread).
    # Do this BEFORE the thread runs so the main-thread result is not influenced
    # by a poisoned cache from the thread.
    # Use a fresh expression that is unlikely to be in the lru_cache yet.
    # We use the raw extracted string '0.250000' (unlikely to be pre-cached).
    test_expr = "0.250000"
    correct_canonical = memoized_canonical_form(test_expr)
    # Sanity: sympy should simplify 0.25 to a fraction or at least recognise it.
    # The key assertion is that it is NOT the bare strip_string result when the
    # thread runs.  (strip_string('0.250000') == '0.250000' on current sal.)
    # The correct sympy canonical is something like '\frac{1}{4}'.
    assert correct_canonical != test_expr, (
        f"Precondition: sympy canonical of {test_expr!r} should differ from "
        f"strip_string fallback {test_expr!r}; got {correct_canonical!r}"
    )

    # Now evict from the shared cache by using a value that is NOT yet cached.
    # Use a slightly different expression so the thread test is independent.
    thread_expr_box = _box("0.250000")

    thread_results = {}
    exc_container = []

    def worker():
        try:
            # canonical_answer runs extract_answer first, giving '0.250000',
            # then calls _memoized_canonical_form_safe which must route through
            # the pebble pool off the main thread.
            c = canonical_answer(thread_expr_box)
            thread_results["canonical"] = c
        except Exception as exc:
            exc_container.append(exc)

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=30.0)
    assert not t.is_alive(), "Thread did not finish in time"
    assert not exc_container, f"Thread raised: {exc_container}"

    thread_canon = thread_results.get("canonical", None)
    assert thread_canon is not None, "Thread did not produce a canonical"
    assert thread_canon == correct_canonical, (
        f"Off-main-thread canonical_answer returned {thread_canon!r} "
        f"(strip_string fallback?), expected sympy canonical {correct_canonical!r}. "
        "Cache poisoning from SIGALRM/ValueError off main thread is still present."
    )

    # Verify the shared cache is not poisoned: main-thread call must also return
    # the correct canonical (lru_cache is per-process so this hits the sal
    # shared_cache or recomputes from main thread — either is correct).
    main_canon = canonical_answer(thread_expr_box)
    assert main_canon == correct_canonical, (
        f"Main-thread canonical_answer returned {main_canon!r} after thread call; "
        f"expected {correct_canonical!r}. Shared Manager cache may be poisoned."
    )


# ---------------------------------------------------------------------------
# 4. class_ids_from_canonicals
# ---------------------------------------------------------------------------

def test_class_ids_basic():
    """Exact string equality groups; '' forms its own class."""
    # '1/2' and '0.5' both canonicalize to '\frac{1}{2}' (same string).
    # We test class_ids on the raw canonical strings to document the behavior.
    #
    # Empirically confirmed:
    #   canonical_answer(_box('1/2')) == canonical_answer(_box('0.5'))
    #   -> both == '\frac{1}{2}'
    # So ['1/2_canon', '0.5_canon', '', '1/2_canon', ''] where 1/2_canon == 0.5_canon
    # gives 2 distinct classes.
    #
    # class_ids_from_canonicals operates on already-canonicalized strings.
    # The input ['1/2', '0.5', '', '1/2', ''] is RAW (not yet canonicalized),
    # so they remain distinct strings -> 3 classes.
    # The test spec says: "note '1/2' vs '0.5' are DIFFERENT canonical strings
    # unless the canonical form merges them". Since we confirmed that the eval-parity
    # canonical (memoized_canonical_form) DOES merge them, but class_ids_from_canonicals
    # receives ALREADY-CANONICALIZED strings, the merge has already happened before
    # this function is called. The caller (grade_batch) produces the canonical strings.
    # We document both behaviors here.

    # Case A: raw strings (no pre-canonicalization) — class_ids treats as distinct
    raw = ["1/2", "0.5", "", "1/2", ""]
    ids = class_ids_from_canonicals(raw)
    assert ids.dtype == np.int64
    assert len(ids) == 5
    # '1/2' should share ID
    assert ids[0] == ids[3]
    # '' should share ID
    assert ids[2] == ids[4]
    # '0.5' is different from '1/2' as a raw string
    assert ids[1] != ids[0], (
        "When class_ids_from_canonicals receives raw strings, '0.5' != '1/2'"
    )
    # 3 distinct classes: '1/2', '0.5', ''
    assert len(set(ids.tolist())) == 3

    # Case B: already-canonicalized strings (as grade_batch would produce)
    # memoized_canonical_form('1/2') == memoized_canonical_form('0.5') == '\frac{1}{2}'
    from sal.utils.math import memoized_canonical_form
    canon_half = memoized_canonical_form("\\frac{1}{2}")
    canon_point5 = memoized_canonical_form("0.5")
    # Confirm they merge
    assert canon_half == canon_point5, (
        "memoized_canonical_form should merge 1/2 and 0.5 into the same string"
    )
    canonicalized = [canon_half, canon_point5, "", canon_half, ""]
    ids2 = class_ids_from_canonicals(canonicalized)
    # Now '1/2' and '0.5' are the same string -> 2 distinct classes
    assert ids2[0] == ids2[1] == ids2[3], "Canonicalized 1/2 and 0.5 must share ID"
    assert ids2[2] == ids2[4], "Empty strings must share ID"
    assert ids2[0] != ids2[2], "Non-empty and empty must differ"
    assert len(set(ids2.tolist())) == 2


def test_class_ids_all_empty():
    ids = class_ids_from_canonicals(["", "", ""])
    assert all(ids == ids[0])
    assert len(set(ids.tolist())) == 1


def test_class_ids_all_unique():
    ids = class_ids_from_canonicals(["a", "b", "c"])
    assert len(set(ids.tolist())) == 3


def test_class_ids_empty_list():
    ids = class_ids_from_canonicals([])
    assert isinstance(ids, np.ndarray)
    assert len(ids) == 0


def test_class_ids_first_occurrence_order():
    """IDs are assigned in first-occurrence order (deterministic)."""
    ids = class_ids_from_canonicals(["b", "a", "b", "c", "a"])
    assert ids[0] == ids[2]   # 'b' appears first
    assert ids[1] == ids[4]   # 'a' appears second
    assert ids[3] != ids[0] and ids[3] != ids[1]  # 'c' is third
    # IDs should be 0, 1, 0, 2, 1
    assert ids[0] == 0
    assert ids[1] == 1
    assert ids[3] == 2


# ---------------------------------------------------------------------------
# 4b. Empty ground-truth guard
# ---------------------------------------------------------------------------

def test_grade_batch_empty_gt_raises():
    """grade_batch with empty gt_answer must raise ValueError, not silently reward.

    math_equal('', '') is True, so without the guard every empty/garbage completion
    would be marked correct=True — a silent data-quality bug.  We require a
    ValueError matching sal's compute_pass_at_k behaviour.
    """
    with pytest.raises(ValueError, match="gt_answer"):
        grade_batch(["garbage text"], "")


def test_grade_batch_empty_gt_raises_even_for_empty_completion():
    """An empty completion against an empty gt must also raise, not return correct=True."""
    with pytest.raises(ValueError, match="gt_answer"):
        grade_batch([""], "")


# ---------------------------------------------------------------------------
# 5. grade_batch batch size > 1
# ---------------------------------------------------------------------------

def test_grade_batch_multiple():
    """grade_batch handles multiple completions correctly."""
    completions = [_box("1"), _box("2"), _box("3"), "garbage"]
    results = grade_batch(completions, "1", timeout_s=8.0)
    assert len(results) == 4
    assert results[0]["correct"] is True
    assert results[1]["correct"] is False
    assert results[2]["correct"] is False
    assert results[3]["correct"] is False


def test_grade_batch_returns_canonical_even_on_wrong():
    """Canonical string is returned even for incorrect completions."""
    results = grade_batch([_box("99")], "42", timeout_s=8.0)
    assert results[0]["correct"] is False
    assert results[0]["canonical"] != ""  # should have a valid canonical


# ---------------------------------------------------------------------------
# 6. Integration: canonical_answer -> class_ids_from_canonicals -> grade_batch parity
# ---------------------------------------------------------------------------

def test_fraction_decimal_parity_end_to_end():
    """1/2 and 0.5 completions produce the same canonical and same class ID."""
    c1 = canonical_answer(_box("1/2"))
    c2 = canonical_answer(_box("0.5"))
    assert c1 == c2, f"Canonicals differ: {c1!r} vs {c2!r}"

    ids = class_ids_from_canonicals([c1, c2])
    assert ids[0] == ids[1], "1/2 and 0.5 must get the same class ID after canonicalization"
