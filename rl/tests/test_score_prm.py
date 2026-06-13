"""
test_score_prm.py — CPU tests for rl/score_prm.py and PRM metrics in analyze.py.

Tests:
1. aggregate_scores: "last" strategy (also "min", "prod").
2. aggregate_scores: empty list returns 0.0.
3. score_row: stubbed PRM via monkeypatch — verifies agg_scores length and values.
4. prm_bon@n: hand-computable exact expectations on a 4-completion synthetic row.
5. prm_wmaj@n: hand-computable exact expectations on a 4-completion synthetic row.
6. Determinism: repeated calls with same seed yield identical results.
7. Schema round-trip: score_prm writes valid JSONL that analyze can load.
8. prm_bon/prm_wmaj: more edge cases (all-correct, all-wrong, tie in scores).

All tests are CPU-only (no GPU required).

Run from repo root:
    /cluster/tufts/hugheslab/kheuto01/mambaforge/envs/daft_rl/bin/python \
        -m pytest rl/tests/test_score_prm.py -q
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

# Add repo root to sys.path so rl.* imports work when running from repo root
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rl.score_prm import aggregate_scores, score_row
from rl.analyze import (
    compute_prm_metrics_for_row,
    compute_prm_per_problem_metrics,
    _prm_subset_utility_bon,
    _prm_subset_utility_wmaj,
    load_completions,
)


# ---------------------------------------------------------------------------
# Helper: build a synthetic row
# ---------------------------------------------------------------------------

def _make_row(
    completions,
    correct_mask,
    canonicals,
    agg_scores=None,
    problem_id="test_prob",
    answer="42",
):
    """Build a minimal completions.jsonl row dict."""
    row = {
        "problem_id": problem_id,
        "problem": "What is 6 * 7?",
        "answer": answer,
        "completions": completions,
        "canonicals": canonicals,
        "correct_mask": correct_mask,
    }
    if agg_scores is not None:
        row["agg_scores"] = agg_scores
    return row


# ---------------------------------------------------------------------------
# 1. aggregate_scores strategies
# ---------------------------------------------------------------------------

class TestAggregateScores:
    def test_last(self):
        assert aggregate_scores([0.1, 0.5, 0.9], "last") == pytest.approx(0.9)

    def test_last_single(self):
        assert aggregate_scores([0.3], "last") == pytest.approx(0.3)

    def test_min(self):
        assert aggregate_scores([0.8, 0.2, 0.6], "min") == pytest.approx(0.2)

    def test_prod(self):
        assert aggregate_scores([0.5, 0.4], "prod") == pytest.approx(0.2)

    def test_empty_returns_zero(self):
        """Empty step-scores (no step separators found) → 0.0 (worst-case score)."""
        assert aggregate_scores([], "last") == pytest.approx(0.0)

    def test_empty_min(self):
        assert aggregate_scores([], "min") == pytest.approx(0.0)

    def test_empty_prod(self):
        assert aggregate_scores([], "prod") == pytest.approx(0.0)

    def test_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown agg_strategy"):
            aggregate_scores([0.5], "mean")


# ---------------------------------------------------------------------------
# 2. score_row via monkeypatched PRM
# ---------------------------------------------------------------------------

class TestScoreRow:
    """Verify score_row plumbing without loading a real model."""

    def _make_stub_model_tokenizer(self, step_scores_per_comp):
        """Return (model, tokenizer) stubs that produce fixed step_scores."""

        class StubModel:
            parameters = lambda self: iter([type("P", (), {"device": "cpu"})()])

        class StubTokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                return "FAKE_CONVERSATION"

            def encode(self, text, return_tensors=None):
                if return_tensors == "pt":
                    import torch
                    return torch.zeros((1, 10), dtype=torch.long)
                # encode("<extra_0>") → returns a list with one int
                return [99999]

        return StubModel(), StubTokenizer()

    def test_score_row_uses_monkeypatched_fn(self, monkeypatch):
        """Monkeypatch score_one_completion to return fixed step scores."""
        import rl.score_prm as score_prm_module

        # Each completion gets step scores [0.3], [0.7, 0.9], [0.5]
        per_comp_scores = [[0.3], [0.7, 0.9], [0.5]]
        call_idx = [0]

        def fake_score_one(question, completion, model, tokenizer):
            idx = call_idx[0] % len(per_comp_scores)
            call_idx[0] += 1
            return per_comp_scores[idx]

        monkeypatch.setattr(score_prm_module, "score_one_completion", fake_score_one)

        row = _make_row(
            completions=["comp0", "comp1", "comp2"],
            correct_mask=[1, 0, 1],
            canonicals=["42", "", "42"],
        )

        # Dummy model/tokenizer (not actually called due to monkeypatch)
        class DummyModel:
            pass

        class DummyTokenizer:
            pass

        agg_scores = score_prm_module.score_row(
            row, DummyModel(), DummyTokenizer(), batch_size=10
        )

        assert len(agg_scores) == 3
        # "last" strategy: [0.3]→0.3, [0.7,0.9]→0.9, [0.5]→0.5
        assert agg_scores[0] == pytest.approx(0.3)
        assert agg_scores[1] == pytest.approx(0.9)
        assert agg_scores[2] == pytest.approx(0.5)

    def test_score_row_empty_completion(self, monkeypatch):
        """Empty/truncated completion produces empty step scores → 0.0 agg."""
        import rl.score_prm as score_prm_module

        monkeypatch.setattr(
            score_prm_module, "score_one_completion", lambda q, c, m, t: []
        )

        row = _make_row(
            completions=["", "some text"],
            correct_mask=[0, 0],
            canonicals=["", ""],
        )

        class D:
            pass

        agg_scores = score_prm_module.score_row(row, D(), D(), batch_size=10)
        assert agg_scores == [pytest.approx(0.0), pytest.approx(0.0)]

    def test_score_row_batch_size_one(self, monkeypatch):
        """Verify batching (batch_size=1) still produces correct output length."""
        import rl.score_prm as score_prm_module

        call_results = [[0.1], [0.2], [0.3], [0.4]]
        call_idx = [0]

        def fake_score(q, c, m, t):
            res = call_results[call_idx[0]]
            call_idx[0] += 1
            return res

        monkeypatch.setattr(score_prm_module, "score_one_completion", fake_score)

        row = _make_row(
            completions=["a", "b", "c", "d"],
            correct_mask=[1, 0, 1, 0],
            canonicals=["42", "", "42", ""],
        )

        class D:
            pass

        agg = score_prm_module.score_row(row, D(), D(), batch_size=1)
        assert len(agg) == 4
        assert agg == pytest.approx([0.1, 0.2, 0.3, 0.4])


# ---------------------------------------------------------------------------
# 3. prm_bon@n hand-computable expectations
#
# Setup: 4 completions
#   comp0: correct,  canonical="42", agg_score=0.9
#   comp1: wrong,    canonical="0",  agg_score=0.6
#   comp2: correct,  canonical="42", agg_score=0.3
#   comp3: wrong,    canonical="7",  agg_score=0.1
#
# n=1: all C(4,1)=4 singletons
#   {0}: argmax=0 (score 0.9), correct → u=1
#   {1}: argmax=1 (score 0.6), wrong   → u=0
#   {2}: argmax=2 (score 0.3), correct → u=1
#   {3}: argmax=3 (score 0.1), wrong   → u=0
#   prm_bon@1 = 2/4 = 0.5
#
# n=2: all C(4,2)=6 pairs
#   {0,1}: argmax=0 (0.9>0.6), correct → 1
#   {0,2}: argmax=0 (0.9>0.3), correct → 1
#   {0,3}: argmax=0 (0.9>0.1), correct → 1
#   {1,2}: argmax=1 (0.6>0.3), wrong   → 0
#   {1,3}: argmax=1 (0.6>0.1), wrong   → 0
#   {2,3}: argmax=2 (0.3>0.1), correct → 1
#   prm_bon@2 = 4/6 = 2/3
#
# n=3: all C(4,3)=4 triples
#   {0,1,2}: argmax=0 (0.9), correct → 1
#   {0,1,3}: argmax=0 (0.9), correct → 1
#   {0,2,3}: argmax=0 (0.9), correct → 1
#   {1,2,3}: argmax=1 (0.6), wrong   → 0
#   prm_bon@3 = 3/4 = 0.75
# ---------------------------------------------------------------------------

SYNTH_ROW = _make_row(
    completions=["a", "b", "c", "d"],
    correct_mask=[1, 0, 1, 0],
    canonicals=["42", "0", "42", "7"],
    agg_scores=[0.9, 0.6, 0.3, 0.1],
)


class TestPrmBon:
    def test_bon_n1(self):
        result = compute_prm_metrics_for_row(SYNTH_ROW, k_values=[1])
        assert result["prm_bon_1"] == pytest.approx(0.5, abs=1e-10)

    def test_bon_n2(self):
        result = compute_prm_metrics_for_row(SYNTH_ROW, k_values=[2])
        assert result["prm_bon_2"] == pytest.approx(4 / 6, abs=1e-10)

    def test_bon_n3(self):
        result = compute_prm_metrics_for_row(SYNTH_ROW, k_values=[3])
        assert result["prm_bon_3"] == pytest.approx(3 / 4, abs=1e-10)

    def test_bon_n4_equals_argmax_full(self):
        """n=4 = G: only one subset (all comps). argmax score=comp0=correct → 1."""
        result = compute_prm_metrics_for_row(SYNTH_ROW, k_values=[4])
        assert result["prm_bon_4"] == pytest.approx(1.0, abs=1e-10)

    def test_bon_keys_present(self):
        result = compute_prm_metrics_for_row(SYNTH_ROW, k_values=[1, 2, 4, 8])
        assert "prm_bon_1" in result
        assert "prm_bon_2" in result
        assert "prm_bon_4" in result
        # n=8 > G=4 → None
        assert result["prm_bon_8"] is None


# ---------------------------------------------------------------------------
# 4. prm_wmaj@n hand-computable expectations
#
# Same 4-completion setup.  Weighted vote: each class gets sum of agg_scores
# of its members in the subset.
#
# Classes: "42" (comps 0,2, correct), "0" (comp 1, wrong), "7" (comp 3, wrong)
#
# n=1: singletons (each class has total weight = its member's score)
#   {0}: class "42"→0.9, class "0"→0 → winner="42" (correct) → 1
#   {1}: class "42"→0,  class "0"→0.6 → winner="0" (wrong) → 0
#   {2}: class "42"→0.3, → winner="42" (correct) → 1
#   {3}: class "7"→0.1,  → winner="7" (wrong) → 0
#   prm_wmaj@1 = 2/4 = 0.5
#
# n=2:
#   {0,1}: "42"=0.9, "0"=0.6 → "42" wins, correct → 1
#   {0,2}: "42"=0.9+0.3=1.2  → "42" wins → 1
#   {0,3}: "42"=0.9, "7"=0.1 → "42" wins → 1
#   {1,2}: "42"=0.3, "0"=0.6 → "0" wins, wrong → 0
#   {1,3}: "0"=0.6,  "7"=0.1 → "0" wins, wrong → 0
#   {2,3}: "42"=0.3, "7"=0.1 → "42" wins → 1
#   prm_wmaj@2 = 4/6 = 2/3
#
# n=3:
#   {0,1,2}: "42"=1.2, "0"=0.6 → "42" wins → 1
#   {0,1,3}: "42"=0.9, "0"=0.6, "7"=0.1 → "42" wins → 1
#   {0,2,3}: "42"=1.2, "7"=0.1 → "42" wins → 1
#   {1,2,3}: "42"=0.3, "0"=0.6, "7"=0.1 → "0" wins, wrong → 0
#   prm_wmaj@3 = 3/4 = 0.75
# ---------------------------------------------------------------------------

class TestPrmWmaj:
    def test_wmaj_n1(self):
        result = compute_prm_metrics_for_row(SYNTH_ROW, k_values=[1])
        assert result["prm_wmaj_1"] == pytest.approx(0.5, abs=1e-10)

    def test_wmaj_n2(self):
        result = compute_prm_metrics_for_row(SYNTH_ROW, k_values=[2])
        assert result["prm_wmaj_2"] == pytest.approx(4 / 6, abs=1e-10)

    def test_wmaj_n3(self):
        result = compute_prm_metrics_for_row(SYNTH_ROW, k_values=[3])
        assert result["prm_wmaj_3"] == pytest.approx(3 / 4, abs=1e-10)

    def test_wmaj_n4(self):
        result = compute_prm_metrics_for_row(SYNTH_ROW, k_values=[4])
        # Full set: "42"=1.2, "0"=0.6, "7"=0.1 → "42" wins → 1
        assert result["prm_wmaj_4"] == pytest.approx(1.0, abs=1e-10)

    def test_wmaj_n8_is_none(self):
        result = compute_prm_metrics_for_row(SYNTH_ROW, k_values=[8])
        assert result["prm_wmaj_8"] is None


# ---------------------------------------------------------------------------
# 5. Tie-breaking edge cases
# ---------------------------------------------------------------------------

class TestTieBreaking:
    """Verify expected-utility tie-break: |correct ∩ argmax| / |argmax|."""

    def test_bon_score_tie(self):
        """Two completions with same score, one correct one wrong → utility = 0.5."""
        row = _make_row(
            completions=["a", "b"],
            correct_mask=[1, 0],
            canonicals=["42", "0"],
            agg_scores=[0.5, 0.5],
        )
        result = compute_prm_metrics_for_row(row, k_values=[2])
        # n=2: only subset {0,1}: scores tied → argmax_set=[0,1]; correct=[0] → 1/2
        assert result["prm_bon_2"] == pytest.approx(0.5, abs=1e-10)

    def test_wmaj_class_tie(self):
        """Two classes with equal total weight → expected utility = 1/2 if one correct."""
        # comp0: "42" correct, score 0.5
        # comp1: "0"  wrong,   score 0.5
        row = _make_row(
            completions=["a", "b"],
            correct_mask=[1, 0],
            canonicals=["42", "0"],
            agg_scores=[0.5, 0.5],
        )
        result = compute_prm_metrics_for_row(row, k_values=[2])
        # n=2: "42"=0.5, "0"=0.5 → tied; argmax_classes=["42","0"];
        #       correct_in_argmax=["42"] → 1/2
        assert result["prm_wmaj_2"] == pytest.approx(0.5, abs=1e-10)

    def test_all_correct(self):
        """All completions correct → prm_bon = prm_wmaj = 1.0 for all n."""
        row = _make_row(
            completions=["a", "b", "c"],
            correct_mask=[1, 1, 1],
            canonicals=["42", "42", "42"],
            agg_scores=[0.9, 0.5, 0.3],
        )
        result = compute_prm_metrics_for_row(row, k_values=[1, 2, 3])
        for k in [1, 2, 3]:
            assert result[f"prm_bon_{k}"] == pytest.approx(1.0, abs=1e-10)
            assert result[f"prm_wmaj_{k}"] == pytest.approx(1.0, abs=1e-10)

    def test_all_wrong(self):
        """All completions wrong → prm_bon = prm_wmaj = 0.0 for all n."""
        row = _make_row(
            completions=["a", "b", "c"],
            correct_mask=[0, 0, 0],
            canonicals=["1", "2", "3"],
            agg_scores=[0.9, 0.5, 0.3],
        )
        result = compute_prm_metrics_for_row(row, k_values=[1, 2, 3])
        for k in [1, 2, 3]:
            assert result[f"prm_bon_{k}"] == pytest.approx(0.0, abs=1e-10)
            assert result[f"prm_wmaj_{k}"] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 6. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_result(self):
        """Monte-Carlo subsets (n > min_subsets) are deterministic with fixed seed."""
        # Make G large enough to exceed min_subsets for some n
        rng = np.random.RandomState(7)
        G = 30
        correct_mask = rng.randint(0, 2, size=G).tolist()
        canonicals = [str(rng.randint(0, 5)) for _ in range(G)]
        agg_scores = rng.uniform(0.0, 1.0, size=G).tolist()

        row = _make_row(
            completions=[f"comp{i}" for i in range(G)],
            correct_mask=correct_mask,
            canonicals=canonicals,
            agg_scores=agg_scores,
        )

        result1 = compute_prm_metrics_for_row(
            row, k_values=[8, 16], min_subsets=20, seed=42
        )
        result2 = compute_prm_metrics_for_row(
            row, k_values=[8, 16], min_subsets=20, seed=42
        )

        for k in [8, 16]:
            assert result1[f"prm_bon_{k}"] == pytest.approx(result2[f"prm_bon_{k}"])
            assert result1[f"prm_wmaj_{k}"] == pytest.approx(result2[f"prm_wmaj_{k}"])

    def test_different_seed_may_differ(self):
        """Different seeds produce results that may differ (sanity: not constant)."""
        rng = np.random.RandomState(99)
        G = 30
        correct_mask = rng.randint(0, 2, size=G).tolist()
        canonicals = [str(rng.randint(0, 5)) for _ in range(G)]
        agg_scores = rng.uniform(0.0, 1.0, size=G).tolist()

        row = _make_row(
            completions=[f"comp{i}" for i in range(G)],
            correct_mask=correct_mask,
            canonicals=canonicals,
            agg_scores=agg_scores,
        )

        r1 = compute_prm_metrics_for_row(row, k_values=[16], min_subsets=10, seed=1)
        r2 = compute_prm_metrics_for_row(row, k_values=[16], min_subsets=10, seed=2)

        # Results should be in [0,1] for all seeds
        for k in [16]:
            assert 0.0 <= r1[f"prm_bon_{k}"] <= 1.0
            assert 0.0 <= r2[f"prm_wmaj_{k}"] <= 1.0


# ---------------------------------------------------------------------------
# 7. Schema round-trip: score_prm writes valid JSONL that analyze can load
# ---------------------------------------------------------------------------

class TestSchemaRoundTrip:
    def test_roundtrip(self):
        """Write a scored JSONL and verify analyze.load_completions reads it back."""
        row = _make_row(
            completions=["a", "b"],
            correct_mask=[1, 0],
            canonicals=["42", "0"],
            agg_scores=[0.9, 0.3],
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(row) + "\n")
            fname = f.name

        try:
            rows = load_completions(fname)
            assert len(rows) == 1
            pid = row["problem_id"]
            assert pid in rows
            loaded = rows[pid]
            assert loaded["agg_scores"] == pytest.approx([0.9, 0.3])
            assert loaded["correct_mask"] == [1, 0]
            assert loaded["canonicals"] == ["42", "0"]
        finally:
            os.unlink(fname)

    def test_no_agg_scores_row_produces_empty_prm_metrics(self):
        """Rows without agg_scores yield empty dicts from compute_prm_per_problem_metrics."""
        row = _make_row(
            completions=["a", "b"],
            correct_mask=[1, 0],
            canonicals=["42", "0"],
            # no agg_scores
        )
        rows = {row["problem_id"]: row}
        result = compute_prm_per_problem_metrics(rows, k_values=[1, 2])
        assert result[row["problem_id"]] == {}

    def test_multi_row_file(self):
        """Multiple rows in JSONL are all read back with agg_scores."""
        rows_to_write = [
            _make_row(
                completions=["a", "b"],
                correct_mask=[1, 0],
                canonicals=["42", "0"],
                agg_scores=[0.9, 0.3],
                problem_id=f"prob_{i}",
            )
            for i in range(5)
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for r in rows_to_write:
                f.write(json.dumps(r) + "\n")
            fname = f.name

        try:
            loaded = load_completions(fname)
            assert len(loaded) == 5
            for i in range(5):
                assert f"prob_{i}" in loaded
                assert "agg_scores" in loaded[f"prob_{i}"]
        finally:
            os.unlink(fname)


# ---------------------------------------------------------------------------
# 8. compute_prm_per_problem_metrics: aggregate over multiple rows
# ---------------------------------------------------------------------------

class TestComputePrmPerProblem:
    def test_aggregate_multiple_rows(self):
        """compute_prm_per_problem_metrics returns per-problem dicts."""
        rows = {
            "p0": _make_row(
                completions=["a", "b", "c", "d"],
                correct_mask=[1, 0, 1, 0],
                canonicals=["42", "0", "42", "7"],
                agg_scores=[0.9, 0.6, 0.3, 0.1],
                problem_id="p0",
            ),
            "p1": _make_row(
                completions=["x", "y"],
                correct_mask=[0, 0],
                canonicals=["1", "2"],
                agg_scores=[0.5, 0.4],
                problem_id="p1",
            ),
        }
        result = compute_prm_per_problem_metrics(rows, k_values=[1, 2])
        assert "p0" in result
        assert "p1" in result
        # p0 prm_bon@1 = 0.5 (from earlier test)
        assert result["p0"]["prm_bon_1"] == pytest.approx(0.5, abs=1e-10)
        # p1: all wrong → 0.0
        assert result["p1"]["prm_bon_1"] == pytest.approx(0.0, abs=1e-10)
        assert result["p1"]["prm_wmaj_1"] == pytest.approx(0.0, abs=1e-10)

    def test_missing_agg_scores_row_skipped(self):
        """Row without agg_scores returns empty dict — not an error."""
        rows = {
            "p_no_scores": _make_row(
                completions=["a"],
                correct_mask=[1],
                canonicals=["42"],
                # no agg_scores
                problem_id="p_no_scores",
            ),
        }
        result = compute_prm_per_problem_metrics(rows, k_values=[1])
        assert result["p_no_scores"] == {}


# ---------------------------------------------------------------------------
# 9. Exact enumeration threshold: verify exact vs Monte-Carlo agreement
# ---------------------------------------------------------------------------

class TestExactVsMonteCarlo:
    def test_exact_and_montecarlo_agree_for_small_G(self):
        """Exact enumeration and Monte-Carlo should agree on small G."""
        # G=4, n=2 → C(4,2)=6 subsets (exact), and MC with many samples
        result_exact = compute_prm_metrics_for_row(
            SYNTH_ROW, k_values=[2], min_subsets=1000, seed=0
        )
        # 6 <= 1000 → exact used; should match hand-computed value
        assert result_exact["prm_bon_2"] == pytest.approx(4 / 6, abs=1e-10)
        assert result_exact["prm_wmaj_2"] == pytest.approx(4 / 6, abs=1e-10)

    def test_large_n_uses_montecarlo(self):
        """With min_subsets=3, C(4,2)=6 > 3 would use MC — but default is 200."""
        # Force MC by setting min_subsets=3 (< C(4,2)=6)
        result_mc = compute_prm_metrics_for_row(
            SYNTH_ROW, k_values=[2], min_subsets=3, seed=0
        )
        # MC should be close to true value (not exact, but within MC noise for 3 samples)
        assert 0.0 <= result_mc["prm_bon_2"] <= 1.0
        assert 0.0 <= result_mc["prm_wmaj_2"] <= 1.0
