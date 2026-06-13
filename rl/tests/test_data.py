"""
test_data.py — Unit tests for prepare_data.py, calibrate_scale.py, and
               related utilities.

All tests are CPU-only and do NOT require GPU.  Tests that need rl.rewards or
rl.advantages use pytest.importorskip so they skip gracefully if those modules
are absent (e.g., rewards.py not yet written by concurrent agent).
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from collections import Counter

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers: locate rl package for importable path
# tests are run from daft/ root (or pytest discovers from there).
# The same sys.path pattern used by test_rewards.py and test_advantages.py.
# ---------------------------------------------------------------------------

_DAFT_DIR = Path(__file__).parent.parent.parent  # .../daft
if str(_DAFT_DIR) not in sys.path:
    sys.path.insert(0, str(_DAFT_DIR))


# ---------------------------------------------------------------------------
# 1. Normalization and 8-gram dedup unit tests (synthetic strings)
# ---------------------------------------------------------------------------

class TestNormalization:
    def _normalize(self, s):
        from rl.prepare_data import normalize_text
        return normalize_text(s)

    def test_lowercase(self):
        from rl.prepare_data import normalize_text
        assert normalize_text("Hello WORLD") == "hello world"

    def test_collapse_whitespace(self):
        from rl.prepare_data import normalize_text
        assert normalize_text("  a   b  ") == "a b"

    def test_strip_dollar(self):
        from rl.prepare_data import normalize_text
        assert normalize_text("$x^2 + $y$") == "x^2 + y"

    def test_combined(self):
        from rl.prepare_data import normalize_text
        result = normalize_text("  Solve $x^2 + 1 = 0$  for $x$. ")
        assert result == "solve x^2 + 1 = 0 for x."

    def test_empty(self):
        from rl.prepare_data import normalize_text
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""


class TestEightGrams:
    def test_short_string_no_grams(self):
        from rl.prepare_data import get_8grams
        short = "one two three four five six seven"  # 7 words
        assert get_8grams(short) == set()

    def test_exact_8_words(self):
        from rl.prepare_data import get_8grams
        s = "one two three four five six seven eight"
        grams = get_8grams(s)
        assert len(grams) == 1
        assert ("one", "two", "three", "four", "five", "six", "seven", "eight") in grams

    def test_longer_string(self):
        from rl.prepare_data import get_8grams
        words = [str(i) for i in range(12)]
        s = " ".join(words)
        grams = get_8grams(s)
        # Should have 12 - 8 + 1 = 5 grams
        assert len(grams) == 5

    def test_shared_8gram_detection(self):
        from rl.prepare_data import texts_share_8gram, normalize_text
        base = "the quick brown fox jumps over the lazy dog today"
        a = normalize_text(base)
        b = normalize_text(base + " and more words after")
        assert texts_share_8gram(a, b)

    def test_no_shared_8gram(self):
        from rl.prepare_data import texts_share_8gram, normalize_text
        a = normalize_text("one two three four five six seven eight alpha")
        b = normalize_text("nine ten eleven twelve thirteen fourteen fifteen sixteen beta")
        assert not texts_share_8gram(a, b)

    def test_contamination_exact_match(self):
        from rl.prepare_data import (
            build_decontam_index, is_contaminated, normalize_text
        )
        refs = ["Solve x plus one equals zero.", "Find the value of y."]
        exact, grams = build_decontam_index(refs)
        assert is_contaminated(normalize_text("Solve x plus one equals zero."), exact, grams)
        assert not is_contaminated(normalize_text("A completely different sentence here."), exact, grams)

    def test_contamination_8gram_match(self):
        from rl.prepare_data import (
            build_decontam_index, is_contaminated, normalize_text
        )
        ref_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        exact, grams = build_decontam_index([ref_text])
        # New problem that shares an 8-gram
        test_text = "prefix alpha beta gamma delta epsilon zeta eta theta iota suffix"
        assert is_contaminated(normalize_text(test_text), exact, grams)

    def test_no_contamination_partial_overlap(self):
        from rl.prepare_data import (
            build_decontam_index, is_contaminated, normalize_text
        )
        ref_text = "alpha beta gamma delta epsilon zeta eta theta"
        exact, grams = build_decontam_index([ref_text])
        # Shares only 7 words with ref — not a full 8-gram match
        test_text = "alpha beta gamma delta epsilon zeta eta kappa"
        # This shares the first 7 words but not an 8-gram
        # The shared prefix is 7 words, the 8th differs
        assert not is_contaminated(normalize_text(test_text), exact, grams)


# ---------------------------------------------------------------------------
# 2. Answer extraction from fixture MATH solutions
# ---------------------------------------------------------------------------

class TestAnswerExtraction:
    def test_simple_boxed(self):
        from rl.prepare_data import extract_answer_from_solution
        sol = r"We compute: $1 + 1 = 2$. Therefore, $\boxed{2}$."
        assert extract_answer_from_solution(sol) == "2"

    def test_nested_braces(self):
        from rl.prepare_data import extract_answer_from_solution
        sol = r"The answer is $\boxed{\frac{1}{2}}$."
        assert extract_answer_from_solution(sol) == r"\frac{1}{2}"

    def test_last_boxed_wins(self):
        from rl.prepare_data import extract_answer_from_solution
        sol = r"First $\boxed{3}$, then $\boxed{7}$."
        assert extract_answer_from_solution(sol) == "7"

    def test_no_boxed_returns_none(self):
        from rl.prepare_data import extract_answer_from_solution
        sol = "The answer is 42 but I forgot to box it."
        assert extract_answer_from_solution(sol) is None

    def test_deeply_nested(self):
        from rl.prepare_data import extract_answer_from_solution
        sol = r"Therefore $\boxed{\left(\frac{a+b}{c}\right)}$"
        ans = extract_answer_from_solution(sol)
        assert ans == r"\left(\frac{a+b}{c}\right)"

    def test_empty_boxed(self):
        from rl.prepare_data import extract_answer_from_solution
        sol = r"The answer is $\boxed{}$."
        ans = extract_answer_from_solution(sol)
        assert ans == ""

    def test_multiline_solution(self):
        from rl.prepare_data import extract_answer_from_solution
        sol = (
            "Step 1: Do something.\n"
            "Step 2: Do more.\n"
            r"Therefore, the final answer is: $\boxed{42}$. I hope it is correct."
        )
        assert extract_answer_from_solution(sol) == "42"


# ---------------------------------------------------------------------------
# 3. Level-stratified sampling determinism
# ---------------------------------------------------------------------------

class TestStratifiedSampling:
    def _make_candidates(self, n_per_level: dict[str, int]) -> list[dict]:
        """Build synthetic candidate list with given level counts."""
        candidates = []
        for lvl, n in n_per_level.items():
            for i in range(n):
                candidates.append({"problem": f"p_{lvl}_{i}", "level": lvl,
                                    "answer": "42", "type": "Algebra"})
        return candidates

    def test_deterministic_seed(self):
        """Same seed produces identical output."""
        import random
        from collections import defaultdict

        candidates = self._make_candidates({"Level 1": 200, "Level 2": 300,
                                             "Level 3": 400, "Level 4": 300,
                                             "Level 5": 300})
        target = 500

        def stratified_sample(candidates, target, seed):
            by_level = defaultdict(list)
            for item in candidates:
                by_level[item["level"]].append(item)
            levels = sorted(by_level.keys())
            total_candidates = len(candidates)
            raw_counts = {lvl: len(by_level[lvl]) * target / total_candidates for lvl in levels}
            floors = {lvl: int(raw_counts[lvl]) for lvl in levels}
            remainder = target - sum(floors.values())
            fracs = sorted(levels, key=lambda lvl: -(raw_counts[lvl] - floors[lvl]))
            for lvl in fracs[:remainder]:
                floors[lvl] += 1
            rng = random.Random(seed)
            sampled = []
            for lvl in levels:
                pool = list(by_level[lvl])
                rng.shuffle(pool)
                sampled.extend(pool[:floors[lvl]])
            return sampled

        s1 = stratified_sample(candidates, target, seed=0)
        s2 = stratified_sample(candidates, target, seed=0)
        assert len(s1) == target
        assert [r["problem"] for r in s1] == [r["problem"] for r in s2]

    def test_different_seed_different_result(self):
        import random
        from collections import defaultdict

        candidates = self._make_candidates({"Level 1": 200, "Level 2": 300,
                                             "Level 3": 500})
        target = 300

        def stratified_sample(candidates, target, seed):
            by_level = defaultdict(list)
            for item in candidates:
                by_level[item["level"]].append(item)
            levels = sorted(by_level.keys())
            total_candidates = len(candidates)
            raw_counts = {lvl: len(by_level[lvl]) * target / total_candidates for lvl in levels}
            floors = {lvl: int(raw_counts[lvl]) for lvl in levels}
            remainder = target - sum(floors.values())
            fracs = sorted(levels, key=lambda lvl: -(raw_counts[lvl] - floors[lvl]))
            for lvl in fracs[:remainder]:
                floors[lvl] += 1
            rng = random.Random(seed)
            sampled = []
            for lvl in levels:
                pool = list(by_level[lvl])
                rng.shuffle(pool)
                sampled.extend(pool[:floors[lvl]])
            return sampled

        s0 = [r["problem"] for r in stratified_sample(candidates, target, seed=0)]
        s1 = [r["problem"] for r in stratified_sample(candidates, target, seed=1)]
        # Very likely to differ with enough candidates
        assert s0 != s1

    def test_level_proportionality(self):
        """Sampled level fractions are approximately proportional to input fractions."""
        import random
        from collections import defaultdict

        level_counts = {"Level 1": 100, "Level 2": 200, "Level 3": 300,
                        "Level 4": 200, "Level 5": 200}
        candidates = self._make_candidates(level_counts)
        total = sum(level_counts.values())
        target = 500

        def stratified_sample(candidates, target, seed):
            by_level = defaultdict(list)
            for item in candidates:
                by_level[item["level"]].append(item)
            levels = sorted(by_level.keys())
            total_candidates = len(candidates)
            raw_counts = {lvl: len(by_level[lvl]) * target / total_candidates for lvl in levels}
            floors = {lvl: int(raw_counts[lvl]) for lvl in levels}
            remainder = target - sum(floors.values())
            fracs = sorted(levels, key=lambda lvl: -(raw_counts[lvl] - floors[lvl]))
            for lvl in fracs[:remainder]:
                floors[lvl] += 1
            rng = random.Random(seed)
            sampled = []
            for lvl in levels:
                pool = list(by_level[lvl])
                rng.shuffle(pool)
                sampled.extend(pool[:floors[lvl]])
            return sampled

        sampled = stratified_sample(candidates, target, seed=0)
        assert len(sampled) == target

        sampled_by_level = Counter(r["level"] for r in sampled)
        for lvl, count in level_counts.items():
            expected = count * target / total
            actual = sampled_by_level[lvl]
            # Allow rounding error of at most 2
            assert abs(actual - expected) <= 2, (
                f"Level {lvl}: expected ~{expected:.1f}, got {actual}"
            )


# ---------------------------------------------------------------------------
# 4. Sweep-row schema validation
# ---------------------------------------------------------------------------

class TestSweepRowSchema:
    REQUIRED_FIELDS = {
        "problem_id", "problem", "answer", "level", "type",
        "C", "G", "correct_mask", "canonicals",
        "flippable_maj8", "mean_len", "trunc_frac",
    }

    def _make_sweep_row(self, **overrides) -> dict:
        row = {
            "problem_id": 0,
            "problem": "What is 2+2?",
            "answer": "4",
            "level": "Level 1",
            "type": "Algebra",
            "C": 8,
            "G": 16,
            "correct_mask": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "canonicals": ["4", "", "4", "", "4", "", "4", "", "4", "", "4", "", "4", "", "4", ""],
            "flippable_maj8": True,
            "mean_len": 150.0,
            "trunc_frac": 0.0,
        }
        row.update(overrides)
        return row

    def test_required_fields_present(self):
        row = self._make_sweep_row()
        missing = self.REQUIRED_FIELDS - set(row.keys())
        assert missing == set(), f"Missing fields: {missing}"

    def test_json_roundtrip(self):
        row = self._make_sweep_row()
        serialized = json.dumps(row)
        loaded = json.loads(serialized)
        assert loaded["problem_id"] == 0
        assert loaded["C"] == 8
        assert loaded["G"] == 16
        assert len(loaded["correct_mask"]) == 16
        assert len(loaded["canonicals"]) == 16
        assert loaded["flippable_maj8"] is True
        assert isinstance(loaded["mean_len"], float)
        assert isinstance(loaded["trunc_frac"], float)

    def test_c_equals_sum_of_mask(self):
        row = self._make_sweep_row(C=8, correct_mask=[1] * 8 + [0] * 8)
        assert row["C"] == sum(row["correct_mask"])

    def test_mask_length_equals_G(self):
        row = self._make_sweep_row(G=16, correct_mask=[1] * 16)
        assert len(row["correct_mask"]) == row["G"]

    def test_canonicals_length_equals_G(self):
        row = self._make_sweep_row(G=16, canonicals=["x"] * 16)
        assert len(row["canonicals"]) == row["G"]

    def test_trunc_frac_range(self):
        row = self._make_sweep_row(trunc_frac=0.0)
        assert 0.0 <= row["trunc_frac"] <= 1.0

    def test_missing_field_detected(self):
        row = self._make_sweep_row()
        del row["C"]
        missing = self.REQUIRED_FIELDS - set(row.keys())
        assert "C" in missing

    def test_rl_train_row_schema(self):
        """RL train JSONL row schema: problem, answer, level, type, C16, weight."""
        rl_row = {
            "problem": "What is 2+2?",
            "answer": "4",
            "level": "Level 1",
            "type": "Algebra",
            "C16": 8,
            "weight": 1.0,
        }
        required_rl = {"problem", "answer", "level", "type", "C16", "weight"}
        missing = required_rl - set(rl_row.keys())
        assert missing == set()
        assert isinstance(rl_row["C16"], int)
        assert isinstance(rl_row["weight"], float)


# ---------------------------------------------------------------------------
# 5. calibrate_scale on synthetic sweep rows
# ---------------------------------------------------------------------------

class TestCalibrateScale:
    def _make_synthetic_sweep_rows(self, n_problems: int, G: int = 16, seed: int = 42) -> list[dict]:
        """Make synthetic sweep rows with known correctness patterns."""
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(n_problems):
            # C ranges from 1 to G-1
            C = int(rng.integers(1, G))
            mask = [0] * G
            for j in rng.choice(G, size=C, replace=False):
                mask[j] = 1
            canonicals = ["4" if mask[j] == 1 else str(j % 3) for j in range(G)]
            rows.append({
                "problem_id": i,
                "problem": f"Problem {i}",
                "answer": "4",
                "level": "Level 1",
                "type": "Algebra",
                "C": C,
                "G": G,
                "correct_mask": mask,
                "canonicals": canonicals,
                "flippable_maj8": True,
                "mean_len": 200.0,
                "trunc_frac": 0.0,
            })
        return rows

    def test_calibrate_returns_all_modes(self):
        advantages = pytest.importorskip("rl.advantages")
        rewards = pytest.importorskip("rl.rewards")
        from rl.calibrate_scale import calibrate, MODES

        rows = self._make_synthetic_sweep_rows(50, G=16)
        constants, diagnostics = calibrate(rows, k=8)
        for mode in MODES:
            assert mode in constants, f"Mode {mode} missing from constants"
            assert mode in diagnostics, f"Mode {mode} missing from diagnostics"

    def test_constants_are_positive_floats(self):
        advantages = pytest.importorskip("rl.advantages")
        rewards = pytest.importorskip("rl.rewards")
        from rl.calibrate_scale import calibrate

        rows = self._make_synthetic_sweep_rows(50, G=16)
        constants, _ = calibrate(rows, k=8)
        for mode, rms in constants.items():
            assert isinstance(rms, float), f"Mode {mode}: rms is not float"
            assert rms >= 0.0, f"Mode {mode}: rms {rms} < 0"
            # For grpo with mixed correct/wrong, RMS must be positive
            if mode == "grpo":
                assert rms > 0.0, "grpo RMS should be positive for mixed correctness"

    def test_grpo_rms_known_value(self):
        """For all-half-correct groups (C=G/2=8), grpo A_i = +-0.5, so RMS = 0.5."""
        advantages = pytest.importorskip("rl.advantages")
        rewards = pytest.importorskip("rl.rewards")
        from rl.calibrate_scale import calibrate

        G = 16
        # All groups: first 8 correct, last 8 wrong → C_i = 0.5, mean = 0.5
        # A_i = 1 - 0.5 = 0.5 for correct, -0.5 for wrong
        # RMS = sqrt(mean(0.25)) = 0.5
        rows = []
        for i in range(20):
            mask = [1] * 8 + [0] * 8
            canonicals = ["4"] * 8 + [str(j) for j in range(8)]
            rows.append({
                "problem_id": i, "problem": f"P{i}", "answer": "4",
                "level": "Level 1", "type": "Algebra",
                "C": 8, "G": G,
                "correct_mask": mask, "canonicals": canonicals,
                "flippable_maj8": True, "mean_len": 100.0, "trunc_frac": 0.0,
            })
        constants, _ = calibrate(rows, k=8)
        assert abs(constants["grpo"] - 0.5) < 1e-9, (
            f"grpo RMS expected 0.5, got {constants['grpo']}"
        )

    def test_diagnostics_structure(self):
        advantages = pytest.importorskip("rl.advantages")
        rewards = pytest.importorskip("rl.rewards")
        from rl.calibrate_scale import calibrate

        rows = self._make_synthetic_sweep_rows(30, G=16)
        _, diagnostics = calibrate(rows, k=8)
        for mode, d in diagnostics.items():
            assert "rms" in d
            assert "frac_zero_groups" in d
            assert "frac_nonzero_samples" in d
            assert 0.0 <= d["frac_zero_groups"] <= 1.0
            assert 0.0 <= d["frac_nonzero_samples"] <= 1.0
            assert "abs_nonzero_pct" in d

    def test_calibrate_output_json_roundtrip(self):
        """Constants can be serialized to JSON and reloaded."""
        advantages = pytest.importorskip("rl.advantages")
        rewards = pytest.importorskip("rl.rewards")
        from rl.calibrate_scale import calibrate

        rows = self._make_synthetic_sweep_rows(20, G=16)
        constants, _ = calibrate(rows, k=8)
        serialized = json.dumps(constants)
        loaded = json.loads(serialized)
        for mode, rms in constants.items():
            assert abs(loaded[mode] - rms) < 1e-12

    def test_all_correct_grpo_zero_rms(self):
        """All-correct groups: all GRPO advantages = 0, RMS = 0."""
        advantages = pytest.importorskip("rl.advantages")
        rewards = pytest.importorskip("rl.rewards")
        from rl.calibrate_scale import calibrate

        G = 16
        rows = []
        for i in range(10):
            mask = [1] * G
            canonicals = ["4"] * G
            rows.append({
                "problem_id": i, "problem": f"P{i}", "answer": "4",
                "level": "Level 1", "type": "Algebra",
                "C": G, "G": G,
                "correct_mask": mask, "canonicals": canonicals,
                "flippable_maj8": False, "mean_len": 100.0, "trunc_frac": 0.0,
            })
        constants, _ = calibrate(rows, k=8)
        assert constants["grpo"] == 0.0, (
            f"All-correct grpo RMS expected 0.0, got {constants['grpo']}"
        )


# ---------------------------------------------------------------------------
# 6. flippable_maj8 logic test (uses advantages.maj_at_k_exact)
# ---------------------------------------------------------------------------

class TestFlippableMaj8:
    def test_all_wrong_not_flippable(self):
        advantages = pytest.importorskip("rl.advantages")
        rewards = pytest.importorskip("rl.rewards")
        from rl.filter_sweep import compute_flippable_maj8

        mask = [0] * 16
        canonicals = [str(i % 4) for i in range(16)]
        assert compute_flippable_maj8(mask, canonicals) is False

    def test_majority_correct_is_flippable(self):
        advantages = pytest.importorskip("rl.advantages")
        rewards = pytest.importorskip("rl.rewards")
        from rl.filter_sweep import compute_flippable_maj8

        # 9 correct out of 16 — can definitely form a majority-correct subset of 8
        mask = [1] * 9 + [0] * 7
        canonicals = ["4"] * 9 + [str(i) for i in range(7)]
        assert compute_flippable_maj8(mask, canonicals) is True

    def test_one_correct_small_group_not_flippable(self):
        """With only 1 correct out of 16, a single correct sample can never win majority of 8."""
        advantages = pytest.importorskip("rl.advantages")
        rewards = pytest.importorskip("rl.rewards")
        from rl.filter_sweep import compute_flippable_maj8

        # 1 correct vs 15 all-same-wrong: the wrong class will dominate
        mask = [1] + [0] * 15
        # All wrong answers are the same class '99' -> they always win plurality
        canonicals = ["4"] + ["99"] * 15
        result = compute_flippable_maj8(mask, canonicals)
        # 15 identical wrong vs 1 correct: wrong always wins — not flippable
        assert result is False

    def test_enough_correct_is_flippable(self):
        """5 correct, 11 wrong — can form subset of 5C+3W with majority correct."""
        advantages = pytest.importorskip("rl.advantages")
        rewards = pytest.importorskip("rl.rewards")
        from rl.filter_sweep import compute_flippable_maj8

        # 5 correct vs 11 wrong (11 all same class)
        # A subset {5 correct, 3 wrong} has majority correct → flippable
        mask = [1] * 5 + [0] * 11
        canonicals = ["4"] * 5 + ["99"] * 11
        assert compute_flippable_maj8(mask, canonicals) is True


# ---------------------------------------------------------------------------
# 7. prepare_data phase2 integration (synthetic sweep files)
# ---------------------------------------------------------------------------

class TestPhase2:
    def _write_sweep_file(self, path: Path, rows: list[dict]) -> None:
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def _make_row(self, C: int, problem_id: int = 0, G: int = 16) -> dict:
        mask = [1] * C + [0] * (G - C)
        canonicals = ["4"] * C + [str(i % 3) for i in range(G - C)]
        return {
            "problem_id": problem_id,
            "problem": f"Problem {problem_id}",
            "answer": "4",
            "level": "Level 1",
            "type": "Algebra",
            "C": C,
            "G": G,
            "correct_mask": mask,
            "canonicals": canonicals,
            "flippable_maj8": C >= 5,
            "mean_len": 200.0,
            "trunc_frac": 0.0,
        }

    def test_phase2_filters_c0_and_cG(self):
        """Phase 2 keeps only rows with 1 <= C <= G-1."""
        import io
        import contextlib
        from rl.prepare_data import run_phase2

        with tempfile.TemporaryDirectory() as tmpdir:
            sweep_path = Path(tmpdir) / "sweep.jsonl"
            rows = [
                self._make_row(0, problem_id=0),   # should be dropped
                self._make_row(8, problem_id=1),   # should be kept
                self._make_row(16, problem_id=2),  # should be dropped
                self._make_row(1, problem_id=3),   # should be kept
                self._make_row(15, problem_id=4),  # should be kept
            ]
            self._write_sweep_file(sweep_path, rows)

            out_path = Path(tmpdir) / "rl_train.jsonl"
            # Monkey-patch DATA_DIR for the test
            import rl.prepare_data as pd_mod
            orig_data_dir = pd_mod.DATA_DIR
            pd_mod.DATA_DIR = Path(tmpdir)
            try:
                run_phase2(str(sweep_path))
            finally:
                pd_mod.DATA_DIR = orig_data_dir

            written_path = Path(tmpdir) / "rl_train.jsonl"
            assert written_path.exists(), "rl_train.jsonl not created"
            written = []
            with open(written_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        written.append(json.loads(line))

            # Expect 3 rows (C=8, C=1, C=15)
            assert len(written) == 3, f"Expected 3 rows, got {len(written)}"
            c16_vals = [r["C16"] for r in written]
            assert sorted(c16_vals) == [1, 8, 15]
            # All weights = 1.0
            for r in written:
                assert r["weight"] == 1.0
                assert "C16" in r
                assert "problem" in r
                assert "answer" in r

    def test_phase2_rl_train_schema(self):
        """rl_train.jsonl rows have the required fields."""
        from rl.prepare_data import run_phase2

        with tempfile.TemporaryDirectory() as tmpdir:
            sweep_path = Path(tmpdir) / "sweep.jsonl"
            rows = [self._make_row(C, problem_id=i) for i, C in enumerate([4, 8, 12])]
            self._write_sweep_file(sweep_path, rows)

            import rl.prepare_data as pd_mod
            orig_data_dir = pd_mod.DATA_DIR
            pd_mod.DATA_DIR = Path(tmpdir)
            try:
                run_phase2(str(sweep_path))
            finally:
                pd_mod.DATA_DIR = orig_data_dir

            written_path = Path(tmpdir) / "rl_train.jsonl"
            required = {"problem", "answer", "level", "type", "C16", "weight"}
            with open(written_path) as f:
                for line in f:
                    row = json.loads(line)
                    missing = required - set(row.keys())
                    assert missing == set(), f"Missing fields: {missing}"
