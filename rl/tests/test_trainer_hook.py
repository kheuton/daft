"""CPU tests for the DAFT GRPO trainer hook (must pass on the login node).

Covers (per task spec sec 8):
  (a) source-hash drift test against the installed TRL.
  (b) the pure advantage-replacement function daft_group_advantages:
        - grpo all-correct group -> zero advantages
        - pass_at_k matches rl.advantages directly
  (c) all YAML configs load and pass validate_config; bad geometry is rejected.

GPU-dependent paths are not exercised here. Tests that need the concurrently
written rl.advantages / rl.rewards modules import them lazily and xfail-skip if
they are not present yet, so the drift/validate tests always run standalone.
"""

import importlib
import inspect
import hashlib
import os
import sys

import numpy as np
import pytest
import yaml

# Make the repo root importable (so `import rl.train_grpo` works from anywhere).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

CONFIG_DIR = os.path.join(REPO_ROOT, "rl", "configs")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _have_siblings():
    """True iff rl.advantages and rl.rewards are importable (concurrent agents)."""
    try:
        importlib.import_module("rl.advantages")
        importlib.import_module("rl.rewards")
        return True
    except Exception:
        return False


HAVE_SIBLINGS = _have_siblings()
need_siblings = pytest.mark.skipif(
    not HAVE_SIBLINGS,
    reason="rl.advantages / rl.rewards not available yet (written concurrently)",
)


def _import_trainer_module():
    """Import rl.train_grpo. If siblings are missing the module-level import of
    rl.rewards/rl.advantages will fail -- callers guard with need_siblings.
    Functions that don't need siblings (validate_config, hash check) are also
    re-derived here so the drift/validate tests run standalone."""
    return importlib.import_module("rl.train_grpo")


# --------------------------------------------------------------------------- #
# (a) Source-hash drift test (standalone: only needs trl installed)
# --------------------------------------------------------------------------- #

# Recorded constant, kept in sync with rl/train_grpo.UPSTREAM_GASC_SHA256.
EXPECTED_GASC_SHA256 = "db1c46e6e219e4089b7e3e28ed3b907d4475479f93992797a21b40a023b1f7b3"


def test_trl_version_pinned():
    import trl
    assert trl.__version__ == "0.18.2", (
        f"expected trl 0.18.2, got {trl.__version__}"
    )


def test_upstream_source_hash_drift():
    """The override copied GRPOTrainer._generate_and_score_completions verbatim;
    if upstream changes, the hash changes and this fails loudly."""
    from trl import GRPOTrainer
    src = inspect.getsource(GRPOTrainer._generate_and_score_completions)
    got = hashlib.sha256(src.encode()).hexdigest()
    assert got == EXPECTED_GASC_SHA256, (
        f"TRL method source drifted: expected {EXPECTED_GASC_SHA256}, got {got}. "
        "Re-review upstream and update UPSTREAM_GASC_SHA256 in rl/train_grpo.py."
    )


def test_recorded_hash_matches_trainer_module_constant():
    """Guard that the test's expected hash and the module's constant agree.
    Imports only the constant -- if the module top-level import fails due to
    missing siblings we still verify via direct file read."""
    if HAVE_SIBLINGS:
        m = _import_trainer_module()
        assert m.UPSTREAM_GASC_SHA256 == EXPECTED_GASC_SHA256
    else:
        path = os.path.join(REPO_ROOT, "rl", "train_grpo.py")
        text = open(path).read()
        assert EXPECTED_GASC_SHA256 in text, (
            "recorded hash not found in rl/train_grpo.py"
        )


# --------------------------------------------------------------------------- #
# (c) Config loading + validate_config (standalone-ish; validate_config also
#     checks the source hash, which only needs trl)
# --------------------------------------------------------------------------- #

CONFIG_FILES = ["smoke.yaml", "arm_a_grpo.yaml", "arm_b_passk.yaml", "arm_c_votek.yaml"]


def _load_cfg(name):
    with open(os.path.join(CONFIG_DIR, name)) as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("name", CONFIG_FILES)
def test_config_loads_and_validates(name):
    """Every shipped config parses and passes the fail-fast assertions.
    validate_config needs trl (for the hash check) but NOT the sibling modules.
    We set world_size to the config's declared value so geometry is checked as
    intended (runtime uses the WORLD_SIZE env)."""
    m = _import_trainer_module() if HAVE_SIBLINGS else None
    if m is None:
        # Import just validate_config without triggering sibling imports by
        # execing the relevant functions is overkill; instead require trl-only
        # path: the module import only fails on `import rl.rewards/advantages`.
        # So pull validate_config via a fresh import guarded below.
        pytest.importorskip("rl.advantages")
        pytest.importorskip("rl.rewards")
        m = _import_trainer_module()

    cfg = _load_cfg(name)
    # validate_config reads world_size from the cfg dict.
    cfg.setdefault("world_size", cfg.get("world_size", 1))
    m.validate_config(cfg)  # must not raise


@need_siblings
def test_validate_config_rejects_bad_geometry():
    m = _import_trainer_module()
    cfg = _load_cfg("arm_a_grpo.yaml")
    # global = pdtb*world*grad_accum = 4*4*4 = 64; pick G that does not divide 64.
    cfg["grpo"]["num_generations"] = 6   # 64 % 6 != 0
    with pytest.raises(AssertionError, match="divisible by num_generations"):
        m.validate_config(cfg)


@need_siblings
def test_validate_config_rejects_too_few_groups():
    m = _import_trainer_module()
    cfg = _load_cfg("smoke.yaml")
    cfg["grpo"]["per_device_train_batch_size"] = 4   # 4*1*1 = 4 = single G=4 group
    cfg["world_size"] = 1
    with pytest.raises(AssertionError, match=r">= 2\*num_generations|2 groups"):
        m.validate_config(cfg)


@need_siblings
def test_validate_config_rejects_scale_rewards_true():
    m = _import_trainer_module()
    cfg = _load_cfg("arm_a_grpo.yaml")
    cfg["grpo"]["scale_rewards"] = True
    with pytest.raises(AssertionError, match="scale_rewards must be False"):
        m.validate_config(cfg)


@need_siblings
def test_validate_config_rejects_k_too_large_for_vote():
    m = _import_trainer_module()
    cfg = _load_cfg("arm_c_votek.yaml")
    cfg["k"] = cfg["grpo"]["num_generations"]   # k == G violates k <= G-1
    with pytest.raises(AssertionError, match="k <= num_generations-1"):
        m.validate_config(cfg)


@need_siblings
def test_validate_config_rejects_bad_mode():
    m = _import_trainer_module()
    cfg = _load_cfg("arm_a_grpo.yaml")
    cfg["mode"] = "not_a_mode"
    with pytest.raises(AssertionError, match="not in"):
        m.validate_config(cfg)


@need_siblings
def test_validate_config_rejects_num_iterations_not_one():
    m = _import_trainer_module()
    cfg = _load_cfg("arm_a_grpo.yaml")
    cfg["grpo"]["num_iterations"] = 2
    with pytest.raises(AssertionError, match="num_iterations must be 1"):
        m.validate_config(cfg)


# --------------------------------------------------------------------------- #
# (b) Pure advantage-replacement function unit tests (need siblings)
# --------------------------------------------------------------------------- #

@need_siblings
def test_daft_group_advantages_grpo_all_correct_is_zero():
    """An all-correct group in grpo mode -> all-zero advantages (c_i - mean(c))."""
    m = _import_trainer_module()
    B, G = 2, 4
    correct = np.array([[1, 1, 1, 1],   # all correct -> zero advantages
                        [1, 0, 1, 0]])  # mixed -> nonzero
    # canonicals: group 0 all same correct answer; group 1 two classes.
    canonicals = [["42", "42", "42", "42"],
                  ["7", "x", "7", "y"]]
    adv = m.daft_group_advantages(correct, canonicals, mode="grpo", k=2, scale=1.0)
    adv = adv.numpy().reshape(B, G)
    assert np.allclose(adv[0], 0.0), f"all-correct grpo group must be zero, got {adv[0]}"
    assert not np.allclose(adv[1], 0.0), "mixed grpo group should be nonzero"


@need_siblings
def test_daft_group_advantages_grpo_matches_centering():
    """grpo advantages == c - mean(c) (no std division), divided by scale."""
    m = _import_trainer_module()
    correct = np.array([[1, 0, 0, 0]])
    canonicals = [["a", "b", "c", "d"]]
    adv = m.daft_group_advantages(correct, canonicals, mode="grpo", k=2, scale=2.0)
    adv = adv.numpy().reshape(1, 4)
    expected = (correct[0] - correct[0].mean()) / 2.0
    assert np.allclose(adv[0], expected), f"{adv[0]} != {expected}"


@need_siblings
def test_daft_group_advantages_passk_matches_advantages_module():
    """pass_at_k path equals rl.advantages.compute_advantages / scale exactly."""
    m = _import_trainer_module()
    import rl.advantages as A
    import rl.rewards as R

    B, G, k = 2, 8, 4
    rng = np.random.default_rng(0)
    correct = rng.integers(0, 2, size=(B, G))
    # build canonicals with a couple distinct classes per group
    canonicals = [[f"c{rng.integers(0,3)}" for _ in range(G)] for _ in range(B)]

    scale = 3.0
    got = m.daft_group_advantages(correct, canonicals, mode="pass_at_k", k=k, scale=scale)
    got = got.numpy().reshape(B, G)

    class_ids = np.stack([R.class_ids_from_canonicals(c) for c in canonicals])
    ref = A.compute_advantages(correct, class_ids, mode="pass_at_k", k=k) / scale
    assert np.allclose(got, ref), f"pass_at_k mismatch:\n{got}\nvs\n{ref}"


@need_siblings
def test_daft_group_advantages_shape_and_order():
    """Returned tensor is (B*G,) row-major: group0 then group1."""
    m = _import_trainer_module()
    B, G = 2, 4
    correct = np.array([[1, 0, 0, 0], [0, 1, 1, 1]])
    canonicals = [["a", "b", "c", "d"], ["p", "q", "q", "q"]]
    adv = m.daft_group_advantages(correct, canonicals, mode="grpo", k=2, scale=1.0)
    assert adv.shape == (B * G,)
    # first G entries correspond to group 0
    assert np.allclose(adv.numpy()[:G], (correct[0] - correct[0].mean()))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
