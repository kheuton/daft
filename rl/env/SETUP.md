# daft_rl environment setup

Conda env `daft_rl` for RL fine-tuning of LLMs (TRL `GRPOTrainer` + vLLM
**colocated** generation) on the Tufts HPC cluster. Stack is the battle-tested
open-r1 combination (vLLM 0.8.5.post1 / torch 2.6.0 cu124 / transformers 4.51.3),
with TRL bumped to 0.18.2 to get colocate mode (see Deviations).

Built on a **login node** (no GPU). Only CPU import checks were run; GPU
validation is deferred to a SLURM smoke test (see bottom).

## Final key versions

| component    | version           |
|--------------|-------------------|
| python       | 3.11.15           |
| torch        | 2.6.0+cu124       |
| torch.version.cuda | 12.4        |
| vllm         | 0.8.5.post1       |
| trl          | 0.18.2            |
| transformers | 4.51.3            |
| accelerate   | 1.13.0            |
| datasets     | 5.0.0             |
| peft         | 0.19.1            |
| wandb        | 0.27.2            |
| matplotlib   | 3.10.9            |
| latex2sympy2 | 1.9.1             |
| sympy        | 1.13.1            |
| fastapi      | 0.136.3           |
| search-and-learn (sal, editable) | 0.1.0 |

Full pinned set: `requirements-lock.txt` (in this directory).

## Exact commands that worked

Home-dir quota is tight (~25.5G / 28.6G used), so build caches/temp were
redirected to hugheslab scratch:

```bash
export TMPDIR=/cluster/tufts/hugheslab/kheuto01/tmp
export PIP_CACHE_DIR=/cluster/tufts/hugheslab/kheuto01/tmp
mkdir -p "$TMPDIR"

MAMBA=/cluster/tufts/hugheslab/kheuto01/mambaforge/bin/mamba
PIP=/cluster/tufts/hugheslab/kheuto01/mambaforge/envs/daft_rl/bin/pip
PY=/cluster/tufts/hugheslab/kheuto01/mambaforge/envs/daft_rl/bin/python

# 1. create env
$MAMBA create -n daft_rl python=3.11 -y

# 2a. vLLM FIRST (pulls torch==2.6.0 cu124 + all CUDA libs)
$PIP install "vllm==0.8.5.post1"

# 2b. the rest (transformers re-pinned here; vllm had pulled transformers 5.x)
$PIP install \
  "trl==0.17.0" \
  "transformers==4.51.3" \
  "accelerate>=1.4" "datasets>=3.0" "peft" "wandb" "matplotlib" "pyyaml" \
  "latex2sympy2==1.9.1" "word2number" "regex" "pebble" "sympy" "fastapi"

# 2c. bump TRL to 0.18.2 for colocate mode, holding the other pins so a
#     transitive resolve cannot clobber them. Only the trl wheel changed;
#     vllm / torch / transformers were "already satisfied".
$PIP install "trl==0.18.2" "transformers==4.51.3" "vllm==0.8.5.post1" "torch==2.6.0"

$PIP check   # -> "No broken requirements found."

# 3. local sal package, WITHOUT deps (so it can't bump pins)
$PIP install -e /cluster/tufts/hugheslab/kheuto01/code/daft/sal --no-deps
```

### Reinstalling from the lock file

`requirements-lock.txt` was produced by `pip freeze`. Note `pip freeze`
recorded `search-and-learn` (sal) as a **git+VCS** line, not an editable local
path, because the repo has a remote. To restore the editable local install,
ignore that line and run the explicit command instead:

```bash
pip install -r requirements-lock.txt        # core stack
pip install -e /cluster/tufts/hugheslab/kheuto01/code/daft/sal --no-deps  # sal editable
```

## Verification results (login node, CPU only)

All imports emit two benign warnings on the login node — expected with no GPU:
`No platform detected, vLLM is running on UnspecifiedPlatform` and
`Failed to import from vllm._C with ImportError('libcuda.so.1: ...')`.
These come from the missing CUDA driver/runtime on the login node, not a broken
install, and must be re-checked on a GPU node.

- **4a** `import torch, trl, vllm, transformers`: OK ->
  torch 2.6.0+cu124 | trl 0.18.2 | vllm 0.8.5.post1 | transformers 4.51.3
- **4b** `from trl import GRPOTrainer, GRPOConfig`: OK
- **4c** colocate support: `use_vllm: True | vllm_mode: True` (both True).
  `vllm_mode` default is `"server"`; set `vllm_mode="colocate"` for in-process
  colocated generation.
- **4d** grader: `extract_answer(...\\boxed{\\frac{1}{2}}...)` -> `\frac{1}{2}`
  (not None); `math_equal("1/2", "0.5")` -> `True`. Completed in <60s, no hang.
- **4e** `torch.version.cuda` -> **12.4**.
- `pip check` -> No broken requirements found (after every install step).

## Deviations from the plan

1. **TRL pinned to 0.18.2, not 0.17.0.** The plan pinned `trl==0.17.0`, but its
   `GRPOTrainer` only supports **server-mode** vLLM (via `VLLMClient` /
   `vllm_server_host`/`vllm_server_port`) — there is no `vllm_mode` field and no
   in-process colocate path. The stated goal is **colocated** generation and
   verification step 4c explicitly requires `vllm_mode` to be a `GRPOConfig`
   field. `vllm_mode` (with `"colocate"` option) was introduced in TRL 0.18.0.
   TRL 0.18.x keeps the same loose deps (`transformers>=4.50.0`, `torch>=2.0.0`),
   so bumping to 0.18.2 left vllm 0.8.5.post1 / torch 2.6.0 cu124 /
   transformers 4.51.3 untouched and `pip check` clean. The 0.7.3 + trl 0.16.1
   fallback path was NOT needed.
2. **vLLM transitively installed transformers 5.11.0**, which was then
   downgraded back to the pinned 4.51.3 in step 2b (and held in 2c). Final
   transformers is 4.51.3 as intended.
3. **No fallback vLLM version was needed** — `vllm==0.8.5.post1` resolved and
   installed on the first try.
4. **`pip freeze` recorded sal as a git URL** rather than an editable path; use
   the explicit `pip install -e .../sal --no-deps` to reproduce the editable
   install (see "Reinstalling from the lock file").

## GPU validation (DEFERRED — run first on a SLURM node)

GPU correctness was NOT validated (login node has no GPU). Run a SLURM smoke
test before trusting the env. Cluster GPU nodes are A100 (sm80) and RTX A6000
(sm86); both are supported by the cu124 wheels' compiled kernels.

**Check this FIRST if vLLM fails on a node:** the cu124 wheels require an NVIDIA
driver **>= 550**. Verify with `nvidia-smi` (driver / "CUDA Version" column) on
the allocated node before deeper debugging. A driver older than 550 will surface
as CUDA init / `libcuda` / kernel-launch errors even though the CPU import check
here passed.

Minimal GPU smoke test (inside a GPU SLURM allocation):

```bash
PY=/cluster/tufts/hugheslab/kheuto01/mambaforge/envs/daft_rl/bin/python
nvidia-smi                                  # confirm driver >= 550
$PY -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
$PY -c "import vllm; from vllm import LLM"   # should import with no libcuda warning
```
