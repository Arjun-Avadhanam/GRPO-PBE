# GRPO-PBE

Train Qwen2.5-1.5B to solve programming-by-example (PBE) data transformation
tasks. Compare two paradigms (supervised fine-tuning on gold labels vs.
reinforcement learning with an execution-based reward) on the same synthetic
dataset, same base model, same LoRA config, and the same 8 GB GPU.

## TL;DR

| Model | Easy | Medium | Hard | Overall | Format compliance | Mean think length |
|---|---|---|---|---|---|---|
| **Base** (zero-shot) | 0.0% | 0.0% | 0.0% | **0.0%** | 2.0% | n/a |
| **SFT** | 97.8% | 95.0% | 100.0% | **97.0%** | 0.0% | n/a |
| **GRPO** | 1.1% | 0.0% | 0.0% | **0.5%** | 100.0% | 118 chars |

200 held-out examples (89 easy, 80 medium, 31 hard). Accuracy is the percentage
of examples whose extracted `<code>` body passes all held-out tests.

**SFT learns the task end-to-end in ~1.5 hours; GRPO with an execution-based
reward learns the format in ~95 minutes but not the underlying
transformations.** See [docs/FINDINGS.md](docs/FINDINGS.md) for the mechanistic
explanation.

## Task

Given 3 demonstration I/O pairs, write a single Python expression (using `x`
as the input) that produces the demonstrated transformation. The expression
is executed against 2 held-out test cases and scored by exact match.

```
Given these input/output examples, write a Python expression that
transforms the input to the output.

Example 1: "2024-01-15" → "Jan 15"
Example 2: "2024-03-20" → "Mar 20"
Example 3: "2024-12-05" → "Dec 05"

Respond with reasoning in <think>...</think> tags, then the Python
expression in <code>...</code> tags. Use `x` as the input variable.
```

25 templates across 7 categories (string slicing, regex, dates, numeric,
list ops, dict ops, chained multi-step). Difficulty mix is roughly 44%
easy, 40% medium, 16% hard. 1300 training examples, 200 evaluation, seeded.

## Quick start

Requires Python 3.12+, an NVIDIA GPU with a CUDA 12.x driver, and ~12 GB of
free disk. Tested on an RTX 4060 Laptop (8 GB VRAM) under WSL2.

```bash
git clone https://github.com/Arjun-Avadhanam/GRPO-PBE
cd GRPO-PBE

# 1. Venv + deps. Install torch from the cu128 wheel index first; PyPI's
#    default torch is cu130 which doesn't initialise on a CUDA 12.x driver.
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"

# 2. Download the pre-quantized 4-bit base model.
mkdir -p models
huggingface-cli download unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit \
    --local-dir models/qwen2.5-1.5b-bnb4

# 3. Generate the synthetic dataset (deterministic, seed=42).
python -m grpo_pbe.data_generator

# 4. Train.
wandb login                              # required for training logs
python -m grpo_pbe.train_grpo            # ~95 min on a 4060 (600 steps)
python -m grpo_pbe.train_sft             # ~1.5 hours (3 epochs)

# 5. Evaluate all three checkpoints.
python -m grpo_pbe.evaluate models/qwen2.5-1.5b-bnb4         base
python -m grpo_pbe.evaluate checkpoints/sft-pbe-1.5b         sft
python -m grpo_pbe.evaluate checkpoints/grpo-pbe-1.5b        grpo
```

## Repository layout

```
grpo_pbe/
  templates/             25 transformation templates (string, regex, date, ...)
  data_generator.py      generates train/eval splits
  prompt.py              prompt formatting (demos to instruction)
  sandbox.py             restricted exec with timeout
  reward.py              training reward (format + correctness)
  train_grpo.py          GRPO training script (Unsloth + TRL)
  train_sft.py           SFT baseline training script
  evaluate.py            inference + scoring on held-out set

tests/                   46 pytest cases (templates, reward, sandbox, eval)
data/                    eval JSONs and (gitignored) training data
checkpoints/             LoRA adapters from training (gitignored)
models/                  local model snapshot (gitignored)
notebooks/analysis.ipynb headline plots + per-template breakdown
docs/
  FINDINGS.md            mechanistic analysis
  superpowers/specs/     design spec
reports/RESULTS.md       standalone results table
```

## Tech stack

* **Base model**: `unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit` (Qwen2.5-1.5B
  pre-quantized to 4-bit nf4)
* **Training**: Unsloth 2025.11 + TRL 0.23 (`GRPOTrainer` / `SFTTrainer`)
* **LoRA**: rank 16, alpha 16, target modules `q,k,v,o,gate,up,down` projections
* **GRPO config**: 600 steps, batch=4 prompts x 4 rollouts, KL coef (β) 0.04,
  lr 5e-6, max completion 512 tokens, adamw_8bit
* **SFT config**: 3 epochs, batch=4, grad-accum=2, lr 2e-4, adamw_8bit
* **Logging**: Weights & Biases

## Training runs

* GRPO: <https://wandb.ai/arjunvijayavadhanam-shiv-nadar-university/grpo-pbe/runs/58t4o2pm>
* SFT:  <https://wandb.ai/arjunvijayavadhanam-shiv-nadar-university/grpo-pbe/runs/kp4g6z1x>

## Notes for re-runners

If `huggingface-cli download` stalls on your network (it does on some networks,
particularly WSL with HF CloudFront edges), `curl` works as a fallback. See
commit `e591b4e` for context. The training and eval scripts set
`HF_HUB_ENABLE_HF_TRANSFER=0` to avoid the parallel-chunk downloader, which
has reproduced this stall on multiple machines.

W&B run names are hard-coded as `grpo-run` and `sft-baseline`. Change them in
the respective `train_*.py` if you want them separated more clearly.
