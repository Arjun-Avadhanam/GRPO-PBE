# GRPO-PBE — Design Spec

**Project:** Train Qwen2.5-1.5B via GRPO to solve programming-by-example (PBE) data transformation tasks, with an SFT baseline comparison.

**Timeline:** 7–10 days, ~22h active work
**Hardware:** RTX 4060 8GB (local), no cloud compute needed

---

## 1. Task Definition

Given 3 input/output demonstration pairs showing a data transformation, generate a `<think>` reasoning trace followed by a `<code>` Python expression that implements the transformation. The expression is executed on 2 held-out test pairs to verify correctness.

**Prompt format (model input):**
```
Given these input/output examples, write a Python expression that transforms the input to the output.

Example 1: "2024-01-15" → "Jan 15"
Example 2: "2024-03-20" → "Mar 20"
Example 3: "2024-12-05" → "Dec 05"

Respond with your reasoning in <think>...</think> tags, then the Python expression in <code>...</code> tags.
The expression should use `x` as the input variable.
```

**Model output format:**
```
<think>The dates are being reformatted from YYYY-MM-DD to abbreviated month + day...</think>
<code>datetime.strptime(x, '%Y-%m-%d').strftime('%b %d')</code>
```

---

## 2. Scope of Transformations (B+)

**Input types:** `str`, `int`, `float`, `list[str]`, `list[int]`, `list[float]`, `dict[str, Any]`
**Output types:** same set
**Code format:** single Python expression (comprehensions, chaining, ternary allowed)
**Pre-imported in sandbox:** `builtins`, `re`, `datetime`
**Not allowed:** `pandas`, `numpy`, multi-line code, additional imports

**~25 transformation templates across these categories:**

| Category | Examples | Input type | Difficulty |
|---|---|---|---|
| String slicing/indexing | first N chars, last word, nth word | `str` | Easy |
| Case transforms | title case, swap case, capitalize after delimiter | `str` | Easy |
| Regex extraction | pull numbers, extract emails, match patterns | `str` | Easy–Medium |
| Date formatting | reformat between date string formats | `str` | Medium |
| Numeric conversion | round, unit convert, percentage format | `int/float` | Easy–Medium |
| Type casting chains | str→int→hex, float→currency string | `str/float` | Medium |
| List operations | cumulative sum, filter by condition, deduplicate | `list` | Medium |
| List string ops | extract/transform elements, sort by criteria | `list[str]` | Medium |
| Dict operations | format values, filter keys, swap key/value | `dict` | Medium–Hard |
| Chained multi-step | split → filter → transform → rejoin | `str/list` | Hard |

**Difficulty distribution:** ~40% easy, ~40% medium, ~20% hard.

**Per template:** ~50–100 random instances (varying concrete values). Total dataset: ~1500–2000 examples.

**Per example:** 5 I/O pairs generated. 3 shown as demonstrations, 2 held out for reward computation.

---

## 3. Reward Function

Two-component reward:

| Component | Signal | Weight |
|---|---|---|
| **Format reward** | Valid `<think>...</think><code>...</code>` tags parsed successfully | 0.1 |
| **Correctness reward** | Fraction of held-out test cases where `exec(code)` output matches gold | 1.0 |

**Maximum reward:** 1.1 (correct format + all test cases pass)
**Minimum reward:** 0.0 (no valid tags AND no correct outputs)

**Execution sandbox:**
- The code expression is wrapped as `lambda x: <expression>` and executed
- Only `builtins`, `re`, `datetime` available (no imports allowed)
- Timeout: 2 seconds per test case
- Any exception (SyntaxError, RuntimeError, TimeoutError) → 0 correctness for that test case
- **Output comparison:** exact match for strings, `abs(a - b) < 1e-6` for floats, `==` for ints/lists/dicts

**Why this works for GRPO:** within a group of G=4 rollouts for the same prompt, correct solutions get positive advantage, incorrect get negative. The format reward provides a weak learning signal even when no rollout solves the task, preventing complete reward starvation early in training.

---

## 4. Model & Training

### GRPO (primary)

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Framework:** Unsloth + TRL `GRPOTrainer`
- **LoRA:** Unsloth's default LoRA-backed GRPO for memory efficiency
- **Hardware:** RTX 4060 8GB local

| Hyperparameter | Value |
|---|---|
| Group size `G` | 4 |
| Max completion length | 512 tokens |
| KL coefficient | 0.04 |
| Learning rate | 5e-6 |
| Batch size | 4 prompts × 4 rollouts |
| Training steps | 500–800 |
| Optimizer | AdamW 8-bit |

**Estimated wall-clock:** 3–5h

**Monitoring during training (check at step 50):**
- Mean reward > 0 (some rollouts solving tasks)
- KL divergence not exploding (< 10)
- Format compliance improving toward 100%
- If reward is flat at 0: stop, debug reward function

### SFT (baseline comparison)

- **Same base model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Framework:** Unsloth + TRL `SFTTrainer`
- **Same LoRA config** as GRPO for fair comparison
- **Training data:** the same prompts, but with gold code as the target (no `<think>` traces — SFT only sees `<code>gold_expression</code>`)
- **Estimated wall-clock:** 1–2h

### Evaluation

- **Held-out set:** ~200 examples not seen during training
- **Metrics per model (base / SFT / GRPO):**
  - Overall accuracy (% held-out test cases passed)
  - Accuracy by difficulty tier (easy / medium / hard)
  - Mean generated code length
  - Mean reasoning trace length (GRPO only)
  - Schema compliance (% valid `<think><code>` format)
- **5 cherry-picked examples** for the blog post showing the model's reasoning on hard transforms

**Logging:** Weights & Biases (free tier). Four tracked curves:
1. Mean reward per step
2. Mean reasoning-trace length per step (the "aha moment" curve)
3. KL divergence
4. Format compliance rate

---

## 5. Deliverables

| Artifact | Description |
|---|---|
| **GitHub repo** | Data generator, reward function, GRPO + SFT training scripts, eval harness, analysis notebook |
| **Two HF models** | `grpo-pbe-1.5b` and `sft-pbe-1.5b` |
| **W&B dashboard** | Training curves, linked from README |
| **Blog post** | ~1500–2000 words on Medium or HF blog |
| **README** | Overview, results table, repro instructions, links |

**Results table format (the headline deliverable):**

| | Base (zero-shot) | SFT | GRPO |
|---|---|---|---|
| Easy accuracy | ? | ? | ? |
| Medium accuracy | ? | ? | ? |
| Hard accuracy | ? | ? | ? |
| Overall accuracy | ? | ? | ? |
| Mean reasoning length | — | — | ? |

---

## 6. Timeline

| Day | Tasks | Active hours |
|---|---|---|
| **Day 1** | Study GRPO concepts (Unsloth guide, DeepSeek-R1, GRPOTrainer API) | 3h |
| **Day 2** | Build synthetic data generator (~25 templates + script). Spot-check 5 templates. | 3h |
| **Day 3** | Reward function + sandbox + manually verify 20 examples. **Hard gate: reward must be correct.** | 2.5h |
| **Day 4** | GRPO training script + launch run. Check at step 50. Write SFT script while training runs. | 1.5h active + 3–5h passive |
| **Day 5** | SFT training run + build eval harness. | 1h active + 1.5h passive |
| **Day 6** | Eval both models + analysis (results table, curves, example outputs). | 2.5h |
| **Day 7–8** | Blog post + README + HF upload. | 3.5h |
| **Day 9** | Buffer: repro check, final polish. | 1.5h |
| **Total** | | **~22h active** |

**Hard gates:**
1. End of Day 2: data generator produces valid triples for all templates
2. End of Day 3: reward function correctly scores 20 hand-picked examples (including edge cases)
3. Day 4, step 50: GRPO reward is not flat

**Scope cuts (drop in order if behind):**
1. Reduce templates from 25 to 15
2. Drop buffer day
3. Shorten blog post to ~1000 words
4. Skip HF model upload

---

## 7. Risks

- **Reward function bugs** — the single biggest risk. An incorrect reward teaches the model the wrong thing and you don't realize until eval. Mitigation: Day 3 hard gate with 20 manually verified examples.
- **1.5B too weak for hard transforms** — possible. If hard-tier accuracy stays near 0% for both SFT and GRPO, that's a finding ("the capacity threshold for multi-step PBE is above 1.5B"), not a failure. Report it honestly.
- **Reasoning-length curve stays flat** — if the task doesn't actually benefit from longer thinking, the "aha moment" doesn't appear. Mitigation: the hard tier (chained multi-step transforms) is specifically designed to need multi-step reasoning. If it's still flat, the blog angle shifts to "when does GRPO reasoning help and when doesn't it?"
- **Code execution security** — `exec()` on model-generated code is inherently risky. Mitigation: sandboxed with restricted builtins, 2-second timeout, no filesystem/network access. This is adequate for a local training run, not production.
- **VRAM OOM on 4060** — Unsloth claims 7GB for 1.5B GRPO. If tight, reduce group size from 4 to 2 or max completion length from 512 to 384. Test in first 10 steps.
- **Any result is a valid result.** GRPO beating SFT is great. SFT beating GRPO is also interesting ("execution rewards don't help when gold labels are available"). Both models failing on hard examples is a capacity finding. The project ships regardless of which direction results go.

---

## 8. What This Project Is NOT

- Not a research contribution (no novel algorithm, no benchmark, no paper)
- Not a production system (sandbox is adequate for training, not deployment)
- Not a hyperparameter study (one config, no sweeps)
- Not a multi-model comparison (just 1.5B)

It IS: a hands-on learning project that teaches GRPO mechanics through a novel domain, produces a portfolio-worthy blog post with real training curves and a comparison table, and ships two models to HF Hub.
