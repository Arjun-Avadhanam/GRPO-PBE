# Results

Held-out evaluation on 200 examples (89 easy, 80 medium, 31 hard) from
`data/eval.jsonl`. Each example provides 3 demo I/O pairs in the prompt
and 2 held-out test cases for scoring. An example is `correct` when the
extracted `<code>` body, run on every held-out test, matches the expected
output exactly.

| Model | Easy (n=89) | Medium (n=80) | Hard (n=31) | Overall (n=200) | Format compliance | Mean think length |
|---|---|---|---|---|---|---|
| Base (zero-shot) | 0.0% | 0.0% | 0.0% | **0.0%** | 2.0% | 110 chars |
| SFT             | 97.8% | 95.0% | 100.0% | **97.0%** | 0.0% | n/a |
| GRPO            | 1.1% | 0.0% | 0.0% | **0.5%** | 100.0% | 118 chars |

`Format compliance` is the strict training-time check (both `<think>` and
`<code>` present). SFT was trained without `<think>` traces by design, so
0.0% is the expected value. GRPO learned the strict format on every single
held-out prompt.

`Mean think length` reports the average character count inside `<think>`
across responses that contain a `<think>` tag. SFT has no entry because it
never emits one.

## Per-model artefacts

* `data/eval_results_base.json`
* `data/eval_results_sft.json`
* `data/eval_results_grpo.json`

Each file has a top-level `metrics` object plus a `results` list of 200
per-example rows. Per-row fields:

* `template_name`, `difficulty`
* `correct` (lenient scoring used for the table above)
* `format_valid` (strict scoring with both tags)
* `response` (raw decoded model output)
* `code` (lenient extraction from `<code>...</code>`, may be `null`)
* `gold_code` (the canonical expression for that example)
* `tests` (the 2 held-out test cases this example was scored against)
* `think_length` (character count inside `<think>` if present, else 0)

## Training runs

* GRPO: <https://wandb.ai/arjunvijayavadhanam-shiv-nadar-university/grpo-pbe/runs/58t4o2pm>
  600 steps, ~95 min, peak `train/reward` 0.143 at step ~500.
* SFT:  <https://wandb.ai/arjunvijayavadhanam-shiv-nadar-university/grpo-pbe/runs/kp4g6z1x>
  3 epochs, monotonic loss decrease.

## Reproducibility

Data generation is seeded (`seed=42` in `grpo_pbe/data_generator.py`),
so `python -m grpo_pbe.data_generator` produces the same 1300/200 split
on any machine. Training is not bitwise-reproducible (`bf16`,
non-deterministic CUDA ops) but the numbers in the table should be
within a couple of percentage points across re-runs.

See `docs/FINDINGS.md` for the mechanistic analysis behind these numbers.
