"""Evaluation harness: run inference on the held-out set, compute metrics."""
import json
import os
import re
from collections import defaultdict
from pathlib import Path

# hf_transfer's parallel-chunk downloader deadlocks on some networks (WSL +
# HF CloudFront edge); standard HF downloads are slower but reliable.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

# torch and unsloth are heavy and touch CUDA on import — load them lazily
# inside the functions that need them so `compute_metrics` stays unit-testable
# without triggering CUDA initialisation.
from grpo_pbe.data_generator import load_dataset
from grpo_pbe.reward import parse_response
from grpo_pbe.sandbox import execute_code


SYSTEM_PROMPT = (
    "You solve data transformation tasks. Given input/output examples, "
    "write a Python expression using `x` as the input variable. "
    "Respond with reasoning in <think>...</think> tags, "
    "then the expression in <code>...</code> tags."
)


_CODE_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL)


def extract_code_lenient(text: str) -> str | None:
    """Extract the first <code>...</code> body from text, regardless of <think>.

    Training-time `compute_reward` requires BOTH <think> and <code> tags so that
    GRPO is pushed to produce structured reasoning. For *evaluation* we just
    want to know whether the model produced an executable expression that
    solves the task — even SFT (trained to emit <code> only, no <think>) should
    be scored fairly.
    """
    m = _CODE_RE.search(text)
    return m.group(1).strip() if m else None


def _outputs_match(predicted, expected) -> bool:
    if isinstance(expected, float) and isinstance(predicted, float):
        return abs(predicted - expected) < 1e-6
    return predicted == expected


def code_passes_all_tests(code: str | None, tests: list[dict]) -> bool:
    """Run `code` (as `lambda x: <code>`) against every held-out test."""
    if not code or not tests:
        return False
    for t in tests:
        result = execute_code(code, t["input"])
        if not (result.success and _outputs_match(result.output, t["output"])):
            return False
    return True


def compute_metrics(results: list[dict]) -> dict:
    """Compute accuracy metrics from evaluation results.

    Args:
        results: list of dicts with "correct" (bool) and "difficulty" keys.
            Optionally "format_valid" (bool) for strict-format compliance.

    Returns:
        dict with overall_accuracy and per-difficulty accuracies, plus
        format_compliance if "format_valid" is present in any result.
        Per-difficulty keys are only present for difficulties that appear in `results`.
    """
    if not results:
        return {}

    by_difficulty = defaultdict(list)
    for r in results:
        by_difficulty[r["difficulty"]].append(r["correct"])

    metrics = {
        "overall_accuracy": sum(r["correct"] for r in results) / len(results),
    }
    for diff, vals in by_difficulty.items():
        metrics[f"{diff}_accuracy"] = sum(vals) / len(vals)

    if any("format_valid" in r for r in results):
        metrics["format_compliance"] = (
            sum(r.get("format_valid", False) for r in results) / len(results)
        )
    return metrics


def _build_chat_prompt(tokenizer, user_prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def run_inference(
    model,
    tokenizer,
    eval_data: list[dict],
    max_new_tokens: int = 512,
    progress_every: int = 20,
) -> list[dict]:
    """Run greedy inference on each eval example and score it.

    Returns a list of per-example result dicts: template_name, difficulty,
    correct, reward, response, think_length, code, gold_code.
    """
    import torch

    results = []
    model.eval()

    for i, example in enumerate(eval_data):
        chat_prompt = _build_chat_prompt(tokenizer, example["prompt"])
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Lenient code extraction: accept <code>...</code> with or without <think>.
        # SFT was trained to emit only <code>; scoring it under the strict
        # training-time format would unfairly zero it out.
        code = extract_code_lenient(response)
        correct = code_passes_all_tests(code, example["tests"])

        # Strict format check (both tags) tracked separately as format_compliance.
        parsed = parse_response(response)

        results.append({
            "template_name": example["template_name"],
            "difficulty": example["difficulty"],
            "correct": correct,
            "format_valid": parsed.format_valid,
            "response": response,
            "think_length": len(parsed.think) if parsed.think else 0,
            "code": code,
            "gold_code": example["gold_code"],
            # Saved so the JSON can be re-graded later without GPU work and
            # without lossy template-matching against the original eval set.
            "tests": example["tests"],
        })

        if (i + 1) % progress_every == 0:
            acc = sum(r["correct"] for r in results) / len(results)
            print(f"  [{i+1}/{len(eval_data)}] running accuracy: {acc:.1%}")

    return results


def evaluate_model(
    checkpoint_path: str,
    eval_path: str = "data/eval.jsonl",
    label: str = "model",
    output_dir: str = "data",
) -> dict:
    """Full evaluation pipeline for one checkpoint or HF model id."""
    from unsloth import FastLanguageModel

    print(f"\n=== Evaluating: {label} ({checkpoint_path}) ===")
    eval_data = load_dataset(eval_path)
    print(f"Loaded {len(eval_data)} eval examples")

    model, tokenizer = FastLanguageModel.from_pretrained(
        checkpoint_path,
        max_seq_length=1024,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    results = run_inference(model, tokenizer, eval_data)
    metrics = compute_metrics(results)

    think_lengths = [r["think_length"] for r in results if r["think_length"] > 0]
    if think_lengths:
        metrics["mean_think_length"] = sum(think_lengths) / len(think_lengths)

    print(f"\nResults for {label}:")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.3f}")

    out_path = Path(output_dir) / f"eval_results_{label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)
    print(f"Wrote {out_path}")

    return metrics


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m grpo_pbe.evaluate <checkpoint_path> [label]")
        print("Example: python -m grpo_pbe.evaluate checkpoints/grpo-pbe-1.5b grpo")
        sys.exit(1)

    checkpoint = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else "model"
    evaluate_model(checkpoint, label=label)
