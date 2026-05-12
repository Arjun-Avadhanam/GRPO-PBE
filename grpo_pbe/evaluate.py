"""Evaluation harness: run inference on the held-out set, compute metrics."""
import json
from collections import defaultdict
from pathlib import Path

# torch and unsloth are heavy and touch CUDA on import — load them lazily
# inside the functions that need them so `compute_metrics` stays unit-testable
# without triggering CUDA initialisation.
from grpo_pbe.data_generator import load_dataset
from grpo_pbe.reward import compute_reward, parse_response


SYSTEM_PROMPT = (
    "You solve data transformation tasks. Given input/output examples, "
    "write a Python expression using `x` as the input variable. "
    "Respond with reasoning in <think>...</think> tags, "
    "then the expression in <code>...</code> tags."
)


def compute_metrics(results: list[dict]) -> dict:
    """Compute accuracy metrics from evaluation results.

    Args:
        results: list of dicts with "correct" (bool) and "difficulty" keys.

    Returns:
        dict with overall_accuracy and per-difficulty accuracies.
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
        reward = compute_reward(response, example["tests"])
        parsed = parse_response(response)

        results.append({
            "template_name": example["template_name"],
            "difficulty": example["difficulty"],
            # All held-out tests must pass: reward >= 1.0 means correctness=1.0
            "correct": reward >= 1.0,
            "reward": reward,
            "response": response,
            "think_length": len(parsed.think) if parsed.think else 0,
            "code": parsed.code,
            "gold_code": example["gold_code"],
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
