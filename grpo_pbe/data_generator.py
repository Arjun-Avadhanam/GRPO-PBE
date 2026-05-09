import json
import random
from pathlib import Path

from grpo_pbe.templates import ALL_TEMPLATES
from grpo_pbe.prompt import format_prompt


def generate_dataset(
    n_per_template: int = 60,
    n_demo: int = 3,
    n_test: int = 2,
    seed: int = 42,
) -> list[dict]:
    random.seed(seed)
    dataset = []

    for template in ALL_TEMPLATES:
        for _ in range(n_per_template):
            example = template.generate_example(n_demo=n_demo, n_test=n_test)
            example["prompt"] = format_prompt(example["demos"])
            dataset.append(example)

    random.shuffle(dataset)
    return dataset


def save_dataset(dataset: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for example in dataset:
            f.write(json.dumps(example) + "\n")


def load_dataset(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    dataset = generate_dataset(n_per_template=60, seed=42)
    eval_set = dataset[-200:]
    train_set = dataset[:-200]

    save_dataset(train_set, "data/train.jsonl")
    save_dataset(eval_set, "data/eval.jsonl")

    from collections import Counter
    diff_counts = Counter(ex["difficulty"] for ex in train_set)
    print(f"Train: {len(train_set)} examples")
    print(f"Eval:  {len(eval_set)} examples")
    print(f"Difficulty distribution (train): {dict(diff_counts)}")
