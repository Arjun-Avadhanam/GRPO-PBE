import random
import string

from grpo_pbe.templates.base import TransformTemplate


class SwapKeysValues(TransformTemplate):
    """Swap keys and values in a dict."""
    name = "swap_keys_values"
    difficulty = "medium"

    def generate_case(self) -> dict:
        keys = random.sample(string.ascii_lowercase, random.randint(3, 5))
        vals = random.sample(range(1, 20), len(keys))
        d = dict(zip(keys, vals))
        return {
            "input": d,
            "gold_code": "{v: k for k, v in x.items()}",
            "output": {v: k for k, v in d.items()},
        }


class FilterByValue(TransformTemplate):
    """Keep only entries where value > threshold."""
    name = "filter_dict_by_value"
    difficulty = "medium"

    def generate_case(self) -> dict:
        keys = random.sample(["alpha", "beta", "gamma", "delta", "epsilon", "zeta"], random.randint(4, 6))
        vals = {k: random.randint(1, 100) for k in keys}
        threshold = random.choice([25, 50, 75])
        return {
            "input": vals,
            "gold_code": f"{{k: v for k, v in x.items() if v > {threshold}}}",
            "output": {k: v for k, v in vals.items() if v > threshold},
        }


class FormatDictValues(TransformTemplate):
    """Format dict as 'key=value' strings joined by separator."""
    name = "format_dict_entries"
    difficulty = "hard"

    def generate_case(self) -> dict:
        keys = random.sample(["name", "age", "city", "role", "dept"], random.randint(3, 5))
        vals = {k: "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 6))) for k in keys}
        sep = random.choice([", ", " | ", "; "])
        return {
            "input": vals,
            "gold_code": f"'{sep}'.join(f'{{k}}={{v}}' for k, v in x.items())",
            "output": sep.join(f"{k}={v}" for k, v in vals.items()),
        }


DICT_TEMPLATES = [
    SwapKeysValues(),
    FilterByValue(),
    FormatDictValues(),
]
