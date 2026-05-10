import re
from dataclasses import dataclass
from typing import Any

from grpo_pbe.sandbox import execute_code


FORMAT_REWARD_WEIGHT = 0.1
CORRECTNESS_REWARD_WEIGHT = 1.0


@dataclass
class ParsedResponse:
    think: str | None
    code: str | None
    format_valid: bool


def parse_response(text: str) -> ParsedResponse:
    """Parse <think>...</think><code>...</code> from model output."""
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    code_match = re.search(r"<code>(.*?)</code>", text, re.DOTALL)

    if think_match and code_match:
        return ParsedResponse(
            think=think_match.group(1).strip(),
            code=code_match.group(1).strip(),
            format_valid=True,
        )
    return ParsedResponse(think=None, code=None, format_valid=False)


def _outputs_match(predicted: Any, expected: Any) -> bool:
    """Compare outputs with type-aware tolerance."""
    if isinstance(expected, float) and isinstance(predicted, float):
        return abs(predicted - expected) < 1e-6
    return predicted == expected


def compute_reward(response: str, tests: list[dict]) -> float:
    """Compute the reward for a model response.

    Args:
        response: raw model output string
        tests: list of {"input": ..., "output": ...} held-out test pairs

    Returns:
        float reward in [0.0, 1.1]
    """
    parsed = parse_response(response)

    format_reward = FORMAT_REWARD_WEIGHT if parsed.format_valid else 0.0

    if not parsed.format_valid or parsed.code is None:
        return format_reward

    n_correct = 0
    for test_case in tests:
        result = execute_code(parsed.code, test_case["input"])
        if result.success and _outputs_match(result.output, test_case["output"]):
            n_correct += 1

    correctness = (n_correct / len(tests)) * CORRECTNESS_REWARD_WEIGHT if tests else 0.0
    return format_reward + correctness
