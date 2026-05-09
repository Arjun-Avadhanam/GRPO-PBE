# tests/test_templates.py
import re
from grpo_pbe.templates.base import TransformTemplate


class FakeTemplate(TransformTemplate):
    name = "fake_upper"
    difficulty = "easy"

    def generate_case(self) -> dict:
        import random
        word = "".join(random.choices("abcdef", k=5))
        return {
            "input": word,
            "gold_code": "x.upper()",
            "output": word.upper(),
        }


def test_template_generate_case_has_required_keys():
    t = FakeTemplate()
    case = t.generate_case()
    assert "input" in case
    assert "gold_code" in case
    assert "output" in case


def test_template_generate_example_splits_correctly():
    t = FakeTemplate()
    example = t.generate_example(n_demo=3, n_test=2)
    assert len(example["demos"]) == 3
    assert len(example["tests"]) == 2
    assert example["difficulty"] == "easy"
    assert example["template_name"] == "fake_upper"


def test_template_gold_code_actually_works():
    t = FakeTemplate()
    case = t.generate_case()
    fn = eval(f"lambda x: {case['gold_code']}")
    assert fn(case["input"]) == case["output"]
