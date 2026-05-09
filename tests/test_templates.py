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


def test_all_string_templates_produce_valid_cases():
    from grpo_pbe.templates.string_ops import STRING_TEMPLATES

    for template in STRING_TEMPLATES:
        for _ in range(5):
            case = template.generate_case()
            fn = eval(f"lambda x: {case['gold_code']}")
            result = fn(case["input"])
            assert result == case["output"], (
                f"{template.name}: expected {case['output']!r}, got {result!r} "
                f"for input {case['input']!r} with code {case['gold_code']!r}"
            )


import re as re_module
import datetime


def _validate_template_list(templates):
    """Helper: verify every case in a template list executes correctly."""
    for template in templates:
        for _ in range(5):
            case = template.generate_case()
            exec_globals = {"__builtins__": __builtins__, "re": re_module, "datetime": datetime}
            fn = eval(f"lambda x: {case['gold_code']}", exec_globals)
            result = fn(case["input"])
            if isinstance(result, float) and isinstance(case["output"], float):
                assert abs(result - case["output"]) < 1e-6, (
                    f"{template.name}: expected {case['output']}, got {result}"
                )
            else:
                assert result == case["output"], (
                    f"{template.name}: expected {case['output']!r}, got {result!r} "
                    f"for input {case['input']!r} with code {case['gold_code']!r}"
                )


def test_all_regex_templates_valid():
    from grpo_pbe.templates.regex_ops import REGEX_TEMPLATES
    _validate_template_list(REGEX_TEMPLATES)


def test_all_date_templates_valid():
    from grpo_pbe.templates.date_ops import DATE_TEMPLATES
    _validate_template_list(DATE_TEMPLATES)


def test_all_numeric_templates_valid():
    from grpo_pbe.templates.numeric_ops import NUMERIC_TEMPLATES
    _validate_template_list(NUMERIC_TEMPLATES)
