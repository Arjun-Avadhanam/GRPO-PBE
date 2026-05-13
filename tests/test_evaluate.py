from grpo_pbe.evaluate import (
    code_passes_all_tests,
    compute_metrics,
    extract_code_lenient,
)


def test_compute_metrics_all_correct():
    results = [
        {"correct": True, "difficulty": "easy"},
        {"correct": True, "difficulty": "medium"},
        {"correct": True, "difficulty": "hard"},
    ]
    metrics = compute_metrics(results)
    assert metrics["overall_accuracy"] == 1.0
    assert metrics["easy_accuracy"] == 1.0
    assert metrics["medium_accuracy"] == 1.0
    assert metrics["hard_accuracy"] == 1.0


def test_compute_metrics_all_wrong():
    results = [
        {"correct": False, "difficulty": "easy"},
        {"correct": False, "difficulty": "medium"},
    ]
    metrics = compute_metrics(results)
    assert metrics["overall_accuracy"] == 0.0
    assert metrics["easy_accuracy"] == 0.0
    assert metrics["medium_accuracy"] == 0.0


def test_compute_metrics_partial():
    results = [
        {"correct": True, "difficulty": "easy"},
        {"correct": False, "difficulty": "easy"},
        {"correct": True, "difficulty": "hard"},
        {"correct": False, "difficulty": "hard"},
    ]
    metrics = compute_metrics(results)
    assert metrics["overall_accuracy"] == 0.5
    assert metrics["easy_accuracy"] == 0.5
    assert metrics["hard_accuracy"] == 0.5


def test_compute_metrics_empty():
    assert compute_metrics([]) == {}


def test_compute_metrics_only_one_difficulty():
    """Per-difficulty keys should only exist for difficulties actually present."""
    results = [
        {"correct": True, "difficulty": "easy"},
        {"correct": False, "difficulty": "easy"},
    ]
    metrics = compute_metrics(results)
    assert metrics["overall_accuracy"] == 0.5
    assert metrics["easy_accuracy"] == 0.5
    assert "medium_accuracy" not in metrics
    assert "hard_accuracy" not in metrics


def test_compute_metrics_format_compliance():
    """When format_valid is present, format_compliance is reported."""
    results = [
        {"correct": True,  "difficulty": "easy", "format_valid": True},
        {"correct": False, "difficulty": "easy", "format_valid": False},
        {"correct": False, "difficulty": "easy", "format_valid": True},
    ]
    metrics = compute_metrics(results)
    assert metrics["format_compliance"] == 2 / 3


def test_compute_metrics_no_format_field():
    """If format_valid is absent, format_compliance is not added."""
    results = [{"correct": True, "difficulty": "easy"}]
    metrics = compute_metrics(results)
    assert "format_compliance" not in metrics


# --- extract_code_lenient ---


def test_extract_code_with_both_tags():
    """Standard GRPO output: <think>...</think><code>...</code>."""
    text = "<think>uppercase the string</think><code>x.upper()</code>"
    assert extract_code_lenient(text) == "x.upper()"


def test_extract_code_without_think_tag():
    """SFT-style output: just <code>...</code>, no <think>."""
    text = "<code>sorted(x, key=len)</code>"
    assert extract_code_lenient(text) == "sorted(x, key=len)"


def test_extract_code_with_extra_text_around():
    """Robust against prefixes/suffixes the model might tack on."""
    text = "Here is my answer: <code>x[:3]</code> hope it's right!"
    assert extract_code_lenient(text) == "x[:3]"


def test_extract_code_missing_returns_none():
    """No <code> tag at all → None (e.g. base model writing markdown blocks)."""
    text = "```python\ndef transform(x):\n    return x.upper()\n```"
    assert extract_code_lenient(text) is None


def test_extract_code_picks_first_when_multiple():
    """If the model emits multiple <code> blocks, take the first."""
    text = "<code>x.upper()</code> ... actually <code>x.lower()</code>"
    assert extract_code_lenient(text) == "x.upper()"


# --- code_passes_all_tests ---


def test_passes_all_tests_happy_path():
    assert code_passes_all_tests(
        "x.upper()",
        [{"input": "hello", "output": "HELLO"}, {"input": "abc", "output": "ABC"}],
    )


def test_passes_all_tests_partial_fail():
    """Code right for some inputs, wrong for others → False (must pass ALL)."""
    assert not code_passes_all_tests(
        "x * 3",
        [{"input": 3, "output": 9}, {"input": -2, "output": 4}],
    )


def test_passes_all_tests_syntax_error():
    """Bad code: sandbox returns success=False → overall False."""
    assert not code_passes_all_tests(
        "x.upper(",
        [{"input": "hello", "output": "HELLO"}],
    )


def test_passes_all_tests_none_code():
    """No extracted code → False, no test execution."""
    assert not code_passes_all_tests(None, [{"input": "x", "output": "X"}])


def test_passes_all_tests_empty_tests_returns_false():
    """No tests = no evidence of correctness → False (not vacuously true)."""
    assert not code_passes_all_tests("x.upper()", [])
