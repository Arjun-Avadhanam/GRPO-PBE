from grpo_pbe.evaluate import compute_metrics


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
