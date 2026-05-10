from grpo_pbe.reward import compute_reward, parse_response


def test_parse_valid_response():
    text = "<think>dates are reformatted</think><code>x.upper()</code>"
    parsed = parse_response(text)
    assert parsed.think == "dates are reformatted"
    assert parsed.code == "x.upper()"
    assert parsed.format_valid is True


def test_parse_missing_tags():
    parsed = parse_response("just some text without tags")
    assert parsed.format_valid is False
    assert parsed.code is None


def test_parse_partial_tags():
    parsed = parse_response("<think>reasoning</think> no code tag")
    assert parsed.format_valid is False


def test_reward_perfect_score():
    tests = [
        {"input": "hello", "output": "HELLO"},
        {"input": "world", "output": "WORLD"},
    ]
    response = "<think>uppercase</think><code>x.upper()</code>"
    reward = compute_reward(response, tests)
    assert abs(reward - 1.1) < 1e-6


def test_reward_correct_code_wrong_format():
    tests = [{"input": "hello", "output": "HELLO"}]
    response = "x.upper()"
    reward = compute_reward(response, tests)
    assert abs(reward - 0.0) < 1e-6


def test_reward_correct_format_wrong_code():
    tests = [
        {"input": "hello", "output": "HELLO"},
        {"input": "world", "output": "WORLD"},
    ]
    response = "<think>lowercase</think><code>x.lower()</code>"
    reward = compute_reward(response, tests)
    assert abs(reward - 0.1) < 1e-6


def test_reward_partial_correctness():
    tests = [
        {"input": 3, "output": 9},
        {"input": -2, "output": 4},
    ]
    response = "<think>multiply</think><code>x * 3</code>"
    reward = compute_reward(response, tests)
    assert abs(reward - 0.6) < 1e-6


def test_reward_syntax_error_zero_correctness():
    tests = [{"input": "hello", "output": "HELLO"}]
    response = "<think>try</think><code>x.upper(</code>"
    reward = compute_reward(response, tests)
    assert abs(reward - 0.1) < 1e-6
