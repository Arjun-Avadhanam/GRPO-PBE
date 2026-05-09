from grpo_pbe.prompt import format_prompt


def test_format_prompt_includes_all_demos():
    demos = [
        {"input": "hello", "output": "HELLO"},
        {"input": "world", "output": "WORLD"},
        {"input": "foo", "output": "FOO"},
    ]
    prompt = format_prompt(demos)
    assert "hello" in prompt
    assert "HELLO" in prompt
    assert "Example 1:" in prompt
    assert "Example 3:" in prompt
    assert "<think>" in prompt
    assert "<code>" in prompt
