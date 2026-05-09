def format_prompt(demos: list[dict]) -> str:
    """Format demonstration I/O pairs into a model prompt."""
    examples = "\n".join(
        f"Example {i+1}: {d['input']!r} → {d['output']!r}"
        for i, d in enumerate(demos)
    )
    return (
        "Given these input/output examples, write a Python expression "
        "that transforms the input to the output.\n\n"
        f"{examples}\n\n"
        "Respond with your reasoning in <think>...</think> tags, "
        "then the Python expression in <code>...</code> tags.\n"
        "The expression should use `x` as the input variable."
    )
