from grpo_pbe.sandbox import execute_code


def test_simple_expression():
    result = execute_code("x.upper()", "hello")
    assert result.success is True
    assert result.output == "HELLO"


def test_math_expression():
    result = execute_code("round(x, 2)", 3.14159)
    assert result.success is True
    assert abs(result.output - 3.14) < 1e-9


def test_list_expression():
    result = execute_code("sorted(x)", [3, 1, 2])
    assert result.success is True
    assert result.output == [1, 2, 3]


def test_syntax_error_returns_failure():
    result = execute_code("x.upper(", "hello")
    assert result.success is False
    assert result.output is None


def test_runtime_error_returns_failure():
    result = execute_code("x / 0", 5)
    assert result.success is False


def test_timeout_returns_failure():
    result = execute_code("[i for i in iter(lambda: 1, 0)]", "x")
    assert result.success is False


def test_import_blocked():
    result = execute_code("__import__('os').listdir('.')", "x")
    assert result.success is False


def test_re_module_available():
    result = execute_code("re.sub(r'\\d', '#', x)", "abc123")
    assert result.success is True
    assert result.output == "abc###"


def test_datetime_module_available():
    result = execute_code("datetime.datetime.strptime(x, '%Y-%m-%d').strftime('%b %d')", "2024-01-15")
    assert result.success is True
    assert result.output == "Jan 15"
