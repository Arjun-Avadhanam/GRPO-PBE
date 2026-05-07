# GRPO-PBE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train Qwen2.5-1.5B via GRPO on synthetic programming-by-example data transformations, with an SFT baseline, producing a blog-post-ready results table and training curves.

**Architecture:** Synthetic data generator (25 templates → 1500+ examples) feeds a reward function (format + execution correctness) into Unsloth's GRPOTrainer. A parallel SFT baseline trains on gold labels. Both are evaluated on a held-out set with accuracy split by difficulty tier.

**Tech Stack:** Python 3.11, Unsloth, TRL (GRPOTrainer + SFTTrainer), PyTorch, Weights & Biases, Pydantic, pytest

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `grpo_pbe/__init__.py`
- Create: `grpo_pbe/templates/__init__.py`
- Create: `data/.gitkeep`
- Create: `notebooks/.gitkeep`
- Create: `reports/.gitkeep`
- Create: `.gitignore`

- [ ] **Step 1: Initialize git repo**

```bash
cd /home/arjun/GRPO_PBE
git init
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[project]
name = "grpo-pbe"
version = "0.1.0"
description = "GRPO training for programming-by-example data transformations"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.4",
    "unsloth",
    "trl",
    "transformers",
    "datasets",
    "peft",
    "accelerate",
    "bitsandbytes",
    "wandb",
    "pydantic>=2.0",
    "pytest",
]

[project.optional-dependencies]
dev = ["ruff", "ipykernel", "matplotlib", "seaborn"]
```

- [ ] **Step 3: Create directory structure and __init__ files**

```python
# grpo_pbe/__init__.py
"""GRPO training for programming-by-example data transformations."""
```

```python
# grpo_pbe/templates/__init__.py
"""Transformation templates for synthetic data generation."""
from grpo_pbe.templates.string_ops import STRING_TEMPLATES
from grpo_pbe.templates.regex_ops import REGEX_TEMPLATES
from grpo_pbe.templates.date_ops import DATE_TEMPLATES
from grpo_pbe.templates.numeric_ops import NUMERIC_TEMPLATES
from grpo_pbe.templates.list_ops import LIST_TEMPLATES
from grpo_pbe.templates.dict_ops import DICT_TEMPLATES
from grpo_pbe.templates.chained_ops import CHAINED_TEMPLATES

ALL_TEMPLATES = (
    STRING_TEMPLATES
    + REGEX_TEMPLATES
    + DATE_TEMPLATES
    + NUMERIC_TEMPLATES
    + LIST_TEMPLATES
    + DICT_TEMPLATES
    + CHAINED_TEMPLATES
)
```

- [ ] **Step 4: Create .gitignore**

```
__pycache__/
*.pyc
.venv/
data/*.json
data/*.jsonl
checkpoints/
wandb/
*.egg-info/
dist/
```

- [ ] **Step 5: Create placeholder directories**

```bash
mkdir -p data notebooks reports tests
touch data/.gitkeep notebooks/.gitkeep reports/.gitkeep tests/__init__.py
```

- [ ] **Step 6: Install project in dev mode**

```bash
cd /home/arjun/GRPO_PBE
pip install -e ".[dev]"
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: project scaffolding"
```

---

### Task 2: Template Base Class

**Files:**
- Create: `grpo_pbe/templates/base.py`
- Create: `tests/test_templates.py`

- [ ] **Step 1: Write the test for the base template interface**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/arjun/GRPO_PBE
pytest tests/test_templates.py -v
```

Expected: FAIL — `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement TransformTemplate base class**

```python
# grpo_pbe/templates/base.py
from abc import ABC, abstractmethod


class TransformTemplate(ABC):
    """Base class for all transformation templates.

    Subclasses must define `name`, `difficulty`, and `generate_case()`.
    """

    name: str = ""
    difficulty: str = ""  # "easy", "medium", or "hard"

    @abstractmethod
    def generate_case(self) -> dict:
        """Generate a single (input, gold_code, output) triple.

        Returns:
            dict with keys: "input", "gold_code", "output"
        """
        ...

    def generate_example(self, n_demo: int = 3, n_test: int = 2) -> dict:
        """Generate a full example with demo and test splits.

        Args:
            n_demo: number of demonstration I/O pairs shown to the model
            n_test: number of held-out I/O pairs for reward computation

        Returns:
            dict with keys: "template_name", "difficulty", "gold_code",
                            "demos" (list of {input, output}),
                            "tests" (list of {input, output})
        """
        cases = [self.generate_case() for _ in range(n_demo + n_test)]
        gold_code = cases[0]["gold_code"]  # same transform for all cases
        return {
            "template_name": self.name,
            "difficulty": self.difficulty,
            "gold_code": gold_code,
            "demos": [{"input": c["input"], "output": c["output"]} for c in cases[:n_demo]],
            "tests": [{"input": c["input"], "output": c["output"]} for c in cases[n_demo:]],
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_templates.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add grpo_pbe/templates/base.py tests/test_templates.py
git commit -m "feat: TransformTemplate base class with generate_example"
```

---

### Task 3: Easy Templates — String & Case Operations

**Files:**
- Create: `grpo_pbe/templates/string_ops.py`

These templates generate randomized string inputs and pair them with a gold Python expression. Each template's `generate_case()` returns a fresh random instance.

- [ ] **Step 1: Add a test that all string templates produce valid, executable cases**

Add to `tests/test_templates.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_templates.py::test_all_string_templates_produce_valid_cases -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement string_ops.py with 5 templates**

```python
# grpo_pbe/templates/string_ops.py
import random
import string

from grpo_pbe.templates.base import TransformTemplate


class FirstNChars(TransformTemplate):
    """Extract first N characters from a string."""
    name = "first_n_chars"
    difficulty = "easy"

    def generate_case(self) -> dict:
        n = random.randint(2, 5)
        word = "".join(random.choices(string.ascii_lowercase, k=random.randint(n + 2, 15)))
        return {
            "input": word,
            "gold_code": f"x[:{n}]",
            "output": word[:n],
        }


class LastWord(TransformTemplate):
    """Extract the last word from a multi-word string."""
    name = "last_word"
    difficulty = "easy"

    def generate_case(self) -> dict:
        words = ["".join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
                 for _ in range(random.randint(2, 5))]
        sentence = " ".join(words)
        return {
            "input": sentence,
            "gold_code": "x.split()[-1]",
            "output": sentence.split()[-1],
        }


class TitleCase(TransformTemplate):
    """Convert a string to title case."""
    name = "title_case"
    difficulty = "easy"

    def generate_case(self) -> dict:
        words = ["".join(random.choices(string.ascii_lowercase, k=random.randint(3, 7)))
                 for _ in range(random.randint(2, 4))]
        sentence = " ".join(words)
        return {
            "input": sentence,
            "gold_code": "x.title()",
            "output": sentence.title(),
        }


class SwapCase(TransformTemplate):
    """Swap the case of each character."""
    name = "swap_case"
    difficulty = "easy"

    def generate_case(self) -> dict:
        word = "".join(random.choices(string.ascii_letters, k=random.randint(5, 12)))
        return {
            "input": word,
            "gold_code": "x.swapcase()",
            "output": word.swapcase(),
        }


class ReverseWords(TransformTemplate):
    """Reverse the order of words in a string."""
    name = "reverse_words"
    difficulty = "easy"

    def generate_case(self) -> dict:
        words = ["".join(random.choices(string.ascii_lowercase, k=random.randint(3, 7)))
                 for _ in range(random.randint(3, 6))]
        sentence = " ".join(words)
        return {
            "input": sentence,
            "gold_code": "' '.join(x.split()[::-1])",
            "output": " ".join(sentence.split()[::-1]),
        }


STRING_TEMPLATES = [
    FirstNChars(),
    LastWord(),
    TitleCase(),
    SwapCase(),
    ReverseWords(),
]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_templates.py -v
```

Expected: all pass (including the new parametric test over 5 templates × 5 random cases)

- [ ] **Step 5: Commit**

```bash
git add grpo_pbe/templates/string_ops.py tests/test_templates.py
git commit -m "feat: 5 easy string transformation templates"
```

---

### Task 4: Medium Templates — Regex, Date, Numeric

**Files:**
- Create: `grpo_pbe/templates/regex_ops.py`
- Create: `grpo_pbe/templates/date_ops.py`
- Create: `grpo_pbe/templates/numeric_ops.py`

- [ ] **Step 1: Add parametric tests for regex, date, and numeric templates**

Add to `tests/test_templates.py`:

```python
import re as re_module
import datetime

def _validate_template_list(templates, allowed_imports=""):
    """Helper: verify every case in a template list executes correctly."""
    for template in templates:
        for _ in range(5):
            case = template.generate_case()
            # Build execution context with allowed modules
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_templates.py -k "regex or date or numeric" -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement regex_ops.py with 3 templates**

```python
# grpo_pbe/templates/regex_ops.py
import random
import re
import string

from grpo_pbe.templates.base import TransformTemplate


class ExtractNumbers(TransformTemplate):
    """Extract all numbers from a string, joined by comma."""
    name = "extract_numbers"
    difficulty = "easy"

    def generate_case(self) -> dict:
        nums = [str(random.randint(1, 999)) for _ in range(random.randint(2, 4))]
        words = [random.choice(["item", "price", "qty", "ref", "code"]) for _ in nums]
        parts = [f"{w}{n}" for w, n in zip(words, nums)]
        text = " ".join(parts)
        return {
            "input": text,
            "gold_code": "','.join(re.findall(r'\\d+', x))",
            "output": ",".join(re.findall(r'\d+', text)),
        }


class ExtractEmails(TransformTemplate):
    """Extract the first email address from a string."""
    name = "extract_email"
    difficulty = "medium"

    def generate_case(self) -> dict:
        user = "".join(random.choices(string.ascii_lowercase, k=random.randint(4, 8)))
        domain = "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 6)))
        tld = random.choice(["com", "org", "net", "io"])
        email = f"{user}@{domain}.{tld}"
        prefix = "".join(random.choices(string.ascii_letters + " ", k=random.randint(5, 20)))
        text = f"{prefix} {email} end"
        return {
            "input": text,
            "gold_code": "re.search(r'[\\w.+-]+@[\\w-]+\\.[\\w.]+', x).group()",
            "output": re.search(r'[\w.+-]+@[\w-]+\.[\w.]+', text).group(),
        }


class ReplaceDigitsWithHash(TransformTemplate):
    """Replace all digits in a string with '#'."""
    name = "replace_digits_hash"
    difficulty = "easy"

    def generate_case(self) -> dict:
        parts = []
        for _ in range(random.randint(3, 6)):
            if random.random() < 0.4:
                parts.append(str(random.randint(0, 99)))
            else:
                parts.append("".join(random.choices(string.ascii_lowercase, k=random.randint(2, 5))))
        text = "".join(parts)
        return {
            "input": text,
            "gold_code": "re.sub(r'\\d', '#', x)",
            "output": re.sub(r'\d', '#', text),
        }


REGEX_TEMPLATES = [
    ExtractNumbers(),
    ExtractEmails(),
    ReplaceDigitsWithHash(),
]
```

- [ ] **Step 4: Implement date_ops.py with 2 templates**

```python
# grpo_pbe/templates/date_ops.py
import random
from datetime import datetime, timedelta

from grpo_pbe.templates.base import TransformTemplate


class DateYMDToMonthDay(TransformTemplate):
    """Reformat YYYY-MM-DD to 'Mon DD' (e.g., 'Jan 15')."""
    name = "date_ymd_to_month_day"
    difficulty = "medium"

    def generate_case(self) -> dict:
        base = datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1800))
        date_str = base.strftime("%Y-%m-%d")
        return {
            "input": date_str,
            "gold_code": "datetime.datetime.strptime(x, '%Y-%m-%d').strftime('%b %d')",
            "output": base.strftime("%b %d"),
        }


class DateDMYToISO(TransformTemplate):
    """Reformat DD/MM/YYYY to YYYY-MM-DD."""
    name = "date_dmy_to_iso"
    difficulty = "medium"

    def generate_case(self) -> dict:
        base = datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1800))
        dmy = base.strftime("%d/%m/%Y")
        iso = base.strftime("%Y-%m-%d")
        return {
            "input": dmy,
            "gold_code": "datetime.datetime.strptime(x, '%d/%m/%Y').strftime('%Y-%m-%d')",
            "output": iso,
        }


DATE_TEMPLATES = [
    DateYMDToMonthDay(),
    DateDMYToISO(),
]
```

- [ ] **Step 5: Implement numeric_ops.py with 4 templates**

```python
# grpo_pbe/templates/numeric_ops.py
import random

from grpo_pbe.templates.base import TransformTemplate


class RoundToN(TransformTemplate):
    """Round a float to N decimal places."""
    name = "round_to_n"
    difficulty = "easy"

    def generate_case(self) -> dict:
        n = random.choice([1, 2, 3])
        val = round(random.uniform(0.001, 999.999), 6)
        return {
            "input": val,
            "gold_code": f"round(x, {n})",
            "output": round(val, n),
        }


class FloatToCurrency(TransformTemplate):
    """Format a float as a currency string like '$12.50'."""
    name = "float_to_currency"
    difficulty = "medium"

    def generate_case(self) -> dict:
        val = round(random.uniform(1.0, 9999.99), 2)
        return {
            "input": val,
            "gold_code": "f'${x:,.2f}'",
            "output": f"${val:,.2f}",
        }


class IntToHex(TransformTemplate):
    """Convert an integer to its hexadecimal string (lowercase, no 0x prefix)."""
    name = "int_to_hex"
    difficulty = "easy"

    def generate_case(self) -> dict:
        val = random.randint(0, 65535)
        return {
            "input": val,
            "gold_code": "hex(x)[2:]",
            "output": hex(val)[2:],
        }


class PercentageFormat(TransformTemplate):
    """Convert a decimal fraction (0.0-1.0) to a percentage string like '45.2%'."""
    name = "percentage_format"
    difficulty = "medium"

    def generate_case(self) -> dict:
        val = round(random.uniform(0.0, 1.0), 4)
        return {
            "input": val,
            "gold_code": "f'{x * 100:.1f}%'",
            "output": f"{val * 100:.1f}%",
        }


NUMERIC_TEMPLATES = [
    RoundToN(),
    FloatToCurrency(),
    IntToHex(),
    PercentageFormat(),
]
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_templates.py -k "regex or date or numeric" -v
```

Expected: all 3 parametric tests pass (3 + 2 + 4 templates × 5 random cases each)

- [ ] **Step 7: Commit**

```bash
git add grpo_pbe/templates/regex_ops.py grpo_pbe/templates/date_ops.py grpo_pbe/templates/numeric_ops.py tests/test_templates.py
git commit -m "feat: regex, date, numeric templates (9 templates, easy-medium)"
```

---

### Task 5: Medium-Hard Templates — List & Dict Operations

**Files:**
- Create: `grpo_pbe/templates/list_ops.py`
- Create: `grpo_pbe/templates/dict_ops.py`

- [ ] **Step 1: Add parametric tests**

Add to `tests/test_templates.py`:

```python
def test_all_list_templates_valid():
    from grpo_pbe.templates.list_ops import LIST_TEMPLATES
    _validate_template_list(LIST_TEMPLATES)


def test_all_dict_templates_valid():
    from grpo_pbe.templates.dict_ops import DICT_TEMPLATES
    _validate_template_list(DICT_TEMPLATES)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_templates.py -k "list or dict" -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement list_ops.py with 5 templates**

```python
# grpo_pbe/templates/list_ops.py
import random
import string

from grpo_pbe.templates.base import TransformTemplate


class SortList(TransformTemplate):
    """Sort a list of integers in ascending order."""
    name = "sort_list"
    difficulty = "easy"

    def generate_case(self) -> dict:
        nums = [random.randint(-50, 50) for _ in range(random.randint(4, 8))]
        return {
            "input": nums,
            "gold_code": "sorted(x)",
            "output": sorted(nums),
        }


class FilterPositive(TransformTemplate):
    """Filter a list to only positive numbers."""
    name = "filter_positive"
    difficulty = "medium"

    def generate_case(self) -> dict:
        nums = [random.randint(-20, 20) for _ in range(random.randint(5, 10))]
        return {
            "input": nums,
            "gold_code": "[n for n in x if n > 0]",
            "output": [n for n in nums if n > 0],
        }


class DeduplicatePreserveOrder(TransformTemplate):
    """Remove duplicates from a list, preserving first occurrence order."""
    name = "deduplicate"
    difficulty = "medium"

    def generate_case(self) -> dict:
        pool = list(range(1, 8))
        nums = [random.choice(pool) for _ in range(random.randint(6, 12))]
        return {
            "input": nums,
            "gold_code": "list(dict.fromkeys(x))",
            "output": list(dict.fromkeys(nums)),
        }


class UppercaseStrings(TransformTemplate):
    """Uppercase all strings in a list."""
    name = "uppercase_strings"
    difficulty = "easy"

    def generate_case(self) -> dict:
        words = ["".join(random.choices(string.ascii_lowercase, k=random.randint(3, 7)))
                 for _ in range(random.randint(3, 6))]
        return {
            "input": words,
            "gold_code": "[s.upper() for s in x]",
            "output": [s.upper() for s in words],
        }


class SortByLength(TransformTemplate):
    """Sort a list of strings by length (shortest first)."""
    name = "sort_by_length"
    difficulty = "medium"

    def generate_case(self) -> dict:
        words = ["".join(random.choices(string.ascii_lowercase, k=random.randint(1, 10)))
                 for _ in range(random.randint(4, 7))]
        return {
            "input": words,
            "gold_code": "sorted(x, key=len)",
            "output": sorted(words, key=len),
        }


LIST_TEMPLATES = [
    SortList(),
    FilterPositive(),
    DeduplicatePreserveOrder(),
    UppercaseStrings(),
    SortByLength(),
]
```

- [ ] **Step 4: Implement dict_ops.py with 3 templates**

```python
# grpo_pbe/templates/dict_ops.py
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
    """Format dict as 'key: value' strings joined by newlines."""
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
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_templates.py -k "list or dict" -v
```

Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add grpo_pbe/templates/list_ops.py grpo_pbe/templates/dict_ops.py tests/test_templates.py
git commit -m "feat: list and dict templates (8 templates, easy-hard)"
```

---

### Task 6: Hard Templates — Chained Multi-Step Operations

**Files:**
- Create: `grpo_pbe/templates/chained_ops.py`

These are the templates designed to benefit from longer reasoning traces — each requires 3+ chained operations.

- [ ] **Step 1: Add parametric test**

Add to `tests/test_templates.py`:

```python
def test_all_chained_templates_valid():
    from grpo_pbe.templates.chained_ops import CHAINED_TEMPLATES
    _validate_template_list(CHAINED_TEMPLATES)
```

- [ ] **Step 2: Implement chained_ops.py with 3 templates**

```python
# grpo_pbe/templates/chained_ops.py
import random
import string

from grpo_pbe.templates.base import TransformTemplate


class SplitFilterJoin(TransformTemplate):
    """Split CSV string, strip whitespace, remove empties, rejoin with pipe."""
    name = "split_filter_join"
    difficulty = "hard"

    def generate_case(self) -> dict:
        words = ["".join(random.choices(string.ascii_lowercase, k=random.randint(2, 6)))
                 for _ in range(random.randint(3, 5))]
        # Insert some blanks
        items = []
        for w in words:
            items.append(f" {w} " if random.random() < 0.3 else w)
            if random.random() < 0.3:
                items.append("")
        csv_str = ",".join(items)
        result = " | ".join(s.strip() for s in csv_str.split(",") if s.strip())
        return {
            "input": csv_str,
            "gold_code": "' | '.join(s.strip() for s in x.split(',') if s.strip())",
            "output": result,
        }


class ExtractSortDedupJoin(TransformTemplate):
    """Extract words from text, lowercase, sort, deduplicate, rejoin."""
    name = "extract_sort_dedup_join"
    difficulty = "hard"

    def generate_case(self) -> dict:
        pool = ["".join(random.choices(string.ascii_lowercase, k=random.randint(3, 6)))
                for _ in range(4)]
        words = [random.choice(pool).upper() if random.random() < 0.3 else random.choice(pool)
                 for _ in range(random.randint(5, 9))]
        text = " ".join(words)
        result = " ".join(sorted(set(w.lower() for w in text.split())))
        return {
            "input": text,
            "gold_code": "' '.join(sorted(set(w.lower() for w in x.split())))",
            "output": result,
        }


class FilterMapReduce(TransformTemplate):
    """From a list of ints, filter evens, square them, sum the result."""
    name = "filter_map_reduce"
    difficulty = "hard"

    def generate_case(self) -> dict:
        nums = [random.randint(1, 20) for _ in range(random.randint(5, 10))]
        result = sum(n ** 2 for n in nums if n % 2 == 0)
        return {
            "input": nums,
            "gold_code": "sum(n ** 2 for n in x if n % 2 == 0)",
            "output": result,
        }


CHAINED_TEMPLATES = [
    SplitFilterJoin(),
    ExtractSortDedupJoin(),
    FilterMapReduce(),
]
```

- [ ] **Step 3: Run all template tests**

```bash
pytest tests/test_templates.py -v
```

Expected: all pass. Confirm ALL_TEMPLATES has 25 templates:

```bash
python -c "from grpo_pbe.templates import ALL_TEMPLATES; print(f'{len(ALL_TEMPLATES)} templates loaded')"
```

Expected: `25 templates loaded`

- [ ] **Step 4: Commit**

```bash
git add grpo_pbe/templates/chained_ops.py tests/test_templates.py
git commit -m "feat: chained multi-step templates (3 hard templates, 25 total)"
```

---

### Task 7: Data Generator + Prompt Formatter

**Files:**
- Create: `grpo_pbe/data_generator.py`
- Create: `grpo_pbe/prompt.py`

- [ ] **Step 1: Write test for prompt formatter**

```python
# tests/test_prompt.py
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
```

- [ ] **Step 2: Implement prompt.py**

```python
# grpo_pbe/prompt.py


def format_prompt(demos: list[dict]) -> str:
    """Format demonstration I/O pairs into a model prompt.

    Args:
        demos: list of {"input": ..., "output": ...} dicts

    Returns:
        Formatted prompt string
    """
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
```

- [ ] **Step 3: Implement data_generator.py**

```python
# grpo_pbe/data_generator.py
import json
import random
from pathlib import Path

from grpo_pbe.templates import ALL_TEMPLATES
from grpo_pbe.prompt import format_prompt


def generate_dataset(
    n_per_template: int = 60,
    n_demo: int = 3,
    n_test: int = 2,
    seed: int = 42,
) -> list[dict]:
    """Generate the full synthetic dataset.

    Args:
        n_per_template: number of examples per template
        n_demo: number of demonstration I/O pairs per example
        n_test: number of held-out test I/O pairs per example
        seed: random seed for reproducibility

    Returns:
        list of example dicts, each with keys:
            template_name, difficulty, gold_code, prompt, demos, tests
    """
    random.seed(seed)
    dataset = []

    for template in ALL_TEMPLATES:
        for _ in range(n_per_template):
            example = template.generate_example(n_demo=n_demo, n_test=n_test)
            example["prompt"] = format_prompt(example["demos"])
            dataset.append(example)

    random.shuffle(dataset)
    return dataset


def save_dataset(dataset: list[dict], path: str | Path) -> None:
    """Save dataset as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for example in dataset:
            f.write(json.dumps(example) + "\n")


def load_dataset(path: str | Path) -> list[dict]:
    """Load dataset from JSONL."""
    with open(path) as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    dataset = generate_dataset(n_per_template=60, seed=42)
    # Split: last 200 for eval, rest for training
    eval_set = dataset[-200:]
    train_set = dataset[:-200]

    save_dataset(train_set, "data/train.jsonl")
    save_dataset(eval_set, "data/eval.jsonl")

    # Print stats
    from collections import Counter
    diff_counts = Counter(ex["difficulty"] for ex in train_set)
    print(f"Train: {len(train_set)} examples")
    print(f"Eval:  {len(eval_set)} examples")
    print(f"Difficulty distribution (train): {dict(diff_counts)}")
```

- [ ] **Step 4: Run prompt test**

```bash
pytest tests/test_prompt.py -v
```

Expected: PASS

- [ ] **Step 5: Generate the dataset and verify**

```bash
cd /home/arjun/GRPO_PBE
python -m grpo_pbe.data_generator
```

Expected output:
```
Train: ~1300 examples
Eval:  200 examples
Difficulty distribution (train): {'easy': ~520, 'medium': ~520, 'hard': ~260}
```

Spot-check 3 examples:
```bash
head -3 data/train.jsonl | python -m json.tool
```

Verify each has `prompt`, `demos`, `tests`, `gold_code`, `difficulty` keys.

- [ ] **Step 6: Commit**

```bash
git add grpo_pbe/prompt.py grpo_pbe/data_generator.py tests/test_prompt.py
git commit -m "feat: data generator + prompt formatter, generate train/eval splits"
```

---

### Task 8: Execution Sandbox

**Files:**
- Create: `grpo_pbe/sandbox.py`
- Create: `tests/test_sandbox.py`

- [ ] **Step 1: Write tests for sandbox**

```python
# tests/test_sandbox.py
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
    result = execute_code("next(i for i in iter(int, 1))", "x")  # infinite iterator
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_sandbox.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement sandbox.py**

```python
# grpo_pbe/sandbox.py
import signal
import re
import datetime
from dataclasses import dataclass
from typing import Any


@dataclass
class ExecutionResult:
    success: bool
    output: Any = None
    error: str | None = None


def _timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out")


def execute_code(code: str, input_val: Any, timeout_seconds: int = 2) -> ExecutionResult:
    """Execute a Python expression in a restricted sandbox.

    The expression can use `x` to refer to the input value.
    Only builtins, `re`, and `datetime` are available.

    Args:
        code: Python expression string (e.g., "x.upper()")
        input_val: the value bound to `x`
        timeout_seconds: max execution time

    Returns:
        ExecutionResult with success flag and output or error
    """
    # Restricted globals: only re, datetime, and safe builtins
    safe_builtins = {
        k: v for k, v in __builtins__.items()
        if k not in ("__import__", "exec", "eval", "compile", "open",
                      "breakpoint", "exit", "quit", "input", "print")
    } if isinstance(__builtins__, dict) else {
        k: getattr(__builtins__, k) for k in dir(__builtins__)
        if k not in ("__import__", "exec", "eval", "compile", "open",
                      "breakpoint", "exit", "quit", "input", "print")
        and not k.startswith("_")
    }

    restricted_globals = {
        "__builtins__": safe_builtins,
        "re": re,
        "datetime": datetime,
    }

    # Set timeout (Unix only — on Windows, skip timeout)
    has_alarm = hasattr(signal, "SIGALRM")
    if has_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)

    try:
        fn = eval(f"lambda x: {code}", restricted_globals)
        result = fn(input_val)
        return ExecutionResult(success=True, output=result)
    except TimeoutError:
        return ExecutionResult(success=False, error="timeout")
    except Exception as e:
        return ExecutionResult(success=False, error=f"{type(e).__name__}: {e}")
    finally:
        if has_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_sandbox.py -v
```

Expected: all 9 tests pass

- [ ] **Step 5: Commit**

```bash
git add grpo_pbe/sandbox.py tests/test_sandbox.py
git commit -m "feat: restricted execution sandbox with timeout"
```

---

### Task 9: Reward Function

**Files:**
- Create: `grpo_pbe/reward.py`
- Create: `tests/test_reward.py`

- [ ] **Step 1: Write tests for reward function**

```python
# tests/test_reward.py
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
    assert abs(reward - 1.1) < 1e-6  # 0.1 format + 1.0 correctness


def test_reward_correct_code_wrong_format():
    tests = [{"input": "hello", "output": "HELLO"}]
    response = "x.upper()"  # no tags
    reward = compute_reward(response, tests)
    assert abs(reward - 0.0) < 1e-6  # 0 format, can't extract code


def test_reward_correct_format_wrong_code():
    tests = [
        {"input": "hello", "output": "HELLO"},
        {"input": "world", "output": "WORLD"},
    ]
    response = "<think>lowercase</think><code>x.lower()</code>"
    reward = compute_reward(response, tests)
    assert abs(reward - 0.1) < 1e-6  # 0.1 format + 0.0 correctness


def test_reward_partial_correctness():
    tests = [
        {"input": 3, "output": 9},
        {"input": -2, "output": 4},
    ]
    # x * 3 works for input=3 (9) but not input=-2 (should be 4, gets -6)
    response = "<think>multiply</think><code>x * 3</code>"
    reward = compute_reward(response, tests)
    assert abs(reward - 0.6) < 1e-6  # 0.1 format + 0.5 correctness (1/2 tests)


def test_reward_syntax_error_zero_correctness():
    tests = [{"input": "hello", "output": "HELLO"}]
    response = "<think>try</think><code>x.upper(</code>"
    reward = compute_reward(response, tests)
    assert abs(reward - 0.1) < 1e-6  # format OK, code fails
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_reward.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement reward.py**

```python
# grpo_pbe/reward.py
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_reward.py -v
```

Expected: all 7 tests pass

- [ ] **Step 5: HARD GATE — Manually verify reward on 20 real examples**

Run this verification script interactively:

```bash
python -c "
from grpo_pbe.data_generator import load_dataset
from grpo_pbe.reward import compute_reward

dataset = load_dataset('data/train.jsonl')
# Test reward on gold code (should get 1.1 for all)
for ex in dataset[:20]:
    fake_response = f'<think>solving</think><code>{ex[\"gold_code\"]}</code>'
    r = compute_reward(fake_response, ex['tests'])
    status = '✓' if abs(r - 1.1) < 1e-6 else '✗ FAIL'
    print(f'{status} {ex[\"template_name\"]}: reward={r:.2f}')
"
```

**Expected:** all 20 lines show `✓` with reward=1.10. If any show `✗ FAIL`, the reward function or template has a bug — fix before proceeding.

- [ ] **Step 6: Commit**

```bash
git add grpo_pbe/reward.py tests/test_reward.py
git commit -m "feat: reward function (format + execution correctness)"
```

---

### Task 10: GRPO Training Script

**Files:**
- Create: `grpo_pbe/train_grpo.py`

- [ ] **Step 1: Implement train_grpo.py**

```python
# grpo_pbe/train_grpo.py
"""GRPO training script using Unsloth + TRL."""
import json
import torch
import wandb
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

from grpo_pbe.data_generator import load_dataset
from grpo_pbe.reward import compute_reward


# --- Config ---
MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 1024
LORA_R = 16
LORA_ALPHA = 16
OUTPUT_DIR = "checkpoints/grpo-pbe-1.5b"
WANDB_PROJECT = "grpo-pbe"

GRPO_CONFIG = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    num_generations=4,  # group size G
    max_completion_length=512,
    learning_rate=5e-6,
    kl_coef=0.04,
    optim="adamw_8bit",
    logging_steps=10,
    save_steps=200,
    report_to="wandb",
    bf16=True,
    seed=42,
)


def build_reward_fn(train_data: list[dict]):
    """Build a reward function that looks up test cases by prompt."""
    prompt_to_tests = {ex["prompt"]: ex["tests"] for ex in train_data}

    def reward_fn(completions: list[str], prompts: list[str] | None = None, **kwargs) -> list[float]:
        rewards = []
        for completion, prompt in zip(completions, prompts or [""]*len(completions)):
            tests = prompt_to_tests.get(prompt, [])
            r = compute_reward(completion, tests)
            rewards.append(r)
        return rewards

    return reward_fn


def main():
    # Load data
    train_data = load_dataset("data/train.jsonl")
    print(f"Loaded {len(train_data)} training examples")

    # Build HF dataset with just the prompts
    hf_dataset = Dataset.from_dict({"prompt": [ex["prompt"] for ex in train_data]})

    # Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )

    # Init W&B
    wandb.init(project=WANDB_PROJECT, name="grpo-run", config=GRPO_CONFIG.to_dict())

    # Build reward function
    reward_fn = build_reward_fn(train_data)

    # Train
    trainer = GRPOTrainer(
        model=model,
        args=GRPO_CONFIG,
        train_dataset=hf_dataset,
        reward_funcs=reward_fn,
        tokenizer=tokenizer,
    )
    trainer.train()

    # Save
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")
    wandb.finish()


if __name__ == "__main__":
    main()
```

**Note:** The exact `GRPOTrainer` API may differ between TRL versions. Check the Unsloth GRPO guide and TRL docs before running. The `reward_funcs` parameter name and signature may need adjustment — consult `https://unsloth.ai/blog/r1-reasoning` for the current API.

- [ ] **Step 2: Smoke test — verify model loads and first step runs**

```bash
cd /home/arjun/GRPO_PBE
python -c "
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    'unsloth/Qwen2.5-1.5B-Instruct', max_seq_length=1024, load_in_4bit=True,
)
print(f'Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB')
"
```

Expected: model loads, VRAM usage reported (~2–3 GB before training)

- [ ] **Step 3: Launch training run**

```bash
cd /home/arjun/GRPO_PBE
python -m grpo_pbe.train_grpo
```

**Monitor at step 50:**
- Check W&B dashboard for mean reward > 0
- Check KL divergence < 10
- If reward is flat at 0 after 50 steps: STOP, debug the reward function integration

Let training run to completion (~3–5h). Continue to Task 11 while waiting.

- [ ] **Step 4: Commit**

```bash
git add grpo_pbe/train_grpo.py
git commit -m "feat: GRPO training script with Unsloth + TRL"
```

---

### Task 11: SFT Baseline Training Script

**Files:**
- Create: `grpo_pbe/train_sft.py`

- [ ] **Step 1: Implement train_sft.py**

```python
# grpo_pbe/train_sft.py
"""SFT baseline training script using Unsloth + TRL."""
import wandb
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

from grpo_pbe.data_generator import load_dataset


# --- Config ---
MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 1024
LORA_R = 16
LORA_ALPHA = 16
OUTPUT_DIR = "checkpoints/sft-pbe-1.5b"
WANDB_PROJECT = "grpo-pbe"

SFT_CONFIG = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    optim="adamw_8bit",
    logging_steps=10,
    save_steps=200,
    report_to="wandb",
    bf16=True,
    seed=42,
    max_seq_length=MAX_SEQ_LENGTH,
)


def format_sft_example(example: dict) -> str:
    """Format a training example as prompt + gold response for SFT."""
    return f"{example['prompt']}\n\n<code>{example['gold_code']}</code>"


def main():
    train_data = load_dataset("data/train.jsonl")
    print(f"Loaded {len(train_data)} training examples")

    # Format as text for SFT
    texts = [format_sft_example(ex) for ex in train_data]
    hf_dataset = Dataset.from_dict({"text": texts})

    # Load model with Unsloth (same config as GRPO for fair comparison)
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )

    wandb.init(project=WANDB_PROJECT, name="sft-baseline", config=SFT_CONFIG.to_dict())

    trainer = SFTTrainer(
        model=model,
        args=SFT_CONFIG,
        train_dataset=hf_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")
    wandb.finish()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Launch SFT training (after GRPO finishes or in parallel if VRAM allows)**

```bash
python -m grpo_pbe.train_sft
```

Expected: ~1–2h wall-clock. Loss should decrease steadily.

- [ ] **Step 3: Commit**

```bash
git add grpo_pbe/train_sft.py
git commit -m "feat: SFT baseline training script"
```

---

### Task 12: Evaluation Harness

**Files:**
- Create: `grpo_pbe/evaluate.py`
- Create: `tests/test_evaluate.py`

- [ ] **Step 1: Write test for metric computation**

```python
# tests/test_evaluate.py
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
```

- [ ] **Step 2: Implement evaluate.py**

```python
# grpo_pbe/evaluate.py
"""Evaluation harness: run inference on held-out set, compute metrics."""
import json
from pathlib import Path
from collections import defaultdict

import torch
from unsloth import FastLanguageModel

from grpo_pbe.data_generator import load_dataset
from grpo_pbe.prompt import format_prompt
from grpo_pbe.reward import compute_reward, parse_response


def compute_metrics(results: list[dict]) -> dict:
    """Compute accuracy metrics from evaluation results.

    Args:
        results: list of dicts with "correct" (bool) and "difficulty" keys

    Returns:
        dict with overall_accuracy and per-difficulty accuracies
    """
    if not results:
        return {}

    by_difficulty = defaultdict(list)
    for r in results:
        by_difficulty[r["difficulty"]].append(r["correct"])

    metrics = {
        "overall_accuracy": sum(r["correct"] for r in results) / len(results),
    }
    for diff, vals in by_difficulty.items():
        metrics[f"{diff}_accuracy"] = sum(vals) / len(vals)

    return metrics


def run_inference(model, tokenizer, eval_data: list[dict], max_new_tokens: int = 512) -> list[dict]:
    """Run model inference on eval set and compute per-example results.

    Args:
        model: loaded model
        tokenizer: loaded tokenizer
        eval_data: list of example dicts from load_dataset
        max_new_tokens: max generation length

    Returns:
        list of result dicts with keys: template_name, difficulty, correct,
            reward, response, think_length, code
    """
    results = []
    model.eval()

    for i, example in enumerate(eval_data):
        prompt = example["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        reward = compute_reward(response, example["tests"])
        parsed = parse_response(response)

        results.append({
            "template_name": example["template_name"],
            "difficulty": example["difficulty"],
            "correct": reward >= 1.0,  # correctness component is 1.0 when all tests pass
            "reward": reward,
            "response": response,
            "think_length": len(parsed.think) if parsed.think else 0,
            "code": parsed.code,
            "gold_code": example["gold_code"],
        })

        if (i + 1) % 20 == 0:
            acc = sum(r["correct"] for r in results) / len(results)
            print(f"  [{i+1}/{len(eval_data)}] running accuracy: {acc:.1%}")

    return results


def evaluate_model(checkpoint_path: str, eval_path: str = "data/eval.jsonl", label: str = "model") -> dict:
    """Full evaluation pipeline for one model checkpoint."""
    print(f"\n=== Evaluating: {label} ({checkpoint_path}) ===")
    eval_data = load_dataset(eval_path)

    model, tokenizer = FastLanguageModel.from_pretrained(
        checkpoint_path,
        max_seq_length=1024,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    results = run_inference(model, tokenizer, eval_data)
    metrics = compute_metrics(results)

    # Add reasoning length stats (GRPO-specific)
    think_lengths = [r["think_length"] for r in results if r["think_length"] > 0]
    if think_lengths:
        metrics["mean_think_length"] = sum(think_lengths) / len(think_lengths)

    print(f"\nResults for {label}:")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.3f}")

    # Save results
    out_dir = Path("data")
    with open(out_dir / f"eval_results_{label}.json", "w") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)

    return metrics


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m grpo_pbe.evaluate <checkpoint_path> [label]")
        print("Example: python -m grpo_pbe.evaluate checkpoints/grpo-pbe-1.5b grpo")
        sys.exit(1)

    checkpoint = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else "model"
    evaluate_model(checkpoint, label=label)
```

- [ ] **Step 3: Run metric tests**

```bash
pytest tests/test_evaluate.py -v
```

Expected: PASS

- [ ] **Step 4: Evaluate all three models**

Run after both training runs complete:

```bash
# Base model (zero-shot)
python -m grpo_pbe.evaluate unsloth/Qwen2.5-1.5B-Instruct base

# SFT baseline
python -m grpo_pbe.evaluate checkpoints/sft-pbe-1.5b sft

# GRPO
python -m grpo_pbe.evaluate checkpoints/grpo-pbe-1.5b grpo
```

Each writes results to `data/eval_results_{label}.json`.

- [ ] **Step 5: Commit**

```bash
git add grpo_pbe/evaluate.py tests/test_evaluate.py
git commit -m "feat: evaluation harness with per-difficulty metrics"
```

---

### Task 13: Analysis + Results Table

**Files:**
- Create: `notebooks/analysis.ipynb`

- [ ] **Step 1: Create analysis notebook**

Create a Jupyter notebook at `notebooks/analysis.ipynb` with these cells:

**Cell 1: Load results**
```python
import json
from pathlib import Path

results = {}
for label in ["base", "sft", "grpo"]:
    path = Path(f"../data/eval_results_{label}.json")
    if path.exists():
        with open(path) as f:
            results[label] = json.load(f)
        print(f"Loaded {label}: {len(results[label]['results'])} examples")
```

**Cell 2: Results table**
```python
import pandas as pd

rows = []
for label, data in results.items():
    m = data["metrics"]
    rows.append({
        "Model": label,
        "Easy": f"{m.get('easy_accuracy', 0):.1%}",
        "Medium": f"{m.get('medium_accuracy', 0):.1%}",
        "Hard": f"{m.get('hard_accuracy', 0):.1%}",
        "Overall": f"{m.get('overall_accuracy', 0):.1%}",
        "Mean Think Length": f"{m.get('mean_think_length', 0):.0f}" if "mean_think_length" in m else "—",
    })

df = pd.DataFrame(rows)
print(df.to_markdown(index=False))
```

**Cell 3: Training curves from W&B (manual — paste screenshots or use wandb API)**
```python
# If using wandb API:
# import wandb
# api = wandb.Api()
# run = api.run("your-username/grpo-pbe/run-id")
# history = run.history()
# Plot: history["reward"], history["kl"], etc.

# Otherwise: export charts from W&B dashboard and embed as images
```

**Cell 4: Cherry-pick 5 interesting examples**
```python
grpo_results = results.get("grpo", {}).get("results", [])

# Find hard examples where GRPO succeeded
hard_correct = [r for r in grpo_results if r["difficulty"] == "hard" and r["correct"]]
hard_wrong = [r for r in grpo_results if r["difficulty"] == "hard" and not r["correct"]]

print("=== HARD examples GRPO got right ===")
for r in hard_correct[:3]:
    print(f"\nTemplate: {r['template_name']}")
    print(f"Think: {r['response'][:200]}...")
    print(f"Code: {r['code']}")
    print(f"Gold: {r['gold_code']}")

print("\n=== HARD examples GRPO got wrong ===")
for r in hard_wrong[:2]:
    print(f"\nTemplate: {r['template_name']}")
    print(f"Response: {r['response'][:200]}...")
```

- [ ] **Step 2: Run the notebook, save outputs**

```bash
cd /home/arjun/GRPO_PBE
jupyter nbconvert --execute notebooks/analysis.ipynb --to notebook --inplace
```

- [ ] **Step 3: Copy the results table into reports/RESULTS.md**

Create `reports/RESULTS.md` with the markdown table output from Cell 2, plus any notable observations.

- [ ] **Step 4: Commit**

```bash
git add notebooks/analysis.ipynb reports/RESULTS.md
git commit -m "feat: analysis notebook + results table"
```

---

### Task 14: README + Blog Post Draft + HF Upload

**Files:**
- Create: `README.md`
- Create: `reports/blog_post_draft.md`

- [ ] **Step 1: Write README.md**

```markdown
# GRPO-PBE: Teaching a 1.5B Model to Program by Example with RL

Train Qwen2.5-1.5B to solve data transformation tasks using GRPO
(Group Relative Policy Optimization) with execution-based rewards.

## Results

[Paste the results table from reports/RESULTS.md]

## Quick Start

```bash
pip install -e .
python -m grpo_pbe.data_generator   # generate synthetic dataset
python -m grpo_pbe.train_grpo       # train GRPO model (~3-5h on RTX 4060)
python -m grpo_pbe.train_sft        # train SFT baseline (~1-2h)
python -m grpo_pbe.evaluate checkpoints/grpo-pbe-1.5b grpo
python -m grpo_pbe.evaluate checkpoints/sft-pbe-1.5b sft
```

## Models

- GRPO: [HuggingFace link]
- SFT baseline: [HuggingFace link]
- W&B training dashboard: [link]

## Blog Post

[Link to Medium/HF blog post]

## How It Works

[Brief description of task, reward function, training setup]
```

- [ ] **Step 2: Draft blog post outline in reports/blog_post_draft.md**

Write the skeleton with section headers and key points. Fill in after results are final.

- [ ] **Step 3: Upload models to HF Hub**

```bash
# After training completes:
huggingface-cli login
python -c "
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained('checkpoints/grpo-pbe-1.5b')
model.push_to_hub('YOUR_USERNAME/grpo-pbe-1.5b')
tokenizer.push_to_hub('YOUR_USERNAME/grpo-pbe-1.5b')
"
# Repeat for sft-pbe-1.5b
```

- [ ] **Step 4: Final commit**

```bash
git add README.md reports/blog_post_draft.md
git commit -m "docs: README, blog post draft, project complete"
```

---

## Self-Review Checklist

- **Spec coverage:**
  - ✅ 25 templates across 10 categories with difficulty tiers
  - ✅ Reward function: format (0.1) + correctness (1.0)
  - ✅ Sandbox with restricted exec, timeout, re + datetime
  - ✅ GRPO training with Unsloth + GRPOTrainer
  - ✅ SFT baseline with same LoRA config
  - ✅ Evaluation: overall + per-difficulty accuracy + reasoning length
  - ✅ Deliverables: repo, 2 HF models, W&B, blog post, README
  - ✅ Hard gates: Day 2 data validation, Day 3 reward verification, Day 4 step-50 check
- **Placeholder scan:** No TBDs. One note in Task 10 about checking the GRPOTrainer API — this is an intentional callout, not a placeholder.
- **Type consistency:** `TransformTemplate.generate_case()` returns `dict` with `input`, `gold_code`, `output` — used consistently in sandbox, reward, data generator. `compute_reward` returns `float` — used in both training reward_fn and evaluation. `parse_response` returns `ParsedResponse` — used in reward.py and evaluate.py.
