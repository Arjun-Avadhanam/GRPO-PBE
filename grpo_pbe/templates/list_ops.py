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
