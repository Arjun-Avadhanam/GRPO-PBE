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
