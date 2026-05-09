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
