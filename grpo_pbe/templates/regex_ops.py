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
