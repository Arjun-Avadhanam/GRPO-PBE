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
