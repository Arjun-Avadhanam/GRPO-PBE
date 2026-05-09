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
