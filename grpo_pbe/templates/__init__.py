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
