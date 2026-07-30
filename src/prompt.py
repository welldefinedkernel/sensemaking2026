"""Prompt builders selected by the run configuration."""

from collections.abc import Callable
from typing import Any

PromptBuilder = Callable[[dict[str, Any], str], str]


def simple_prompt_zero_shot(inp: dict[str, Any], mask_token: str) -> str:
    return "\n".join(
        [
            f"Context: {inp.get('context', '')}",
            f"Question: {inp.get('question', '')}",
            f"Answer: {inp.get('answer', '')}",
            f"The chosen label is: {mask_token}",
        ]
    )


def rubric_prompt_zero_shot(inp: dict[str, Any], mask_token: str) -> str:
    rubrics = inp.get("rubrics", {})
    return "\n".join(
        [
            f"Context: {inp.get('context', '')}",
            f"Question: {inp.get('question', '')}",
            "Rubric:\n"
            f"0 (No Credit): {rubrics.get('NC', '')}\n"
            f"1 (Partial Credit): {rubrics.get('PC', '')}\n"
            f"2 (Full Credit): {rubrics.get('FC', '')}",
            f"Answer: {inp.get('answer', '')}",
            f"The chosen label is: {mask_token}",
        ]
    )


PROMPTS: dict[str, PromptBuilder] = {
    "simple/zero_shot": simple_prompt_zero_shot,
    "rubric/zero_shot": rubric_prompt_zero_shot,
}


def build_prompt(name: str, inp: dict[str, Any], mask_token: str) -> str:
    """Build a configured prompt."""
    try:
        builder = PROMPTS[name]
    except KeyError:
        raise ValueError(
            f"prompt={name!r} is not implemented; available: {sorted(PROMPTS)}"
        ) from None
    return builder(inp, mask_token)
