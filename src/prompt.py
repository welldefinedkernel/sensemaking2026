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


# Task description shown to the model: what the score means and how to pick it
SIMPLE_INSTRUCTION = (
    "You grade how well an Answer responds to the Question given the Context.\n"
    "Choose exactly one label on a 0-4 scale:\n"
    "0: incorrect or unrelated; contradicts the Context.\n"
    "1: mostly wrong; only a minor relevant element.\n"
    "2: partially correct; misses key information or mixes in errors.\n"
    "3: largely correct; supported by the Context with minor omissions.\n"
    "4: fully correct and complete; well supported by the Context."
)

# One worked demonstration per label
SIMPLE_LABEL_EXAMPLES: list[str] = [
    "\n".join(
        [
            "Context: Water boils at 100 degrees Celsius at sea level.",
            "Question: At what temperature does water boil at sea level?",
            "Answer: Water never boils; it only freezes.",
            "The chosen label is: 0",
        ]
    ),
    "\n".join(
        [
            "Context: Water boils at 100 degrees Celsius at sea level.",
            "Question: At what temperature does water boil at sea level?",
            "Answer: It has something to do with heat.",
            "The chosen label is: 1",
        ]
    ),
    "\n".join(
        [
            "Context: Water boils at 100 degrees Celsius at sea level.",
            "Question: At what temperature does water boil at sea level?",
            "Answer: Around 90 degrees Celsius.",
            "The chosen label is: 2",
        ]
    ),
    "\n".join(
        [
            "Context: Water boils at 100 degrees Celsius at sea level.",
            "Question: At what temperature does water boil at sea level?",
            "Answer: About 100 degrees.",
            "The chosen label is: 3",
        ]
    ),
    "\n".join(
        [
            "Context: Water boils at 100 degrees Celsius at sea level.",
            "Question: At what temperature does water boil at sea level?",
            "Answer: 100 degrees Celsius at sea level.",
            "The chosen label is: 4",
        ]
    ),
]


def simple_prompt_examples(inp: dict[str, Any], mask_token: str) -> str:
    """Few-shot prompt: task instruction plus one example per label (0-4)."""
    query = "\n".join(
        [
            f"Context: {inp.get('context', '')}",
            f"Question: {inp.get('question', '')}",
            f"Answer: {inp.get('answer', '')}",
            f"The chosen label is: {mask_token}",
        ]
    )
    blocks = [SIMPLE_INSTRUCTION, *SIMPLE_LABEL_EXAMPLES, query]
    return "\n\n".join(blocks)


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
    "simple/examples": simple_prompt_examples,
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
