"""Prompt builders selected by the run configuration."""

import time
from collections.abc import Callable
from functools import lru_cache
from typing import Any

PromptBuilder = Callable[[dict[str, Any], str, str], str]

# Field labels, translated separately from the values they introduce
CONTEXT_LABEL = "Context:"
QUESTION_LABEL = "Question:"
ANSWER_LABEL = "Answer:"
RUBRIC_LABEL = "Rubric:"
CHOSEN_LABEL = "The chosen label is:"


@lru_cache(maxsize=None)
def translate(text: str, lang: str) -> str:
    """Translate a single prompt component via Google Translate.

    Cached per (text, lang): labels, instructions and examples are fixed
    strings reused on every query, so translation only needs to happen once
    per language rather than once per query.
    """
    if lang == "en" or not text.strip():
        return text
    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source="en", target=lang)
    for attempt in range(3):
        try:
            translated = translator.translate(text)
        except Exception as exc:  # transient API failures / empty responses
            if attempt == 2:
                print(f"[warn] translation to {lang!r} failed ({exc}); using English.")
                return text
            time.sleep(1 + attempt)
            continue
        if translated and translated.strip():
            return translated
    print(f"[warn] empty translation to {lang!r}; using English.")
    return text


def translate_lines(text: str, lang: str) -> str:
    """Translate a multi-line block one line at a time (translators cap input length)."""
    return "\n".join(translate(line, lang) for line in text.split("\n"))


def _field(label: str, value: str, lang: str) -> str:
    """Render a `Label: value` line with only the label translated."""
    return f"{translate(label, lang)} {value}"


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


def _simple_block(inp: dict[str, Any], mask_token: str, lang: str) -> str:
    """The fields of a single graded item, without the task instruction."""
    return "\n".join(
        [
            _field(CONTEXT_LABEL, inp.get("context", ""), lang),
            _field(QUESTION_LABEL, inp.get("question", ""), lang),
            _field(ANSWER_LABEL, inp.get("answer", ""), lang),
            _field(CHOSEN_LABEL, mask_token, lang),
        ]
    )


def simple_prompt_zero_shot(inp: dict[str, Any], mask_token: str, lang: str) -> str:
    instruction = translate_lines(SIMPLE_INSTRUCTION, lang)
    return f"{instruction}\n\n{_simple_block(inp, mask_token, lang)}"


# One worked demonstration per label, authored in English
SIMPLE_LABEL_EXAMPLES: list[dict[str, str]] = [
    {
        "context": "Water boils at 100 degrees Celsius at sea level.",
        "question": "At what temperature does water boil at sea level?",
        "answer": "Water never boils; it only freezes.",
        "label": "0",
    },
    {
        "context": "Water boils at 100 degrees Celsius at sea level.",
        "question": "At what temperature does water boil at sea level?",
        "answer": "It has something to do with heat.",
        "label": "1",
    },
    {
        "context": "Water boils at 100 degrees Celsius at sea level.",
        "question": "At what temperature does water boil at sea level?",
        "answer": "Around 90 degrees Celsius.",
        "label": "2",
    },
    {
        "context": "Water boils at 100 degrees Celsius at sea level.",
        "question": "At what temperature does water boil at sea level?",
        "answer": "About 100 degrees.",
        "label": "3",
    },
    {
        "context": "Water boils at 100 degrees Celsius at sea level.",
        "question": "At what temperature does water boil at sea level?",
        "answer": "100 degrees Celsius at sea level.",
        "label": "4",
    },
]


@lru_cache(maxsize=None)
def _simple_preamble(lang: str) -> str:
    """Instruction + one worked example per label, translated once per language."""
    examples = [
        _simple_block(
            {
                "context": translate(ex["context"], lang),
                "question": translate(ex["question"], lang),
                "answer": translate(ex["answer"], lang),
            },
            ex["label"],
            lang,
        )
        for ex in SIMPLE_LABEL_EXAMPLES
    ]
    return "\n\n".join([translate_lines(SIMPLE_INSTRUCTION, lang), *examples])


def simple_prompt_examples(inp: dict[str, Any], mask_token: str, lang: str) -> str:
    """Few-shot prompt: cached instruction + examples, plus the current query."""
    query = _simple_block(inp, mask_token, lang)
    return f"{_simple_preamble(lang)}\n\n{query}"


RUBRIC_LEVELS = [
    ("NC", "0 (No Credit):"),
    ("PC", "1 (Partial Credit):"),
    ("FC", "2 (Full Credit):"),
]

# Task description shown to the model: what the score means and how to pick it
RUBRIC_INSTRUCTION = (
    "You grade how well an Answer responds to the Question given the Context,\n"
    "following the Rubric written for this Question.\n"
    "Choose exactly one label on a 0-2 scale:\n"
    "0: the Answer meets the No Credit criterion.\n"
    "1: the Answer meets the Partial Credit criterion.\n"
    "2: the Answer meets the Full Credit criterion."
)


def _rubric_block(inp: dict[str, Any], mask_token: str, lang: str) -> str:
    """The fields of a single graded item, without the task instruction."""
    rubrics = inp.get("rubrics", {})
    rubric_block = "\n".join(
        [translate(RUBRIC_LABEL, lang)]
        + [_field(level, rubrics.get(key, ""), lang) for key, level in RUBRIC_LEVELS]
    )
    return "\n".join(
        [
            _field(CONTEXT_LABEL, inp.get("context", ""), lang),
            _field(QUESTION_LABEL, inp.get("question", ""), lang),
            rubric_block,
            _field(ANSWER_LABEL, inp.get("answer", ""), lang),
            _field(CHOSEN_LABEL, mask_token, lang),
        ]
    )


def rubric_prompt_zero_shot(inp: dict[str, Any], mask_token: str, lang: str) -> str:
    instruction = translate_lines(RUBRIC_INSTRUCTION, lang)
    return f"{instruction}\n\n{_rubric_block(inp, mask_token, lang)}"


# One worked demonstration per label, authored in English
RUBRIC_EXAMPLE_RUBRICS: dict[str, str] = {
    "NC": "The answer gives no temperature, or one that contradicts the Context.",
    "PC": "The answer refers to boiling or heat but not to 100 degrees Celsius.",
    "FC": "The answer states 100 degrees Celsius.",
}
RUBRIC_LABEL_EXAMPLES: list[dict[str, str]] = [
    {
        "context": "Water boils at 100 degrees Celsius at sea level.",
        "question": "At what temperature does water boil at sea level?",
        "answer": "Water never boils; it only freezes.",
        "label": "0",
    },
    {
        "context": "Water boils at 100 degrees Celsius at sea level.",
        "question": "At what temperature does water boil at sea level?",
        "answer": "It boils once it gets hot enough.",
        "label": "1",
    },
    {
        "context": "Water boils at 100 degrees Celsius at sea level.",
        "question": "At what temperature does water boil at sea level?",
        "answer": "100 degrees Celsius at sea level.",
        "label": "2",
    },
]


@lru_cache(maxsize=None)
def _rubric_preamble(lang: str) -> str:
    """Instruction + one worked example per label, translated once per language."""
    rubrics = {k: translate(v, lang) for k, v in RUBRIC_EXAMPLE_RUBRICS.items()}
    examples = [
        _rubric_block(
            {
                "context": translate(ex["context"], lang),
                "question": translate(ex["question"], lang),
                "answer": translate(ex["answer"], lang),
                "rubrics": rubrics,
            },
            ex["label"],
            lang,
        )
        for ex in RUBRIC_LABEL_EXAMPLES
    ]
    return "\n\n".join([translate_lines(RUBRIC_INSTRUCTION, lang), *examples])


def rubric_prompt_examples(inp: dict[str, Any], mask_token: str, lang: str) -> str:
    """Few-shot prompt: cached instruction + examples, plus the current query."""
    query = _rubric_block(inp, mask_token, lang)
    return f"{_rubric_preamble(lang)}\n\n{query}"


PROMPTS: dict[str, PromptBuilder] = {
    "simple/zero_shot": simple_prompt_zero_shot,
    "simple/examples": simple_prompt_examples,
    "rubric/zero_shot": rubric_prompt_zero_shot,
    "rubric/examples": rubric_prompt_examples,
}


def build_prompt(
    name: str, inp: dict[str, Any], mask_token: str, lang: str = "en"
) -> str:
    """Build a configured prompt with each component translated into `lang`."""
    try:
        builder = PROMPTS[name]
    except KeyError:
        raise ValueError(
            f"prompt={name!r} is not implemented; available: {sorted(PROMPTS)}"
        ) from None
    return builder(inp, mask_token, lang)
