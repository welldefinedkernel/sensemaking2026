# Sensemaking 2026

Answer-grading systems for the Sensemaking shared task, built on monolingual HPLT encoders.

Two tracks:

| track | input | labels |
|-------|-------|--------|
| `simple` | context, question, answer | 0–4 |
| `rubric` | context, question, answer, rubric (NC/PC/FC) | 0–2 |

Scored with quadratic weighted kappa, averaged over domains.

## Idea

Everything reduces to *reading a digit at one position*. A prompt is built from the data
point, ending in `The chosen label is: <mask>`, and the model has to fill that slot with a
label digit. This lets pretrained heads be reused directly — no randomly initialised layers
in the zero-shot setting.

Three ways to obtain that digit:

- `mlm` — masked-LM cloze: take the logits at the mask position, restrict them to the digit
  tokens, argmax. No training.
- `cls` — a sequence-classification head over the `[CLS]` representation. Requires
  finetuning (the head starts random).
- `instruction_tune` — causal generation with GPT-BERT; the gold digit is appended after the
  prompt and it is the only supervised token.

Prompts come in two flavours (`zero_shot`, `examples` — the latter prepends worked
examples) and are machine-translated into the target language, component by component, with
the translations cached per language.

## Models

- HPLT 2.0 BERT — `HPLT/hplt_bert_base_2_0_<lng-Script>` (512 positions)
- HPLT 3.0 GPT-BERT — `HPLT/hplt_gpt_bert_base_3_0_<lng_Script>` (long context)

The 2.0 models are only used on the `simple` track: the rubric prompt alone overruns their
512-token window before any context is added.

## How runs were made

One TOML config = one run. `src/run.py` loads the config, groups the eval set by language,
loads the matching checkpoint per language, optionally finetunes on the training split, and
writes predictions.

```
configs/<track>/<family>.<method>.<prompt>.<dataset>.toml
```

with `family ∈ {hplt2, hplt3}`, `method ∈ {mlm, cls, instruction_tune}`,
`prompt ∈ {zero_shot, examples}` and `dataset ∈ {regular, balanced}` — 28 configs in total,
which is the full grid that was run. `regular` is the released train/dev split; `balanced`
is a re-split that fixes the near-zero domain overlap of the released one.

Run:

```sh
uv run python src/run.py --config configs/simple/hplt3.cls.examples.balanced.toml
```

## Output

Predictions land in `results/<track>/<model>/<prompt>/<dataset>.json` as the submission
format, with the checkpoint name and the per-digit probabilities kept in `misc`:

```json
{"id": "dev:28238", "label": 4,
 "misc": {"model": "HPLT/hplt_gpt_bert_base_3_0_gle_Latn", "probs": [0.045, 0.111, 0.234, 0.090, 0.520]}}
```

Scoring uses the organisers' script:

```sh
uv run python data/scripts/evaluate.py --ref data/devset/dev.simple.json \
    --pred results/simple/3.0_gpt_bert_cls/examples/regular.json
```

(add `--is_rubric` for the rubric track).

## Setup

```sh
uv sync
```
