# Architecture — Sensemaking @ CLEF 2026

Automatic evaluation of answers to open questions using **HPLT monolingual models**
(one model per language). This document describes the intended code architecture, not
a finished implementation.

## 1. Problem framing

Two tracks, both cast as **ordinal classification**:

| Track  | Inputs                                   | Labels                          | Metric |
|--------|------------------------------------------|---------------------------------|--------|
| Simple | context, question, answer                | 0–4 (5 classes)                 | QWK    |
| Rubric | context, question, rubric{NC,PC,FC}, answer | 0=NC, 1=PC, 2=FC (3 classes) | QWK    |

- **Metric**: Quadratic Weighted Kappa, computed per domain and averaged over domains.
  (`data/scripts/evaluate.py`.)
- **Per-language models**: route each example by its `lang` tag to the matching HPLT
  checkpoint. Ordinal structure of labels should be exploited (weighted/ordinal loss).

## 2. Model families

Three HPLT families are explored in parallel (see `/memories/repo/hplt-models.md` for
the full language→checkpoint map):

1. **HPLT 2.0 BERT** — encoder-only (LTG-BERT), `HPLT/hplt_bert_base_2_0_<lang-Script>`
   (`trust_remote_code=True`).
2. **HPLT 3.0 GPT-BERT** — hybrid causal+MLM encoder, `HPLT/hplt_gpt_bert_base_3_0_<lang_Script>`.

Both encoder families share one scoring mechanism:

- **Default — MLM cloze (PET-style).** Append a template ending in the model's mask
  token, e.g. `The chosen label is: <mask>`. Read the logits **at the mask position**
  over just the digit tokens (`0`–`2` / `0`–`4`), softmax + argmax. Reuses the pretrained
  MLM head (no randomly-initialized parameters), converges fast, strong in low-data. Uses
  `tokenizer.mask_token` (not a literal `[MASK]`). Fine-tune by continuing the MLM
  objective toward the gold digit at the mask position.
- **Ablation — CLS classification head.** Linear head over the pooled/first-token
  representation → 3 or 5 logits, fine-tuned with CE / ordinal loss. Conventional and
  often strongest with full training data, but the head is randomly initialized and it
  diverges from the shared digit-logit path. Kept as an ablation to test vs. the cloze
  verbalizer.

3. **HPLT 3.0 T5** — encoder-decoder, `HPLT/hplt_t5_base_3_0_<lang_Script>`.
   Used **text-to-text** (its pretraining objective). Labels are the **numeric digits**
   `0`–`2` (rubric) / `0`–`4` (simple), chosen to be single tokens. Prediction reads the
   first decoder step's logits over just those digit tokens, softmax + argmax — one
   forward pass, no free-form generation. The per-digit probabilities give the ordinal
   signal for QWK and go into `misc`. Meaning of the digits is anchored by the prompt.

**Unifying design:** all three families reduce to *"read the logits over the digit tokens
at one position"* (mask position for encoders, first decoder step for T5). This lets a
single shared reader in `common/` produce the label + per-digit probabilities (for `misc`)
regardless of family.

## 3. Design principle: one config-driven pipeline

Adapted from the `czech_embedding_benchmark` repo (`model_factory` + `run` + `evaluate`).
**There are no separate per-model sub-projects.** All three HPLT families live behind a
single pipeline; a **TOML config selects the family, track, language(s), head, and
hyperparameters**. A `model_factory.build_model(cfg)` dispatches on `cfg.model.family`
(`bert` / `gpt_bert` / `t5`) and `cfg.model.head` (`cloze` / `cls`). One config = one run.

```
configs/                       # one TOML per run (model family × track × lang set)
  bert/eng.simple.toml
  gpt_bert/eng.rubric.toml
  t5/multi.simple.toml
  _example.toml                # documented template
src/                           # single installable package, model-agnostic core
  config.py                    #   dataclasses + load_config(path) -> Config (TOML)
  model_factory.py             #   build_model(cfg) -> dispatch on family + head
  models/
    encoder_cloze.py           #     BERT 2.0 + GPT-BERT, MLM cloze digit reader
    encoder_cls.py             #     CLS-head ablation
    t5_text2text.py            #     T5 first-decoder-step digit reader
  data.py                      #   load JSON, schema, domain from fileid -> HF Dataset
  prompt.py                    #   input serialization + context truncation policy
  verbalizer.py                #   digit label tokens + shared digit-logit reader
  routing.py                   #   lang tag -> HPLT checkpoint id (see memory table)
  metrics.py                   #   per-domain QWK (mirrors data/scripts/evaluate.py)
  ordinal.py                   #   ordinal loss / label utilities
  train.py                     #   ENTRY: fine-tune from a config
  run.py                       #   ENTRY: predict a dataset from a config
  submit.py                    #   write {id, label, misc} JSON
jobs/
  scripts/run_aic.sh           # SLURM submission: sbatch --export=...,CONFIG=... 
results/
  <track>/<model_tag>/         # metrics.json, predictions.json (with misc), config.toml
data/
  devset/ trainset/ testset/   # provided data (see /memories/repo/task-overview.md)
  scripts/evaluate.py          # official-style evaluator (QWK)
```

### Config schema (dataclasses in `config.py`)

```toml
[run]
seed = 42
device = "cuda"
batch_size = 16
output_dir = "results"
mode = "train"          # or "predict"

[task]
track = "rubric"        # "simple" (0-4) | "rubric" (0-2)
train_path = "data/trainset/train.rubric.json"
eval_path  = "data/devset/dev.rubric.json"
langs = ["en"]          # subset routed to this run; [] = all

[model]
family = "bert"         # "bert" | "gpt_bert" | "t5"
head   = "cloze"        # encoders: "cloze" (default) | "cls";  t5: ignored
version = "2_0"         # resolves checkpoint via routing.py + lang
max_length = 512
trust_remote_code = true

[train]
epochs = 3
lr = 2e-5
loss = "ce"             # "ce" | "ordinal"
```

`routing.py` turns (`family`, `version`, `lang`) into the concrete `HPLT/...` id, so one
config can fan out to per-language checkpoints without listing them.

## 4. Model factory

`build_model(cfg) -> Model` returns an object with a uniform interface:

- `predict(batch) -> (labels, per_digit_probs)` — used by `run.py` and eval.
- `train_step(batch) / fit(...)` — used by `train.py`.

Dispatch:

| `family` | `head`  | implementation        | scoring position          |
|----------|---------|-----------------------|---------------------------|
| `bert`   | `cloze` | `encoder_cloze.py`    | mask-token logits         |
| `bert`   | `cls`   | `encoder_cls.py`      | pooled repr → linear head |
| `gpt_bert`| `cloze`/`cls` | same encoder modules | as above             |
| `t5`     | —       | `t5_text2text.py`     | first decoder-step logits |

All paths funnel through the shared `verbalizer.py` digit-logit reader, so `misc`
(per-digit probabilities + model name) is produced identically everywhere.

## 5. Data pipeline

1. **Load** JSON list; each item has `id`, `input`, optional `label`, `lang`, `fileid`.
2. **Domain** = parsed from `fileid` path segment (e.g. `popular_science`); used only for
   metric grouping, never fed to the model.
3. **Serialize input** (`prompt.py`):
   - Simple: `QUESTION … ANSWER … CONTEXT …`
   - Rubric: adds `NC/PC/FC` rubric descriptions.
   - Encoders append the cloze template (`… The chosen label is: <mask>`).
   - Truncation: preserve question + answer (+ rubric) + template; truncate the long
     context to fit `model.max_length`. Configurable.
4. **Routing** (`routing.py`): `lang` → checkpoint. Models are lazy-loaded and cached per
   language within a run.

## 6. Train / predict / evaluate

- **`train.py --config …`**: load config → `build_model` → load `train_path` (+ eval) →
  fine-tune (cloze: MLM objective toward gold digit; cls: CE/ordinal head) → save under
  `results/<track>/<model_tag>/`.
- **`run.py --config …`**: load config → `build_model` (or load fine-tuned) → predict
  `eval_path` / test batch → write `predictions.json` as a list of `{"id","label","misc"}`.
  `misc` MUST carry intermediary output (model name, per-digit probabilities). Never omit.
- **`metrics.py` / `data/scripts/evaluate.py`**: per-domain QWK on dev; the repo script is
  the source of truth (`--ref … --pred … [--is_rubric]`).

## 7. Experiment tracking

- Each run writes its resolved `config.toml` + `metrics.json` next to predictions, so runs
  are reproducible and comparable. Primary comparison axis: BERT 2.0 vs. GPT-BERT 3.0 vs.
  T5 3.0, cloze vs. cls, and loss variants — on dev QWK per domain.
- Research question (task.md): can small encoder-only models reliably judge grounded
  answers, and how do the three HPLT families compare?


## 8. Open decisions

- Long-context handling for large articles (truncate vs. chunk+pool vs. retrieval).
- Cross-lingual strategy for machine-translated domains.
- Shared multilingual head vs. strictly per-language fine-tunes.
