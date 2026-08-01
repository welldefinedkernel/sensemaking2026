# Results & Methods Log — Sensemaking @ CLEF 2026

Running log of methods tried and results achieved. Intended as raw material for a full
report, not the report itself — see [task.md](task.md) for the task definition and
[ARCHITECTURE.md](ARCHITECTURE.md) for the intended code design.

Each experiment entry below uses a fixed key set so it can be parsed mechanically
(e.g. `grep '^- id:'` or a simple YAML-ish scan). Add new entries by copying the template
at the bottom.

## Task recap

- Two tracks, both ordinal classification, scored with **Quadratic Weighted Kappa (QWK)**,
  computed per domain and averaged over domains (per [task.md](task.md)):
  - **Simple**: (context, question, answer) → label 0–4.
  - **Rubric**: (context, question, rubric{NC,PC,FC}, answer) → label 0=NC, 1=PC, 2=FC.
- 13 languages: en, cs, sr, pt, ro, sv, hu, uk, da, el, fi, ga, de.
- Models: HPLT monolingual encoder/encoder-decoder models, one checkpoint per language
  (no big general-purpose LLMs). See `/memories/repo/hplt-models.md` for the checkpoint map.

## Method A: zero-shot MLM cloze prompting (no fine-tuning)

Implemented in [src/run.py](../src/run.py). For HPLT 2.0 BERT / 3.0 GPT-BERT
(`AutoModelForMaskedLM`), the prompt serializes context/question/(rubric)/answer as text
ending in the tokenizer's mask token (`build_prompt`). The model's existing MLM head is
read **only** at the mask position, restricted to the single-token digit ids valid for the
track (0–4 simple, 0–2 rubric); softmax + argmax over just those logits gives the label
(`digit_token_ids`, `predict`). No parameters are fine-tuned or randomly initialized.
Per-digit softmax probabilities are stored in `misc.probs`; `misc.model` records the
per-language checkpoint used.

## Evaluation method

[data/scripts/evaluate.py](../data/scripts/evaluate.py) (`sklearn.cohen_kappa_score`,
`weights="quadratic"`) computes accuracy, linear/quadratic Cohen's kappa, and MAE.
Invocation used for the runs below:

```
python data/scripts/evaluate.py --ref=data/devset/dev.simple.json --pred=results/simple/2.0_mlm
```

(`--pred` must be a **directory** — the single-file/stdin code paths in `evaluate.py` are
currently broken, see Known Issues below.)

## Summary (simple track, full dev set, all 13 langs, global metrics)

| model            | accuracy | linear κ | quadratic κ (QWK) | MAE    |
|------------------|----------|----------|--------------------|--------|
| 2.0/mlm          | 0.2091   | -0.0025  | -0.0001            | 1.5393 |
| 3.0/gpt_bert/mlm | 0.1691   | -0.0015  | -0.0021            | 1.7511 |
| majority-class baseline | ~0.428 | — | 0 (by definition) | —  |

Both zero-shot MLM baselines score at chance on QWK and below the majority-class
baseline on accuracy; GPT-BERT 3.0 is worse than BERT 2.0 on every metric. See
per-experiment entries below for details.

## Experiments

- id: exp-2026-07-25-hplt2-mlm-simple-dev-full
- date: 2026-07-25 (predictions present in repo; exact run date not logged elsewhere)
- config: [configs/hplt2_baseline.simple.toml](../configs/hplt2_baseline.simple.toml)
- model: `2.0/mlm` (HPLT 2.0 BERT, `HPLT/hplt_bert_base_2_0_<lang-Script>`, per-language)
- method: Method A, zero-shot, no fine-tuning
- track: simple (5-way)
- data: `data/devset/dev.simple.json`, full dev set, all 13 languages, n=4146
- eval granularity: `granularity_5` (native 5-class, no label merging)
- results:
  - accuracy: 0.2091
  - linear_cohen_kappa: -0.0025
  - quadratic_cohen_kappa: -0.0001
  - mae: 1.5393
- baseline for comparison: majority-class (always predict label 4, 42.8% of dev set)
  would score accuracy ≈ 0.428, i.e. **higher** than this model's accuracy.
- interpretation: QWK ≈ 0 → predictions are statistically indistinguishable from chance,
  and worse than a trivial majority-class baseline on accuracy. Zero-shot MLM-cloze
  prompting on HPLT 2.0 BERT, with this prompt template and no fine-tuning, does not work.
- artifacts: `results/simple/2.0_mlm/predictions.json`
- status: DONE

- id: exp-2026-07-29-hplt3-gptbert-mlm-simple-dev-full
- date: 2026-07-29 (model-key mismatch since fixed; `MODEL_CHECKPOINTS` now has an
  explicit `"3.0/gpt_bert/mlm"` entry distinct from `"3.0/gpt_bert/e2e"`)
- config: [configs/hplt3_baseline.simple.toml](../configs/hplt3_baseline.simple.toml)
- model: `3.0/gpt_bert/mlm` (HPLT 3.0 GPT-BERT, `HPLT/hplt_gpt_bert_base_3_0_<lang_Script>`)
- method: Method A, zero-shot, no fine-tuning
- track: simple (5-way)
- data: `data/devset/dev.simple.json`, full dev set, all 13 languages, n=4146
- eval granularity: `granularity_5`
- results:
  - accuracy: 0.1691
  - linear_cohen_kappa: -0.0015
  - quadratic_cohen_kappa: -0.0021
  - mae: 1.7511
- interpretation: worse than HPLT 2.0 BERT MLM across every metric (lower accuracy,
  more negative QWK, higher MAE), and further below the majority-class baseline
  (≈0.428 accuracy). Zero-shot MLM-cloze prompting is not just weak but *worse* on
  GPT-BERT 3.0 than on BERT 2.0 with the same prompt/verbalizer.
- artifacts: `results/simple/3.0_gpt_bert_mlm/predictions.json`
- status: DONE

<!-- Template for new entries — copy below this line:
- id: exp-<date>-<model>-<track>-<split>
- date:
- config:
- model:
- method:
- track:
- data:
- eval granularity:
- results:
  - accuracy:
  - linear_cohen_kappa:
  - quadratic_cohen_kappa:
  - mae:
- interpretation:
- artifacts:
- status: DONE | BLOCKED | IN PROGRESS
-->

## Known issues / gaps (relevant to interpreting results)

1. **No domain-averaged QWK yet.** The task metric is QWK computed per domain then
   averaged; `evaluate.py` currently reports one global quadratic kappa over the whole
   reference set, with no per-domain breakdown. Numbers above are global, not the
   official metric.
2. **`evaluate.py` single-file/stdin path is broken.** When `--pred` is a single file
   (not a directory) or stdin, `load_labels()` returns a flat `{id: label}` dict, but
   `evaluate()` does `for sf, name, predictions in pred:`, which expects an iterable of
   3-tuples (as produced by the directory branch). Always pass a **directory** as
   `--pred` until this is fixed.
3. **T5 3.0 still unimplemented**: `MODEL_CHECKPOINTS["3.0/t5/"]` key doesn't match any
   documented config `model` string and T5 needs a different (seq2seq) prediction path
   than the shared `AutoModelForMaskedLM` cloze code — no run attempted yet.
4. **Rubric track**: no experiments run yet (simple track only so far).
5. **Fine-tuning / CLS-head ablation**: not implemented, design only
   (see [ARCHITECTURE.md](ARCHITECTURE.md) §2).
6. Only one prompt template has been tried (`build_prompt` in `src/run.py`); no
   prompt-variation ablations yet.
7. Both zero-shot MLM baselines so far (2.0 BERT, 3.0 GPT-BERT) score at/below chance
   and below a majority-class baseline — no working baseline yet on the simple track.

## Candidate next steps

- Wire up a T5 3.0 zero-shot baseline (needs a seq2seq first-decoder-step digit reader,
  distinct from the MLM cloze path used for 2.0 BERT / 3.0 GPT-BERT).
- Implement per-domain QWK averaging in (or alongside) `evaluate.py`.
- Run a rubric-track zero-shot baseline for comparison.
- Investigate why both zero-shot MLM baselines underperform a majority-class baseline
  (prompt wording, label-token choice, calibration) before concluding cloze prompting
  doesn't work at all — current evidence suggests the prompt/verbalizer, not the model
  family, is the likely bottleneck since both HPLT 2.0 BERT and 3.0 GPT-BERT fail
  similarly.
- Try the CLS-head fine-tuning ablation once training loop exists.
