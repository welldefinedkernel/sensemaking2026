"""Reads a TOML config, loads a task JSON (filtered by language), predicts
each answer's label based on the chosen model and method, and writes
submission-format predictions.

Usage:
    python src/run.py --config configs/eng.simple.toml
"""

import argparse
import json
import random
import sys
import tomllib
import torch

from collections import defaultdict
from pathlib import Path
from prompt import build_prompt
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from typing import Any


LANG_TO_ISO3_SCRIPT: dict[str, tuple[str, str]] = {
    "en": ("eng", "Latn"),
    "cs": ("ces", "Latn"),
    "sr": ("srp", "Cyrl"),
    "pt": ("por", "Latn"),
    "ro": ("ron", "Latn"),
    "sv": ("swe", "Latn"),
    "hu": ("hun", "Latn"),
    "uk": ("ukr", "Cyrl"),
    "da": ("dan", "Latn"),
    "el": ("ell", "Grek"),
    "fi": ("fin", "Latn"),
    "ga": ("gle", "Latn"),
    "de": ("deu", "Latn"),
}
TRACK_LABELS = {"simple": [0, 1, 2, 3, 4], "rubric": [0, 1, 2]}
MODEL_CHECKPOINTS = {
    "2.0/mlm": "HPLT/hplt_bert_base_2_0_{iso3}-{script}",
    "2.0/cls": "HPLT/hplt_bert_base_2_0_{iso3}-{script}",
    "3.0/gpt_bert/mlm": "HPLT/hplt_gpt_bert_base_3_0_{iso3}_{script}",
    "3.0/gpt_bert/cls": "HPLT/hplt_gpt_bert_base_3_0_{iso3}_{script}",
    "3.0/gpt_bert/instruction_tune": "HPLT/hplt_gpt_bert_base_3_0_{iso3}_{script}",
    "3.0/t5/instruction_tune": "HPLT/hplt_t5_base_3_0_{iso3}_{script}",
}


def checkpoint_id(model: str, lang: str) -> str:
    """Resolve a checkpoint id for a given model."""
    if model not in MODEL_CHECKPOINTS:
        raise NotImplementedError(
            f"model={model!r} is not implemented yet; "
            f"available: {sorted(MODEL_CHECKPOINTS)}"
        )
    iso3, script = LANG_TO_ISO3_SCRIPT[lang]
    return MODEL_CHECKPOINTS[model].format(iso3=iso3, script=script)


def load_dataset(
    path: str | Path, langs: list[str], limit: int
) -> dict[str, list[dict[str, Any]]]:
    """Load a task JSON, optionally filtering by language and capping the count."""
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)

    subsets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    count = 0
    max = limit if limit else len(data)

    for it in data:
        if langs and it.get("lang") not in langs:
            continue
        lang = it.get("lang")

        # domain lives in the fileid path, e.g. ".../popular_science/..."
        segs = [p for p in it.get("fileid", "").split("/") if p]
        domain = next(
            (p for p in segs if p not in {"devset", "trainset", "testset"}), "unknown"
        )
        subsets[lang].append(
            {
                "id": it["id"],
                "input": it["input"],
                "label": it.get("label"),
                "lang": it.get("lang"),
                "domain": domain,
            }
        )

        if (count := count + 1) == max:
            break

    return subsets


def digit_token_ids(tokenizer, track: str) -> list[int]:
    """Token id for each candidate digit; each must map to a single token."""
    ids = []
    for label in TRACK_LABELS[track]:
        toks = tokenizer.encode(str(label), add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(f"Digit {label} is not a single token: {toks}")
        ids.append(toks[0])
    return ids


def fit_prompt(
    name: str,
    inp: dict[str, Any],
    mask_token: str,
    tokenizer,
    max_length: int,
    lang: str,
) -> str:
    """Build a prompt, right-truncating only the context so it fits."""
    prompt = build_prompt(name, inp, mask_token, lang)
    if len(tokenizer(prompt).input_ids) <= max_length:
        return prompt

    # Cost of everything except the context text
    overhead = len(
        tokenizer(build_prompt(name, {**inp, "context": ""}, mask_token, lang)).input_ids
    )

    # Small margin
    budget = max_length - overhead - 8
    if budget <= 0:
        # Non-context parts alone exceed max_length; tokenizer truncation is the backstop
        return prompt

    ctx_ids = tokenizer(inp.get("context", ""), add_special_tokens=False).input_ids
    kept_ctx = tokenizer.decode(ctx_ids[:budget], skip_special_tokens=True)
    return build_prompt(name, {**inp, "context": kept_ctx}, mask_token, lang)


def parse_digit(text: str, labels: list[int]) -> int | None:
    """First character of the generated text that is a valid label digit."""
    for ch in text:
        if ch.isdigit() and int(ch) in labels:
            return int(ch)
    return None


def dataset_name(cfg: dict[str, Any]) -> str:
    """Which data split the run used; becomes the results filename."""
    if "dataset" in cfg:
        return cfg["dataset"]
    return "balanced" if "balanced" in Path(cfg["eval_path"]).parts else "regular"


def plain_prompts(
    cfg: dict[str, Any], batch: list[dict[str, Any]], tokenizer, lang: str
) -> list[str]:
    """Prompts ending right where the label belongs, with no mask token."""
    return [
        fit_prompt(
            name=cfg["prompt"],
            inp=it["input"],
            mask_token="",
            tokenizer=tokenizer,
            max_length=cfg["max_length"],
            lang=lang,
        ).rstrip()
        for it in batch
    ]


def causal_batch(
    cfg: dict[str, Any], batch: list[dict[str, Any]], tokenizer, lang: str, device: str
) -> dict[str, torch.Tensor]:
    """Tokenize prompts with the gold digit appended; supervise only that digit."""
    digit_ids = digit_token_ids(tokenizer, cfg["track"])
    label_values = TRACK_LABELS[cfg["track"]]

    rows = [
        tokenizer(prompt, truncation=True, max_length=cfg["max_length"]).input_ids
        + [digit_ids[label_values.index(it["label"])]]
        for prompt, it in zip(plain_prompts(cfg, batch, tokenizer, lang), batch)
    ]

    width = max(len(r) for r in rows)
    input_ids = torch.full((len(rows), width), tokenizer.pad_token_id)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    labels = torch.full((len(rows), width), -100)
    for i, row in enumerate(rows):
        input_ids[i, : len(row)] = torch.tensor(row)
        attention_mask[i, : len(row)] = 1
        labels[i, len(row) - 1] = row[-1]

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
    }


def cls_batch(
    cfg: dict[str, Any], batch: list[dict[str, Any]], tokenizer, lang: str, device: str
) -> dict[str, torch.Tensor]:
    """Tokenize prompts and attach the gold label index for the classifier head."""
    label_values = TRACK_LABELS[cfg["track"]]
    enc = tokenizer(
        plain_prompts(cfg, batch, tokenizer, lang),
        padding=True,
        truncation=True,
        max_length=cfg["max_length"],
        return_tensors="pt",
    ).to(device)
    labels = torch.tensor(
        [label_values.index(it["label"]) for it in batch], device=device
    )
    return {**enc, "labels": labels}


class _MaskedSoftmax:
    """Drop-in for the 2.0 remote code's custom autograd Function, whose backward
    calls torch._softmax_backward_data with a signature removed years ago."""

    @staticmethod
    def apply(x, mask, dim):
        return torch.softmax(x.masked_fill(mask, float("-inf")), dim).masked_fill(
            mask, 0.0
        )


def patch_masked_softmax(model) -> None:
    """Make HPLT 2.0 encoders trainable; their forward works but backward raises."""
    module = sys.modules.get(type(model).__module__)
    if module is not None and hasattr(module, "MaskedSoftmax"):
        module.MaskedSoftmax = _MaskedSoftmax


@torch.enable_grad()  # the callers run under torch.no_grad()
def finetune(
    model,
    cfg: dict[str, Any],
    subset: list[dict[str, Any]],
    lang: str,
    device: str,
    make_batch,
) -> None:
    """Fine-tune on the model's own loss; make_batch turns rows into forward kwargs."""
    if not subset:
        raise ValueError(
            f"no training items for lang={lang!r}; the eval set covers a language "
            f"the train set does not (check `langs` / `train_limit`)"
        )

    epochs = cfg.get("epochs", 1)
    bs = cfg.get("train_batch_size", cfg["batch_size"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-5))

    # backward keeps every layer's attention matrix, which is O(seq_len^2); recompute
    # them instead so full-length prompts fit
    if cfg.get("gradient_checkpointing", True) and hasattr(model, "gradient_checkpointing"):
        model.gradient_checkpointing = True

    model.train()
    for epoch in range(epochs):
        random.shuffle(subset)
        total, steps = 0.0, 0
        for start in range(0, len(subset), bs):
            loss = model(**make_batch(subset[start : start + bs])).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total, steps = total + loss.item(), steps + 1
            if steps % 20 == 0:
                print(f"  epoch {epoch} step {steps}: loss {total / steps:.4f}")
        print(f"  epoch {epoch}: mean loss {total / max(steps, 1):.4f}")

    model.eval()


def resolve_device(cfg: dict[str, Any]) -> str:
    """Fall back to CPU when the requested accelerator is unavailable."""
    device = cfg.get("device", "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    if device.startswith("mps") and not torch.mps.is_available():
        device = "cpu"
    return device


@torch.no_grad()
def predict_mlm(
    cfg: dict[str, Any], dataset: dict[str, list[dict[str, Any]]], device: str, train_dataset: dict[str, list[dict[str, Any]]] | None = None
) -> list[dict[str, Any]]:
    """Masked-LM: score digit logits at the mask position."""
    track = cfg["track"]
    model_sel = cfg["model"]
    labels = TRACK_LABELS[track]
    bs = cfg["batch_size"]

    results: list[dict[str, Any]] = []
    for lang, subset in dataset.items():
        ckpt = checkpoint_id(model_sel, lang)
        print(f"Loading {ckpt} (mlm) on {device} ...")
        tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        tokenizer.truncation_side = "left"  # we do not want to truncate away [MASK]
        load_kwargs: dict[str, Any] = {"trust_remote_code": True}
        load_kwargs["use_safetensors"] = False
        model = AutoModelForMaskedLM.from_pretrained(ckpt, **load_kwargs)
        model.to(device).eval()

        if train_dataset:
            print(f"Finetuning {ckpt} on {len(train_dataset[lang])} items ...")
            

        digit_ids = digit_token_ids(tokenizer, track)
        mask_id = tokenizer.mask_token_id

        for start in range(0, len(subset), bs):
            batch = subset[start : start + bs]
            prompts = [
                fit_prompt(
                    name=cfg["prompt"],
                    inp=it["input"],
                    mask_token=tokenizer.mask_token,
                    tokenizer=tokenizer,
                    max_length=cfg["max_length"],
                    lang=lang,
                )
                for it in batch
            ]
            enc = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=cfg["max_length"],
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits  # [B, T, V]

            for row, it in enumerate(batch):
                pos = int(
                    (enc["input_ids"][row] == mask_id)
                    .nonzero(as_tuple=True)[0][0]
                    .item()
                )
                probs = torch.softmax(logits[row, pos, digit_ids], dim=-1)
                idx = int(torch.argmax(probs).item())
                results.append(
                    {
                        "id": it["id"],
                        "label": labels[idx],
                        "misc": {"model": ckpt, "probs": probs.tolist()},
                    }
                )
            del enc, logits, probs
    return results


@torch.no_grad()
def predict_causal(
    cfg: dict[str, Any],
    dataset: dict[str, list[dict[str, Any]]],
    device: str,
    train_dataset: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """End-to-end: let the model generate a continuation and parse a digit out of it."""
    track = cfg["track"]
    model_sel = cfg["model"]
    labels = TRACK_LABELS[track]
    bs = cfg["batch_size"]
    max_new_tokens = cfg.get("max_new_tokens", 8)
    fallback = labels[len(labels) // 2]

    results: list[dict[str, Any]] = []
    for lang, subset in dataset.items():
        ckpt = checkpoint_id(model_sel, lang)
        print(f"Loading {ckpt} (e2e) on {device} ...")
        tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        tokenizer.truncation_side = "left"  # keep the instruction at the end
        tokenizer.padding_side = "left"  # generation continues from the last position
        model = AutoModelForCausalLM.from_pretrained(
            ckpt, trust_remote_code=True, use_safetensors=False
        )
        model.to(device).eval()

        if train_dataset:
            print(f"Finetuning {ckpt} on {len(train_dataset[lang])} items ...")
            tokenizer.padding_side = "right"  # the digit sits at the end of the real tokens
            finetune(
                model,
                cfg,
                train_dataset[lang],
                lang,
                device,
                lambda rows: causal_batch(cfg, rows, tokenizer, lang, device),
            )
            tokenizer.padding_side = "left"

        for start in range(0, len(subset), bs):
            batch = subset[start : start + bs]
            prompts = plain_prompts(cfg, batch, tokenizer, lang)
            enc = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=cfg["max_length"],
                return_tensors="pt",
            ).to(device)
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=False,  # GptBertForCausalLM has no KV cache
                pad_token_id=tokenizer.pad_token_id,
            )

            prompt_len = enc["input_ids"].shape[1]
            for row, it in enumerate(batch):
                text = tokenizer.decode(
                    out[row, prompt_len:], skip_special_tokens=True
                )
                digit = parse_digit(text, labels)
                results.append(
                    {
                        "id": it["id"],
                        "label": digit if digit is not None else fallback,
                        "misc": {
                            "model": ckpt,
                            "generated": text,
                            "parsed": digit is not None,
                        },
                    }
                )
            del enc, out
    return results


@torch.no_grad()
def predict_cls(
    cfg: dict[str, Any],
    dataset: dict[str, list[dict[str, Any]]],
    device: str,
    train_dataset: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Classification head over the [CLS] position."""
    track = cfg["track"]
    model_sel = cfg["model"]
    labels = TRACK_LABELS[track]
    bs = cfg["batch_size"]

    if not train_dataset:
        raise ValueError(
            f"model={model_sel!r} needs `train_path`: the classification head is "
            f"randomly initialised, so without fine-tuning it predicts noise"
        )

    results: list[dict[str, Any]] = []
    for lang, subset in dataset.items():
        ckpt = checkpoint_id(model_sel, lang)
        print(f"Loading {ckpt} (cls) on {device} ...")
        tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        # the head reads position 0, so padding must not displace [CLS]; the tokenizer
        # re-adds [CLS] after truncating, so cut from the left to keep the answer
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "left"
        model = AutoModelForSequenceClassification.from_pretrained(
            ckpt,
            trust_remote_code=True,
            use_safetensors=False,
            num_labels=len(labels),
        )
        model.to(device)
        patch_masked_softmax(model)

        print(f"Finetuning {ckpt} on {len(train_dataset[lang])} items ...")
        finetune(
            model,
            cfg,
            train_dataset[lang],
            lang,
            device,
            lambda rows: cls_batch(cfg, rows, tokenizer, lang, device),
        )

        for start in range(0, len(subset), bs):
            batch = subset[start : start + bs]
            enc = tokenizer(
                plain_prompts(cfg, batch, tokenizer, lang),
                padding=True,
                truncation=True,
                max_length=cfg["max_length"],
                return_tensors="pt",
            ).to(device)
            probs = torch.softmax(model(**enc).logits, dim=-1)

            for row, it in enumerate(batch):
                idx = int(torch.argmax(probs[row]).item())
                results.append(
                    {
                        "id": it["id"],
                        "label": labels[idx],
                        "misc": {"model": ckpt, "probs": probs[row].tolist()},
                    }
                )
            del enc, probs
    return results


def predict(
    cfg: dict[str, Any], 
    dataset: dict[str, list[dict[str, Any]]],
    train_dataset: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Choose inference method using the last `model` field segment."""
    device = resolve_device(cfg)
    method = cfg["model"].split("/")[-1]
    if train_dataset:
        print(
            f"Training dataset provided, finetuning will be performed on {sum([len(subset) for subset in train_dataset.values()])} items."
        )
    print(f"Predicting with {method} on {device} ...")
    if method == "mlm":
        return predict_mlm(cfg, dataset, device, train_dataset=train_dataset)
    if method == "cls":
        return predict_cls(cfg, dataset, device, train_dataset=train_dataset)
    if method in ("e2e", "instruction_tune"):
        return predict_causal(cfg, dataset, device, train_dataset=train_dataset)
    raise NotImplementedError(
        f"inference method={method!r} is not implemented, check example config'"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a TOML config.")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    assert all(item in cfg for item in ["eval_path", "track", "prompt", "model"])

    if "train_path" in cfg:
        train_dataset = load_dataset(
            path=cfg["train_path"],
            langs=cfg.get("langs", []),
            limit=cfg.get("train_limit", 0),
        )
        print(
            f"Loaded {sum([len(subset) for subset in train_dataset.values()])} items from {cfg['train_path']} (langs={cfg.get('langs') or 'all'})."
        )

    dataset = load_dataset(
        path=cfg["eval_path"], langs=cfg.get("langs", []), limit=cfg.get("limit", 0)
    )
    print(
        f"Loaded {sum([len(subset) for subset in dataset.values()])} items from {cfg['eval_path']} (langs={cfg.get('langs') or 'all'})."
    )

    preds = predict(cfg, dataset, train_dataset if "train_dataset" in locals() else None)
    out_dir = Path(cfg["output_dir"]) / cfg["track"] / cfg["model"].replace("/", "_") / cfg["prompt"].split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset_name(cfg)}.json"
    out_path.write_text(
        json.dumps(preds, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote predictions -> {out_path}")


if __name__ == "__main__":
    main()
