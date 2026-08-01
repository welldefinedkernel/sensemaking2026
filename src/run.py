"""Reads a TOML config, loads a task JSON (filtered by language), predicts
each answer's label based on the chosen model and method, and writes
submission-format predictions.

Usage:
    python src/run.py --config configs/eng.simple.toml
"""

import argparse
import json
import tomllib
import torch

from collections import defaultdict
from pathlib import Path
from prompt import build_prompt
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
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
    "3.0/gpt_bert/e2e": "HPLT/hplt_gpt_bert_base_3_0_{iso3}_{script}",
    "3.0/gpt_bert/mlm": "HPLT/hplt_gpt_bert_base_3_0_{iso3}_{script}",
    "3.0/t5/": "HPLT/hplt_t5_base_3_0_{iso3}_{script}",
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
) -> str:
    """Build a prompt, right-truncating only the context so it fits."""
    prompt = build_prompt(name, inp, mask_token)
    if len(tokenizer(prompt).input_ids) <= max_length:
        return prompt

    # Cost of everything except the context text
    overhead = len(
        tokenizer(build_prompt(name, {**inp, "context": ""}, mask_token)).input_ids
    )

    # Small margin
    budget = max_length - overhead - 8
    if budget <= 0:
        # Non-context parts already fit
        return prompt

    ctx_ids = tokenizer(inp.get("context", ""), add_special_tokens=False).input_ids
    kept_ctx = tokenizer.decode(ctx_ids[:budget], skip_special_tokens=True)
    return build_prompt(name, {**inp, "context": kept_ctx}, mask_token)


def translate_prompt(prompt: str, lang: str) -> str:
    """Translate a prompt into the sample's language via Google Translate."""
    if lang == "en":
        return prompt
    from deep_translator import GoogleTranslator

    return GoogleTranslator(source="auto", target=lang).translate(prompt)


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
    cfg: dict[str, Any], dataset: dict[str, list[dict[str, Any]]], device: str
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
                )
                for it in batch
            ]
            prompts = [translate_prompt(p, lang) for p in prompts]
            enc = tokenizer(
                prompts,
                padding=True,
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
    return results


@torch.no_grad()
def predict_causal(
    cfg: dict[str, Any], dataset: dict[str, list[dict[str, Any]]], device: str
) -> list[dict[str, Any]]:
    """Causal-LM: score digit logits at the next-token position."""
    track = cfg["track"]
    model_sel = cfg["model"]
    labels = TRACK_LABELS[track]
    bs = cfg["batch_size"]

    results: list[dict[str, Any]] = []
    for lang, subset in dataset.items():
        ckpt = checkpoint_id(model_sel, lang)
        print(f"Loading {ckpt} (e2e) on {device} ...")
        tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        tokenizer.truncation_side = "left"  # keep the trailing prompt intact
        tokenizer.padding_side = "right"  # last real token sits at attn_mask.sum()-1
        load_kwargs: dict[str, Any] = {"trust_remote_code": True}
        load_kwargs["use_safetensors"] = False
        model: Any = AutoModelForCausalLM.from_pretrained(ckpt, **load_kwargs)
        model.to(device)
        model.eval()

        digit_ids = digit_token_ids(tokenizer, track)

        for start in range(0, len(subset), bs):
            batch = subset[start : start + bs]
            prompts = [
                fit_prompt(
                    name=cfg["prompt"],
                    inp=it["input"],
                    mask_token="",
                    tokenizer=tokenizer,
                    max_length=cfg["max_length"],
                )
                for it in batch
            ]
            if cfg.get("translate", False):
                prompts = [translate_prompt(p, lang) for p in prompts]
            enc = tokenizer(
                prompts,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits  # [B, T, V]

            for row, it in enumerate(batch):
                pos = int(enc["attention_mask"][row].sum().item()) - 1
                probs = torch.softmax(logits[row, pos, digit_ids], dim=-1)
                idx = int(torch.argmax(probs).item())
                results.append(
                    {
                        "id": it["id"],
                        "label": labels[idx],
                        "misc": {"model": ckpt, "probs": probs.tolist()},
                    }
                )
    return results


def predict(
    cfg: dict[str, Any], dataset: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Choose inference method using the last `model` field segment."""
    device = resolve_device(cfg)
    method = cfg["model"].split("/")[-1]
    print(f"Predicting with {method} on {device} ...")
    if method == "mlm":
        return predict_mlm(cfg, dataset, device)
    if method == "e2e":
        return predict_causal(cfg, dataset, device)
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

    dataset = load_dataset(
        path=cfg["eval_path"], langs=cfg.get("langs", []), limit=cfg.get("limit", 0)
    )
    print(
        f"Loaded {sum([len(subset) for subset in dataset.values()])} items from {cfg['eval_path']} (langs={cfg.get('langs') or 'all'})."
    )

    preds = predict(cfg, dataset)
    out_dir = Path(cfg["output_dir"]) / cfg["track"] / cfg["model"].replace("/", "_") / cfg["prompt"].split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.json").write_text(
        json.dumps(preds, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote predictions -> {out_dir / 'predictions.json'}")


if __name__ == "__main__":
    main()
