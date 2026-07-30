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
from transformers import AutoModelForMaskedLM, AutoTokenizer
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


@torch.no_grad()
def predict(
    cfg: dict[str, Any], dataset: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Predict labels for one language subset."""
    device = cfg["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    if device.startswith("mps") and not torch.mps.is_available():
        device = "cpu"

    track = cfg["track"]
    model_sel = cfg["model"]
    labels = TRACK_LABELS[track]
    bs = cfg["batch_size"]

    results: list[dict[str, Any]] = []
    for lang, subset in dataset.items():
        ckpt = checkpoint_id(model_sel, lang)
        print(f"Loading {ckpt} on {device} ...")
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
                build_prompt(
                    name=cfg["prompt"],
                    inp=it["input"],
                    mask_token=tokenizer.mask_token,
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
    return results


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
    out_dir = Path(cfg["output_dir"]) / cfg["track"] / cfg["model"].replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.json").write_text(
        json.dumps(preds, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote predictions -> {out_dir / 'predictions.json'}")


if __name__ == "__main__":
    main()
