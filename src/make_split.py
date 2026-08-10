"""Build a domain-balanced train/dev split from the released train+dev data.

The released split has almost no domain overlap (train is 82% ukr-biology, dev is
four domains never seen in training), so dev QWK measures domain transfer rather
than task skill. This pools both files and re-splits 80/20 inside every
(domain, language) cell.

Splitting happens at question-group level: each question carries ~6 graded
answers, so a row-level split would put the same question on both sides.

Usage:
    python src/make_split.py --track simple
"""

import argparse
import collections
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV_FRACTION = 0.2
# ukr-biology is 63% of the pooled rows; cap it relative to the next largest domain
UKR_BIOLOGY_CAP_FACTOR = 2.0


def domain_of(fileid: str) -> str:
    """Domain name from a fileid, ignoring split prefixes and document numbers."""
    segs = [p for p in fileid.split("/") if p and p not in {"devset", "trainset", "testset"}]
    return re.sub(r"-\d+$", "", segs[0]) if segs else "unknown"


def load_pool(track: str, source: str | None = None) -> list[dict]:
    """Pooled rows from both input files, with ids made globally unique."""
    rows = []
    if source:
        sources = [("train", f"{source}/train.{track}.json"), ("dev", f"{source}/dev.{track}.json")]
    else:
        sources = [("train", f"data/trainset/train.{track}.json"), ("dev", f"data/devset/dev.{track}.json")]
    for src, rel in sources:
        for it in json.loads((ROOT / rel).read_text(encoding="utf-8")):
            rows.append(
                {
                    # released ids are only unique within a file: 1256 collide across the two
                    "id": f"{src}:{it['id']}",
                    "input": it["input"],
                    "label": it["label"],
                    "lang": it["lang"],
                    "fileid": it["fileid"],
                    "domain": domain_of(it["fileid"]),
                }
            )
    return rows


def question_groups(rows: list[dict]) -> dict[tuple, list[dict]]:
    """Rows keyed by (domain, lang, question); models are monolingual, so the
    same question translated into another language is not leakage."""
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["domain"], r["lang"], r["input"]["question"])].append(r)
    return groups


def cap_domain(groups: dict[tuple, list[dict]], domain: str, budget: int, rng) -> None:
    """Drop whole question groups from `domain` until it fits `budget` rows."""
    per_lang = collections.defaultdict(list)
    for key in [k for k in groups if k[0] == domain]:
        per_lang[key[1]].append(key)

    total = sum(len(groups[k]) for lang in per_lang for k in per_lang[lang])
    for lang, keys in per_lang.items():
        lang_rows = sum(len(groups[k]) for k in keys)
        lang_budget = round(budget * lang_rows / total)
        rng.shuffle(keys)
        kept = 0
        for k in keys:
            if kept >= lang_budget:
                del groups[k]
            else:
                kept += len(groups[k])


def split(groups: dict[tuple, list[dict]], rng) -> tuple[list[dict], list[dict]]:
    """Assign whole question groups to dev until each cell reaches DEV_FRACTION."""
    by_cell = collections.defaultdict(list)
    for key in groups:
        by_cell[(key[0], key[1])].append(key)

    train, dev = [], []
    for cell, keys in sorted(by_cell.items()):
        rng.shuffle(keys)
        target = DEV_FRACTION * sum(len(groups[k]) for k in keys)
        taken = 0
        for i, k in enumerate(keys):
            size = len(groups[k])
            # take the group only while it moves dev closer to the target share
            closer = abs(taken + size - target) < abs(taken - target)
            if closer and i < len(keys) - 1:  # always leave one group for training
                dev.extend(groups[k])
                taken += size
            else:
                train.extend(groups[k])
    return train, dev


def report(train: list[dict], dev: list[dict]) -> None:
    doms = sorted({r["domain"] for r in train} | {r["domain"] for r in dev})
    labels = sorted({r["label"] for r in train} | {r["label"] for r in dev})
    width = 5 * len(labels)
    print(
        f"{'domain':17s}{'train':>7s}{'dev':>7s}{'dev%':>6s}   "
        f"{'train label %':<{width}s}   dev label %"
    )
    for d in doms:
        t = [r for r in train if r["domain"] == d]
        v = [r for r in dev if r["domain"] == d]
        n = len(t) + len(v)
        tc = collections.Counter(r["label"] for r in t)
        vc = collections.Counter(r["label"] for r in v)
        tp = " ".join(f"{100 * tc[i] / max(len(t), 1):4.0f}" for i in labels)
        vp = " ".join(f"{100 * vc[i] / max(len(v), 1):4.0f}" for i in labels)
        print(f"{d:17s}{len(t):7d}{len(v):7d}{100 * len(v) / n:5.0f}%   {tp}   {vp}")
    print(f"{'TOTAL':17s}{len(train):7d}{len(dev):7d}{100 * len(dev) / (len(train) + len(dev)):5.0f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", default="simple", choices=["simple", "rubric"])
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--source", default=None, help="Directory holding train.<track>.json and dev.<track>.json.")
    parser.add_argument("--out", default="data/balanced")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = load_pool(args.track, args.source)
    groups = question_groups(rows)

    sizes = collections.Counter(r["domain"] for r in rows)
    largest, second = [n for _, n in sizes.most_common(2)]
    if largest > UKR_BIOLOGY_CAP_FACTOR * second:
        domain = sizes.most_common(1)[0][0]
        budget = int(UKR_BIOLOGY_CAP_FACTOR * second)
        print(f"Capping {domain}: {largest} -> ~{budget} rows")
        cap_domain(groups, domain, budget, rng)

    train, dev = split(groups, rng)
    rng.shuffle(train)
    rng.shuffle(dev)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    for name, data in [("train", train), ("dev", dev)]:
        path = out / f"{name}.{args.track}.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {len(data):6d} rows -> {path}")
    print()
    report(train, dev)


if __name__ == "__main__":
    main()
