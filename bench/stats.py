"""
Quick inspection / stats tool for any pipeline output file.
Usage:
  python inspect.py data/01_filtered.jsonl
  python inspect.py data/03_annotated.jsonl --show-demands
  python inspect.py data/final/bench_singleturn.jsonl --sample 3
"""

import json
import argparse
import random
from pathlib import Path
from collections import defaultdict, Counter


def load(path: str) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def stats_filtered(items):
    print(f"Total sessions: {len(items)}")
    subcats = Counter(s.get("_meta", {}).get("subcategory", "?") for s in items)
    langs   = Counter(s.get("_meta", {}).get("language", "?") for s in items)
    density = Counter(s.get("_meta", {}).get("demand_density", "?") for s in items)
    print("Subcategory:", dict(subcats.most_common()))
    print("Language:",    dict(langs.most_common()))
    print("Demand density:", dict(density.most_common()))
    qs = [s.get("_meta", {}).get("quality_score", 0) for s in items]
    print(f"Quality score: min={min(qs):.2f} max={max(qs):.2f} avg={sum(qs)/len(qs):.2f}")


def stats_turns(items):
    print(f"Total sessions: {len(items)}")
    n_turns = [s.get("n_turns", 0) for s in items]
    subcats = Counter(s.get("subcategory", "?") for s in items)
    langs   = Counter(s.get("language", "?") for s in items)
    sources = Counter(s.get("source", "?") for s in items)
    print(f"Turns: min={min(n_turns)} max={max(n_turns)} avg={sum(n_turns)/len(n_turns):.1f}")
    print("Subcategory:", dict(subcats.most_common()))
    print("Language:",    dict(langs.most_common()))
    print("Source:",      dict(sources.most_common()))


def stats_annotated(items, show_demands=False):
    stats_turns(items)
    all_demands = []
    for s in items:
        for t in s.get("turns", []):
            ann = t.get("annotation", {})
            all_demands.extend(ann.get("demands", []))
    total_turns = sum(len(s.get("turns", [])) for s in items)
    turns_with  = sum(
        1 for s in items for t in s.get("turns", [])
        if t.get("annotation", {}).get("has_demand", False)
    )
    dtype_counts = Counter(d.get("demand_type", "?") for d in all_demands)
    cat_counts   = Counter(d.get("category", "?") for d in all_demands)
    print(f"\nAnnotation stats:")
    print(f"  Total demands: {len(all_demands)}")
    print(f"  Turns with demand: {turns_with} / {total_turns} ({turns_with/max(total_turns,1)*100:.1f}%)")
    print(f"  Category: {dict(cat_counts.most_common())}")
    print(f"  Top demand types: {dict(dtype_counts.most_common(8))}")
    if show_demands and all_demands:
        print("\nSample demand:")
        d = random.choice(all_demands)
        print(json.dumps(d, ensure_ascii=False, indent=2))


def stats_singleturn(items):
    print(f"Total items: {len(items)}")
    labels  = Counter(i.get("label", "?") for i in items)
    subcats = Counter(i.get("subcategory", "?") for i in items)
    langs   = Counter(i.get("language", "?") for i in items)
    sources = Counter(i.get("source", "?") for i in items)
    print(f"Labels: {dict(labels.most_common())}")
    print(f"Subcategory: {dict(subcats.most_common())}")
    print(f"Language: {dict(langs.most_common())}")
    print(f"Source: {dict(sources.most_common())}")


def sample_items(items, n, kind):
    sample = random.sample(items, min(n, len(items)))
    for i, item in enumerate(sample):
        print(f"\n{'='*60} Sample {i+1} {'='*60}")
        if kind == "singleturn":
            print(f"ID: {item.get('item_id')}  Label: {item.get('label')}  Subcat: {item.get('subcategory')}")
            print(f"Context (last turn):\n{item.get('context', '')[-300:]}")
            print(f"\nCurrent turn:\n{item.get('current_turn', '')[:400]}")
            if item.get("demands"):
                print(f"\nDemands: {json.dumps(item['demands'][:2], ensure_ascii=False)}")
        else:
            print(json.dumps(item, ensure_ascii=False, indent=2)[:800])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to data file")
    parser.add_argument("--show-demands", action="store_true")
    parser.add_argument("--sample", type=int, default=0, help="Show N random samples")
    args = parser.parse_args()

    items = load(args.file)
    fname = Path(args.file).name

    if "filtered" in fname:
        stats_filtered(items)
    elif "singleturn" in fname:
        stats_singleturn(items)
        if args.sample:
            sample_items(items, args.sample, "singleturn")
    elif "annotated" in fname or "annotated_synth" in fname:
        stats_annotated(items, args.show_demands)
        if args.sample:
            sample_items(items, args.sample, "annotated")
    else:
        stats_turns(items)
        if args.sample:
            sample_items(items, args.sample, "turns")
