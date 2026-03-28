"""
Generate comparison report from scored results.

Reads {model}_{level}.jsonl files from scores/ and results/ dirs.
Reports best-level score per model + detailed breakdown.

Usage:
    python -m eval.report
"""

import json
from pathlib import Path
from collections import defaultdict

from eval.config import MODELS

EVAL_DIR    = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
SCORES_DIR  = EVAL_DIR / "scores"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def main():
    # Discover all scored result files
    model_map = {m["name"]: m for m in MODELS}
    entries = []  # (model_cfg, level, scores, results)

    for score_file in sorted(SCORES_DIR.glob("*.jsonl")):
        stem = score_file.stem  # e.g. "gpt-5-mini_encouraging"
        # Parse model name and level from stem
        for mname in sorted(model_map, key=len, reverse=True):
            if stem.startswith(mname + "_"):
                level = stem[len(mname) + 1:]
                scores = load_jsonl(score_file)
                results = load_jsonl(RESULTS_DIR / f"{stem}.jsonl")
                if scores:
                    entries.append((model_map[mname], level, scores, results))
                break

    if not entries:
        print("No scored results found.")
        return

    # Group by model, pick best level by balanced score (1:1 demand/non-demand)
    import random
    random.seed(42)

    by_model = defaultdict(list)
    for mcfg, level, scores, results in entries:
        by_model[mcfg["name"]].append((mcfg, level, scores, results))

    rows = []
    for mname, variants in by_model.items():
        best = None
        best_bal = -1
        for mcfg, level, scores, results in variants:
            # Compute balanced score
            demand_correct = demand_total = 0
            nodemand_correct = nodemand_total = 0
            for s in scores:
                # Determine if this turn had demand from label
                if s["label"] in ("fn", "tp", "tp_bad"):
                    demand_total += 1
                    if s["score"] == 1:
                        demand_correct += 1
                else:
                    nodemand_total += 1
                    if s["score"] == 1:
                        nodemand_correct += 1
            d_acc = demand_correct / demand_total if demand_total else 0
            nd_acc = nodemand_correct / nodemand_total if nodemand_total else 0
            bal = (d_acc + nd_acc) / 2 * 100

            if bal > best_bal:
                best_bal = bal
                best = (mcfg, level, scores, results, bal)

        mcfg, level, scores, results, bal = best
        total = len(scores)
        raw_score = sum(s["score"] for s in scores) / total * 100

        labels = defaultdict(int)
        for s in scores:
            labels[s["label"]] += 1

        # Cost & latency
        if results:
            total_in = sum(r.get("input_tokens", 0) for r in results)
            total_out = sum(r.get("output_tokens", 0) for r in results)
            cost = (total_in * mcfg["input"] + total_out * mcfg["output"]) / 1_000_000
            lats = [r.get("latency_ms", 0) for r in results if r.get("latency_ms", 0) > 0]
            avg_lat = round(sum(lats) / len(lats)) if lats else 0
            p95_lat = round(sorted(lats)[int(len(lats) * 0.95)]) if lats else 0
        else:
            total_in = total_out = 0
            cost = avg_lat = p95_lat = 0

        rows.append({
            "model": mcfg,
            "level": level,
            "score": bal,
            "raw_score": raw_score,
            "total": total,
            "labels": labels,
            "total_in": total_in,
            "total_out": total_out,
            "cost": cost,
            "avg_lat": avg_lat,
            "p95_lat": p95_lat,
        })

    rows.sort(key=lambda x: -x["score"])

    # ---- Main Table ----
    print("=" * 95)
    print(f"{'Model':<25} {'Level':<13} {'Balanced':>8} {'Raw':>5} {'TN':>5} {'FN':>5} {'TP':>5} {'TP✗':>5} {'FP✓':>5} {'FP✗':>5}")
    print("-" * 95)
    for r in rows:
        lb = r["labels"]
        print(
            f"{r['model']['name']:<25} "
            f"{r['level']:<13} "
            f"{r['score']:>7.1f} "
            f"{r['raw_score']:>5.1f} "
            f"{lb.get('tn', 0):>5} "
            f"{lb.get('fn', 0):>5} "
            f"{lb.get('tp', 0):>5} "
            f"{lb.get('tp_bad', 0):>5} "
            f"{lb.get('fp_accepted', 0):>5} "
            f"{lb.get('fp_rejected', 0):>5}"
        )
    print("=" * 95)

    # ---- Cost & Latency ----
    print(f"\n{'Model':<25} {'Cost$':>7} {'AvgMs':>7} {'P95Ms':>7} {'InTok':>9} {'OutTok':>9}")
    print("-" * 70)
    for r in rows:
        print(
            f"{r['model']['name']:<25} "
            f"{r['cost']:>7.3f} "
            f"{r['avg_lat']:>7} "
            f"{r['p95_lat']:>7} "
            f"{r['total_in']:>9} "
            f"{r['total_out']:>9}"
        )

    # ---- Per-subcategory for best model ----
    best = rows[0]
    scores = load_jsonl(SCORES_DIR / f"{best['model']['name']}_{best['level']}.jsonl")
    by_sc = defaultdict(list)
    for s in scores:
        by_sc[s["subcategory"]].append(s["score"])

    print(f"\nPer-subcategory ({best['model']['name']}, {best['level']}, balanced={best['score']:.1f}):")
    print(f"{'Subcat':<8} {'Score':>6} {'N':>5}")
    print("-" * 22)
    for sc in sorted(by_sc):
        vals = by_sc[sc]
        print(f"{sc:<8} {sum(vals)/len(vals)*100:>5.1f} {len(vals):>5}")

    # ---- All levels comparison ----
    print(f"\n--- All levels tested ---")
    print(f"{'Model':<25} {'Level':<13} {'Balanced':>8}")
    print("-" * 50)
    for mcfg, level, scores, results in entries:
        demand_c = demand_t = nd_c = nd_t = 0
        for s in scores:
            if s["label"] in ("fn", "tp", "tp_bad"):
                demand_t += 1
                if s["score"] == 1: demand_c += 1
            else:
                nd_t += 1
                if s["score"] == 1: nd_c += 1
        d_acc = demand_c / demand_t if demand_t else 0
        nd_acc = nd_c / nd_t if nd_t else 0
        bal = (d_acc + nd_acc) / 2 * 100
        print(f"{mcfg['name']:<25} {level:<13} {bal:>7.1f}")


if __name__ == "__main__":
    main()
