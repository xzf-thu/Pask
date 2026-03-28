"""
Step 6: Format final benchmark output.

Inputs:
  data/03_annotated.jsonl          (real_raw multi-turn, annotated)
  data/03_annotated_clean.jsonl    (real_clean multi-turn, annotated)
  data/05_annotated_synth.jsonl    (synthetic multi-turn, annotated)

Outputs (in data/final/):
  bench_multiturn_real_raw.jsonl       ~1000 sessions
  bench_multiturn_real_clean.jsonl     ~1000 sessions
  bench_multiturn_synthetic.jsonl      ~1000 sessions
  bench_singleturn.jsonl               ~1000 items (1:1 pos/neg, from all three)

Multi-turn format (per session):
{
  "session_id": "...",
  "source": "real_raw|real_clean|synthetic",
  "subcategory": "W1",
  "language": "cn|en|mixed",
  "n_turns": 10,
  "turns": [
    {
      "turn_id": 0,
      "turn_text": "...",
      "annotation": {
        "demands": [...],
        "has_demand": true
      }
    },
    ...
  ]
}

Single-turn format (per item):
{
  "item_id": "...",
  "source": "...",
  "subcategory": "W1",
  "language": "cn|en|mixed",
  "context": "prior 3 turns (text only)",
  "current_turn": "...",
  "label": 1,  // 1=has demand, 0=no demand
  "demands": [...],  // populated only when label=1
}
"""

import json
import random
import logging
import argparse
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR  = Path(__file__).parent / "data"
FINAL_DIR = DATA_DIR / "final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Multi-turn formatting
# ------------------------------------------------------------------

def format_multiturn_session(session: dict) -> dict:
    """Strip internal metadata, keep benchmark-facing fields."""
    turns_out = []
    for t in session.get("turns", []):
        annotation = t.get("annotation", {"demands": [], "has_demand": False})
        turns_out.append({
            "turn_id":    t.get("turn_id", len(turns_out)),
            "turn_text":  t.get("turn_text", ""),
            "language":   t.get("language", session.get("language", "cn")),
            "annotation": {
                "demands":   annotation.get("demands", []),
                "has_demand": annotation.get("has_demand", False),
            },
        })
    return {
        "session_id":   session.get("session_id", ""),
        "source":       session.get("source", ""),
        "subcategory":  session.get("subcategory", ""),
        "language":     session.get("language", "cn"),
        "n_turns":      len(turns_out),
        "turns":        turns_out,
    }


# ------------------------------------------------------------------
# Single-turn extraction
# ------------------------------------------------------------------

def extract_singleturn_items(
    sessions: list[dict],
    source: str,
    pos_target: int,
    neg_target: int,
    context_window: int = 3,
) -> list[dict]:
    """
    Extract (context, current_turn, label) items.
    pos_target: how many positive items to extract
    neg_target: how many negative items to extract
    """
    positives = []
    negatives = []

    for session in sessions:
        turns = session.get("turns", [])
        for ti, turn in enumerate(turns):
            annotation = turn.get("annotation", {})
            has_demand = annotation.get("has_demand", False)
            demands    = annotation.get("demands", [])

            # Build context from prior turns
            ctx_turns = turns[max(0, ti - context_window): ti]
            context   = "\n\n".join(
                f"[Turn {t.get('turn_id', i)+1}]\n{t.get('turn_text', '')}"
                for i, t in enumerate(ctx_turns)
            )

            item = {
                "item_id":      f"{session.get('session_id', '')}_{ti}",
                "source":       source,
                "subcategory":  session.get("subcategory", ""),
                "language":     turn.get("language", session.get("language", "cn")),
                "context":      context,
                "current_turn": turn.get("turn_text", ""),
                "label":        1 if has_demand else 0,
                "demands":      demands if has_demand else [],
            }

            if has_demand:
                positives.append(item)
            else:
                negatives.append(item)

    random.shuffle(positives)
    random.shuffle(negatives)
    return positives[:pos_target] + negatives[:neg_target]


# ------------------------------------------------------------------
# Balance helpers
# ------------------------------------------------------------------

def balance_by_subcategory(sessions: list[dict], target_per_subcat: int) -> list[dict]:
    """Subsample to balance subcategory distribution."""
    by_subcat = defaultdict(list)
    for s in sessions:
        by_subcat[s.get("subcategory", "")].append(s)

    result = []
    for subcat, items in by_subcat.items():
        random.shuffle(items)
        result.extend(items[:target_per_subcat])
    return result


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run(
    real_raw_file:   Path = DATA_DIR / "03_annotated.jsonl",
    real_clean_file: Path = DATA_DIR / "03_annotated_clean.jsonl",
    synthetic_file:  Path = DATA_DIR / "05_annotated_synth.jsonl",
    target_per_bench: int = 1000,
    singleturn_total: int = 1000,
    seed: int = 42,
):
    random.seed(seed)

    def load(path: Path) -> list[dict]:
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return []
        items = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        logger.info(f"Loaded {len(items)} sessions from {path.name}")
        return items

    raw_sessions   = load(real_raw_file)
    clean_sessions = load(real_clean_file)
    synth_sessions = load(synthetic_file)

    # Per-subcat target
    n_subcats = 12
    tps = target_per_bench // n_subcats

    def write_multiturn(sessions, source, out_name):
        sessions = balance_by_subcategory(sessions, tps)
        random.shuffle(sessions)
        formatted = [format_multiturn_session(s) for s in sessions]
        out = FINAL_DIR / out_name
        with open(out, "w") as f:
            for s in formatted:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info(f"  Wrote {len(formatted)} sessions → {out.name}")
        return sessions  # return balanced for single-turn extraction

    logger.info("Writing multi-turn benchmarks ...")
    balanced_raw   = write_multiturn(raw_sessions,   "real_raw",   "bench_multiturn_real_raw.jsonl")
    balanced_clean = write_multiturn(clean_sessions, "real_clean", "bench_multiturn_real_clean.jsonl")
    balanced_synth = write_multiturn(synth_sessions, "synthetic",  "bench_multiturn_synthetic.jsonl")

    # Single-turn: 1:1 pos/neg, ~1/3 from each source
    logger.info("Extracting single-turn items ...")
    per_source     = singleturn_total // 3
    half           = per_source // 2

    st_items = []
    st_items += extract_singleturn_items(balanced_raw,   "real_raw",   half, half)
    st_items += extract_singleturn_items(balanced_clean, "real_clean", half, half)
    st_items += extract_singleturn_items(balanced_synth, "synthetic",  half, half)

    random.shuffle(st_items)

    # Balance: ensure exactly 1:1 pos/neg
    pos = [i for i in st_items if i["label"] == 1]
    neg = [i for i in st_items if i["label"] == 0]
    min_count = min(len(pos), len(neg), singleturn_total // 2)
    st_items = pos[:min_count] + neg[:min_count]
    random.shuffle(st_items)

    # Add sequential item_ids
    for i, item in enumerate(st_items):
        item["item_id"] = f"st_{i:04d}"

    st_out = FINAL_DIR / "bench_singleturn.jsonl"
    with open(st_out, "w") as f:
        for item in st_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"  Wrote {len(st_items)} single-turn items → {st_out.name}")
    logger.info(f"  Pos: {sum(1 for i in st_items if i['label']==1)}  Neg: {sum(1 for i in st_items if i['label']==0)}")

    # Summary stats
    logger.info("\n=== Final Benchmark Summary ===")
    for fname in ["bench_multiturn_real_raw.jsonl", "bench_multiturn_real_clean.jsonl",
                  "bench_multiturn_synthetic.jsonl", "bench_singleturn.jsonl"]:
        p = FINAL_DIR / fname
        if p.exists():
            lines = sum(1 for _ in open(p))
            logger.info(f"  {fname}: {lines} items")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format final benchmark output")
    parser.add_argument("--real-raw",       default=str(DATA_DIR / "03_annotated.jsonl"))
    parser.add_argument("--real-clean",     default=str(DATA_DIR / "03_annotated_clean.jsonl"))
    parser.add_argument("--synthetic",      default=str(DATA_DIR / "05_annotated_synth.jsonl"))
    parser.add_argument("--target",         type=int, default=1000)
    parser.add_argument("--singleturn",     type=int, default=1000)
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    run(
        real_raw_file=Path(args.real_raw),
        real_clean_file=Path(args.real_clean),
        synthetic_file=Path(args.synthetic),
        target_per_bench=args.target,
        singleturn_total=args.singleturn,
        seed=args.seed,
    )
