"""
Step 5: Generate synthetic conversations.

Output: data/05_synthetic.jsonl  (~1000 sessions)

Strategy:
- LLM generates full multi-turn conversations from scratch
- Each conversation targets a specific subcategory
- Built-in demand triggers (both Req and Ins) woven into dialogue naturally
- Language distribution: ~60% CN, ~20% EN, ~20% mixed (mirrors real data)
- Turn count: 8-15 turns per session
- 5-8 segments per turn (simulate real turn structure)
"""

import json
import asyncio
import logging
import random
import argparse
from pathlib import Path

from llm_client import AsyncLLMClient
from taxonomy import (
    CATEGORIES,
    SUBCATEGORY_DEMANDS,
    get_all_subcategory_codes,
    describe_subcategory,
    describe_demand_type,
    DEMAND_TYPES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR    = Path(__file__).parent / "data"
OUTPUT_FILE = DATA_DIR / "05_synthetic.jsonl"

TOTAL_TARGET  = 1000
TURNS_MIN     = 8
TURNS_MAX     = 15
SEGS_PER_TURN = (5, 8)

# Language distribution per subcategory (cn, en, mixed probabilities)
LANG_DIST = {
    "W1": (0.7, 0.1, 0.2), "W2": (0.4, 0.2, 0.4), "W3": (0.5, 0.2, 0.3), "W4": (0.7, 0.1, 0.2),
    "L1": (0.5, 0.2, 0.3), "L2": (0.4, 0.3, 0.3), "L3": (0.5, 0.2, 0.3), "L4": (0.3, 0.4, 0.3),
    "D1": (0.6, 0.2, 0.2), "D2": (0.7, 0.1, 0.2), "D3": (0.7, 0.1, 0.2), "D4": (0.5, 0.2, 0.3),
}


def pick_language(subcat: str) -> str:
    probs = LANG_DIST.get(subcat, (0.6, 0.2, 0.2))
    return random.choices(["cn", "en", "mixed"], weights=probs)[0]


# ------------------------------------------------------------------
# Synthetic generation prompts
# ------------------------------------------------------------------

SYNTH_SYSTEM = """You are a data generation expert creating realistic conversational AI training data.

Generate a realistic multi-turn conversation matching the given specifications.
The conversation should:
1. Sound natural, with realistic speech patterns (not overly formal)
2. Embed DEMAND TRIGGERS organically — the user's words should naturally invite proactive AI assistance
3. Mix Req (clear explicit) and Ins (subtle implicit) demand opportunities across turns
4. Follow the subcategory's typical scenario and domain

Output format — JSON with this structure:
{
  "scenario_description": "...",
  "turns": [
    {
      "turn_id": 0,
      "segments": [
        {"speaker": "microphone", "content": "..."},
        {"speaker": "system", "content": "..."},
        ...
      ],
      "embedded_demands": [
        {"demand_type": "decision_support", "category": "Req", "trigger_hint": "user mentioned X"}
      ]
    },
    ...
  ]
}

IMPORTANT: Do NOT include obvious explicit requests like "can you help me with X". Instead, weave triggers naturally into conversation."""


def make_synth_messages(
    subcat: str,
    language: str,
    n_turns: int,
    seed_demands: list[str],
) -> list[dict]:
    subcat_desc = describe_subcategory(subcat)
    demand_descs = "\n".join(f"  - {describe_demand_type(d)}" for d in seed_demands)
    lang_instruction = {
        "cn":    "Write the entire conversation in Chinese (Mandarin).",
        "en":    "Write the entire conversation in English.",
        "mixed": "Write in a natural mix of Chinese and English (code-switching), as tech professionals often do.",
    }[language]

    user_content = f"""Generate a {n_turns}-turn conversation for subcategory: {subcat_desc}

Language: {lang_instruction}

Embed at least 3-5 demand triggers across the turns. Focus on these demand types:
{demand_descs}

Each turn should have {SEGS_PER_TURN[0]}-{SEGS_PER_TURN[1]} segments (alternating or grouped speech).
Speakers: "microphone" = user (human), "system" = AI assistant or environment.

Make the conversation realistic — include hesitations, topic shifts, and natural dialogue flow."""

    return [
        {"role": "system", "content": SYNTH_SYSTEM},
        {"role": "user",   "content": user_content},
    ]


SYNTH_SCHEMA = '{"scenario_description": "...", "turns": [{"turn_id": 0, "segments": [{"speaker": "microphone", "content": "..."}], "embedded_demands": []}]}'


def post_process_synthetic(raw: dict, subcat: str, language: str, idx: int) -> dict:
    """Convert LLM output to standard session format."""
    turns_raw = raw.get("turns", [])
    turns = []
    for t in turns_raw:
        segs = t.get("segments", [])
        parts = []
        for s in segs:
            speaker = s.get("speaker", "microphone")
            label = "User" if speaker == "microphone" else "System"
            content = s.get("content", "").strip()
            if content:
                parts.append(f"[{label}] {content}")

        turn_dict = {
            "session_id":    f"synth_{idx:04d}",
            "turn_id":       t.get("turn_id", len(turns)),
            "turn_text":     "\n".join(parts),
            "n_segments":    len(segs),
            "segment_ids":   list(range(len(segs))),
            "speaker_mix":   {
                "user":   sum(1 for s in segs if s.get("speaker") == "microphone"),
                "system": sum(1 for s in segs if s.get("speaker") != "microphone"),
            },
            "language":      language,
            "embedded_demands": t.get("embedded_demands", []),
        }
        turns.append(turn_dict)

    return {
        "session_id":       f"synth_{idx:04d}",
        "subcategory":      subcat,
        "quality_score":    0.8,  # synthetic default
        "language":         language,
        "demand_density":   "high",
        "n_turns":          len(turns),
        "turns":            turns,
        "source":           "synthetic",
        "scenario_description": raw.get("scenario_description", ""),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

async def run(
    output_file: Path = OUTPUT_FILE,
    total: int = TOTAL_TARGET,
    concurrency: int = 6,  # slightly lower for generation tasks
):
    subcats = get_all_subcategory_codes()
    per_subcat = total // len(subcats)  # ~83 per subcategory
    remainder  = total - per_subcat * len(subcats)

    # Build task list
    tasks      = []
    tasks_meta = []  # (subcat, language, idx)
    idx = 0

    for i, subcat in enumerate(subcats):
        count = per_subcat + (1 if i < remainder else 0)
        demand_types = SUBCATEGORY_DEMANDS.get(subcat, [])
        # Exclude "other" from seeding
        seedable = [d for d in demand_types if d != "other"]

        for _ in range(count):
            language = pick_language(subcat)
            n_turns  = random.randint(TURNS_MIN, TURNS_MAX)

            # Randomly sample 3-4 demand types for this conversation
            n_seed = min(4, len(seedable))
            seed = random.sample(seedable, n_seed) if len(seedable) >= n_seed else seedable

            tasks.append({
                "messages":    make_synth_messages(subcat, language, n_turns, seed),
                "schema_hint": SYNTH_SCHEMA,
                "max_tokens":  3000,
                "temperature": 0.85,  # higher for diversity
            })
            tasks_meta.append((subcat, language, idx))
            idx += 1

    logger.info(f"Generating {len(tasks)} synthetic sessions ...")
    client = AsyncLLMClient(concurrency=concurrency, temperature=0.85, max_tokens=3000)
    results = await client.batch(tasks)

    output_sessions = []
    failures = 0
    for (subcat, language, sidx), raw in zip(tasks_meta, results):
        if raw is None or not isinstance(raw, dict) or not raw.get("turns"):
            failures += 1
            continue
        try:
            session = post_process_synthetic(raw, subcat, language, sidx)
            if session["n_turns"] >= 3:
                output_sessions.append(session)
        except Exception as e:
            logger.warning(f"Post-processing failed for synth_{sidx:04d}: {e}")
            failures += 1

    with open(output_file, "w") as f:
        for s in output_sessions:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(output_sessions)} synthetic sessions to {output_file}")
    logger.info(f"Failures: {failures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic conversations")
    parser.add_argument("--output",      default=str(OUTPUT_FILE))
    parser.add_argument("--total",       type=int, default=TOTAL_TARGET)
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()

    asyncio.run(run(
        output_file=Path(args.output),
        total=args.total,
        concurrency=args.concurrency,
    ))
