"""
Step 2: Build semantic turns from sub-sessions.

Input:  data/01b_subsessions.jsonl
Output: data/02_turns.jsonl

Turn definition:
  - Group consecutive segments into sentence-groups
  - Each turn: ends at a sentence-ending segment AND has accumulated >= MIN_TURN_CHARS
  - Result: each turn ~ 40-120 chars (one complete thought)

Part splitting (for multi-part sessions):
  - Each session is capped at MAX_TURNS_PER_PART turns per benchmark item
  - If a session has more turns, split into Part 0, Part 1, ...
  - Part N>0 gets a `memory` field = gpt-5-nano summary of all previous parts

Memory generation:
  - Input: all turns from previous parts (text only)
  - Output: ~200-300 char summary capturing key topics, decisions, open questions
"""

import json
import re
import asyncio
import logging
import argparse
from pathlib import Path

from llm_client import AsyncLLMClient, NANO_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR    = Path(__file__).parent / "data"
INPUT_FILE  = DATA_DIR / "01b_subsessions.jsonl"
OUTPUT_FILE = DATA_DIR / "02_turns.jsonl"

MIN_TURN_CHARS    = 40    # minimum chars before a turn can end
MAX_TURN_CHARS    = 150   # force-end turn regardless of punctuation
MAX_TURNS_PER_PART = 60   # benchmark item turn cap


# ------------------------------------------------------------------
# Semantic sentence-group turn splitting
# ------------------------------------------------------------------

_SENT_END = re.compile(r'[。！？.!?]["」)）]?\s*$')


def build_semantic_turns(segs: list[dict]) -> list[list[dict]]:
    """
    Group segments into semantic turns.
    A turn ends when:
      - accumulated chars >= MIN_TURN_CHARS AND current segment ends a sentence
      - OR accumulated chars >= MAX_TURN_CHARS (force break)
    """
    turns = []
    current = []
    current_chars = 0

    for seg in segs:
        content = seg.get('content', '').strip()
        current.append(seg)
        current_chars += len(content)

        at_sentence_end = bool(_SENT_END.search(content))
        force_break     = current_chars >= MAX_TURN_CHARS

        if (at_sentence_end and current_chars >= MIN_TURN_CHARS) or force_break:
            turns.append(current)
            current = []
            current_chars = 0

    if current:  # flush remainder
        if turns and len(current) < 3:
            turns[-1] = turns[-1] + current  # merge tiny tail
        else:
            turns.append(current)

    return turns


def format_turn(turn_idx: int, segs: list[dict], language: str) -> dict:
    parts = []
    for s in segs:
        label   = "User" if s.get('speaker') == 'microphone' else "System"
        content = s.get('content', '').strip()
        if content:
            parts.append(f"[{label}] {content}")
    return {
        "turn_id":    turn_idx,
        "turn_text":  "\n".join(parts),
        "n_segments": len(segs),
        "language":   language,
    }


# ------------------------------------------------------------------
# Memory generation
# ------------------------------------------------------------------

MEMORY_SYSTEM = """Summarize the following conversation transcript in 2-4 sentences (200-300 characters).
Capture: main topics discussed, key facts or decisions established, any unresolved questions.
Be concise and factual. Output the summary only, no preamble."""


def make_memory_messages(turns: list[dict]) -> list[dict]:
    text = "\n\n".join(
        f"[Turn {t['turn_id']+1}] {t['turn_text'][:200]}"
        for t in turns
    )
    return [
        {"role": "system", "content": MEMORY_SYSTEM},
        {"role": "user",   "content": text},
    ]


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

async def run(
    input_file:  Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE,
    concurrency: int  = 6,
):
    logger.info(f"Loading from {input_file}")
    subsessions = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                subsessions.append(json.loads(line))

    logger.info(f"Building semantic turns for {len(subsessions)} sub-sessions ...")

    # Step 1: build turns for all sub-sessions (no LLM needed)
    all_records = []
    memory_tasks      = []   # (record_index, turns_to_summarize)
    memory_task_meta  = []

    for ss in subsessions:
        segs     = ss.get('segments', [])
        language = ss.get('language', 'cn')
        turns_segs = build_semantic_turns(segs)

        # Format all turns
        all_turns = [format_turn(i, s, language) for i, s in enumerate(turns_segs)]

        # Split into parts of MAX_TURNS_PER_PART
        n_parts = max(1, (len(all_turns) + MAX_TURNS_PER_PART - 1) // MAX_TURNS_PER_PART)

        for p in range(n_parts):
            start  = p * MAX_TURNS_PER_PART
            end    = min(start + MAX_TURNS_PER_PART, len(all_turns))
            chunk  = all_turns[start:end]

            # Re-index turn_ids within this part
            for j, t in enumerate(chunk):
                t = dict(t)
                t['turn_id'] = j
                chunk[j] = t

            record = {
                "session_id":          f"{ss['session_id']}_p{p}",
                "original_session_id": ss.get('original_session_id', ss['session_id']),
                "subcategory":         ss.get('subcategory', ''),
                "quality_score":       ss.get('quality_score', 0),
                "language":            language,
                "demand_density":      ss.get('demand_density', ''),
                "source":              ss.get('source', 'real_raw'),
                "part_index":          p,
                "total_parts":         n_parts,
                "memory":              None,   # filled in later for p > 0
                "n_turns":             len(chunk),
                "turns":               chunk,
            }
            all_records.append(record)

            # Schedule memory generation for non-first parts
            if p > 0:
                prev_turns = all_turns[:start]  # all turns before this part
                memory_tasks.append({"messages": make_memory_messages(prev_turns), "max_tokens": 200, "temperature": 0.2})
                memory_task_meta.append(len(all_records) - 1)  # index in all_records

    # Step 2: generate memories in batch
    if memory_tasks:
        logger.info(f"Generating {len(memory_tasks)} memory summaries ...")
        client = AsyncLLMClient(model=NANO_MODEL, concurrency=concurrency, temperature=0.2, max_tokens=200)
        memories = await client.batch(memory_tasks)
        for rec_idx, memory in zip(memory_task_meta, memories):
            if memory:
                all_records[rec_idx]['memory'] = memory.strip()

    # Write output
    with open(output_file, 'w') as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    total_turns = sum(r['n_turns'] for r in all_records)
    with_memory = sum(1 for r in all_records if r['memory'])
    logger.info(f"Saved {len(all_records)} benchmark sessions to {output_file}")
    logger.info(f"  Total turns: {total_turns}  |  Sessions with memory: {with_memory}")
    turn_counts = [r['n_turns'] for r in all_records]
    logger.info(f"  Turns/session: min={min(turn_counts)} max={max(turn_counts)} avg={sum(turn_counts)/len(turn_counts):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       default=str(INPUT_FILE))
    parser.add_argument("--output",      default=str(OUTPUT_FILE))
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()
    asyncio.run(run(Path(args.input), Path(args.output), args.concurrency))
