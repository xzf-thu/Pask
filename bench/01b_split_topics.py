"""
Step 1b: Split sessions into topic-coherent sub-sessions.

Input:  data/01_filtered.jsonl
Output: data/01b_subsessions.jsonl

Strategy:
- Slide a window through the session segments
- Use gpt-5-nano to detect topic boundary at candidate points
- Candidate points: every ~50 segments (at a sentence-ending segment)
- Hard constraints: sub-session must be 3k-12k chars
- Very long stretches without a topic change → force-split at 12k chars

Each sub-session gets:
  session_id: "{original_id}_s{n}"
  part metadata preserved from parent
"""

import json
import asyncio
import logging
import argparse
import re
from pathlib import Path

from llm_client import AsyncLLMClient, NANO_MODEL
from taxonomy import get_all_subcategory_codes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR    = Path(__file__).parent / "data"
INPUT_FILE  = DATA_DIR / "01_filtered.jsonl"
OUTPUT_FILE = DATA_DIR / "01b_subsessions.jsonl"

MIN_SUBSES_CHARS = 3_000
MAX_SUBSES_CHARS = 12_000
WINDOW_SEGS      = 40   # check for topic change every ~40 segments
CONTEXT_SEGS     = 10   # segments of overlap shown to LLM for context


# ------------------------------------------------------------------
# Heuristic: find sentence-ending segment near a candidate index
# ------------------------------------------------------------------

_SENT_END = re.compile(r'[。！？.!?]\s*$')

def find_sentence_end(segs: list[dict], start: int, search_range: int = 15) -> int:
    """Find nearest sentence-ending segment at or after `start` within range."""
    for i in range(start, min(start + search_range, len(segs))):
        if _SENT_END.search(segs[i].get('content', '')):
            return i
    return start  # fallback: use start as-is


# ------------------------------------------------------------------
# LLM topic boundary detection
# ------------------------------------------------------------------

TOPIC_SYSTEM = """You are analyzing a conversation transcript to find topic boundaries.

Given two consecutive text windows (A = just before the candidate split point, B = just after),
decide if there is a meaningful topic change between them.

A "topic change" means the conversation shifts to a substantially different subject, question, or task.
Minor tangents or continuations of the same theme are NOT topic changes.

Respond with JSON only: {"is_topic_change": true/false, "confidence": 0.0-1.0, "reason": "one sentence"}"""

def make_topic_detection_messages(segs_before: list[dict], segs_after: list[dict]) -> list[dict]:
    def fmt(segs):
        return " ".join(s.get('content', '').strip() for s in segs if s.get('content', '').strip())

    text_a = fmt(segs_before[-CONTEXT_SEGS:])[:400]
    text_b = fmt(segs_after[:CONTEXT_SEGS])[:400]

    return [
        {"role": "system", "content": TOPIC_SYSTEM},
        {"role": "user",   "content": f"Window A (before split):\n{text_a}\n\nWindow B (after split):\n{text_b}"},
    ]


# ------------------------------------------------------------------
# Main splitting logic
# ------------------------------------------------------------------

def chars_of(segs: list[dict]) -> int:
    return sum(len(s.get('content', '')) for s in segs)


async def split_session(
    session: dict,
    client: AsyncLLMClient,
) -> list[dict]:
    """Split a single session into topic-coherent sub-sessions."""
    segs = session.get('segments', [])
    total_chars = chars_of(segs)

    # Short session: no split needed
    if total_chars <= MAX_SUBSES_CHARS:
        return [_make_subsession(session, segs, 0)]

    # Find candidate split points: every WINDOW_SEGS segments (at sentence ends)
    candidates = []
    i = WINDOW_SEGS
    while i < len(segs) - WINDOW_SEGS:
        # Only candidate if enough chars have accumulated
        chars_so_far = chars_of(segs[:i])
        if chars_so_far >= MIN_SUBSES_CHARS:
            actual = find_sentence_end(segs, i)
            candidates.append(actual)
        i += WINDOW_SEGS

    if not candidates:
        return [_make_subsession(session, segs, 0)]

    # Ask LLM about each candidate (in batch)
    tasks = []
    for c in candidates:
        msgs = make_topic_detection_messages(segs[:c], segs[c:])
        tasks.append({"messages": msgs, "schema_hint": '{"is_topic_change": true, "confidence": 0.8, "reason": "..."}', "max_tokens": 100, "temperature": 0.1})

    results = await client.batch(tasks)

    # Select split points: topic changes with confidence > 0.6 AND size constraints
    split_points = [0]  # always start at 0
    for c, result in zip(candidates, results):
        if result is None:
            continue
        is_change   = result.get('is_topic_change', False)
        confidence  = float(result.get('confidence', 0))

        chars_since_last = chars_of(segs[split_points[-1]:c])
        chars_remaining  = chars_of(segs[c:])

        # Force split if current chunk exceeds MAX even without topic change
        if chars_since_last >= MAX_SUBSES_CHARS:
            split_points.append(c)
        elif is_change and confidence >= 0.6 and chars_since_last >= MIN_SUBSES_CHARS and chars_remaining >= MIN_SUBSES_CHARS:
            split_points.append(c)

    split_points.append(len(segs))  # end

    # Build sub-sessions
    subsessions = []
    for idx in range(len(split_points) - 1):
        start = split_points[idx]
        end   = split_points[idx + 1]
        chunk = segs[start:end]
        if chars_of(chunk) >= MIN_SUBSES_CHARS // 2:  # skip tiny tail chunks
            subsessions.append(_make_subsession(session, chunk, idx))

    return subsessions if subsessions else [_make_subsession(session, segs, 0)]


def _make_subsession(session: dict, segs: list[dict], idx: int) -> dict:
    meta = session.get('_meta', {})
    return {
        'session_id':          f"{session['id']}_s{idx}",
        'original_session_id': session['id'],
        'sub_index':           idx,
        'subcategory':         meta.get('subcategory', ''),
        'quality_score':       meta.get('quality_score', 0),
        'language':            meta.get('language', session.get('language', 'cn')),
        'demand_density':      meta.get('demand_density', ''),
        'n_segments':          len(segs),
        'total_chars':         chars_of(segs),
        'segments':            segs,
        'source':              'real_raw',
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

async def run(
    input_file:  Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE,
    concurrency: int  = 6,
):
    logger.info(f"Loading from {input_file}")
    sessions = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                sessions.append(json.loads(line))

    logger.info(f"Splitting {len(sessions)} sessions into topic-coherent sub-sessions ...")
    client = AsyncLLMClient(model=NANO_MODEL, concurrency=concurrency, temperature=0.1, max_tokens=100)

    # Process sessions sequentially but LLM calls within each session are batched
    all_subsessions = []
    for session in sessions:
        subs = await split_session(session, client)
        all_subsessions.extend(subs)

    with open(output_file, 'w') as f:
        for s in all_subsessions:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    logger.info(f"Saved {len(all_subsessions)} sub-sessions to {output_file}")
    logger.info(f"  Original sessions: {len(sessions)}  →  Sub-sessions: {len(all_subsessions)}  (×{len(all_subsessions)/len(sessions):.1f})")

    # Size distribution
    sizes = [s['total_chars'] for s in all_subsessions]
    logger.info(f"  Sub-session chars: min={min(sizes)} max={max(sizes):,} avg={sum(sizes)/len(sizes):.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       default=str(INPUT_FILE))
    parser.add_argument("--output",      default=str(OUTPUT_FILE))
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()
    asyncio.run(run(Path(args.input), Path(args.output), args.concurrency))
