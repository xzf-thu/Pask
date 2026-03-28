"""
Step 3c: Memory generation — create rich contextual memory for multi-part sessions.

Input:  data/03b_rewritten.jsonl (clean sessions with scene blueprints)
Output: data/03c_memory.jsonl (sessions with updated memory field)

For multi-part sessions (part_index > 0), generates memory that:
  - References characters by name
  - Notes unresolved questions, pending decisions
  - Records key facts/data points
  - Captures emotional states and relationships
  - Tracks what the primary_user knows/doesn't know

For part_index == 0 sessions, copies through unchanged.
"""

import json
import asyncio
import logging
import argparse
from pathlib import Path

from llm_client import AsyncLLMClient, DEFAULT_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR    = Path(__file__).parent / "data"
INPUT_FILE  = DATA_DIR / "03b_rewritten.jsonl"
OUTPUT_FILE = DATA_DIR / "03c_memory.jsonl"


MEMORY_SYSTEM = """You generate rich contextual memory for a conversation that continues from a prior session.

You will be given:
1. Scene blueprint (characters, primary user)
2. The prior part's conversation (clean, attributed dialogue)
3. The current part's first few turns (to understand what comes next)

Generate a MEMORY document that the AI assistant would have from observing the prior conversation.
This memory should be written from the perspective of an AI serving the PRIMARY USER.

Include:
- What was discussed (key topics, decisions, data points) — reference characters by name
- Unresolved questions or pending decisions ("张明 proposed Plan A but 李薇 hasn't agreed yet")
- Important facts/numbers that might be relevant later
- Emotional dynamics ("张明 seemed frustrated about the timeline")
- What the primary user knows/believes at this point
- What the primary user might NOT realize (blind spots, things others hinted at)

Write in the conversation's language. Be specific and concrete, not generic summaries.
Output a single JSON object: {"memory": "the memory text"}"""


def build_memory_prompt(prior_session: dict, current_session: dict) -> list[dict]:
    """Build prompt for memory generation."""
    scene = current_session.get('scene', prior_session.get('scene', {}))

    # Scene context
    characters = scene.get('characters', [])
    char_desc = "\n".join(
        f"  - {c['name']} ({c.get('role','')}): {c.get('background','')}"
        for c in characters
    )
    primary = scene.get('primary_user', {})
    primary_name = primary.get('name', '?')

    # Prior conversation
    prior_turns = prior_session.get('turns', [])
    prior_text = "\n".join(
        f"{t.get('speaker','?')}: {t.get('text','')[:200]}"
        for t in prior_turns
    )

    # Current first few turns (for context)
    current_turns = current_session.get('turns', [])[:5]
    current_preview = "\n".join(
        f"{t.get('speaker','?')}: {t.get('text','')[:200]}"
        for t in current_turns
    ) if current_turns else "(no preview available)"

    user_content = f"""[Characters]
{char_desc}

[Primary user: {primary_name}]

[PRIOR CONVERSATION (what happened before)]
{prior_text}

[NEXT PART PREVIEW (first few turns of what comes next)]
{current_preview}

Generate a detailed memory document for the AI assistant serving {primary_name}."""

    return [
        {"role": "system", "content": MEMORY_SYSTEM},
        {"role": "user",   "content": user_content},
    ]


async def run(
    input_file:   Path = INPUT_FILE,
    output_file:  Path = OUTPUT_FILE,
    concurrency:  int  = 8,
    max_sessions: int  = 0,
):
    logger.info(f"Loading from {input_file}")
    sessions = []
    with open(input_file) as f:
        for line in f:
            if line.strip():
                sessions.append(json.loads(line))

    if max_sessions > 0:
        sessions = sessions[:max_sessions]

    # Group by original_session_id to find prior parts
    by_original = {}
    for s in sessions:
        oid = s.get('original_session_id', s.get('session_id', ''))
        pi = s.get('part_index', 0)
        by_original.setdefault(oid, {})[pi] = s

    output_file.write_text("")
    client = AsyncLLMClient(model=DEFAULT_MODEL, concurrency=concurrency, temperature=0.2, max_tokens=2048)
    total = len(sessions)
    mem_count = 0

    for si, session in enumerate(sessions):
        sid = session.get("session_id", f"s{si}")
        part_index = session.get("part_index", 0)

        if part_index == 0:
            # First part — no memory needed
            session["memory"] = ""
            with open(output_file, "a") as f:
                f.write(json.dumps(session, ensure_ascii=False) + "\n")
            logger.info(f"[{si+1}/{total}] {sid[:40]}: part 0, no memory needed")
            continue

        # Find prior part
        oid = session.get('original_session_id', '')
        prior = by_original.get(oid, {}).get(part_index - 1)

        if not prior:
            # No prior part found, use existing memory or empty
            session.setdefault("memory", "")
            with open(output_file, "a") as f:
                f.write(json.dumps(session, ensure_ascii=False) + "\n")
            logger.info(f"[{si+1}/{total}] {sid[:40]}: part {part_index}, no prior part found")
            continue

        # Generate memory
        try:
            messages = build_memory_prompt(prior, session)
            result = await client.complete_json(messages, schema_hint='{"memory": "..."}')
            memory_text = result.get("memory", "") if result else ""
        except Exception as e:
            logger.warning(f"Memory generation failed for {sid}: {e}")
            memory_text = session.get("memory", "")

        session["memory"] = memory_text
        mem_count += 1

        with open(output_file, "a") as f:
            f.write(json.dumps(session, ensure_ascii=False) + "\n")

        logger.info(f"[{si+1}/{total}] {sid[:40]}: part {part_index}, memory generated ({len(memory_text)} chars)")

    logger.info(f"Done. {total} sessions, {mem_count} memories generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",        default=str(INPUT_FILE))
    parser.add_argument("--output",       default=str(OUTPUT_FILE))
    parser.add_argument("--concurrency",  type=int, default=8)
    parser.add_argument("--max-sessions", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(run(Path(args.input), Path(args.output), args.concurrency, args.max_sessions))
