"""
Step 3a: Scene analysis — understand the session, identify characters, designate primary user.

Input:  data/03_annotated.jsonl (or any session file with turns)
Output: data/03a_scenes.jsonl (session_id → scene blueprint)

For each session, produces:
  - scene: what is this conversation about
  - characters: list of {name, role, background, personality}
  - primary_user: which character the AI serves, and why
  - speaker_diarization: which segments likely belong to which character
  - content_outline: key topics and turning points
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
INPUT_FILE  = DATA_DIR / "03_annotated.jsonl"
OUTPUT_FILE = DATA_DIR / "03a_scenes.jsonl"


SCENE_SYSTEM = """You are an expert conversation analyst. Given a raw ASR transcript, you must:

1. Figure out how many distinct people are speaking, and who they are.
   IMPORTANT: The tags [System]/[User] are just audio channels, NOT people.
   A single channel often contains multiple speakers mixed together.
   Identify speakers from CONTENT (speech style, pronouns, topic knowledge, dialogue cues like
   "你说得对", "I think...", question-answer patterns, etc.)

2. Create realistic character profiles with fictional names (for privacy).
   - Use culturally appropriate names matching the conversation language
   - Chinese conversations → Chinese names (张明, 李薇, etc.)
   - English conversations → English names (Alex, Sarah, etc.)
   - Mixed → match the dominant language

3. Designate a PRIMARY USER — the person the AI assistant would serve.
   - In a lecture/class: the student or learner
   - In a podcast/show: one of the hosts or a designated listener persona
   - In a meeting: one specific participant (pick whoever would benefit most from proactive help)
   - In a monologue (single speaker, e.g. watching a video): create a listener persona
     as the primary user who is consuming this content

4. Try to map segments to speakers (speaker diarization).
   This doesn't need to be perfect — do your best from content cues.

5. Outline the content structure: key topics, transitions, and notable moments.

Respond in the conversation's primary language. Output JSON only."""

SCENE_SCHEMA = """{
  "scene": "1-2 sentence scene description",
  "n_speakers": 2,
  "characters": [
    {"name": "张明", "role": "host", "background": "tech podcast host, 5+ years experience", "personality": "curious, structured thinker"},
    {"name": "李薇", "role": "guest", "background": "robotics engineer at a startup", "personality": "enthusiastic, detail-oriented"}
  ],
  "primary_user": {"name": "张明", "reason": "as the host, he needs proactive support to guide the conversation and catch important points"},
  "speaker_diarization_notes": "The transcript is single-channel [System]. Speaker changes are identifiable by dialogue cues...",
  "content_outline": [
    {"topic": "intro & background", "turns": "0-3", "notes": "host introduces guest"},
    {"topic": "core discussion", "turns": "4-20", "notes": "deep dive into robotics hands"}
  ]
}"""


def build_transcript_for_analysis(session: dict) -> str:
    """Build a readable transcript from session turns for scene analysis."""
    lines = []
    turns = session.get('turns', [])

    # Show all turns (truncate very long sessions)
    max_turns = min(len(turns), 60)
    for t in turns[:max_turns]:
        text = t.get('turn_text', t.get('text', ''))[:300]
        lines.append(f"[Turn {t['turn_id']}] {text}")

    if len(turns) > max_turns:
        lines.append(f"... ({len(turns) - max_turns} more turns omitted)")

    return "\n".join(lines)


def make_scene_messages(session: dict) -> list[dict]:
    """Build prompt for scene analysis."""
    transcript = build_transcript_for_analysis(session)

    memory = session.get('memory', '')
    memory_block = f"\n[Prior context / memory from earlier parts]\n{memory}\n" if memory else ""

    subcat = session.get('subcategory', '')
    lang = session.get('language', 'cn')
    n_turns = len(session.get('turns', []))
    part_info = f"Part {session.get('part_index', 0) + 1}/{session.get('total_parts', 1)}"

    user_content = f"""Analyze this conversation transcript and produce a scene blueprint.

Session info: subcategory={subcat}, language={lang}, {part_info}, {n_turns} turns
{memory_block}
[Full transcript]
{transcript}"""

    return [
        {"role": "system", "content": SCENE_SYSTEM},
        {"role": "user",   "content": user_content},
    ]


# ------------------------------------------------------------------
# Checkpoint
# ------------------------------------------------------------------

def load_checkpoint(output_file: Path) -> dict[str, dict]:
    done = {}
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    sid = obj.get("session_id", "")
                    if sid:
                        done[sid] = obj
    return done


def append_result(output_file: Path, obj: dict):
    with open(output_file, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

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

    done = load_checkpoint(output_file)
    if done:
        logger.info(f"Checkpoint: {len(done)} scenes already done, resuming...")
    else:
        output_file.write_text("")

    client = AsyncLLMClient(model=DEFAULT_MODEL, concurrency=concurrency, temperature=0.3, max_tokens=2048)
    total = len(sessions)

    for si, session in enumerate(sessions):
        sid = session.get("session_id", f"s{si}")

        if sid in done:
            continue

        try:
            messages = make_scene_messages(session)
            result = await client.complete_json(messages, schema_hint=SCENE_SCHEMA)
            if result is None:
                result = {}
        except Exception as e:
            logger.warning(f"Scene analysis failed for {sid}: {e}")
            result = {}

        obj = {"session_id": sid, "scene": result}
        append_result(output_file, obj)

        n_chars = len(result.get("characters", []))
        primary = result.get("primary_user", {}).get("name", "?")
        logger.info(f"[{si+1}/{total}] {sid[:40]}: {n_chars} characters, primary={primary}")

    logger.info(f"Done. {total} sessions processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",        default=str(INPUT_FILE))
    parser.add_argument("--output",       default=str(OUTPUT_FILE))
    parser.add_argument("--concurrency",  type=int, default=8)
    parser.add_argument("--max-sessions", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(run(Path(args.input), Path(args.output), args.concurrency, args.max_sessions))
