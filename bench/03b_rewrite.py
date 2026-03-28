"""
Step 3b: Conversation rewrite — clean ASR text, add character attribution, remove privacy info.

Input:  data/03_annotated.jsonl (raw turns) + data/03a_scenes.jsonl (scene blueprints)
Output: data/03b_rewritten.jsonl (clean sessions with character-attributed turns)

Each session is rewritten in chunks of ~15 turns. The LLM:
  - Fixes ASR errors while preserving meaning
  - Attributes each utterance to a named character from the blueprint
  - Removes/anonymizes privacy-sensitive content
  - May merge fragment turns or split overly long ones
  - Maintains natural conversation flow
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
SCENE_FILE  = DATA_DIR / "03a_scenes.jsonl"
OUTPUT_FILE = DATA_DIR / "03b_rewritten.jsonl"

CHUNK_SIZE  = 15  # turns per rewrite chunk
OVERLAP     = 2   # overlap with previous chunk for continuity


REWRITE_SYSTEM = """You are a professional conversation editor. You will be given:
1. A scene blueprint (characters, setting, speaker notes)
2. A chunk of raw ASR transcript (noisy, may have errors, fragments, no proper speaker labels)

Your task: rewrite this chunk into a clean, natural conversation with proper speaker attribution.

RULES:
- Attribute each line to a specific character from the blueprint, using their name
- Format: "CharacterName: dialogue text" — one speaker per line
- Fix ASR errors (garbled text, repeated phrases, broken sentences) while preserving the MEANING
- Merge very short fragments into complete sentences when they clearly belong together
- Split a turn if it clearly contains two different speakers
- Remove or replace any real names, phone numbers, company names, addresses with fictional ones
- Keep the conversation's language (Chinese stays Chinese, English stays English, mixed stays mixed)
- Do NOT add content that wasn't in the original — only clean and restructure
- For segments that are completely unintelligible, write: "[unintelligible]" and skip
- Keep the original tone and register (casual stays casual, formal stays formal)
- If there's a primary user who is a listener/observer (not speaking), you may add brief
  reactions or internal thoughts in parentheses: "PrimaryUser: (thinking: this contradicts what was said earlier)"
  — but ONLY where it's natural and adds value, not every turn

Output a JSON array of turns. Each turn: {"speaker": "CharacterName", "text": "clean dialogue"}
Output JSON only."""


def build_rewrite_prompt(session: dict, scene: dict, chunk_start: int, chunk_end: int, prev_context: str) -> list[dict]:
    """Build prompt for rewriting a chunk of turns."""
    turns = session.get('turns', [])
    chunk = turns[chunk_start:chunk_end]

    # Build scene context
    characters = scene.get('characters', [])
    char_desc = "\n".join(
        f"  - {c['name']} ({c.get('role','')}): {c.get('background','')}"
        for c in characters
    )
    primary = scene.get('primary_user', {})
    primary_desc = f"{primary.get('name','?')} — {primary.get('reason','')}" if primary else "unknown"
    diarization = scene.get('speaker_diarization_notes', '')

    # Build raw chunk text
    raw_lines = []
    for t in chunk:
        text = t.get('turn_text', t.get('text', ''))
        raw_lines.append(f"[Turn {t['turn_id']}] {text}")
    raw_text = "\n".join(raw_lines)

    user_content = f"""[Scene]
{scene.get('scene', '')}

[Characters]
{char_desc}

[Primary user (AI serves this person)]
{primary_desc}

[Speaker identification notes]
{diarization[:500]}

{f"[Previous context (last 2 turns of prior chunk)]{chr(10)}{prev_context}{chr(10)}" if prev_context else ""}
[Raw ASR transcript to rewrite — {len(chunk)} turns]
{raw_text}

Rewrite this into clean, properly attributed dialogue. Output JSON array only."""

    return [
        {"role": "system", "content": REWRITE_SYSTEM},
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
                    if sid and obj.get("turns"):
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
    scene_file:   Path = SCENE_FILE,
    output_file:  Path = OUTPUT_FILE,
    concurrency:  int  = 8,
    max_sessions: int  = 0,
):
    logger.info(f"Loading sessions from {input_file}")
    sessions = []
    with open(input_file) as f:
        for line in f:
            if line.strip():
                sessions.append(json.loads(line))

    logger.info(f"Loading scenes from {scene_file}")
    scenes = {}
    with open(scene_file) as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                scenes[obj["session_id"]] = obj.get("scene", {})

    if max_sessions > 0:
        sessions = sessions[:max_sessions]

    done = load_checkpoint(output_file)
    if done:
        logger.info(f"Checkpoint: {len(done)} sessions already rewritten, resuming...")
    else:
        output_file.write_text("")

    client = AsyncLLMClient(model=DEFAULT_MODEL, concurrency=concurrency, temperature=0.25, max_tokens=4096)
    total = len(sessions)

    for si, session in enumerate(sessions):
        sid = session.get("session_id", f"s{si}")

        if sid in done:
            continue

        scene = scenes.get(sid, {})
        if not scene:
            logger.warning(f"No scene for {sid}, skipping")
            continue

        turns = session.get('turns', [])
        n_turns = len(turns)

        # Process in chunks
        all_rewritten = []
        prev_context = ""

        for chunk_start in range(0, n_turns, CHUNK_SIZE - OVERLAP):
            chunk_end = min(chunk_start + CHUNK_SIZE, n_turns)
            # Skip if this chunk is entirely overlap from previous
            if chunk_start > 0 and chunk_end <= chunk_start + OVERLAP:
                break

            messages = build_rewrite_prompt(session, scene, chunk_start, chunk_end, prev_context)

            try:
                result = await client.complete_json(messages)
                if result is None:
                    result = []
            except Exception as e:
                logger.warning(f"Rewrite failed for {sid} chunk {chunk_start}-{chunk_end}: {e}")
                result = []

            # If this isn't the first chunk, skip overlap turns
            if chunk_start > 0 and isinstance(result, list) and len(result) > OVERLAP:
                result = result[OVERLAP:]
            elif chunk_start > 0 and isinstance(result, list):
                pass  # chunk too short, keep all

            if isinstance(result, list):
                all_rewritten.extend(result)

            # Build prev_context for next chunk
            if isinstance(result, list) and len(result) >= 2:
                last_two = result[-2:]
                prev_context = "\n".join(
                    f"{t.get('speaker','?')}: {t.get('text','')[:100]}"
                    for t in last_two
                )
            elif isinstance(result, list) and len(result) == 1:
                prev_context = f"{result[0].get('speaker','?')}: {result[0].get('text','')[:100]}"

        # Assign turn_ids
        for i, t in enumerate(all_rewritten):
            t['turn_id'] = i

        # Build output session
        out_session = {
            "session_id": sid,
            "original_session_id": session.get("original_session_id", ""),
            "subcategory": session.get("subcategory", ""),
            "language": session.get("language", ""),
            "part_index": session.get("part_index", 0),
            "total_parts": session.get("total_parts", 1),
            "scene": scene,
            "n_turns": len(all_rewritten),
            "turns": all_rewritten,
        }

        append_result(output_file, out_session)

        logger.info(f"[{si+1}/{total}] {sid[:40]}: {n_turns} raw → {len(all_rewritten)} clean turns")

    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",        default=str(INPUT_FILE))
    parser.add_argument("--scenes",       default=str(SCENE_FILE))
    parser.add_argument("--output",       default=str(OUTPUT_FILE))
    parser.add_argument("--concurrency",  type=int, default=8)
    parser.add_argument("--max-sessions", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(run(Path(args.input), Path(args.scenes), Path(args.output), args.concurrency, args.max_sessions))
