"""
Step 3: LLM demand annotation (gpt-5.2 via OpenRouter).

Input:  data/02_turns.jsonl
Output: data/03_annotated.jsonl

Supports checkpoint/resume: saves after each session, skips already-annotated sessions on restart.
"""

import json
import asyncio
import logging
import argparse
from pathlib import Path

from llm_client import AsyncLLMClient, DEFAULT_MODEL
from taxonomy import (
    DEMAND_TYPES,
    get_demand_types_for,
    describe_subcategory,
    describe_demand_type,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR    = Path(__file__).parent / "data"
INPUT_FILE  = DATA_DIR / "02_turns.jsonl"
OUTPUT_FILE = DATA_DIR / "03_annotated.jsonl"


# ------------------------------------------------------------------
# Session metadata prompt (topic + participant profiles)
# ------------------------------------------------------------------

META_SYSTEM = """You analyze conversation transcripts and extract structured metadata.
Output JSON only."""

META_SCHEMA = '{"topic": "1-2 sentence summary of the conversation topic", "participants": [{"role": "e.g. host/guest/lecturer/student", "description": "brief background/expertise description"}], "speaker_map": {"[System]": "who this label represents", "[Microphone]": "who this label represents"}}'


def make_meta_messages(session: dict) -> list[dict]:
    """Build prompt to extract session metadata (topic + participant profiles + speaker mapping)."""
    turns = session['turns']
    # Use first 15 turns + memory for metadata extraction
    sample_turns = turns[:min(15, len(turns))]
    transcript = "\n".join(f"[Turn {t['turn_id']+1}] {t['turn_text'][:200]}" for t in sample_turns)

    memory = session.get('memory', '')
    memory_block = f"\n[Prior context summary]\n{memory}\n" if memory else ""

    subcat = session.get('subcategory', '')
    lang = session.get('language', 'cn')

    # Collect unique speaker tags from turns
    speaker_tags = set()
    import re as _re
    for t in turns:
        found = _re.findall(r'\[(?:System|User|Microphone|Speaker\d*)\]', t.get('turn_text', ''))
        speaker_tags.update(found)

    speaker_hint = ", ".join(sorted(speaker_tags)) if speaker_tags else "[System]"

    user_content = f"""Analyze this conversation and extract:
1. **topic**: What is this conversation about? (1-2 sentences, use the conversation's language: {lang})
2. **participants**: Who are the participants? Their roles and brief background.
   Infer from content — e.g. if someone explains code, they're likely a developer/instructor.
3. **speaker_map**: The transcript uses ASR audio channel tags: {speaker_hint}.
   [System] = system audio channel (podcast playback, remote meeting audio, video, etc.)
   [User] = local microphone channel (person physically near the recording device)
   Some sessions only have one channel (single audio source).
   When both exist, [System] is the content being played/received, [User] is the local person.
   Based on the CONTENT, infer who each tag actually represents.
   e.g. if [System] delivers a lecture and [User] asks questions →
   speaker_map: {{"[System]": "lecturer", "[User]": "student"}}
   If only [System] exists with multiple speakers, note that in the mapping.

Subcategory: {describe_subcategory(subcat) if subcat else 'unknown'}
{memory_block}
[Transcript sample]
{transcript}"""

    return [
        {"role": "system", "content": META_SYSTEM},
        {"role": "user",   "content": user_content},
    ]


# ------------------------------------------------------------------
# Annotation prompt (with reasoning chain)
# ------------------------------------------------------------------

ANNOTATE_SYSTEM = """You are an expert annotator for a proactive AI demand-detection benchmark.

You will be given: session background (topic, participants, memory), prior conversation context,
and the CURRENT turn to annotate. Your job: decide whether an AI assistant should proactively
speak up at this moment.

REASONING PROCESS — think through these steps before deciding:
1. **Conversation state**: What stage is the conversation at? What has been established so far?
2. **New information**: What new content/claims/questions appeared in THIS turn?
3. **State transition**: Is there a shift in topic, emotion, direction, or pace?
4. **User intent**: What is the user thinking, doing, or about to do? What do they NOT realize?
5. **Final check**: Even if you found something interesting, ask yourself —
   would the user genuinely NEED the AI to point this out? Would they thank the AI,
   or would it feel like unnecessary interruption? Only proceed if the answer is clearly YES.

TARGET: ~35-45% of turns should have demand. Not every turn deserves annotation —
a normal conversation has stretches where things flow naturally and no intervention is needed.
When two consecutive turns already have demands, be extra skeptical about the third.

GUIDELINES:
- callback_reminder: THIS turn connects to a specific earlier point the user would likely miss.
  Requires a concrete prior reference, not just thematic similarity.
- context_synthesis: fragmented info has accumulated and user may have lost the big picture.
- Req demands: clear articulable need (decision, planning, problem, lookup, summary).
  Aim for Req to be at least 35% of all demands — don't overlook explicit needs.
- Ins demands: subtle need the user hasn't recognized (risk, trend, knowledge gap, callback).
- Use "other" only as last resort.

Output JSON only."""

ANNOTATE_SCHEMA = '{"reasoning": "your step-by-step reasoning (state → new info → transition → intent → decision)", "demands": [{"demand_type": "...", "category": "Req|Ins", "trigger_text": "the specific text that triggered this demand", "prior_reference": "for callback_reminder: what earlier point this connects to, else null", "proposed_response": "what the AI should say (1-2 sentences)", "confidence": 0.8}], "has_demand": true}'


def build_context_block(session: dict, current_turn_idx: int) -> str:
    """Build context string: background + memory + prior turns."""
    lines = []

    # Session background (generated metadata)
    meta = session.get('_meta', {})
    if meta:
        lines.append("[Session Background]")
        if meta.get('topic'):
            lines.append(f"Topic: {meta['topic']}")
        if meta.get('participants'):
            parts = "; ".join(
                f"{p.get('role','?')}: {p.get('description','')}"
                for p in meta['participants']
            )
            lines.append(f"Participants: {parts}")
        if meta.get('speaker_map'):
            sm = ", ".join(f"{tag} = {desc}" for tag, desc in meta['speaker_map'].items())
            lines.append(f"Speaker mapping: {sm}")
        lines.append("")

    memory = session.get('memory')
    if memory:
        lines.append(f"[Prior Context Summary]\n{memory}\n")

    prior_turns = session['turns'][:current_turn_idx]
    if prior_turns:
        lines.append("[Conversation History]")
        show_full  = prior_turns[-10:]
        show_brief = prior_turns[:-10]
        if show_brief:
            brief_text = " / ".join(t['turn_text'][:60] for t in show_brief[-5:])
            lines.append(f"(earlier) {brief_text}")
        for t in show_full:
            lines.append(f"[Turn {t['turn_id']+1}] {t['turn_text']}")

    return "\n".join(lines) if lines else "This is the first turn. No prior context."


def make_annotate_messages(session: dict, turn_idx: int) -> list[dict]:
    turn       = session['turns'][turn_idx]
    subcat     = session.get('subcategory', '')
    context    = build_context_block(session, turn_idx)
    dtype_hints = "\n".join(
        f"  - {describe_demand_type(d)}"
        for d in get_demand_types_for(subcat)
    )
    subcat_desc = describe_subcategory(subcat) if subcat else "unknown"
    n_turns     = session['n_turns']
    part_info   = f"Part {session.get('part_index',0)+1}/{session.get('total_parts',1)}"

    user_content = f"""Subcategory: {subcat_desc}
Session: {part_info}, Turn {turn_idx+1}/{n_turns}
Language: {turn.get('language', 'cn')}

{context}

[CURRENT TURN — annotate this]
{turn['turn_text']}

Relevant demand types for this subcategory:
{dtype_hints}"""

    return [
        {"role": "system", "content": ANNOTATE_SYSTEM},
        {"role": "user",   "content": user_content},
    ]


def postprocess_annotation(ann: dict) -> dict:
    """Validate and filter a single annotation."""
    if ann is None:
        return {"reasoning": "", "demands": [], "has_demand": False}
    for d in ann.get("demands", []):
        d["confidence"] = max(0.0, min(1.0, float(d.get("confidence", 0.5))))
        dtype = d.get("demand_type", "other")
        if dtype in DEMAND_TYPES:
            d["category"] = DEMAND_TYPES[dtype]["type"]
    ann["demands"] = [
        d for d in ann.get("demands", [])
        if d.get("confidence", 0) >= (0.80 if d.get("demand_type") == "other" else 0.65)
    ]
    ann["has_demand"] = len(ann["demands"]) > 0
    return ann


def postprocess_consecutive(session: dict):
    """Break consecutive demand runs > 2."""
    run = 0
    for t in session["turns"]:
        ann = t.get("annotation", {})
        if ann.get("has_demand"):
            run += 1
            if run > 2:
                best = max(ann["demands"], key=lambda d: d.get("confidence", 0), default=None)
                if best and best.get("confidence", 0) >= 0.80:
                    ann["demands"] = [best]
                else:
                    ann["demands"] = []
                    ann["has_demand"] = False
                    run = 0
        else:
            run = 0


# ------------------------------------------------------------------
# Checkpoint helpers
# ------------------------------------------------------------------

def load_checkpoint(output_file: Path) -> dict[str, dict]:
    """Load already-annotated sessions from output file. Returns {session_id: session}."""
    done = {}
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    s = json.loads(line)
                    sid = s.get("session_id", "")
                    # Only count as done if at least one turn has annotation
                    if any("annotation" in t for t in s.get("turns", [])):
                        done[sid] = s
    return done


def append_session(output_file: Path, session: dict):
    """Append one annotated session to output file."""
    with open(output_file, "a") as f:
        f.write(json.dumps(session, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

async def generate_meta(client: AsyncLLMClient, session: dict) -> dict:
    """Generate session metadata (topic + participants) via LLM."""
    try:
        messages = make_meta_messages(session)
        result = await client.complete_json(messages, schema_hint=META_SCHEMA)
        if result is None:
            return {"topic": "", "participants": []}
        return result
    except Exception as e:
        logger.warning(f"Meta generation failed: {e}")
        return {"topic": "", "participants": []}


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
            line = line.strip()
            if line:
                sessions.append(json.loads(line))

    if max_sessions > 0:
        sessions = sessions[:max_sessions]

    # Load checkpoint
    done = load_checkpoint(output_file)
    if done:
        logger.info(f"Checkpoint: {len(done)} sessions already annotated, resuming ...")
    else:
        # Start fresh — clear output file
        output_file.write_text("")

    total_sessions = len(sessions)
    skipped = 0
    client = AsyncLLMClient(model=DEFAULT_MODEL, concurrency=concurrency, temperature=0.15, max_tokens=1024)

    for si, session in enumerate(sessions):
        sid = session.get("session_id", f"s{si}")

        # Skip if already done
        if sid in done:
            skipped += 1
            continue

        n_turns = len(session['turns'])

        # Step 1: Generate session metadata (topic + participants)
        meta = await generate_meta(client, session)
        session['_meta'] = meta

        # Step 2: Build annotation tasks for all turns
        tasks = []
        for ti in range(n_turns):
            tasks.append({
                "messages":    make_annotate_messages(session, ti),
                "schema_hint": ANNOTATE_SCHEMA,
            })

        # Run batch for this session
        results = await client.batch(tasks)

        # Apply annotations
        for ti, ann in enumerate(results):
            session["turns"][ti]["annotation"] = postprocess_annotation(ann)

        # Post-processing: consecutive runs
        postprocess_consecutive(session)

        # Save immediately (include meta in output)
        append_session(output_file, session)

        completed = si + 1 - skipped
        total_demands = sum(1 for t in session["turns"] if t.get("annotation", {}).get("has_demand"))
        logger.info(f"[{completed + len(done)}/{total_sessions}] {sid}: {n_turns} turns, {total_demands} demands")

    # Final stats
    total_turns = 0
    turns_with_demand = 0
    # Re-read the full output
    all_sessions = []
    with open(output_file) as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                all_sessions.append(s)
                for t in s.get("turns", []):
                    total_turns += 1
                    if t.get("annotation", {}).get("has_demand"):
                        turns_with_demand += 1

    logger.info(f"Done. {len(all_sessions)} sessions, {turns_with_demand}/{total_turns} turns with demand ({turns_with_demand/max(total_turns,1)*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",        default=str(INPUT_FILE))
    parser.add_argument("--output",       default=str(OUTPUT_FILE))
    parser.add_argument("--concurrency",  type=int, default=8)
    parser.add_argument("--max-sessions", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(run(Path(args.input), Path(args.output), args.concurrency, args.max_sessions))
