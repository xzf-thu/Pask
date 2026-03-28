"""
Step 3d: Demand annotation v2 — dual-perspective annotation on clean data.

Input:  data/03c_memory.jsonl (clean sessions with scene + memory)
Output: data/LatentNeeds-Bench.jsonl (annotated sessions)

For each turn, the annotator:
  1. Takes the primary user's first-person perspective (what do I need/want/not-realize?)
  2. Takes the god's-eye perspective (what info exists that the user should know?)
  3. Decides whether proactive AI intervention is genuinely valuable
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
INPUT_FILE  = DATA_DIR / "03c_memory.jsonl"
OUTPUT_FILE = DATA_DIR / "LatentNeeds-Bench.jsonl"


ANNOTATE_SYSTEM = """You are an expert annotator for a proactive AI demand-detection benchmark.

You serve a specific PRIMARY USER in this conversation. Your job: at each turn, decide whether
the AI should proactively speak up to help the primary user.

CRITICAL: Most turns (65-75%) should have NO demand. Proactive AI intervention is RARE and
only warranted when there is a CLEAR, SPECIFIC benefit. Normal conversation flow — where
people are talking, explaining, responding to each other — does NOT need AI intervention.

DUAL-PERSPECTIVE REASONING — think through both before deciding:

1. **User's first-person perspective** (站在用户的角度):
   "I am [primary_user_name]. What am I thinking right now? What do I need?"

2. **God's-eye perspective** (上帝视角):
   "Is there something concrete the user is MISSING or would be SURPRISED to learn?"

3. **Decision**: Apply the INTERRUPTION TEST — if someone tapped the user on the shoulder
   right now to tell them this, would the user say "thanks, that's really helpful!" or
   would they say "yeah, I know" / "don't interrupt me"?

WHEN TO MARK A DEMAND (must meet ALL criteria):
✓ There is a SPECIFIC, ACTIONABLE piece of information or insight
✓ The primary user is UNLIKELY to realize this on their own from the conversation
✓ Telling them NOW (at this turn) matters — waiting would cause them to miss it or make a mistake
✓ The AI's intervention would be WELCOMED, not annoying

WHEN NOT TO MARK A DEMAND:
✗ Someone is just explaining something — that's normal conversation, not a demand
✗ The conversation is flowing naturally and the user is engaged — don't interrupt
✗ The information is already obvious from context
✗ A similar demand was already triggered 1-2 turns ago — avoid repetition
✗ The "insight" is vague or generic (e.g., "this is complex" or "there are tradeoffs")

DEMAND TYPE GUIDANCE:
- callback_reminder: ONLY when current turn CONTRADICTS or significantly UPDATES a specific earlier point that the user likely forgot. NOT for normal topic continuation.
- context_synthesis: ONLY after 8+ turns of fragmented discussion where key threads are genuinely getting lost. NOT for every few turns.
- knowledge_gap: ONLY when the user has a concrete, identifiable blind spot. NOT when someone is just learning.
- risk_warning: ONLY for concrete, specific risks the user hasn't considered. NOT for general uncertainty.
- task_planning / decision_support: ONLY when there's a clear decision point or planning need.

RULES:
- Output AT MOST 1 demand per turn. If you see multiple, pick the single strongest one.
- The proposed_response should be addressed TO the primary user, in natural language.
- Set confidence honestly — 0.9+ only for clear-cut cases, 0.7-0.8 for reasonable cases.
- When in doubt, mark has_demand: false. False negatives are better than false positives.

Output JSON only."""

ANNOTATE_SCHEMA = '{"user_perspective": "what the primary user is thinking/needing right now", "god_perspective": "what an omniscient observer sees that the user might miss", "demands": [{"demand_type": "...", "category": "Req|Ins", "trigger_text": "the exact phrase that triggers this demand", "prior_reference": "turn number or quote from earlier, if applicable", "proposed_response": "what the AI would say to the user", "confidence": 0.85}], "has_demand": false}'


def build_context_block(session: dict, current_turn_idx: int) -> str:
    """Build context: scene + memory + prior turns."""
    lines = []

    # Scene info
    scene = session.get('scene', {})
    if scene:
        lines.append("[Scene]")
        lines.append(scene.get('scene', ''))

        characters = scene.get('characters', [])
        if characters:
            lines.append("\n[Characters]")
            for c in characters:
                lines.append(f"  {c.get('name','?')} ({c.get('role','')}): {c.get('background','')}")

        primary = scene.get('primary_user', {})
        if primary:
            lines.append(f"\n[Primary user — YOU are serving this person]")
            lines.append(f"  {primary.get('name','?')}: {primary.get('reason','')}")
        lines.append("")

    # Memory
    memory = session.get('memory', '')
    if memory:
        lines.append(f"[Memory from prior conversation]\n{memory}\n")

    # Prior turns
    prior_turns = session['turns'][:current_turn_idx]
    if prior_turns:
        lines.append("[Conversation so far]")
        show_full = prior_turns[-12:]
        show_brief = prior_turns[:-12]
        if show_brief:
            brief = " / ".join(
                f"{t.get('speaker','?')}: {t.get('text','')[:50]}"
                for t in show_brief[-5:]
            )
            lines.append(f"(earlier) {brief}")
        for t in show_full:
            speaker = t.get('speaker', '?')
            text = t.get('text', t.get('turn_text', ''))
            lines.append(f"  {speaker}: {text}")

    return "\n".join(lines) if lines else "This is the first turn. No prior context."


def make_annotate_messages(session: dict, turn_idx: int) -> list[dict]:
    turn = session['turns'][turn_idx]
    subcat = session.get('subcategory', '')
    context = build_context_block(session, turn_idx)
    dtype_hints = "\n".join(
        f"  - {describe_demand_type(d)}"
        for d in get_demand_types_for(subcat)
    )
    n_turns = session.get('n_turns', len(session['turns']))

    speaker = turn.get('speaker', '?')
    text = turn.get('text', turn.get('turn_text', ''))
    primary_name = session.get('scene', {}).get('primary_user', {}).get('name', '?')

    user_content = f"""Subcategory: {describe_subcategory(subcat) if subcat else 'unknown'}
Turn {turn_idx+1}/{n_turns}
Primary user: {primary_name}

{context}

[CURRENT TURN — annotate this]
{speaker}: {text}

Relevant demand types:
{dtype_hints}"""

    return [
        {"role": "system", "content": ANNOTATE_SYSTEM},
        {"role": "user",   "content": user_content},
    ]


def postprocess_annotation(ann: dict) -> dict:
    if ann is None:
        return {"user_perspective": "", "god_perspective": "", "demands": [], "has_demand": False}
    for d in ann.get("demands", []):
        d["confidence"] = max(0.0, min(1.0, float(d.get("confidence", 0.5))))
        dtype = d.get("demand_type", "other")
        if dtype in DEMAND_TYPES:
            d["category"] = DEMAND_TYPES[dtype]["type"]
        else:
            d["demand_type"] = "other"
            d["category"] = "Ins"
    # Filter by confidence threshold — stricter for over-represented types
    CONF_THRESHOLDS = {"other": 0.85, "knowledge_gap": 0.82, "context_synthesis": 0.82}
    ann["demands"] = [
        d for d in ann.get("demands", [])
        if d.get("confidence", 0) >= CONF_THRESHOLDS.get(d.get("demand_type", ""), 0.78)
    ]
    # Keep only the single best demand per turn
    if len(ann["demands"]) > 1:
        ann["demands"] = [max(ann["demands"], key=lambda d: d.get("confidence", 0))]
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
                best = ann["demands"][0] if ann["demands"] else None
                if best and best.get("confidence", 0) >= 0.90:
                    pass  # Keep very high confidence demands
                else:
                    ann["demands"] = []
                    ann["has_demand"] = False
                    run = 0
        else:
            run = 0


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
                    s = json.loads(line)
                    sid = s.get("session_id", "")
                    if any("annotation" in t for t in s.get("turns", [])):
                        done[sid] = s
    return done


def append_session(output_file: Path, session: dict):
    with open(output_file, "a") as f:
        f.write(json.dumps(session, ensure_ascii=False) + "\n")


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
        logger.info(f"Checkpoint: {len(done)} sessions already annotated, resuming...")
    else:
        output_file.write_text("")

    total = len(sessions)
    client = AsyncLLMClient(model=DEFAULT_MODEL, concurrency=concurrency, temperature=0.15, max_tokens=1024)

    for si, session in enumerate(sessions):
        sid = session.get("session_id", f"s{si}")

        if sid in done:
            continue

        n_turns = len(session['turns'])

        # Build tasks for all turns
        tasks = []
        for ti in range(n_turns):
            tasks.append({
                "messages": make_annotate_messages(session, ti),
                "schema_hint": ANNOTATE_SCHEMA,
            })

        results = await client.batch(tasks)

        for ti, ann in enumerate(results):
            session["turns"][ti]["annotation"] = postprocess_annotation(ann)

        postprocess_consecutive(session)
        append_session(output_file, session)

        total_demands = sum(1 for t in session["turns"] if t.get("annotation", {}).get("has_demand"))
        logger.info(f"[{si+1}/{total}] {sid[:40]}: {n_turns} turns, {total_demands} demands")

    # Final stats
    total_turns = 0
    turns_with_demand = 0
    with open(output_file) as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                for t in s.get("turns", []):
                    total_turns += 1
                    if t.get("annotation", {}).get("has_demand"):
                        turns_with_demand += 1

    logger.info(f"Done. {total} sessions, {turns_with_demand}/{total_turns} turns with demand ({turns_with_demand/max(total_turns,1)*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",        default=str(INPUT_FILE))
    parser.add_argument("--output",       default=str(OUTPUT_FILE))
    parser.add_argument("--concurrency",  type=int, default=8)
    parser.add_argument("--max-sessions", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(run(Path(args.input), Path(args.output), args.concurrency, args.max_sessions))
