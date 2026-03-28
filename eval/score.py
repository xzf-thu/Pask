"""
Score evaluation results using LLM-as-judge.

Scoring per turn:
  - TN (annotation=no demand, model=no demand): 1
  - FN (annotation=demand, model=no demand): 0
  - Model responded: judge decides 0/1
    - Round 1: judge sees current turn + annotation + model response (no context)
    - Round 2 (if judge uncertain): add memory + prior turns, re-judge

Final score = sum(turn_scores) / total_turns * 100

Usage:
    python -m eval.score                           # score all result files
    python -m eval.score --models gpt-5-nano       # specific model (all levels)
"""

import json
import asyncio
import logging
import argparse
from pathlib import Path

import sys
PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT / "bench"))

from openai import AsyncOpenAI
from llm_client import _pick_credentials
from eval.config import MODELS, JUDGE_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EVAL_DIR    = PROJ_ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"
SCORES_DIR  = EVAL_DIR / "scores"
DATA_DIR    = PROJ_ROOT / "bench" / "data"

# ---- Judge prompts ----

JUDGE_R1_SYSTEM = """You are evaluating whether an AI assistant's proactive intervention is valuable.

You will see:
- The current conversation turn (with speaker attribution)
- The annotation (ground truth: annotator's reasoning and expected demand, if any)
- The model's actual response

NOTE: You do NOT have the full conversation context. Judge based on what you can see.

Rules:
- If the annotation says a demand exists: check if the model's response addresses a similar need. It doesn't need to match exactly, but must be relevant and useful.
- If the annotation says NO demand: check if the model's response is nonetheless genuinely valuable and actionable for this turn's content.
- When the annotation says NO demand, be skeptical — the annotator had full context. Only approve if the model's response is clearly valuable even from this limited view.

Output ONLY one of:
- "1" if the response is valuable
- "0" if the response is not valuable
- "NEED_CONTEXT" if you cannot judge without seeing the prior conversation"""

JUDGE_R2_SYSTEM = """You are evaluating whether an AI assistant's proactive intervention is valuable.

You now have the FULL conversation context (memory + prior turns). The annotation was made WITH this full context.

Rules:
- If the annotation says a demand exists: check if the model's response addresses a similar need.
- If the annotation says NO demand: the annotator (with full context) decided no proactive intervention was needed. Only override this if the model's response is clearly and specifically valuable. When in doubt, trust the annotation.

Output ONLY "1" (valuable) or "0" (not valuable)."""


def _format_turn(turn: dict) -> str:
    """Format turn as 'speaker: text'."""
    return f"{turn.get('speaker', '?')}: {turn.get('text', '')}"


def build_judge_r1_msg(item: dict) -> str:
    """Build Round 1 judge user message."""
    gt = item["ground_truth"]

    # Annotation section with dual perspective
    if gt["has_demand"]:
        ann_parts = []
        if item.get("user_perspective"):
            ann_parts.append(f"User perspective: {item['user_perspective']}")
        if item.get("god_perspective"):
            ann_parts.append(f"Observer perspective: {item['god_perspective']}")
        for d in gt.get("demands", []):
            ann_parts.append(f"- [{d.get('demand_type', '?')}] {d.get('proposed_response', '')[:200]}")
        ann_text = "Annotated demand:\n" + "\n".join(ann_parts)
    else:
        ann_text = "No proactive demand annotated for this turn."

    return (
        f"[Current turn]\n{item.get('turn_text', '(not available)')}\n\n"
        f"[Annotation]\n{ann_text}\n\n"
        f"[Model response]\n{item['response'][:500]}"
    )


def build_judge_r2_msg(item: dict, context: str) -> str:
    """Build Round 2 judge user message (with full context)."""
    r1_msg = build_judge_r1_msg(item)
    return f"[Full conversation context]\n{context}\n\n{r1_msg}"


def load_sessions_map(input_file: Path) -> dict:
    """Load annotated sessions as {session_id: session}."""
    sessions = {}
    with open(input_file) as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                sessions[s["session_id"]] = s
    return sessions


def get_turn_context(session: dict, turn_id: int) -> tuple[str, str]:
    """Get (context_text, turn_text) for a turn."""
    turns = session.get("turns", [])
    memory = session.get("memory")

    ctx_parts = []
    if memory:
        ctx_parts.append(f"[Memory]\n{memory}")
    for t in turns[:turn_id]:
        ctx_parts.append(_format_turn(t))
    context = "\n\n".join(ctx_parts) if ctx_parts else "(conversation start)"

    turn_text = _format_turn(turns[turn_id]) if turn_id < len(turns) else ""
    return context, turn_text


async def judge_batch(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    system: str,
    messages: list[str],
) -> list[str]:
    """Run judge on a batch of user messages."""
    async def _call(user_msg):
        async with semaphore:
            for attempt in range(3):
                try:
                    resp = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=JUDGE_MODEL,
                            messages=[
                                {"role": "system", "content": system},
                                {"role": "user", "content": user_msg},
                            ],
                            temperature=0.0,
                            max_tokens=16,
                        ),
                        timeout=30,
                    )
                    return (resp.choices[0].message.content or "").strip()
                except Exception as e:
                    logger.warning(f"Judge call failed (attempt {attempt+1}): {e}")
                    if attempt < 2:
                        await asyncio.sleep(2 * (attempt + 1))
            return "0"

    return await asyncio.gather(*[_call(m) for m in messages])


async def score_result_file(result_file: Path, sessions_map: dict, concurrency: int):
    """Score one result file (model + level combination)."""
    stem = result_file.stem  # e.g. "gpt-5-mini_encouraging"
    score_file = SCORES_DIR / f"{stem}.jsonl"

    # Load results
    items = []
    with open(result_file) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    # Load checkpoint
    done_keys = set()
    if score_file.exists():
        with open(score_file) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done_keys.add((r["session_id"], r["turn_id"]))

    remaining = [it for it in items if (it["session_id"], it["turn_id"]) not in done_keys]
    if not remaining:
        logger.info(f"[{stem}] Already scored ({len(items)} items)")
        return

    logger.info(f"[{stem}] Scoring {len(remaining)} items ({len(done_keys)} done)")

    # Enrich items with turn_text and annotation perspectives from sessions
    for it in remaining:
        sid = it["session_id"]
        tid = it["turn_id"]
        if sid in sessions_map:
            session = sessions_map[sid]
            turns = session.get("turns", [])
            if tid < len(turns):
                turn = turns[tid]
                it["turn_text"] = _format_turn(turn)
                ann = turn.get("annotation", {})
                it["user_perspective"] = ann.get("user_perspective", "")
                it["god_perspective"] = ann.get("god_perspective", "")

    # Split: items needing judge vs auto-scored
    need_judge = []
    auto_scored = []

    for it in remaining:
        if it["has_response"]:
            need_judge.append(it)
        else:
            gt_demand = it["ground_truth"]["has_demand"]
            score = 1 if not gt_demand else 0
            auto_scored.append((it, score, "tn" if not gt_demand else "fn"))

    logger.info(f"[{stem}] Auto: {len(auto_scored)} (TN/FN), Judge: {len(need_judge)}")

    # ---- Round 1: judge without context ----
    base_url, api_key = _pick_credentials(JUDGE_MODEL)
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    semaphore = asyncio.Semaphore(concurrency)

    r1_messages = [build_judge_r1_msg(it) for it in need_judge]
    r1_results = await judge_batch(client, semaphore, JUDGE_R1_SYSTEM, r1_messages)

    # Collect results and find items needing Round 2
    judged = {}
    need_r2_indices = []

    for i, (it, r1) in enumerate(zip(need_judge, r1_results)):
        gt_demand = it["ground_truth"]["has_demand"]
        if r1.startswith("1"):
            label = "tp" if gt_demand else "fp_accepted"
            judged[i] = (1, label)
        elif r1.startswith("0"):
            label = "tp_bad" if gt_demand else "fp_rejected"
            judged[i] = (0, label)
        else:
            need_r2_indices.append(i)

    logger.info(f"[{stem}] R1 done. Need R2: {len(need_r2_indices)}")

    # ---- Round 2: with full context ----
    if need_r2_indices:
        r2_messages = []
        for i in need_r2_indices:
            it = need_judge[i]
            sid = it["session_id"]
            tid = it["turn_id"]
            if sid in sessions_map:
                context, _ = get_turn_context(sessions_map[sid], tid)
            else:
                context = "(context not available)"
            r2_messages.append(build_judge_r2_msg(it, context))

        r2_results = await judge_batch(client, semaphore, JUDGE_R2_SYSTEM, r2_messages)

        for i, r2 in zip(need_r2_indices, r2_results):
            it = need_judge[i]
            gt_demand = it["ground_truth"]["has_demand"]
            score = 1 if r2.startswith("1") else 0
            if score == 1:
                label = "tp" if gt_demand else "fp_accepted"
            else:
                label = "tp_bad" if gt_demand else "fp_rejected"
            judged[i] = (score, label)

    # ---- Write all scores ----
    with open(score_file, "a") as f:
        for it, score, label in auto_scored:
            out = {
                "session_id": it["session_id"],
                "turn_id": it["turn_id"],
                "subcategory": it["subcategory"],
                "score": score,
                "label": label,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

        for i, it in enumerate(need_judge):
            score, label = judged[i]
            out = {
                "session_id": it["session_id"],
                "turn_id": it["turn_id"],
                "subcategory": it["subcategory"],
                "score": score,
                "label": label,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    logger.info(f"[{stem}] Scoring complete")


async def main(model_filter: list[str] | None, concurrency: int):
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    sessions_map = load_sessions_map(DATA_DIR / "LatentNeeds-Bench.jsonl")
    logger.info(f"Loaded {len(sessions_map)} annotated sessions")

    # Find all result files, optionally filtered by model name
    for result_file in sorted(RESULTS_DIR.glob("*.jsonl")):
        if model_filter:
            stem = result_file.stem
            if not any(stem.startswith(m) for m in model_filter):
                continue
        await score_result_file(result_file, sessions_map, concurrency)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", help="Filter by model name prefix")
    parser.add_argument("--concurrency", type=int, default=12)
    args = parser.parse_args()

    asyncio.run(main(args.models, args.concurrency))
