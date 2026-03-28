"""
Step 1: Filter & classify raw sessions.

Input:  ../raw_new.jsonl  (2042 sessions)
Output: data/01_filtered.jsonl  (~800-1000 sessions with category labels)

Pipeline:
1. Heuristic pre-filter (segment count, speaker balance, language detection)
2. LLM classification into 12 subcategories
3. Quality score assignment
4. Save with metadata
"""

import json
import asyncio
import logging
import argparse
from pathlib import Path
from collections import defaultdict

from llm_client import AsyncLLMClient
from taxonomy import CATEGORIES, get_all_subcategory_codes, describe_subcategory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

INPUT_FILE  = Path(__file__).parent.parent / "raw_new.jsonl"
OUTPUT_FILE = DATA_DIR / "01_filtered.jsonl"

# ------------------------------------------------------------------
# Heuristic filters
# ------------------------------------------------------------------

MIN_SEGMENTS = 15        # need enough content for multi-turn
MAX_SEGMENTS = 2000      # avoid transcription-dump outliers
# Note: mic_ratio filter removed — most sessions are podcasts where
# the user passively listens (mic_ratio=0); demand triggers are
# inferred from content regardless of speaker label.


def heuristic_filter(session: dict) -> tuple[bool, str]:
    """Return (pass, reason)."""
    segs = session.get("segments", [])
    n = len(segs)

    if n < MIN_SEGMENTS:
        return False, f"too_short ({n} segs)"
    if n > MAX_SEGMENTS:
        return False, f"too_long ({n} segs)"

    # Check content length (avoid empty/boilerplate)
    total_chars = sum(len(s.get("content", "")) for s in segs)
    if total_chars < 500:
        return False, "low_content"

    return True, "ok"


def extract_sample_text(session: dict, max_segs: int = 20) -> str:
    """Extract a representative text sample for LLM classification."""
    segs = session.get("segments", [])
    # Take first 10 + middle 5 + last 5 to cover session arc
    n = len(segs)
    indices = list(range(min(10, n)))
    if n > 20:
        mid = n // 2
        indices += list(range(mid - 2, min(mid + 3, n)))
        indices += list(range(max(n - 5, 0), n))
    indices = sorted(set(indices))[:max_segs]

    lines = []
    for i in indices:
        s = segs[i]
        speaker = "User" if s.get("speaker") == "microphone" else "System"
        content = s.get("content", "").strip()
        if content:
            lines.append(f"[{speaker}] {content[:200]}")
    return "\n".join(lines)


# ------------------------------------------------------------------
# LLM classification
# ------------------------------------------------------------------

SUBCAT_LIST = "\n".join(f"  {describe_subcategory(c)}" for c in get_all_subcategory_codes())

CLASSIFY_SYSTEM = f"""You are a conversation classification expert. Given a transcript excerpt, classify it into exactly ONE of these 12 subcategories:

{SUBCAT_LIST}

Also provide:
- quality_score: float 0-1 (higher = more natural, content-rich, demand-dense)
- language: "cn", "en", or "mixed"
- demand_density: "high", "medium", "low" (estimate of how many proactive demands could be detected)
- reasoning: one sentence

Respond with JSON only."""

CLASSIFY_SCHEMA = '{"subcategory": "W1", "quality_score": 0.85, "language": "cn", "demand_density": "high", "reasoning": "..."}'


def make_classify_messages(session: dict) -> list[dict]:
    sample = extract_sample_text(session)
    scenario = session.get("scenario", "")
    lang = session.get("language", "")
    user_msg = f"Scenario: {scenario}\nLanguage hint: {lang}\n\nTranscript excerpt:\n{sample}"
    return [
        {"role": "system", "content": CLASSIFY_SYSTEM},
        {"role": "user",   "content": user_msg},
    ]


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

async def run(
    input_file: Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE,
    concurrency: int = 8,
    max_sessions: int = 0,  # 0 = no limit
    min_quality: float = 0.5,
    target_per_subcat: int = 100,
):
    logger.info(f"Loading sessions from {input_file}")
    sessions = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                sessions.append(json.loads(line))

    logger.info(f"Total sessions: {len(sessions)}")

    # Step 1: heuristic filter
    heuristic_pass = []
    heuristic_fail_reasons = defaultdict(int)
    for s in sessions:
        ok, reason = heuristic_filter(s)
        if ok:
            heuristic_pass.append(s)
        else:
            heuristic_fail_reasons[reason.split("(")[0]] += 1

    logger.info(f"After heuristic filter: {len(heuristic_pass)} / {len(sessions)}")
    for r, c in sorted(heuristic_fail_reasons.items(), key=lambda x: -x[1]):
        logger.info(f"  Filtered ({r}): {c}")

    if max_sessions > 0:
        heuristic_pass = heuristic_pass[:max_sessions]

    # Step 2: LLM classification (async batch)
    logger.info(f"Running LLM classification on {len(heuristic_pass)} sessions ...")
    client = AsyncLLMClient(concurrency=concurrency, temperature=0.1, max_tokens=256)

    tasks = []
    for s in heuristic_pass:
        tasks.append({
            "messages": make_classify_messages(s),
            "schema_hint": CLASSIFY_SCHEMA,
        })

    results = await client.batch(tasks)

    # Step 3: Merge + filter by quality + balance subcategories
    subcat_counts = defaultdict(int)
    output_sessions = []

    for session, clf in zip(heuristic_pass, results):
        if clf is None:
            logger.warning(f"  Classification failed for session {session.get('id')}, skipping")
            continue

        quality = clf.get("quality_score", 0)
        subcat  = clf.get("subcategory", "")
        density = clf.get("demand_density", "low")

        # Normalize subcategory: extract code (e.g. "W1 MeetingCollab" → "W1")
        for code in get_all_subcategory_codes():
            if subcat.startswith(code):
                subcat = code
                break

        if quality < min_quality:
            continue
        # Note: density filter removed — demand density is determined in annotation step
        if subcat not in get_all_subcategory_codes():
            logger.warning(f"  Unknown subcategory '{subcat}', skipping")
            continue
        if subcat_counts[subcat] >= target_per_subcat:
            continue

        session["_meta"] = {
            "subcategory": subcat,
            "quality_score": quality,
            "language": clf.get("language", session.get("language", "cn")),
            "demand_density": density,
            "clf_reasoning": clf.get("reasoning", ""),
            "n_segments": len(session.get("segments", [])),
        }
        output_sessions.append(session)
        subcat_counts[subcat] += 1

    # Step 4: Write output
    with open(output_file, "w") as f:
        for s in output_sessions:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(output_sessions)} sessions to {output_file}")
    logger.info("Subcategory distribution:")
    for code in get_all_subcategory_codes():
        logger.info(f"  {code}: {subcat_counts.get(code, 0)}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and classify raw sessions")
    parser.add_argument("--input",          default=str(INPUT_FILE))
    parser.add_argument("--output",         default=str(OUTPUT_FILE))
    parser.add_argument("--concurrency",    type=int,   default=8)
    parser.add_argument("--max-sessions",   type=int,   default=0,   help="Limit for debugging (0=all)")
    parser.add_argument("--min-quality",    type=float, default=0.5)
    parser.add_argument("--target-per-subcat", type=int, default=100)
    args = parser.parse_args()

    asyncio.run(run(
        input_file=Path(args.input),
        output_file=Path(args.output),
        concurrency=args.concurrency,
        max_sessions=args.max_sessions,
        min_quality=args.min_quality,
        target_per_subcat=args.target_per_subcat,
    ))
