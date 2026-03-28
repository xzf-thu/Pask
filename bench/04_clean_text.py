"""
Step 4: Clean/polish turn text to create the "real_clean" bench version.

Input:  data/02_turns.jsonl          (raw turns)
Output: data/02_turns_clean.jsonl    (cleaned turns, same structure)

Cleaning operations (LLM-based):
- Remove filler words, disfluencies ("嗯", "啊", "um", "uh", repetitions)
- Fix broken sentences from ASR errors
- Normalize mixed CN/EN spacing
- Preserve all semantic content and named entities

The cleaned turns are then passed to 03_annotate.py separately.
"""

import json
import asyncio
import logging
import argparse
import re
from pathlib import Path

from llm_client import AsyncLLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR    = Path(__file__).parent / "data"
INPUT_FILE  = DATA_DIR / "02_turns.jsonl"
OUTPUT_FILE = DATA_DIR / "02_turns_clean.jsonl"

# ------------------------------------------------------------------
# Heuristic pre-cleaning (fast, no LLM)
# ------------------------------------------------------------------

_FILLER_CN = re.compile(
    r"(?:嗯+|啊+|哦+|哈+|呃+|额+|那个那个|就是就是|然后然后|对对对|好好好)"
    r"(?:[，,。.！!？?]|\s|$)",
    re.UNICODE
)
_FILLER_EN = re.compile(
    r"\b(?:um+|uh+|er+|ah+|like like|you know you know|so so)\b",
    re.IGNORECASE
)
_REPEAT_WORDS = re.compile(r"\b(\w{2,})\s+\1\b", re.UNICODE)  # word word → word


def heuristic_clean(text: str) -> str:
    text = _FILLER_CN.sub(" ", text)
    text = _FILLER_EN.sub(" ", text)
    text = _REPEAT_WORDS.sub(r"\1", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    # Fix spacing around CN/EN boundaries
    text = re.sub(r"([a-zA-Z0-9])([\u4e00-\u9fff])", r"\1 \2", text)
    text = re.sub(r"([\u4e00-\u9fff])([a-zA-Z0-9])", r"\1 \2", text)
    return text.strip()


# ------------------------------------------------------------------
# LLM cleaning prompt
# ------------------------------------------------------------------

CLEAN_SYSTEM = """You are a text cleaning assistant for conversational AI data.

Your task: Clean the given conversation turn text while preserving ALL semantic content.

Cleaning rules:
1. Remove ASR disfluencies: filler words (嗯, 啊, um, uh), false starts, repetitions
2. Fix broken/incomplete sentences caused by ASR errors
3. Normalize mixed Chinese-English: proper spacing between scripts
4. Remove non-lexical sounds: "[laughter]", "[pause]", etc.
5. DO NOT: add new content, change facts, alter named entities, change meaning
6. Keep speaker labels "[User]" and "[System]" exactly as-is

Output the cleaned turn text only, no explanation."""


def make_clean_messages(turn_text: str) -> list[dict]:
    return [
        {"role": "system", "content": CLEAN_SYSTEM},
        {"role": "user",   "content": f"Clean this turn:\n\n{turn_text}"},
    ]


# ------------------------------------------------------------------
# Per-turn cleaning score (decide if LLM cleaning is needed)
# ------------------------------------------------------------------

_NOISE_PATTERNS = re.compile(
    r"嗯{2,}|啊{2,}|哦{2,}|um{2,}|uh{2,}"  # repeated fillers
    r"|(?:那个){2,}|(?:就是){2,}"             # repeated CN fillers
    r"|\[[a-z]+\]"                             # ASR tags
    r"|\b(\w+)\s+\1\b",                        # repeated words
    re.IGNORECASE | re.UNICODE
)


def noise_score(text: str) -> float:
    """Estimate noise level 0-1. Higher = noisier."""
    if not text:
        return 0.0
    matches = len(_NOISE_PATTERNS.findall(text))
    return min(1.0, matches / max(1, len(text.split()) / 20))


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

async def run(
    input_file:   Path  = INPUT_FILE,
    output_file:  Path  = OUTPUT_FILE,
    concurrency:  int   = 8,
    noise_threshold: float = 0.1,  # only send to LLM if noisy enough
    max_sessions: int   = 0,
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

    logger.info(f"Cleaning {len(sessions)} sessions")

    # Step 1: heuristic clean for all turns
    for session in sessions:
        for turn in session["turns"]:
            turn["turn_text_original"] = turn["turn_text"]
            turn["turn_text"] = heuristic_clean(turn["turn_text"])

    # Step 2: identify high-noise turns for LLM cleaning
    llm_tasks     = []
    llm_task_meta = []

    for si, session in enumerate(sessions):
        for ti, turn in enumerate(session["turns"]):
            ns = noise_score(turn["turn_text"])
            if ns >= noise_threshold:
                llm_task_meta.append((si, ti))
                llm_tasks.append({
                    "messages": make_clean_messages(turn["turn_text"]),
                })

    logger.info(f"Sending {len(llm_tasks)} high-noise turns to LLM for deep cleaning")

    if llm_tasks:
        client = AsyncLLMClient(concurrency=concurrency, temperature=0.1, max_tokens=2048)
        results = await client.batch(llm_tasks)

        for (si, ti), cleaned in zip(llm_task_meta, results):
            if cleaned:
                sessions[si]["turns"][ti]["turn_text"] = cleaned.strip()
                sessions[si]["turns"][ti]["llm_cleaned"] = True

    # Update source tag
    for session in sessions:
        session["source"] = "real_clean"

    with open(output_file, "w") as f:
        for s in sessions:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(f"Saved cleaned sessions to {output_file}")
    logger.info(f"LLM-cleaned turns: {len(llm_tasks)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean turn text for real_clean bench")
    parser.add_argument("--input",           default=str(INPUT_FILE))
    parser.add_argument("--output",          default=str(OUTPUT_FILE))
    parser.add_argument("--concurrency",     type=int,   default=8)
    parser.add_argument("--noise-threshold", type=float, default=0.1)
    parser.add_argument("--max-sessions",    type=int,   default=0)
    args = parser.parse_args()

    asyncio.run(run(
        input_file=Path(args.input),
        output_file=Path(args.output),
        concurrency=args.concurrency,
        noise_threshold=args.noise_threshold,
        max_sessions=args.max_sessions,
    ))
