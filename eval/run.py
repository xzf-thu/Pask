"""
Run evaluation: test each model on all annotated turns.

Usage:
    python -m eval.run                                    # all models, neutral
    python -m eval.run --level encouraging                # specific prompt level
    python -m eval.run --models gpt-5-nano --level all    # all 3 levels for one model
    python -m eval.run --concurrency 20                   # higher concurrency
"""

import json
import time
import asyncio
import logging
import argparse
from pathlib import Path

import sys
PROJ_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = PROJ_ROOT / "bench"
sys.path.insert(0, str(BENCH_DIR))

from openai import AsyncOpenAI
from llm_client import _pick_credentials
from eval.config import MODELS, NO_DEMAND_TOKEN
from eval.prompts import build_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR    = BENCH_DIR / "data"
RESULTS_DIR = PROJ_ROOT / "eval" / "results"

LEVELS = ["encouraging", "neutral", "suppressing"]


def load_sessions(path: Path) -> list[dict]:
    sessions = []
    with open(path) as f:
        for line in f:
            if line.strip():
                sessions.append(json.loads(line))
    return sessions


def build_tasks(sessions: list[dict], level: str) -> list[dict]:
    """Build one eval task per turn across all sessions."""
    tasks = []
    for s in sessions:
        sid = s["session_id"]
        subcat = s.get("subcategory", "")
        for i, turn in enumerate(s["turns"]):
            ann = turn.get("annotation", {})
            tasks.append({
                "session_id": sid,
                "subcategory": subcat,
                "turn_id": i,
                "messages": build_prompt(s, i, level=level),
                "ground_truth": {
                    "has_demand": ann.get("has_demand", False),
                    "demands": ann.get("demands", []),
                },
            })
    return tasks


def parse_response(text: str) -> dict:
    """Parse model output into structured result."""
    text = (text or "").strip()
    no_demand = (
        not text
        or NO_DEMAND_TOKEN in text
        or text.lower() in ("no demand", "no_demand", "none")
    )
    return {
        "has_response": not no_demand,
        "response": "" if no_demand else text,
    }


async def call_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model_id: str,
    messages: list[dict],
    reasoning_effort: str | None = None,
    max_tokens: int = 4000,
    max_retries: int = 3,
) -> dict:
    """Call API once, return {text, input_tokens, output_tokens, latency_ms}."""
    async with semaphore:
        for attempt in range(max_retries):
            try:
                t0 = time.monotonic()
                kwargs = dict(
                    model=model_id,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=max_tokens,
                )
                if reasoning_effort:
                    kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
                resp = await asyncio.wait_for(
                    client.chat.completions.create(**kwargs),
                    timeout=60,
                )
                latency_ms = round((time.monotonic() - t0) * 1000)

                usage = resp.usage
                return {
                    "text": (resp.choices[0].message.content or "").strip(),
                    "input_tokens": usage.prompt_tokens if usage else 0,
                    "output_tokens": usage.completion_tokens if usage else 0,
                    "latency_ms": latency_ms,
                }
            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 * (attempt + 1))
                else:
                    return {"text": "", "input_tokens": 0, "output_tokens": 0, "latency_ms": 0}


async def eval_model(
    model_cfg: dict,
    tasks: list[dict],
    output_file: Path,
    concurrency: int,
):
    """Run all tasks through one model, save results incrementally."""
    model_id = model_cfg["id"]
    model_name = model_cfg["name"]

    # Load checkpoint
    done_keys = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done_keys.add((r["session_id"], r["turn_id"]))

    remaining = [t for t in tasks if (t["session_id"], t["turn_id"]) not in done_keys]
    if not remaining:
        logger.info(f"[{model_name}] Already complete ({len(tasks)} turns)")
        return

    logger.info(f"[{model_name}] {len(done_keys)} done, {len(remaining)} remaining")

    if "base_url" in model_cfg:
        base_url = model_cfg["base_url"]
        api_key = model_cfg.get("api_key", "EMPTY")
    else:
        base_url, api_key = _pick_credentials(model_id)
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    semaphore = asyncio.Semaphore(concurrency)
    reasoning_effort = model_cfg.get("reasoning")
    max_tokens = model_cfg.get("max_tokens", 4000)

    # Process in batches to save progress
    batch_size = 50
    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start:batch_start + batch_size]

        coros = [
            call_one(client, semaphore, model_id, t["messages"], reasoning_effort, max_tokens)
            for t in batch
        ]
        api_results = await asyncio.gather(*coros)

        with open(output_file, "a") as f:
            for task, api_res in zip(batch, api_results):
                parsed = parse_response(api_res["text"])
                result = {
                    "session_id":    task["session_id"],
                    "turn_id":       task["turn_id"],
                    "subcategory":   task["subcategory"],
                    "ground_truth":  task["ground_truth"],
                    "input_tokens":  api_res["input_tokens"],
                    "output_tokens": api_res["output_tokens"],
                    "latency_ms":    api_res["latency_ms"],
                    **parsed,
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        done_so_far = len(done_keys) + batch_start + len(batch)
        logger.info(f"[{model_name}] {done_so_far}/{len(tasks)} turns")


async def main(
    input_file: Path,
    models: list[dict],
    levels: list[str],
    concurrency: int,
    max_sessions: int = 0,
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_sessions(input_file)
    if max_sessions:
        sessions = sessions[:max_sessions]

    for level in levels:
        tasks = build_tasks(sessions, level)
        logger.info(f"=== Level: {level} | {len(sessions)} sessions, {len(tasks)} turns ===")

        for model_cfg in models:
            output_file = RESULTS_DIR / f"{model_cfg['name']}_{level}.jsonl"
            await eval_model(model_cfg, tasks, output_file, concurrency)

    logger.info("All models complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DATA_DIR / "LatentNeeds-Bench.jsonl"))
    parser.add_argument("--models", nargs="*", help="Filter by model name (default: all)")
    parser.add_argument("--level", default="neutral",
                        help="Prompt level: encouraging/neutral/suppressing/all")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--max-sessions", type=int, default=0, help="Limit number of sessions (0=all)")
    args = parser.parse_args()

    if args.models:
        selected = [m for m in MODELS if m["name"] in args.models]
    else:
        selected = MODELS

    levels = LEVELS if args.level == "all" else [args.level]
    asyncio.run(main(Path(args.input), selected, levels, args.concurrency, args.max_sessions))
