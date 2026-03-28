"""Patch existing annotated sessions with updated metadata (speaker_map)."""

import json
import asyncio
import logging
import argparse
from pathlib import Path

from llm_client import AsyncLLMClient, DEFAULT_MODEL

# Reuse the updated make_meta_messages
from importlib import import_module
annotate_mod = import_module("03_annotate")
make_meta_messages = annotate_mod.make_meta_messages
META_SCHEMA = annotate_mod.META_SCHEMA

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def patch(input_file: Path, concurrency: int = 12):
    # Load all sessions
    sessions = []
    with open(input_file) as f:
        for line in f:
            if line.strip():
                sessions.append(json.loads(line))

    logger.info(f"Loaded {len(sessions)} sessions, regenerating metadata...")

    client = AsyncLLMClient(model=DEFAULT_MODEL, concurrency=concurrency, temperature=0.15, max_tokens=1024)

    # Build all meta tasks
    async def gen_meta(session):
        try:
            messages = make_meta_messages(session)
            result = await client.complete_json(messages, schema_hint=META_SCHEMA)
            return result if result else {"topic": "", "participants": [], "speaker_map": {}}
        except Exception as e:
            logger.warning(f"Meta failed for {session.get('session_id','?')}: {e}")
            return {"topic": "", "participants": [], "speaker_map": {}}

    results = await asyncio.gather(*[gen_meta(s) for s in sessions])

    # Patch and write back
    for s, meta in zip(sessions, results):
        s['_meta'] = meta

    with open(input_file, 'w') as f:
        for s in sessions:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Show samples
    for s in sessions[:3]:
        sid = s.get('session_id', '?')[:40]
        meta = s.get('_meta', {})
        sm = meta.get('speaker_map', {})
        logger.info(f"{sid}: speaker_map={sm}")

    logger.info(f"Done. Patched {len(sessions)} sessions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/03_annotated.jsonl")
    parser.add_argument("--concurrency", type=int, default=12)
    args = parser.parse_args()
    asyncio.run(patch(Path(args.input), args.concurrency))
