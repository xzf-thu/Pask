"""
Re-cluster subcategories based on actual data distribution.

Input:  data/02_turns.jsonl (1639 sessions)
Output: data/02_turns_reclustered.jsonl (~1000 balanced sessions)

Steps:
1. For each main category (W/L/D), sample sessions and ask LLM to propose 3-4 subcategories
2. Reclassify all sessions into new subcategories (nano)
3. Select ~1000 balanced items across subcategories and languages
"""

import json
import asyncio
import logging
import random
from pathlib import Path
from collections import defaultdict

from llm_client import AsyncLLMClient, DEFAULT_MODEL, NANO_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR    = Path(__file__).parent / "data"
INPUT_FILE  = DATA_DIR / "02_turns.jsonl"
OUTPUT_FILE = DATA_DIR / "02_turns_reclustered.jsonl"

MAIN_CATEGORIES = {
    "W": "Work (工作场景): meetings, tech development, business, workplace communication",
    "L": "Learning (学习场景): courses, research, skill training, language learning",
    "D": "Daily (日常场景): media/entertainment, life management, social/emotional, knowledge exploration",
}

# ------------------------------------------------------------------
# Step 1: Discover subcategories
# ------------------------------------------------------------------

DISCOVER_SYSTEM = """You are a data taxonomy expert. Given sample conversations from a main category, propose exactly {n} subcategories that best cover the data.

Requirements:
1. Each subcategory must be clearly distinct from others
2. Subcategories should reflect ACTUAL content patterns, not theoretical divisions
3. Each subcategory needs a short code (e.g. W1), English name, and Chinese name
4. Aim for roughly equal coverage — don't create a subcategory for <5% of data

Output JSON only."""

DISCOVER_SCHEMA = '{"subcategories": [{"code": "W1", "name_en": "ProductDiscussion", "name_cn": "产品讨论", "description": "...", "keywords": ["product", "feature", "design"]}]}'


def make_discover_messages(main_cat: str, desc: str, samples: list[str], n_subcats: int) -> list[dict]:
    sample_text = "\n---\n".join(samples[:40])  # up to 40 samples
    return [
        {"role": "system", "content": DISCOVER_SYSTEM.format(n=n_subcats)},
        {"role": "user", "content": f"""Main category: {main_cat} — {desc}

Total sessions in this category: {len(samples)}

Here are representative session excerpts:

{sample_text}

Propose exactly {n_subcats} subcategories with codes {main_cat}1, {main_cat}2, ... {main_cat}{n_subcats}."""},
    ]


# ------------------------------------------------------------------
# Step 2: Reclassify
# ------------------------------------------------------------------

CLASSIFY_SYSTEM = """You are a conversation classifier. Given a conversation excerpt, classify it into exactly ONE of the provided subcategories.

Output JSON only: {{"subcategory": "W1", "confidence": 0.85}}"""


def make_classify_messages(session: dict, subcategories: list[dict]) -> list[dict]:
    subcat_desc = "\n".join(
        f"  {s['code']}: {s['name_en']} ({s['name_cn']}) — {s['description']}"
        for s in subcategories
    )
    # Get representative content
    turns = session.get("turns", [])
    sample = "\n".join(t["turn_text"][:150] for t in turns[:5])
    lang = session.get("language", "")

    return [
        {"role": "system", "content": CLASSIFY_SYSTEM},
        {"role": "user", "content": f"""Subcategories:
{subcat_desc}

Language: {lang}
Conversation excerpt:
{sample}

Classify into one subcategory."""},
    ]


# ------------------------------------------------------------------
# Step 3: Balance selection
# ------------------------------------------------------------------

def select_balanced(sessions: list[dict], target_total: int = 1000) -> list[dict]:
    """Select balanced subset across subcategories and languages."""
    by_subcat = defaultdict(list)
    for s in sessions:
        by_subcat[s["subcategory"]].append(s)

    n_subcats = len(by_subcat)
    per_subcat = target_total // n_subcats

    selected = []
    for sc, items in sorted(by_subcat.items()):
        # Within each subcategory, balance languages
        by_lang = defaultdict(list)
        for s in items:
            by_lang[s.get("language", "cn")].append(s)

        # Aim for roughly balanced language within each subcategory
        n_langs = len(by_lang)
        per_lang = per_subcat // max(n_langs, 1)

        subcat_selected = []
        remaining_quota = per_subcat

        # First pass: take up to per_lang from each language
        for lang, lang_items in sorted(by_lang.items()):
            random.shuffle(lang_items)
            take = min(per_lang, len(lang_items), remaining_quota)
            subcat_selected.extend(lang_items[:take])
            remaining_quota -= take

        # Second pass: fill remaining from any language
        if remaining_quota > 0:
            all_remaining = [s for s in items if s not in subcat_selected]
            random.shuffle(all_remaining)
            subcat_selected.extend(all_remaining[:remaining_quota])

        selected.extend(subcat_selected[:per_subcat])

    return selected


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

async def run(target_total: int = 1000):
    random.seed(42)

    logger.info(f"Loading from {INPUT_FILE}")
    sessions = []
    with open(INPUT_FILE) as f:
        for line in f:
            if line.strip():
                sessions.append(json.loads(line))

    logger.info(f"Loaded {len(sessions)} sessions")

    # Group by main category
    by_main = defaultdict(list)
    for s in sessions:
        sc = s.get("subcategory", "?")
        main = sc[0]
        by_main[main].append(s)

    # Step 1: Discover subcategories per main category
    logger.info("Step 1: Discovering subcategories from data ...")
    client_main = AsyncLLMClient(model=DEFAULT_MODEL, concurrency=3, temperature=0.3, max_tokens=1024)

    # Decide how many subcats per main category based on data volume
    subcat_counts = {"W": 4, "L": 3, "D": 3}  # total = 10 subcats

    discover_tasks = []
    main_cats_order = []
    for main_cat in ["W", "L", "D"]:
        items = by_main[main_cat]
        # Sample representative excerpts
        random.shuffle(items)
        samples = []
        for s in items[:50]:
            turns = s.get("turns", [])
            excerpt = " | ".join(t["turn_text"][:100] for t in turns[:3])
            samples.append(f"[{s.get('language','')}] {excerpt[:400]}")

        discover_tasks.append({
            "messages": make_discover_messages(
                main_cat, MAIN_CATEGORIES[main_cat], samples, subcat_counts[main_cat]
            ),
            "schema_hint": DISCOVER_SCHEMA,
        })
        main_cats_order.append(main_cat)

    discover_results = await client_main.batch(discover_tasks)

    # Parse discovered subcategories
    all_subcats = {}  # main_cat -> list of subcategory dicts
    for main_cat, result in zip(main_cats_order, discover_results):
        if result and "subcategories" in result:
            subcats = result["subcategories"]
            all_subcats[main_cat] = subcats
            logger.info(f"  {main_cat}: {len(subcats)} subcategories")
            for sc in subcats:
                logger.info(f"    {sc['code']}: {sc['name_en']} ({sc['name_cn']})")
        else:
            logger.error(f"  {main_cat}: Discovery failed, using fallback")
            all_subcats[main_cat] = [{"code": f"{main_cat}1", "name_en": "General", "name_cn": "通用", "description": "all"}]

    # Step 2: Reclassify all sessions
    logger.info("Step 2: Reclassifying sessions with nano ...")
    client_nano = AsyncLLMClient(model=NANO_MODEL, concurrency=8, temperature=0.1, max_tokens=64)

    classify_tasks = []
    classify_meta = []  # (session_idx, main_cat)
    for si, s in enumerate(sessions):
        main = s.get("subcategory", "?")[0]
        subcats = all_subcats.get(main, [])
        if len(subcats) <= 1:
            # Only one subcategory, no need to classify
            classify_meta.append((si, main, subcats[0]["code"] if subcats else f"{main}1"))
            continue
        classify_tasks.append({
            "messages": make_classify_messages(s, subcats),
            "schema_hint": '{"subcategory": "W1", "confidence": 0.85}',
        })
        classify_meta.append((si, main, None))  # None = needs classification

    # Run classification
    task_idx = 0
    results_iter = iter(await client_nano.batch(classify_tasks)) if classify_tasks else iter([])

    for si, main, preset_code in classify_meta:
        if preset_code:
            sessions[si]["subcategory"] = preset_code
        else:
            result = next(results_iter, None)
            if result and isinstance(result, dict) and "subcategory" in result:
                code = result["subcategory"]
                # Validate code belongs to this main category
                valid_codes = [sc["code"] for sc in all_subcats.get(main, [])]
                if code in valid_codes:
                    sessions[si]["subcategory"] = code
                else:
                    sessions[si]["subcategory"] = valid_codes[0] if valid_codes else f"{main}1"
            else:
                sessions[si]["subcategory"] = f"{main}1"

    # Show new distribution
    new_dist = defaultdict(int)
    for s in sessions:
        new_dist[s["subcategory"]] += 1
    logger.info("New subcategory distribution:")
    for sc, cnt in sorted(new_dist.items()):
        logger.info(f"  {sc}: {cnt}")

    # Step 3: Select balanced subset
    logger.info(f"Step 3: Selecting balanced {target_total} items ...")
    selected = select_balanced(sessions, target_total)
    random.shuffle(selected)

    # Show final distribution
    final_dist = defaultdict(lambda: {"total": 0, "cn": 0, "en": 0, "mixed": 0})
    total_turns = 0
    for s in selected:
        sc = s["subcategory"]
        lang = s.get("language", "cn")
        final_dist[sc]["total"] += 1
        final_dist[sc][lang] = final_dist[sc].get(lang, 0) + 1
        total_turns += len(s["turns"])

    logger.info(f"Final selection: {len(selected)} sessions, {total_turns} turns")
    for sc in sorted(final_dist.keys()):
        d = final_dist[sc]
        logger.info(f"  {sc}: {d['total']} (cn={d.get('cn',0)} en={d.get('en',0)} mixed={d.get('mixed',0)})")

    # Save new taxonomy
    taxonomy = {"main_categories": MAIN_CATEGORIES, "subcategories": all_subcats}
    with open(DATA_DIR / "taxonomy_new.json", "w") as f:
        json.dump(taxonomy, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved new taxonomy to {DATA_DIR / 'taxonomy_new.json'}")

    # Save selected sessions
    with open(OUTPUT_FILE, "w") as f:
        for s in selected:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(selected)} sessions to {OUTPUT_FILE}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=1000)
    args = parser.parse_args()
    asyncio.run(run(target_total=args.target))
