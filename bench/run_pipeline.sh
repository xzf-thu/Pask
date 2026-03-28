#!/usr/bin/env bash
# Full benchmark construction pipeline.
# Run from the bench/ directory: bash run_pipeline.sh
#
# Steps:
#  01  → Filter & classify raw sessions (gpt-5-nano)
#  01b → Split sessions at topic boundaries (gpt-5-nano)
#  02  → Build semantic turns + memory (gpt-5-nano)
#  03a → Annotate real_raw turns (gpt-5.2)
#  04  → Clean text → real_clean version (gpt-5-nano)
#  03b → Annotate real_clean turns (gpt-5.2)
#  05  → Generate synthetic conversations (gpt-5.2)
#  03c → Annotate synthetic turns (gpt-5.2)
#  06  → Format final output

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

CONCURRENCY=${CONCURRENCY:-8}
MAX_SESSIONS=${MAX_SESSIONS:-0}

log "=== IntentFlow Benchmark Pipeline ==="
log "Concurrency: $CONCURRENCY | Debug limit: $MAX_SESSIONS"

log "Step 01: Filter & classify"
python 01_filter.py --concurrency "$CONCURRENCY" --max-sessions "$MAX_SESSIONS"

log "Step 01b: Topic-boundary split"
python 01b_split_topics.py --concurrency "$CONCURRENCY"

log "Step 02: Build semantic turns + memory"
python 02_build_turns.py --concurrency "$CONCURRENCY"

log "Step 03a: Annotate real_raw"
python 03_annotate.py --input data/02_turns.jsonl --output data/03_annotated.jsonl --concurrency "$CONCURRENCY"

log "Step 04: Clean text"
python 04_clean_text.py --input data/02_turns.jsonl --output data/02_turns_clean.jsonl --concurrency "$CONCURRENCY"

log "Step 03b: Annotate real_clean"
python 03_annotate.py --input data/02_turns_clean.jsonl --output data/03_annotated_clean.jsonl --concurrency "$CONCURRENCY"

log "Step 05: Synthesize"
python 05_synthesize.py --concurrency 6

log "Step 03c: Annotate synthetic"
python 03_annotate.py --input data/05_synthetic.jsonl --output data/05_annotated_synth.jsonl --concurrency "$CONCURRENCY"

log "Step 06: Format final output"
python 06_format_output.py

log "=== Pipeline complete ==="
ls -lh "$SCRIPT_DIR/data/final/"
