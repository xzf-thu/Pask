# IntentFlow Benchmark Construction Pipeline

## Overview

The benchmark evaluates **proactive demand detection** in multi-turn conversations — the ability of an AI agent to identify what a user needs *without being explicitly asked*. It consists of three parallel corpora derived from different sources, each supporting both multi-turn and single-turn evaluation.

### Final Benchmark Composition

| Corpus | Source | Sessions | Turns/Session | Description |
|--------|--------|----------|---------------|-------------|
| `real_raw` | Real user speech (raw ASR) | ~1,000 | up to 60 | Natural speech with disfluencies |
| `real_clean` | Real user speech (polished) | ~1,000 | up to 60 | Same content, cleaned text |
| `synthetic` | LLM-generated | ~1,000 | 8–15 | Constructed with embedded demand triggers |

Single-turn items are derived directly from the multi-turn annotations: each (history, turn, label) triple from the multi-turn data becomes one single-turn item.

---

## Demand Taxonomy

Demands are organized into **3 categories × 4 subcategories = 12 subcategories**, each associated with specific demand types.

### Categories and Subcategories

| Code | Subcategory | Domain |
|------|-------------|--------|
| W1 | MeetingCollab (会议协作) | Team meetings, action items, collaboration |
| W2 | TechDev (技术研发) | Engineering, debugging, technical discussion |
| W3 | BusinessInsight (商业洞察) | Market analysis, strategy, competitive intel |
| W4 | WorkplaceComm (职场沟通) | HR, interpersonal dynamics, workplace communication |
| L1 | CourseLecture (课程讲座) | Formal lectures, educational content |
| L2 | ResearchDiscuss (研究探讨) | Academic discussion, paper review, research |
| L3 | SkillTraining (技能训练) | Tutorials, hands-on skill building |
| L4 | LanguageLearning (语言学习) | Language study, translation, vocabulary |
| D1 | MediaEntertain (媒体娱乐) | Podcasts, entertainment, casual content |
| D2 | LifeManagement (生活管理) | Planning, health, personal finance |
| D3 | SocialEmotion (社交情感) | Personal conversations, emotional support |
| D4 | KnowledgeExplore (知识探索) | Curiosity-driven discussion, general knowledge |

### Demand Types

Demands split into two types, reflecting the **Req/Ins distinction** used in model evaluation:

**Req (Requirement)** — explicit, clear-trigger needs:
- `decision_support` — recommendation or decision help
- `information_lookup` — fact retrieval, definitions
- `task_planning` — step-by-step planning, scheduling
- `problem_solving` — debugging, issue resolution
- `summarization` — key point extraction

**Ins (Insight)** — implicit, interpretive needs:
- `trend_insight` — emerging patterns without explicit ask
- `risk_warning` — proactively surfacing hidden risks
- `sentiment_analysis` — emotional state reading
- `knowledge_gap` — detecting and filling user's knowledge gap
- `context_synthesis` — synthesizing across multiple turns

---

## Data Source

Raw input: `/raw_new.jsonl`
- 2,042 sessions, all `speech_podcast` template
- Fields: `id`, `conversation_type`, `scenario`, `language`, `segments[{segment_id, content, speaker}]`
- Speakers: `microphone` (user's own voice) / `system` (external audio, other speakers)
- Languages: Chinese (cn), English (en), mixed
- After heuristic filtering: **~710 usable sessions** (≥15 segments, ≥500 chars)

---

## Pipeline Steps

```
raw_new.jsonl
     │
  [01] Filter & Classify          (gpt-5-nano)
     │
  [01b] Topic-Boundary Split      (gpt-5-nano)
     │
  [02] Semantic Turn Building     (gpt-5-nano for memory)
     │              │
  [03] Annotate   [04] Clean Text  (gpt-5-nano)
     │              │
     │           [03] Annotate     (gpt-5.2)
     │              │
  [05] Synthesize                  (gpt-5.2)
     │
  [03] Annotate                    (gpt-5.2)
     │
  [06] Format Output
     │
  data/final/
```

---

### Step 01 — Filter & Classify (`01_filter.py`)

**Purpose:** Remove low-quality sessions and assign each to a subcategory.

**Heuristic pre-filter:**
- Minimum 15 segments (sufficient for multi-turn structure)
- Maximum 2,000 segments (exclude transcription dumps)
- Minimum 500 characters total content

**LLM classification** (gpt-5-nano):
- Assigns subcategory (W1–D4), quality score (0–1), language, demand density
- Sessions with `demand_density=low` or `quality_score<0.5` are discarded
- Balanced across subcategories (up to 100 per subcategory)

**Output:** `data/01_filtered.jsonl`

---

### Step 01b — Topic-Boundary Split (`01b_split_topics.py`)

**Purpose:** Split long sessions into topically coherent sub-sessions.

**Motivation:** A single session often spans multiple unrelated topics. Splitting at topic boundaries ensures each benchmark item covers a coherent conversation thread, and allows more complete content coverage of long sessions.

**Algorithm:**
1. Slide a window every ~40 segments through the session
2. At each candidate split point (anchored to a sentence-ending segment), query gpt-5-nano: *"Is there a meaningful topic change between window A and window B?"*
3. Accept split if: topic change detected (confidence ≥ 0.6) AND both sides have ≥ 3,000 chars
4. Force-split at 12,000 chars even without a detected topic change

**Size constraints:** 3,000–12,000 chars per sub-session

**Effect:** ~710 sessions → ~1,150 sub-sessions (×1.6 expansion)

**Output:** `data/01b_subsessions.jsonl`

---

### Step 02 — Build Semantic Turns (`02_build_turns.py`)

**Purpose:** Convert raw segments into semantically coherent turns, then split into 60-turn benchmark items with memory.

#### Turn Definition

A **turn** is a sentence-group: consecutive segments accumulated until:
- Accumulated chars ≥ 40 AND the current segment ends with sentence-final punctuation (。？！.?!), **OR**
- Accumulated chars ≥ 150 (force break)

This yields turns of ~40–150 characters — roughly one complete thought — mirroring the decision window of a real-time speech AI.

#### Part Splitting with Memory

Each sub-session is capped at **60 turns per benchmark item**. If a sub-session produces more than 60 turns, it is split into consecutive parts:

- **Part 0:** Turns 1–60, `memory = null`
- **Part 1:** Turns 61–120, `memory = <LLM summary of turns 1–60>`
- **Part N:** Turns 60N+1–60(N+1), `memory = <LLM summary of all prior turns>`

Memory summaries (generated by gpt-5-nano, ~200–300 chars) capture:
- Main topics discussed
- Key decisions or facts established
- Unresolved questions or open threads

This design reflects how a deployed agent would maintain a running compressed context for long conversations.

**Output:** `data/02_turns.jsonl`

---

### Step 03 — Demand Annotation (`03_annotate.py`)

**Purpose:** For each turn, determine whether the AI should proactively intervene and what it should say.

**Model:** gpt-5.2 (complex reasoning required)

**Context passed to annotator:**
```
[Memory]           ← summary of previous parts (if any)
[History]          ← last 10 turns in full + brief summary of earlier turns
[Current Turn]     ← the turn being annotated
[Demand Type Hints]← subcategory-specific demand types
```

**Annotation schema per turn:**
```json
{
  "demands": [
    {
      "demand_type": "risk_warning",
      "category": "Ins",
      "trigger_text": "exact span from turn that motivated this",
      "proposed_response": "what the agent should proactively say (1-2 sentences)",
      "confidence": 0.82
    }
  ],
  "has_demand": true,
  "annotator_notes": "..."
}
```

**Selectivity:** The prompt explicitly instructs the model that *most turns should have no demand* (target rate: ~25–35%). Demands with confidence < 0.65 are automatically discarded in post-processing.

**Output:** `data/03_annotated.jsonl`

---

### Step 04 — Text Cleaning (`04_clean_text.py`)

**Purpose:** Produce the `real_clean` corpus by denoising raw ASR text.

**Two-pass cleaning:**
1. **Heuristic pass** (no LLM): remove repeated fillers (嗯嗯, um, uh), normalize CN/EN spacing, collapse repeated words
2. **LLM pass** (gpt-5-nano): deep cleaning for high-noise turns — fix broken sentences, remove non-lexical ASR artifacts, preserve all semantic content

All semantic content and named entities are preserved; only surface disfluencies are removed.

**Output:** `data/02_turns_clean.jsonl` → fed into Step 03 for `real_clean` annotation

---

### Step 05 — Synthetic Generation (`05_synthesize.py`)

**Purpose:** Generate ~1,000 synthetic conversations with known embedded demand triggers.

**Model:** gpt-5.2

**Per conversation:**
- Subcategory sampled uniformly across all 12
- Language sampled per subcategory distribution (≈60% CN, 20% EN, 20% mixed)
- 8–15 turns, 5–8 segments per turn
- 3–4 demand types seeded into the generation prompt
- Demands are woven organically into dialogue (not as explicit user requests)

Synthetic sessions are passed through the same Step 03 annotation pipeline to produce standardized labels.

**Output:** `data/05_synthetic.jsonl` → `data/05_annotated_synth.jsonl`

---

### Step 06 — Format Final Output (`06_format_output.py`)

**Purpose:** Merge, balance, and package the three corpora into final benchmark files.

**Multi-turn format:** Each session record:
```json
{
  "session_id": "speech_85d43_s0_p0",
  "source": "real_raw",
  "subcategory": "W2",
  "language": "cn",
  "part_index": 0,
  "memory": null,
  "n_turns": 60,
  "turns": [
    {
      "turn_id": 0,
      "turn_text": "[System] 那这三个东西怎么样就能预测准？靠监督还是靠什么呢？",
      "annotation": {
        "has_demand": false,
        "demands": []
      }
    },
    ...
  ]
}
```

**Single-turn derivation:** Each annotated turn becomes a single-turn item:
```json
{
  "item_id": "st_0042",
  "source": "real_raw",
  "subcategory": "W2",
  "language": "cn",
  "context": "[prior turns concatenated]",
  "current_turn": "[System] 我的赫兹可能是每秒只属于测5个TOKEN...",
  "label": 1,
  "demands": [...]
}
```

**Output files** in `data/final/`:

| File | Description |
|------|-------------|
| `bench_multiturn_real_raw.jsonl` | Multi-turn, real speech, raw ASR |
| `bench_multiturn_real_clean.jsonl` | Multi-turn, real speech, cleaned text |
| `bench_multiturn_synthetic.jsonl` | Multi-turn, synthetic conversations |
| `bench_singleturn.jsonl` | Single-turn items (derived, balanced) |

---

## Model Usage Summary

| Step | Task | Model | Justification |
|------|------|-------|---------------|
| 01 | Session classification | gpt-5-nano | Structured classification, low ambiguity |
| 01b | Topic boundary detection | gpt-5-nano | Binary judgment with short context |
| 02 | Memory summarization | gpt-5-nano | Straightforward extractive summarization |
| 03 | Demand annotation | gpt-5.2 | Requires nuanced pragmatic reasoning |
| 04 | Text cleaning | gpt-5-nano | Rule-following transformation task |
| 05 | Synthetic generation | gpt-5.2 | Creative generation with embedded constraints |

---

## Running the Pipeline

```bash
cd bench/
pip install -r requirements.txt

# Full run
bash run_pipeline.sh

# Debug run (first 10 sessions only)
MAX_SESSIONS=10 bash run_pipeline.sh

# Step by step
python 01_filter.py --max-sessions 20
python 01b_split_topics.py
python 02_build_turns.py
python 03_annotate.py --max-sessions 5
```

**Inspect outputs at any stage:**
```bash
python stats.py data/01_filtered.jsonl
python stats.py data/02_turns.jsonl
python stats.py data/03_annotated.jsonl --show-demands
python stats.py data/final/bench_singleturn.jsonl --sample 3
```

---

## Expected Scale

| Stage | Count | Notes |
|-------|-------|-------|
| Raw sessions | 2,042 | All input |
| After heuristic filter | ~710 | ≥15 segs, ≥500 chars |
| After quality filter | ~500–600 | LLM quality + density filter |
| After topic split | ~800–1,000 | ×1.6 expansion |
| Benchmark items (multi-turn) | ~3,000 | 3 corpora × ~1,000 each |
| Turns total | ~150,000 | Across all multi-turn items |
| Demands annotated | ~40,000 | At ~25–35% demand density |
| Single-turn items | ~2,000–3,000 | Derived, balanced pos/neg |
