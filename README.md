# IntentFlow

Proactive demand detection in multi-turn conversations — identifying what the user needs *without being explicitly asked*.

## Project Structure

```
bench/          # Benchmark construction pipeline
  data/         # LatentNeeds-Bench.jsonl (final benchmark)
  llm_client.py # API client (OpenAI / OpenRouter / local vLLM)
  01_filter.py → 03d_annotate.py  # Data pipeline steps

eval/           # Model evaluation framework
  config.py     # Model definitions (9 baselines + IntentFlow)
  prompts.py    # 3-level prompt templates (encouraging/neutral/suppressing)
  run.py        # Run inference across models
  score.py      # Two-round LLM-as-judge scoring
  report.py     # Generate comparison report

latex/          # Paper artifacts
  latex_fill.py # Auto-generate LaTeX tables from score data
  plot.py       # Publication-quality figures
  output/       # Generated .tex files
  figure/       # Generated figures (PDF)
```

## LatentNeeds-Bench

100 sessions, 3,936 turns, 10 subcategories across 3 domains (Work, Learning, Daily). Each turn is annotated with demand labels (Requirement / Insight types). See `bench/PIPELINE.md` for construction details.

## Results

### Main Results

| Model | Type | Best Level | Balanced Acc |
|-------|------|-----------|-------------|
| **IntentFlow** | Ours | neutral | **84.2** |
| Gemini-3-Flash | API | suppressing | 80.8 |
| GPT-5-Mini | API | neutral | 77.2 |
| GPT-5-Nano | API | encouraging | 71.5 |
| GPT-oss-120b | Open | suppressing | 70.3 |
| Claude-Haiku-4.5 | API | encouraging | 66.2 |
| DeepSeek-V3.2 | Open | encouraging | 61.6 |
| Qwen3.5-Flash | API | neutral | 61.1 |
| Qwen3-30B-A3B | Local | encouraging | 58.9 |
| Gemini-2.5-Flash-Lite | API | encouraging | 52.1 |

### Accuracy vs. Conversation Depth

![Multi-turn accuracy degradation](latex/figure/multi_turn_combined.png)

*Left: per-bucket accuracy (4 turns each). Right: cumulative accuracy. IntentFlow maintains >80% balanced accuracy across all turn positions, while baselines degrade significantly.*

## Evaluation

```bash
# Run a model on all 3 prompt levels
python -m eval.run --models gpt-5-mini --level all

# Run local vLLM model
python -m eval.run --models qwen3-30b-a3b --level all

# Score results with LLM-as-judge
python -m eval.score --models gpt-5-mini

# Print comparison report
python -m eval.report

# Generate LaTeX tables and figures
python -m latex.latex_fill
python -m latex.plot
```
