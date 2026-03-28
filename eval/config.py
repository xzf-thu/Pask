"""Model definitions and evaluation config."""

import os

MODELS = [
    {"id": "google/gemini-3-flash-preview",          "name": "gemini-3-flash",         "input": 0.5,   "output": 3.0},
    {"id": "google/gemini-2.5-flash-lite",           "name": "gemini-2.5-flash-lite",  "input": 0.1,   "output": 0.4},
    {"id": "gpt-5-mini",                              "name": "gpt-5-mini",             "input": 0.25,  "output": 2.0,  "reasoning": "minimal"},
    {"id": "gpt-5-nano",                              "name": "gpt-5-nano",             "input": 0.05,  "output": 0.4,  "reasoning": "minimal"},
    {"id": "openai/gpt-oss-120b",                    "name": "gpt-oss-120b",           "input": 0.0,   "output": 0.0,  "reasoning": "minimal"},
    {"id": "anthropic/claude-haiku-4.5",             "name": "claude-haiku-4.5",       "input": 1.0,   "output": 5.0},
    {"id": "qwen/qwen3.5-flash-02-23",              "name": "qwen3.5-flash",          "input": 0.1,   "output": 0.4,  "max_tokens": 4000},
    {"id": "deepseek/deepseek-v3.2",                "name": "deepseek-v3.2",          "input": 0.25,  "output": 0.4},
    # Local vLLM model — set VLLM_BASE_URL env var before running (e.g. http://localhost:9000/v1)
    {"id": "qwen3-30b-a3b",                         "name": "qwen3-30b-a3b",          "input": 0.0,   "output": 0.0,  "base_url": os.environ.get("VLLM_BASE_URL", "http://localhost:9000/v1"), "api_key": "EMPTY", "max_tokens": 4000},
    # {"id": "google/gemini-3.1-flash-lite-preview",  "name": "gemini-3.1-flash-lite",  "input": 0.25,  "output": 1.5},
    # {"id": "minimax/minimax-m2.1",                   "name": "minimax-m2.1",           "input": 0.27,  "output": 0.95, "max_tokens": 4000},
    # {"id": "z-ai/glm-4.7-flash",                    "name": "glm-4.7-flash",          "input": 0.06,  "output": 0.4,  "max_tokens": 4000},
    # {"id": "bytedance-seed/seed-1.6-flash",         "name": "seed-1.6-flash",         "input": 0.075, "output": 0.3,  "max_tokens": 4000},
]

# Judge model for scoring
JUDGE_MODEL = "gpt-4.1-mini"

NO_DEMAND_TOKEN = "[NO_DEMAND]"
