"""
LLM client wrapper for OpenRouter / gpt-5.2.

Supports:
- Single completion calls
- Structured JSON output (with retry + validation)
- Async batch calls with concurrency control
- Streaming (optional)
"""

import os
import json
import time
import asyncio
import logging
from typing import Any, Optional
from pathlib import Path

from openai import OpenAI, AsyncOpenAI

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

def _load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

_load_env()

OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY  = os.environ.get("OPENROUTER_API_KEY", "")
OPENAI_BASE_URL     = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY      = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_MODEL       = "gpt-5.2"
NANO_MODEL          = "gpt-4.1-nano"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS  = 2048


def _pick_credentials(model: str) -> tuple[str, str]:
    """Pick API base_url and key based on model name.
    Models with provider prefix (contain '/') use OpenRouter; plain model names use OpenAI endpoint."""
    if "/" in model:
        return OPENROUTER_BASE_URL, OPENROUTER_API_KEY
    return OPENAI_BASE_URL, OPENAI_API_KEY

# ------------------------------------------------------------------
# Sync client
# ------------------------------------------------------------------

class LLMClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        base_url, api_key = _pick_credentials(model)
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, messages: list[dict], **kwargs) -> str:
        """Return raw text completion."""
        params = dict(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(**params)
                content = resp.choices[0].message.content
                if content is None:
                    raise ValueError("LLM returned empty content")
                return content.strip()
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise

    def complete_json(
        self,
        messages: list[dict],
        schema_hint: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Return parsed JSON. Retries on parse failure."""
        if schema_hint:
            messages = list(messages)
            messages[-1] = dict(messages[-1])
            messages[-1]["content"] += f"\n\nRespond ONLY with valid JSON matching: {schema_hint}"
        for attempt in range(self.max_retries):
            text = self.complete(messages, **kwargs)
            # Strip markdown code fences if present
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
                text = text.rsplit("```", 1)[0]
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse failed (attempt {attempt+1}): {e}\nraw: {text[:200]}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise ValueError(f"Failed to parse JSON after {self.max_retries} attempts") from e


# ------------------------------------------------------------------
# Async client
# ------------------------------------------------------------------

class AsyncLLMClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        concurrency: int = 8,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._semaphore = asyncio.Semaphore(concurrency)
        base_url, api_key = _pick_credentials(model)
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(self, messages: list[dict], **kwargs) -> str:
        params = dict(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        async with self._semaphore:
            for attempt in range(self.max_retries):
                try:
                    resp = await self._client.chat.completions.create(**params)
                    content = resp.choices[0].message.content
                    if content is None:
                        raise ValueError("LLM returned empty content")
                    return content.strip()
                except Exception as e:
                    logger.warning(f"Async LLM call failed (attempt {attempt+1}): {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                    else:
                        raise

    async def complete_json(self, messages: list[dict], schema_hint: Optional[str] = None, **kwargs) -> Any:
        if schema_hint:
            messages = list(messages)
            messages[-1] = dict(messages[-1])
            messages[-1]["content"] += f"\n\nRespond ONLY with valid JSON matching: {schema_hint}"
        for attempt in range(self.max_retries):
            text = await self.complete(messages, **kwargs)
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
                text = text.rsplit("```", 1)[0]
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise

    async def batch(self, tasks: list[dict]) -> list[Any]:
        """
        Run many tasks concurrently. Each task: {"messages": [...], "schema_hint": ..., **kwargs}
        Returns list of results in same order (None on unrecoverable error).
        """
        async def _run(task):
            try:
                task = dict(task)  # shallow copy to avoid mutating original
                schema_hint = task.pop("schema_hint", None)
                messages = task.pop("messages")
                if schema_hint is not None:
                    return await self.complete_json(messages, schema_hint=schema_hint, **task)
                return await self.complete(messages, **task)
            except Exception as e:
                logger.error(f"Batch task failed: {e}")
                return None

        return await asyncio.gather(*[_run(t) for t in tasks])


# ------------------------------------------------------------------
# Convenience factory
# ------------------------------------------------------------------

def get_client(async_mode: bool = False, nano: bool = False, **kwargs) -> "LLMClient | AsyncLLMClient":
    if "model" not in kwargs:
        kwargs["model"] = NANO_MODEL if nano else DEFAULT_MODEL
    if async_mode:
        return AsyncLLMClient(**kwargs)
    return LLMClient(**kwargs)
