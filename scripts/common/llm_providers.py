# -*- coding: utf-8 -*-
"""
llm_providers.py -- provider abstraction shared by the model-call scripts.

Three providers, selected with --provider:
  openai    -> GPT-4o           (OpenAI SDK)          env OPENAI_API_KEY
  deepseek  -> deepseek-chat    (OpenAI SDK, diff base_url)  env DEEPSEEK_API_KEY
  anthropic -> claude-opus-4-8  (Anthropic SDK)       env ANTHROPIC_API_KEY

Design goals
  - Each provider writes to a separate output file when a suffix is needed,
    so existing response artifacts are not overwritten accidentally.
  - openai/deepseek share one code path (DeepSeek is OpenAI-compatible: same
    SDK, only base_url + model + key differ).
  - anthropic uses its own SDK. It has NO `seed` parameter; repeats vary via
    temperature only. `max_tokens` is required (already set tiny).
  - A shared (raw_text, error) return contract keeps the run loops uniform.
"""
from __future__ import annotations

import os
import random
import sys
import time
from typing import Any


# ---- provider defaults --------------------------------------------------
PROVIDER_DEFAULTS = {
    "openai":    {"model": "gpt-4o",          "env": "OPENAI_API_KEY",    "base_url": None},
    "deepseek":  {"model": "deepseek-chat",   "env": "DEEPSEEK_API_KEY",  "base_url": "https://api.deepseek.com"},
    "anthropic": {"model": "claude-opus-4-8", "env": "ANTHROPIC_API_KEY", "base_url": None},
}

# filename suffix per provider (openai keeps the original bare names)
PROVIDER_SUFFIX = {"openai": "", "deepseek": "_deepseek", "anthropic": "_claude"}


def default_model(provider: str) -> str:
    return PROVIDER_DEFAULTS[provider]["model"]


def out_path_for(base_live: str, provider: str) -> str:
    """responses.jsonl + deepseek -> responses_deepseek.jsonl ; openai -> unchanged."""
    suffix = PROVIDER_SUFFIX[provider]
    if not suffix:
        return base_live
    root, dot, ext = base_live.rpartition(".")
    return f"{root}{suffix}.{ext}" if dot else f"{base_live}{suffix}"


def make_client(provider: str) -> Any:
    """Build and return an SDK client for the provider (reads key from env)."""
    cfg = PROVIDER_DEFAULTS[provider]
    key = os.environ.get(cfg["env"])
    if not key:
        sys.exit(f"{cfg['env']} not set in environment. Set it (do NOT hardcode) and re-run.")

    if provider in ("openai", "deepseek"):
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("openai package not installed: pip install openai")
        if cfg["base_url"]:
            return OpenAI(api_key=key, base_url=cfg["base_url"])
        return OpenAI(api_key=key)

    if provider == "anthropic":
        try:
            from anthropic import Anthropic
        except ImportError:
            sys.exit("anthropic package not installed: pip install anthropic")
        return Anthropic(api_key=key)

    sys.exit(f"unknown provider: {provider}")


def _is_auth_error(msg: str) -> bool:
    low = msg.lower()
    return "401" in msg or "invalid_api_key" in low or "authentication" in low


def call_model(
    client: Any,
    provider: str,
    *,
    prompt: str,
    seed: int,
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int = 5,
    backoff_base: float = 2.0,
) -> tuple[str | None, str | None]:
    """One live call with retry/backoff. Returns (raw_text, error_or_None).

    `seed` is passed through for openai/deepseek and ignored for anthropic
    (Anthropic has no seed param; repeats vary via temperature).
    """
    for attempt in range(max_retries):
        try:
            if provider in ("openai", "deepseek"):
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                )
                return resp.choices[0].message.content, None

            # anthropic
            # NOTE: Opus 4.8 (and current-gen Claude models generally) run
            # adaptive thinking by default, and the API REJECTS a non-default
            # temperature while thinking is active (400 invalid_request_error:
            # "temperature may only be set to 1 when thinking is enabled").
            # Thinking cannot be disabled on this model family, so we omit
            # temperature entirely rather than sending 0.7. Repeat-to-repeat
            # variation for the R=3 median comes from the model's own decoding
            # stochasticity, not from a temperature knob -- note this in the
            # repository notes as a cross-provider caveat.
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            # content is a list of blocks; concatenate any text blocks
            parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
            return ("".join(parts) if parts else ""), None

        except Exception as e:  # noqa: BLE001 - SDKs raise heterogeneous types
            msg = str(e)
            if _is_auth_error(msg):
                return None, f"AUTH ERROR: {msg}"
            wait = backoff_base * (2 ** attempt) + random.uniform(0, 1)
            sys.stderr.write(
                f"  retry {attempt + 1}/{max_retries} after error: {msg} "
                f"(sleep {wait:.1f}s)\n"
            )
            time.sleep(wait)
    return None, "MAX_RETRIES exceeded"
