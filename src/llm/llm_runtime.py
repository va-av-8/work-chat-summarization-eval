from __future__ import annotations

import json
import os
import random
import time
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import requests


# =========================
# Small utilities
# =========================

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def jsonl_append(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def build_cache_key(model_name: str, system_prompt: str, user_prompt: str) -> str:
    return sha256_text(model_name + "\n" + system_prompt + "\n" + user_prompt)


def retry_sleep(attempt: int, base: float = 1.0, cap: float = 30.0) -> None:
    t = min(cap, base * (2 ** attempt))
    t = t * (0.75 + random.random() * 0.5)
    time.sleep(t)


# =========================
# Backend protocol + configs
# =========================

class LLMBackend(Protocol):
    name: str
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str: ...


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    timeout_s: int = 180


@dataclass
class OpenAIConfig:
    api_key: Optional[str] = None
    timeout_s: int = 180


@dataclass
class MistralConfig:
    api_key: Optional[str] = None
    base_url: str = "https://api.mistral.ai/v1"
    timeout_s: int = 180


# =========================
# Concrete backends
# =========================

class OllamaChatBackend:
    """
    Ollama /api/chat backend.
    """
    def __init__(self, model: str, cfg: Optional[OllamaConfig] = None):
        self.model = model
        self.cfg = cfg or OllamaConfig()
        self.name = f"ollama:{model}"

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        url = self.cfg.base_url.rstrip("/") + "/api/chat"

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }

        # allow ollama options (temperature, top_p, num_predict, etc.)
        opts = kwargs.get("options")
        if isinstance(opts, dict):
            payload["options"] = opts

        r = requests.post(url, json=payload, timeout=self.cfg.timeout_s)
        r.raise_for_status()
        data = r.json()

        msg = (data.get("message") or {}).get("content")
        if isinstance(msg, str):
            return msg.strip()

        # fallback shapes
        if isinstance(data.get("response"), str):
            return data["response"].strip()

        return json.dumps(data, ensure_ascii=False)


class OpenAIResponsesBackend:
    """
    OpenAI Responses API backend (official SDK).
    """
    def __init__(self, model: str, cfg: Optional[OpenAIConfig] = None):
        self.model = model
        self.cfg = cfg or OpenAIConfig(api_key=os.getenv("OPENAI_API_KEY"))
        self.name = f"openai:{model}"

        from openai import OpenAI
        self._client = OpenAI(api_key=self.cfg.api_key)

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        temperature = kwargs.get("temperature", 0.0)

        resp = self._client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=user_prompt,
            temperature=temperature,
            max_output_tokens=kwargs.get("max_output_tokens"),
            store=kwargs.get("store", False),
        )
        return (resp.output_text or "").strip()


class MistralChatBackend:
    """
    Mistral Chat Completions via HTTP.
    """
    def __init__(self, model: str, cfg: Optional[MistralConfig] = None):
        self.model = model
        self.cfg = cfg or MistralConfig(api_key=os.getenv("MISTRAL_API_KEY"))
        self.name = f"mistral:{model}"

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": kwargs.get("temperature", 0.0),
        }

        max_tokens = kwargs.get("max_tokens")
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        r = requests.post(url, headers=headers, json=payload, timeout=self.cfg.timeout_s)
        r.raise_for_status()
        data = r.json()

        choices = data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            msg = (choices[0].get("message") or {}).get("content")
            if isinstance(msg, str):
                return msg.strip()

        return json.dumps(data, ensure_ascii=False)


# =========================
# Generator runner with cache + retries
# =========================

@dataclass
class GenerationResult:
    raw_text: str
    latency_s: float
    cache_hit: bool
    error: Optional[str] = None


class DiskCache:
    """
    Simple JSON cache: key -> raw_text
    """
    def __init__(self, path: str):
        self.path = path
        self.data: Dict[str, str] = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.data = json.load(f) or {}
            except Exception:
                self.data = {}

    def get(self, key: str) -> Optional[str]:
        return self.data.get(key)

    def set(self, key: str, value: str) -> None:
        self.data[key] = value

    def flush(self) -> None:
        ensure_dir(os.path.dirname(self.path) or ".")
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)


def generate_with_cache_and_retries(
    backend: LLMBackend,
    cache: DiskCache,
    system_prompt: str,
    user_prompt: str,
    *,
    max_retries: int = 3,
    retry_base_s: float = 1.0,
    retry_cap_s: float = 30.0,
    **gen_kwargs,
) -> GenerationResult:
    key = build_cache_key(backend.name, system_prompt, user_prompt)
    cached = cache.get(key)
    if cached is not None:
        return GenerationResult(raw_text=cached, latency_s=0.0, cache_hit=True, error=None)

    last_err: Optional[str] = None
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            text = backend.generate(system_prompt=system_prompt, user_prompt=user_prompt, **gen_kwargs)
            dt = time.time() - t0
            cache.set(key, text)
            return GenerationResult(raw_text=text, latency_s=dt, cache_hit=False, error=None)
        except Exception as e:
            last_err = repr(e)
            if attempt < max_retries - 1:
                retry_sleep(attempt, base=retry_base_s, cap=retry_cap_s)

    return GenerationResult(raw_text="", latency_s=0.0, cache_hit=False, error=last_err)
