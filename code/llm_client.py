from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, repr=False)
class LLMProvider:
    name: str
    api_key_env: str
    base_url: str
    model: str
    api_style: str

    def __repr__(self) -> str:
        return (
            f"LLMProvider(name={self.name!r}, api_key_env={self.api_key_env!r}, "
            f"model={self.model!r}, api_style={self.api_style!r})"
        )


PROVIDERS = [
    ("openai", "OPENAI_API_KEY", "https://api.openai.com/v1", "gpt-4o-mini", "openai"),
    ("anthropic", "ANTHROPIC_API_KEY", "https://api.anthropic.com/v1", "claude-3-5-haiku-latest", "anthropic"),
    ("gemini", "GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta", "gemini-3-flash-preview", "gemini"),
    ("gemini", "GOOGLE_API_KEY", "https://generativelanguage.googleapis.com/v1beta", "gemini-3.1-flash-lite-preview", "gemini"),
    ("openrouter", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "openai/gpt-4o-mini", "openai"),
    ("groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1", "llama-3.1-8b-instant", "openai"),
    ("mistral", "MISTRAL_API_KEY", "https://api.mistral.ai/v1", "mistral-small-latest", "openai"),
    ("together", "TOGETHER_API_KEY", "https://api.together.xyz/v1", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", "openai"),
    ("cohere", "COHERE_API_KEY", "https://api.cohere.com/v2", "command-r", "cohere"),
    ("azure_openai", "AZURE_OPENAI_API_KEY", "", "", "azure_openai"),
    ("openai_compatible", "OPENAI_COMPATIBLE_API_KEY", "", "gpt-4o-mini", "openai"),
]


def detect_provider(env: Mapping[str, str] | None = None) -> LLMProvider | None:
    values = env if env is not None else os.environ
    for name, key_env, default_base_url, default_model, api_style in PROVIDERS:
        if not values.get(key_env):
            continue
        if name == "azure_openai":
            endpoint = values.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
            deployment = values.get("AZURE_OPENAI_DEPLOYMENT", "")
            if endpoint and deployment:
                return LLMProvider(name, key_env, endpoint, deployment, api_style)
            continue
        if name == "openai_compatible":
            base_url = values.get("OPENAI_COMPATIBLE_BASE_URL", "").rstrip("/")
            if not base_url:
                continue
            model = values.get("OPENAI_COMPATIBLE_MODEL", default_model)
            return LLMProvider(name, key_env, base_url, model, api_style)
        model = values.get(f"{key_env.removesuffix('_API_KEY')}_MODEL", default_model)
        base_url = values.get(f"{key_env.removesuffix('_API_KEY')}_BASE_URL", default_base_url).rstrip("/")
        return LLMProvider(name, key_env, base_url, model, api_style)
    return None


class OptionalLLMClient:
    def __init__(self, env: Mapping[str, str] | None = None, timeout_seconds: int = 20):
        self.env = env if env is not None else os.environ
        self.provider = detect_provider(self.env)
        self.timeout_seconds = timeout_seconds
        self.generation_attempts = 0
        self.successful_generations = 0

    @property
    def enabled(self) -> bool:
        return self.provider is not None

    @property
    def provider_name(self) -> str:
        return self.provider.name if self.provider is not None else "none"

    @property
    def generation_used(self) -> bool:
        return self.successful_generations > 0

    def generate(self, prompt: str) -> str | None:
        if self.provider is None:
            return None
        self.generation_attempts += 1
        try:
            text = None
            if self.provider.api_style in {"openai", "azure_openai"}:
                text = self._chat_completions(prompt)
            if self.provider.api_style == "anthropic":
                text = self._anthropic(prompt)
            if self.provider.api_style == "gemini":
                text = self._gemini(prompt)
            if self.provider.api_style == "cohere":
                text = self._cohere(prompt)
            if text:
                self.successful_generations += 1
                return text
        except (OSError, urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError):
            return None
        return None

    def _chat_completions(self, prompt: str) -> str | None:
        assert self.provider is not None
        if self.provider.api_style == "azure_openai":
            version = self.env.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
            url = (
                f"{self.provider.base_url}/openai/deployments/{self.provider.model}"
                f"/chat/completions?api-version={version}"
            )
            headers = {"api-key": self._api_key(), "Content-Type": "application/json"}
            model_field = self.provider.model
        else:
            url = f"{self.provider.base_url}/chat/completions"
            headers = {"Authorization": f"Bearer {self._api_key()}", "Content-Type": "application/json"}
            model_field = self.provider.model
        payload = {
            "model": model_field,
            "temperature": 0.0,
            "max_tokens": 350,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You rewrite support replies using only the supplied local context. "
                        "Do not add facts, policies, links, or steps that are not present."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        data = self._post_json(url, headers, payload)
        return data["choices"][0]["message"]["content"].strip()

    def _anthropic(self, prompt: str) -> str | None:
        assert self.provider is not None
        payload = {
            "model": self.provider.model,
            "max_tokens": 350,
            "temperature": 0.0,
            "system": "Use only the supplied local support context. Do not invent policy.",
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": self._api_key(),
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        data = self._post_json(f"{self.provider.base_url}/messages", headers, payload)
        return data["content"][0]["text"].strip()

    def _gemini(self, prompt: str) -> str | None:
        assert self.provider is not None
        url = f"{self.provider.base_url}/models/{self.provider.model}:generateContent?key={self._api_key()}"
        payload = {
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 350},
            "contents": [{"parts": [{"text": prompt}]}],
        }
        data = self._post_json(url, {"Content-Type": "application/json"}, payload)
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    def _cohere(self, prompt: str) -> str | None:
        assert self.provider is not None
        payload = {
            "model": self.provider.model,
            "temperature": 0.0,
            "max_tokens": 350,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {"Authorization": f"Bearer {self._api_key()}", "Content-Type": "application/json"}
        data = self._post_json(f"{self.provider.base_url}/chat", headers, payload)
        return data["message"]["content"][0]["text"].strip()

    def _post_json(self, url: str, headers: dict[str, str], payload: dict[str, object]) -> dict[str, object]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def _api_key(self) -> str:
        assert self.provider is not None
        value = self.env.get(self.provider.api_key_env, "")
        if not value:
            raise ValueError("LLM provider is missing its configured API key")
        return value
