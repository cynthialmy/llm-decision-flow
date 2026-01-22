"""Groq LLM client wrapper (OpenAI-compatible)."""
from __future__ import annotations

from typing import Optional, Dict, Any
from openai import OpenAI
import hashlib

from src.config import settings


class GroqClient:
    """Thin wrapper for Groq chat completions."""

    def __init__(self):
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is required for Groq client.")
        self.client = OpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = settings.groq_model

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        max_tokens = max_tokens or settings.claim_max_tokens
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=settings.frontier_timeout_s,
        )
        content = response.choices[0].message.content or ""

        return {
            "content": content,
            "model": self.model,
            "prompt_hash": self._hash_prompt(system_prompt or "", prompt),
        }

    @staticmethod
    def _hash_prompt(system_prompt: str, user_prompt: str) -> str:
        raw = f"{system_prompt}\n\n{user_prompt}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()
