"""Zentropi SLM client wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import httpx

from src.config import settings


@dataclass
class ZentropiResult:
    label: Optional[str]
    confidence: float
    raw: Dict[str, Any]


class ZentropiClient:
    """Thin client for Zentropi label API."""

    def __init__(self):
        self.api_key = self._clean(settings.zentropi_api_key)
        self.labeler_id = self._clean(settings.zentropi_labeler_id)
        self.labeler_version_id = self._clean(settings.zentropi_labeler_version_id)
        self.base_url = "https://api.zentropi.ai/v1/label"

    def is_configured(self) -> bool:
        return all([self.api_key, self.labeler_id, self.labeler_version_id])

    @staticmethod
    def _clean(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None

    def label(self, content_text: str, criteria_text: Optional[str] = None) -> ZentropiResult:
        if not self.is_configured():
            if not self.api_key:
                raise ValueError("Zentropi is not configured. Set ZENTROPI_API_KEY.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "content_text": content_text,
        }
        if self.labeler_id and self.labeler_version_id:
            payload["labeler_id"] = self.labeler_id
            payload["labeler_version_id"] = self.labeler_version_id
        if criteria_text:
            payload["criteria_text"] = criteria_text

        timeout = settings.slm_timeout_s
        with httpx.Client(timeout=timeout) as client:
            response = client.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        label = data.get("label") or data.get("predicted_label")
        confidence = data.get("confidence") or data.get("score") or 0.0
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        return ZentropiResult(label=label, confidence=confidence, raw=data)
