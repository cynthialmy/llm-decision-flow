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
        self.api_key = settings.zentropi_api_key
        self.labeler_id = settings.zentropi_labeler_id
        self.labeler_version_id = settings.zentropi_labeler_version_id
        self.base_url = "https://api.zentropi.ai/v1/label"

    def is_configured(self) -> bool:
        return all([self.api_key, self.labeler_id, self.labeler_version_id])

    def label(self, content_text: str) -> ZentropiResult:
        if not self.is_configured():
            raise ValueError("Zentropi is not configured. Set ZENTROPI_API_KEY and labeler IDs.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "content_text": content_text,
            "labeler_id": self.labeler_id,
            "labeler_version_id": self.labeler_version_id,
        }

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
