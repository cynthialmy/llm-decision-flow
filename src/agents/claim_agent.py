"""Claim Agent: Extracts factual claims and tags domains."""
from typing import List, Tuple
from pydantic import BaseModel
import time
from src.agents.base import BaseAgent
from src.agents.prompt_registry import render_prompt
from src.governance.system_config_store import get_prompt_overrides
from src.models.schemas import Claim, Domain, AgentExecutionDetail
from src.llm.groq_client import GroqClient
from src.config import get_settings


class ClaimAgent(BaseAgent):
    """Agent for extracting factual claims from transcripts."""

    def process(self, transcript: str) -> Tuple[List[Claim], AgentExecutionDetail]:
        """
        Extract explicit and implicit factual claims from transcript.

        Args:
            transcript: Content transcript text

        Returns:
            List of extracted claims with domain tags
        """
        prompt_overrides = get_prompt_overrides()
        system_prompt = render_prompt("claim", "system_prompt", {}, overrides=prompt_overrides)
        user_prompt = render_prompt(
            "claim",
            "user_prompt",
            {"transcript": transcript},
            overrides=prompt_overrides
        )

        # Define output model
        class ClaimResponse(BaseModel):
            claims: List[Claim]

        groq = GroqClient()
        start_time = time.perf_counter()
        response_data = groq.chat(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=get_settings().claim_max_tokens
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        response = self._parse_structured_output(response_data["content"], ClaimResponse)

        detail = AgentExecutionDetail(
            agent_name="Claim Agent",
            agent_type="claim",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=response_data.get("model"),
            model_provider="groq",
            prompt_hash=response_data.get("prompt_hash"),
            confidence=self._aggregate_claim_confidence(response.claims),
            execution_time_ms=elapsed_ms,
            status="completed",
            policy_version=get_settings().policy_version
        )

        return response.claims, detail

    @staticmethod
    def _aggregate_claim_confidence(claims: List[Claim]) -> float:
        if not claims:
            return 0.0
        return sum([claim.confidence for claim in claims]) / len(claims)
