"""Claim Agent: Extracts factual claims and tags domains."""
from typing import List, Tuple
from pydantic import BaseModel
import time
import logging
from src.agents.base import BaseAgent
from src.agents.prompt_registry import render_prompt
from src.governance.system_config_store import get_prompt_overrides
from src.models.schemas import Claim, Domain, AgentExecutionDetail
from src.llm.groq_client import GroqClient
from src.config import get_settings

logger = logging.getLogger(__name__)


class ClaimAgent(BaseAgent):
    """Agent for extracting factual claims from transcripts."""

    def process(self, transcript: str) -> Tuple[List[Claim], AgentExecutionDetail]:
        """
        Extract explicit and implicit factual claims from transcript.
        Uses Groq when GROQ_API_KEY is set and valid; falls back to Azure on failure.
        """
        prompt_overrides = get_prompt_overrides()
        system_prompt = render_prompt("claim", "system_prompt", {}, overrides=prompt_overrides)
        user_prompt = render_prompt(
            "claim",
            "user_prompt",
            {"transcript": transcript},
            overrides=prompt_overrides
        )

        class ClaimResponse(BaseModel):
            claims: List[Claim]

        settings = get_settings()
        start_time = time.perf_counter()
        content: str
        model_name: str
        model_provider: str
        prompt_hash: str

        if settings.groq_api_key and settings.groq_api_key.strip():
            try:
                groq = GroqClient()
                response_data = groq.chat(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.2,
                    max_tokens=settings.claim_max_tokens
                )
                content = response_data["content"]
                model_name = response_data.get("model", settings.groq_model or "groq")
                model_provider = "groq"
                prompt_hash = response_data.get("prompt_hash", "")
            except Exception as e:
                err_str = str(e).lower()
                if "401" in err_str or "invalid" in err_str or "api_key" in err_str or "api key" in err_str:
                    logger.warning(
                        "Groq returned invalid API key or 401; falling back to Azure for claim extraction. "
                        "Set a valid GROQ_API_KEY in Secrets or remove it to use Azure only."
                    )
                else:
                    logger.warning("Groq call failed, falling back to Azure: %s", e)
                content = self._call_llm(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.2,
                    max_tokens=settings.claim_max_tokens
                )
                model_name = settings.azure_openai_deployment_name or "azure"
                model_provider = "azure_openai"
                prompt_hash = ""
        else:
            content = self._call_llm(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.2,
                max_tokens=settings.claim_max_tokens
            )
            model_name = settings.azure_openai_deployment_name or "azure"
            model_provider = "azure_openai"
            prompt_hash = ""

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        response = self._parse_structured_output(content, ClaimResponse)

        detail = AgentExecutionDetail(
            agent_name="Claim Agent",
            agent_type="claim",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=model_name,
            model_provider=model_provider,
            prompt_hash=prompt_hash,
            confidence=self._aggregate_claim_confidence(response.claims),
            execution_time_ms=elapsed_ms,
            status="completed",
            policy_version=settings.policy_version
        )

        return response.claims, detail

    @staticmethod
    def _aggregate_claim_confidence(claims: List[Claim]) -> float:
        if not claims:
            return 0.0
        return sum([claim.confidence for claim in claims]) / len(claims)
