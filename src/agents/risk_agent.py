"""Risk Agent: Assesses risk tier based on potential harm, exposure, and vulnerable populations."""
from typing import Tuple, Optional
import time
from src.agents.base import BaseAgent
from src.models.schemas import RiskAssessment, RiskTier, Claim, AgentExecutionDetail
from src.llm.zentropi_client import ZentropiClient
from src.config import settings


class RiskAgent(BaseAgent):
    """Agent for assessing risk tier of content."""

    def process(self, transcript: str, claims: list[Claim]) -> Tuple[RiskAssessment, AgentExecutionDetail]:
        """
        Assess risk tier based on potential harm, exposure, and vulnerable populations.

        IMPORTANT: This agent does NOT see evidence or apply policy.
        It only assesses risk based on the content itself.

        Args:
            transcript: Content transcript text
            claims: List of extracted claims

        Returns:
            RiskAssessment with tier and reasoning
        """
        system_prompt = """You are a risk assessment agent. Your role is to assess the potential risk of content based on:

1. Potential harm: What harm could this content cause if false or misleading?
2. Estimated exposure: How many people might see this content?
3. Vulnerable populations: Which groups might be particularly affected?

IMPORTANT CONSTRAINTS:
- You do NOT have access to evidence about truthfulness
- You do NOT apply policy rules
- You assess risk based solely on the content's potential impact
- Risk tiers: Low, Medium, High
- Be conservative - err on the side of higher risk if uncertain

Return a JSON object with:
- "tier": "Low", "Medium", or "High"
- "reasoning": explanation of risk assessment
- "confidence": float between 0.0 and 1.0
- "potential_harm": description of potential harm
- "estimated_exposure": description of exposure level
- "vulnerable_populations": array of affected vulnerable groups"""

        claims_text = "\n".join([f"- {claim.text} ({claim.domain.value})" for claim in claims])

        user_prompt = f"""Assess the risk of the following content:

Transcript:
{transcript}

Extracted Claims:
{claims_text}

Return a JSON object with this structure:
{{
  "tier": "Low|Medium|High",
  "reasoning": "detailed reasoning",
  "confidence": 0.72,
  "potential_harm": "description of potential harm",
  "estimated_exposure": "description of exposure level",
  "vulnerable_populations": ["group1", "group2"]
}}"""

        start_time = time.perf_counter()
        zentropi = ZentropiClient()
        slm_result: Optional[RiskAssessment] = None
        fallback_used = False
        route_reason = "slm_primary"

        if zentropi.is_configured():
            try:
                content = f"{transcript}\n\nClaims:\n{claims_text}"
                slm_response = zentropi.label(content)
                slm_tier = self._map_label_to_risk_tier(slm_response.label)
                if slm_tier:
                    slm_result = RiskAssessment(
                        tier=slm_tier,
                        reasoning=str(slm_response.raw.get("reasoning") or "SLM risk label output."),
                        confidence=slm_response.confidence,
                        potential_harm="SLM risk classification",
                        estimated_exposure="SLM risk classification",
                        vulnerable_populations=[],
                        route_reason="slm_primary",
                        model_used="zentropi"
                    )
            except Exception:
                slm_result = None

        if slm_result is None or slm_result.confidence < settings.risk_confidence_threshold:
            fallback_used = True
            route_reason = "fallback_frontier"
            response = self._call_llm_structured(
                prompt=user_prompt,
                system_prompt=system_prompt,
                output_model=RiskAssessment,
                temperature=0.3,
                max_tokens=settings.frontier_max_tokens
            )
            response.route_reason = route_reason
            response.model_used = settings.azure_openai_deployment_name
            response.confidence = response.confidence if response.confidence else 0.0
        else:
            response = slm_result

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        detail = AgentExecutionDetail(
            agent_name="Risk Agent",
            agent_type="risk",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=response.model_used,
            model_provider="zentropi" if response.model_used == "zentropi" else "azure_openai",
            prompt_hash=self._prompt_hash(system_prompt, user_prompt),
            confidence=response.confidence,
            route_reason=response.route_reason,
            fallback_used=fallback_used,
            policy_version=settings.policy_version,
            execution_time_ms=elapsed_ms,
            status="completed"
        )

        return response, detail

    @staticmethod
    def _map_label_to_risk_tier(label: Optional[str]) -> Optional[RiskTier]:
        if not label:
            return None
        normalized = label.strip().lower()
        if "high" in normalized:
            return RiskTier.HIGH
        if "medium" in normalized:
            return RiskTier.MEDIUM
        if "low" in normalized:
            return RiskTier.LOW
        return None
