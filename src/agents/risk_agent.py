"""Risk Agent: Assesses risk tier based on potential harm, exposure, and vulnerable populations."""
from typing import Tuple, Optional
import time
from src.agents.base import BaseAgent
from src.agents.prompt_registry import render_prompt
from src.governance.system_config_store import get_prompt_overrides, get_threshold_value
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
        prompt_overrides = get_prompt_overrides()
        system_prompt = render_prompt("risk", "system_prompt", {}, overrides=prompt_overrides)

        claims_text = "\n".join([f"- {claim.text} ({claim.domain.value})" for claim in claims])
        user_prompt = render_prompt(
            "risk",
            "user_prompt",
            {"transcript": transcript, "claims_text": claims_text},
            overrides=prompt_overrides
        )

        start_time = time.perf_counter()
        zentropi = ZentropiClient()
        slm_result: Optional[RiskAssessment] = None
        fallback_used = False
        route_reason = "slm_primary"
        slm_error: Optional[str] = None

        if zentropi.is_configured():
            try:
                content = f"{transcript}\n\nClaims:\n{claims_text}"
                criteria_text = "Label the content risk tier as one of: Low, Medium, High."
                slm_response = zentropi.label(content, criteria_text=criteria_text)
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
                else:
                    route_reason = f"slm_unmapped_label:{slm_response.label}"
                    slm_error = f"Zentropi label unmapped: {slm_response.label}"
            except Exception as exc:
                slm_result = None
                slm_error = f"Zentropi call failed: {exc}"

        risk_threshold = get_threshold_value("risk_confidence_threshold", settings.risk_confidence_threshold)
        if slm_result is None or slm_result.confidence < risk_threshold:
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
            route_reason=response.route_reason or route_reason,
            fallback_used=fallback_used,
            policy_version=settings.policy_version,
            execution_time_ms=elapsed_ms,
            status="completed",
            error=slm_error
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
