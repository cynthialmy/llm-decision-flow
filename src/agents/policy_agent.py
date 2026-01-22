"""Policy Interpretation Agent: Interprets policy text and determines violations."""
import os
import time
from typing import Tuple, Optional
from src.agents.base import BaseAgent
from src.agents.prompt_registry import render_prompt
from src.governance.system_config_store import get_prompt_overrides, get_threshold_value
from src.models.schemas import PolicyInterpretation, ViolationStatus, Claim, FactualityAssessment, RiskAssessment, AgentExecutionDetail
from src.config import settings
from src.llm.zentropi_client import ZentropiClient


class PolicyAgent(BaseAgent):
    """Agent for interpreting policy and determining violations."""

    def __init__(self):
        """Initialize Policy Agent and load policy text."""
        super().__init__()
        self.policy_text = self._load_policy()

    def _load_policy(self) -> str:
        """
        Load policy text from file.

        Returns:
            Policy text as string
        """
        policy_path = settings.policy_file_path

        if not os.path.exists(policy_path):
            # Return default policy if file doesn't exist
            return """Platform Misinformation Policy:

1. Health Misinformation: Content that makes false or misleading health claims that could cause harm is prohibited, except when clearly marked as personal experience or opinion.

2. Civic Misinformation: False information about elections, voting, or democratic processes is prohibited.

3. Financial Misinformation: False or misleading financial advice that could cause financial harm is prohibited.

4. Contextual Exceptions: Satire, clearly labeled opinion, and personal experiences are generally allowed even if factually incorrect.

5. Risk-Based Enforcement: Higher risk content requires stricter enforcement."""

        try:
            with open(policy_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Error loading policy file: {e}")

    def process(
        self,
        claims: list[Claim],
        factuality_assessments: list[FactualityAssessment],
        risk_assessment: RiskAssessment
    ) -> Tuple[PolicyInterpretation, AgentExecutionDetail]:
        """
        Interpret policy and determine if content violates it.

        IMPORTANT: Policy text is treated as input, not hard-coded logic.
        This agent has no enforcement authority - it only interprets policy.

        Args:
            claims: List of claims
            factuality_assessments: Factuality assessments for claims
            risk_assessment: Risk assessment

        Returns:
            PolicyInterpretation with violation status and reasoning
        """
        prompt_overrides = get_prompt_overrides()
        system_prompt = render_prompt("policy", "system_prompt", {}, overrides=prompt_overrides)

        # Format inputs
        claims_text = "\n".join([f"- {claim.text} ({claim.domain.value})" for claim in claims])

        factuality_text = "\n".join([
            f"- {fa.claim_text}: {fa.status.value} (confidence: {fa.confidence:.2f})"
            for fa in factuality_assessments
        ])

        user_prompt = render_prompt(
            "policy",
            "user_prompt",
            {
                "policy_text": self.policy_text,
                "claims_text": claims_text,
                "factuality_text": factuality_text,
                "risk_tier": risk_assessment.tier.value,
                "risk_reasoning": risk_assessment.reasoning,
            },
            overrides=prompt_overrides
        )

        start_time = time.perf_counter()
        zentropi = ZentropiClient()
        slm_result: Optional[PolicyInterpretation] = None
        fallback_used = False
        route_reason = "slm_primary"
        slm_error: Optional[str] = None

        if zentropi.is_configured():
            try:
                content = f"{claims_text}\n\nFactuality:\n{factuality_text}\n\nRisk:{risk_assessment.tier.value}"
                criteria_text = "Label the policy outcome as one of: Yes, No, Contextual."
                slm_response = zentropi.label(content, criteria_text=criteria_text)
                violation = self._map_label_to_violation(slm_response.label)
                if violation:
                    slm_result = PolicyInterpretation(
                        violation=violation,
                        violation_type=None,
                        policy_confidence=slm_response.confidence,
                        allowed_contexts=[],
                        reasoning=str(slm_response.raw.get("reasoning") or "SLM policy label output."),
                        conflict_detected=False,
                        model_used="zentropi",
                        route_reason="slm_primary"
                    )
                else:
                    route_reason = f"slm_unmapped_label:{slm_response.label}"
                    slm_error = f"Zentropi label unmapped: {slm_response.label}"
            except Exception as exc:
                slm_result = None
                slm_error = f"Zentropi call failed: {exc}"

        policy_threshold = get_threshold_value("policy_confidence_threshold", settings.policy_confidence_threshold)
        if slm_result is None or slm_result.policy_confidence < policy_threshold:
            fallback_used = True
            route_reason = "fallback_frontier"
            response = self._call_llm_structured(
                prompt=user_prompt,
                system_prompt=system_prompt,
                output_model=PolicyInterpretation,
                temperature=0.3,
                max_tokens=settings.frontier_max_tokens
            )
            response.route_reason = route_reason
            response.model_used = settings.azure_openai_deployment_name
        else:
            response = slm_result

        response.conflict_detected = self._detect_conflict(response)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        detail = AgentExecutionDetail(
            agent_name="Policy Interpretation Agent",
            agent_type="policy",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=response.model_used,
            model_provider="zentropi" if response.model_used == "zentropi" else "azure_openai",
            prompt_hash=self._prompt_hash(system_prompt, user_prompt),
            confidence=response.policy_confidence,
            route_reason=response.route_reason or route_reason,
            fallback_used=fallback_used,
            policy_version=settings.policy_version,
            execution_time_ms=elapsed_ms,
            status="completed",
            error=slm_error
        )

        return response, detail

    @staticmethod
    def _map_label_to_violation(label: Optional[str]) -> Optional[ViolationStatus]:
        if not label:
            return None
        normalized = label.strip().lower()
        if "context" in normalized:
            return ViolationStatus.CONTEXTUAL
        if "yes" in normalized or "violate" in normalized:
            return ViolationStatus.YES
        if "no" in normalized:
            return ViolationStatus.NO
        return None

    @staticmethod
    def _detect_conflict(result: PolicyInterpretation) -> bool:
        if result.violation == ViolationStatus.YES and result.allowed_contexts:
            return True
        return False
