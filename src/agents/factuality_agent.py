"""Factuality Agent: Assesses factual status of claims against evidence."""
from typing import List, Tuple
from pydantic import BaseModel
from src.agents.base import BaseAgent
from src.models.schemas import FactualityAssessment, FactualityStatus, Claim, Evidence, AgentExecutionDetail
from src.config import settings


class FactualityAgent(BaseAgent):
    """Agent for assessing claim factuality."""

    def process(self, claims: List[Claim], evidence: Evidence) -> Tuple[List[FactualityAssessment], AgentExecutionDetail]:
        """
        Assess factual status of each claim against evidence.

        IMPORTANT: Factuality ≠ Policy Violation. This agent only assesses truthfulness,
        not whether content violates platform policy.

        Args:
            claims: List of claims to assess
            evidence: Retrieved evidence

        Returns:
            List of factuality assessments
        """
        system_prompt = """You are a factuality assessment agent. Your role is to assess whether claims are likely true, likely false, or uncertain based on available evidence.

IMPORTANT CONSTRAINTS:
- Assess ONLY factual truthfulness, NOT policy violations
- Use the evidence provided to make your assessment
- If evidence conflicts, mark as "Uncertain / Disputed"
- Be conservative - mark as uncertain if evidence is insufficient
- Provide clear reasoning for your assessment
- Assign confidence scores (0.0 to 1.0) based on evidence strength

Return a JSON object with a "assessments" array. Each assessment should have:
- "claim_text": the claim being assessed
- "status": "Likely True", "Likely False", or "Uncertain / Disputed"
- "confidence": float between 0.0 and 1.0
- "reasoning": explanation of assessment
- "evidence_summary": summary of evidence considered"""

        # Format evidence for prompt
        supporting_text = "\n".join([f"- {item.text} (Source: {item.source})" for item in evidence.supporting[:5]])
        contradicting_text = "\n".join([f"- {item.text} (Source: {item.source})" for item in evidence.contradicting[:5]])

        claims_text = "\n".join([f"- {claim.text}" for claim in claims])

        user_prompt = f"""Assess the factuality of the following claims based on the provided evidence:

Claims to Assess:
{claims_text}

Supporting Evidence:
{supporting_text if supporting_text else "None"}

Contradicting Evidence:
{contradicting_text if contradicting_text else "None"}

Return a JSON object with this structure:
{{
  "assessments": [
    {{
      "claim_text": "claim text",
      "status": "Likely True|Likely False|Uncertain / Disputed",
      "confidence": 0.75,
      "reasoning": "detailed reasoning",
      "evidence_summary": "summary of evidence"
    }}
  ]
}}"""

        # Define output model
        class FactualityResponse(BaseModel):
            assessments: List[FactualityAssessment]

        response, elapsed_ms = self._call_llm_structured_with_timing(
            prompt=user_prompt,
            system_prompt=system_prompt,
            output_model=FactualityResponse,
            temperature=0.3,
            max_tokens=settings.frontier_max_tokens
        )

        detail = AgentExecutionDetail(
            agent_name="Factuality Agent",
            agent_type="factuality",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=settings.azure_openai_deployment_name,
            model_provider="azure_openai",
            prompt_hash=self._prompt_hash(system_prompt, user_prompt),
            confidence=self._aggregate_confidence(response.assessments),
            route_reason="frontier_primary",
            fallback_used=False,
            policy_version=settings.policy_version,
            execution_time_ms=elapsed_ms,
            status="completed"
        )

        return response.assessments, detail

    @staticmethod
    def _aggregate_confidence(assessments: List[FactualityAssessment]) -> float:
        if not assessments:
            return 0.0
        return sum([item.confidence for item in assessments]) / len(assessments)
