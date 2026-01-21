"""Policy Interpretation Agent: Interprets policy text and determines violations."""
import os
from src.agents.base import BaseAgent
from src.models.schemas import PolicyInterpretation, ViolationStatus, Claim, FactualityAssessment, RiskAssessment
from src.config import settings


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
    ) -> PolicyInterpretation:
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
        system_prompt = """You are a policy interpretation agent. Your role is to interpret platform policy text and determine if content violates it.

IMPORTANT CONSTRAINTS:
- Policy text is provided as input - interpret it, don't apply hard-coded rules
- Consider factuality, but factuality alone does not determine violations
- Consider context (satire, personal experience, opinion)
- Consider risk level in policy interpretation
- You have NO enforcement authority - you only interpret policy
- Provide confidence scores based on policy clarity

Return a JSON object with:
- "violation": "Yes", "No", or "Contextual"
- "violation_type": type of violation if applicable (null if no violation)
- "policy_confidence": float between 0.0 and 1.0
- "allowed_contexts": array of allowed contexts (e.g., ["satire", "personal experience"])
- "reasoning": detailed reasoning for interpretation"""

        # Format inputs
        claims_text = "\n".join([f"- {claim.text} ({claim.domain.value})" for claim in claims])

        factuality_text = "\n".join([
            f"- {fa.claim_text}: {fa.status.value} (confidence: {fa.confidence:.2f})"
            for fa in factuality_assessments
        ])

        user_prompt = f"""Interpret the following policy and determine if the content violates it:

POLICY TEXT:
{self.policy_text}

CONTENT ANALYSIS:
Claims:
{claims_text}

Factuality Assessments:
{factuality_text}

Risk Assessment: {risk_assessment.tier.value}
Risk Reasoning: {risk_assessment.reasoning}

Return a JSON object with this structure:
{{
  "violation": "Yes|No|Contextual",
  "violation_type": "violation type or null",
  "policy_confidence": 0.85,
  "allowed_contexts": ["satire", "personal experience"],
  "reasoning": "detailed reasoning"
}}"""

        response = self._call_llm_structured(
            prompt=user_prompt,
            system_prompt=system_prompt,
            output_model=PolicyInterpretation,
            temperature=0.3
        )

        return response
