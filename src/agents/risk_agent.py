"""Risk Agent: Assesses risk tier based on potential harm, exposure, and vulnerable populations."""
from src.agents.base import BaseAgent
from src.models.schemas import RiskAssessment, RiskTier, Claim


class RiskAgent(BaseAgent):
    """Agent for assessing risk tier of content."""

    def process(self, transcript: str, claims: list[Claim]) -> RiskAssessment:
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
  "potential_harm": "description of potential harm",
  "estimated_exposure": "description of exposure level",
  "vulnerable_populations": ["group1", "group2"]
}}"""

        response = self._call_llm_structured(
            prompt=user_prompt,
            system_prompt=system_prompt,
            output_model=RiskAssessment,
            temperature=0.3
        )

        return response
