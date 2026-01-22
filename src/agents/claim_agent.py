"""Claim Agent: Extracts factual claims and tags domains."""
from typing import List, Tuple
from pydantic import BaseModel
import time
from src.agents.base import BaseAgent
from src.models.schemas import Claim, Domain, AgentExecutionDetail
from src.llm.groq_client import GroqClient
from src.config import settings


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
        system_prompt = """You are a conservative claim extraction agent. Your role is to identify factual claims in text and decompose compound claims into atomic sub-claims.

IMPORTANT CONSTRAINTS:
- Extract ONLY factual claims (statements that can be verified as true or false)
- Tag each claim with its domain: health, civic, finance, or other
- Be conservative - only extract clear factual statements
- Do NOT infer intent or judge truthfulness
- Distinguish between explicit claims (directly stated) and implicit claims (implied)
- Assign confidence scores (0.0 to 1.0) based on how clear the claim is
- For compound claims, include atomic sub-claims that are independently checkable
- Sub-claims must inherit the domain of the parent claim
- If a claim is atomic, return an empty subclaims array

Return a JSON object with a "claims" array. Each claim should have:
- "text": the claim text
- "domain": one of "health", "civic", "finance", "other"
- "is_explicit": boolean (true for explicit, false for implicit)
- "confidence": float between 0.0 and 1.0
- "subclaims": array of atomic sub-claims with the same fields
- "parent_claim": optional, set for sub-claims (use parent text)
- "decomposition_method": optional string describing how decomposition was done"""

        user_prompt = f"""Extract all factual claims from the following transcript:

{transcript}

Return the claims as a JSON object with this structure:
{{
  "claims": [
    {{
      "text": "claim text here",
      "domain": "health|civic|finance|other",
      "is_explicit": true,
      "confidence": 0.85,
      "subclaims": [
        {{
          "text": "atomic sub-claim text",
          "domain": "health|civic|finance|other",
          "is_explicit": true,
          "confidence": 0.85,
          "subclaims": [],
          "parent_claim": "parent claim text",
          "decomposition_method": "llm_atomic_decomposition"
        }}
      ],
      "parent_claim": null,
      "decomposition_method": "llm_atomic_decomposition"
    }}
  ]
}}"""

        # Define output model
        class ClaimResponse(BaseModel):
            claims: List[Claim]

        groq = GroqClient()
        start_time = time.perf_counter()
        response_data = groq.chat(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=settings.claim_max_tokens
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
            policy_version=settings.policy_version
        )

        return response.claims, detail

    @staticmethod
    def _aggregate_claim_confidence(claims: List[Claim]) -> float:
        if not claims:
            return 0.0
        return sum([claim.confidence for claim in claims]) / len(claims)
