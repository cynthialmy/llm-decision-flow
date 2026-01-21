"""Claim Agent: Extracts factual claims and tags domains."""
from typing import List
from src.agents.base import BaseAgent
from src.models.schemas import Claim, Domain


class ClaimAgent(BaseAgent):
    """Agent for extracting factual claims from transcripts."""

    def process(self, transcript: str) -> List[Claim]:
        """
        Extract explicit and implicit factual claims from transcript.

        Args:
            transcript: Content transcript text

        Returns:
            List of extracted claims with domain tags
        """
        system_prompt = """You are a conservative claim extraction agent. Your role is to identify factual claims in text.

IMPORTANT CONSTRAINTS:
- Extract ONLY factual claims (statements that can be verified as true or false)
- Tag each claim with its domain: health, civic, finance, or other
- Be conservative - only extract clear factual statements
- Do NOT infer intent or judge truthfulness
- Distinguish between explicit claims (directly stated) and implicit claims (implied)
- Assign confidence scores (0.0 to 1.0) based on how clear the claim is

Return a JSON object with a "claims" array. Each claim should have:
- "text": the claim text
- "domain": one of "health", "civic", "finance", "other"
- "is_explicit": boolean (true for explicit, false for implicit)
- "confidence": float between 0.0 and 1.0"""

        user_prompt = f"""Extract all factual claims from the following transcript:

{transcript}

Return the claims as a JSON object with this structure:
{{
  "claims": [
    {{
      "text": "claim text here",
      "domain": "health|civic|finance|other",
      "is_explicit": true,
      "confidence": 0.85
    }}
  ]
}}"""

        # Define output model
        class ClaimResponse(BaseModel):
            claims: List[Claim]

        response = self._call_llm_structured(
            prompt=user_prompt,
            system_prompt=system_prompt,
            output_model=ClaimResponse,
            temperature=0.2  # Low temperature for consistent extraction
        )

        return response.claims
