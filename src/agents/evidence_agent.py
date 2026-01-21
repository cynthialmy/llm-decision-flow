"""Evidence Agent: Retrieves evidence using RAG system."""
import time
from typing import Tuple
from src.agents.base import BaseAgent
from src.models.schemas import Evidence, Claim, AgentExecutionDetail
from src.rag.evidence_retriever import EvidenceRetriever
from src.rag.vector_store import VectorStore


class EvidenceAgent(BaseAgent):
    """Agent for retrieving evidence using RAG."""

    def __init__(self):
        """Initialize Evidence Agent with RAG components."""
        super().__init__()
        vector_store = VectorStore()
        self.retriever = EvidenceRetriever(vector_store)

    def process(self, claims: list[Claim]) -> Tuple[Evidence, AgentExecutionDetail]:
        """
        Retrieve evidence for claims using RAG.

        IMPORTANT: This agent preserves conflicting evidence and does NOT synthesize into "truth".

        Args:
            claims: List of claims to find evidence for

        Returns:
            Evidence object with supporting and contradicting evidence
        """
        start_time = time.perf_counter()
        if not claims:
            evidence = Evidence(
                supporting=[],
                contradicting=[],
                evidence_confidence=0.0,
                conflicts_present=False
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            detail = AgentExecutionDetail(
                agent_name="Evidence Agent",
                agent_type="evidence",
                system_prompt="",
                user_prompt="No claims provided; skipping retrieval.",
                execution_time_ms=elapsed_ms,
                status="skipped"
            )
            return evidence, detail

        # Use RAG to retrieve evidence
        evidence = self.retriever.retrieve_evidence(claims, n_results=10)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        detail = AgentExecutionDetail(
            agent_name="Evidence Agent",
            agent_type="evidence",
            system_prompt="",
            user_prompt="Retrieve evidence for extracted claims.",
            execution_time_ms=elapsed_ms,
            status="completed"
        )

        return evidence, detail
