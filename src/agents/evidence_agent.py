"""Evidence Agent: Retrieves evidence using RAG system."""
from src.agents.base import BaseAgent
from src.models.schemas import Evidence, Claim
from src.rag.evidence_retriever import EvidenceRetriever
from src.rag.vector_store import VectorStore


class EvidenceAgent(BaseAgent):
    """Agent for retrieving evidence using RAG."""

    def __init__(self):
        """Initialize Evidence Agent with RAG components."""
        super().__init__()
        vector_store = VectorStore()
        self.retriever = EvidenceRetriever(vector_store)

    def process(self, claims: list[Claim]) -> Evidence:
        """
        Retrieve evidence for claims using RAG.

        IMPORTANT: This agent preserves conflicting evidence and does NOT synthesize into "truth".

        Args:
            claims: List of claims to find evidence for

        Returns:
            Evidence object with supporting and contradicting evidence
        """
        if not claims:
            return Evidence(
                supporting=[],
                contradicting=[],
                evidence_confidence=0.0,
                conflicts_present=False
            )

        # Use RAG to retrieve evidence
        evidence = self.retriever.retrieve_evidence(claims, n_results=10)

        return evidence
