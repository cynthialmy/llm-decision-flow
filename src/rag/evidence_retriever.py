"""Evidence retriever using RAG to find supporting and contradicting evidence."""
from typing import List, Optional
from datetime import datetime
from src.rag.vector_store import VectorStore
from src.models.schemas import Evidence, EvidenceItem, Claim, SourceType
from src.config import settings
from src.governance.system_config_store import get_threshold_value, get_weightings_with_overrides


class EvidenceRetriever:
    """Retrieves evidence for claims using vector search."""

    def __init__(self, vector_store: VectorStore):
        """
        Initialize evidence retriever.

        Args:
            vector_store: VectorStore instance
        """
        self.vector_store = vector_store

    def retrieve_evidence(self, claims: List[Claim], n_results: int = 10) -> Evidence:
        """
        Retrieve evidence for claims, separating supporting and contradicting evidence.

        Args:
            claims: List of claims to find evidence for
            n_results: Number of evidence items to retrieve per claim

        Returns:
            Evidence object with supporting and contradicting evidence
        """
        all_supporting = []
        all_contradicting = []
        credible_items = 0

        for claim in claims:
            # Search for evidence related to this claim
            search_results = self.vector_store.search(
                query=claim.text,
                n_results=n_results,
                where={"index_version": settings.evidence_index_version} if settings.evidence_index_version else None
            )

            # Classify evidence as supporting or contradicting
            # This is a simplified approach - in production, you'd use a classifier
            for result in search_results:
                relevance_score = 1.0 - (result['distance'] or 0.0)
                cutoff = get_threshold_value("evidence_similarity_cutoff", settings.evidence_similarity_cutoff)
                metadata = result.get('metadata', {}) or {}
                source_type = self._infer_source_type(metadata, result.get('document', ''), metadata.get('source'))
                weights = get_weightings_with_overrides()
                weight_key = source_type.value if source_type else "external"
                weight_multiplier = weights.get(weight_key, 1.0)
                weighted_score = min(relevance_score * weight_multiplier, 1.0)
                if weighted_score < cutoff:
                    continue

                evidence_item = EvidenceItem(
                    text=result['document'],
                    source=metadata.get('source', 'unknown'),
                    source_quality=metadata.get('quality', 'unknown'),
                    source_type=source_type,
                    url=metadata.get('url'),
                    timestamp=self._parse_timestamp(metadata.get('timestamp')),
                    relevance_score=weighted_score
                )
                if source_type and source_type != SourceType.EXTERNAL:
                    credible_items += 1

                # Simple heuristic: if distance is low, it's likely supporting
                # In production, use a classifier to determine support vs contradiction
                if result['distance'] is not None and result['distance'] < 0.3:
                    all_supporting.append(evidence_item)
                else:
                    # For demo purposes, we'll add to contradicting
                    # In production, use semantic analysis to determine
                    all_contradicting.append(evidence_item)

        # Calculate overall evidence confidence
        # Higher confidence when we have more supporting evidence and less contradiction
        total_items = len(all_supporting) + len(all_contradicting)
        if total_items == 0:
            evidence_confidence = 0.0
        else:
            support_ratio = len(all_supporting) / total_items if total_items > 0 else 0.0
            evidence_confidence = support_ratio * 0.8  # Cap at 0.8 to account for uncertainty

        conflicts_present = len(all_contradicting) > 0
        evidence_gap = total_items == 0 or credible_items == 0
        if total_items == 0:
            evidence_gap_reason = "No matching internal evidence found."
        elif credible_items == 0:
            evidence_gap_reason = "No evidence met the credibility threshold."
        else:
            evidence_gap_reason = None

        return Evidence(
            supporting=all_supporting[:10],  # Limit to top 10
            contradicting=all_contradicting[:10],  # Limit to top 10
            contextual=[],
            evidence_confidence=evidence_confidence,
            conflicts_present=conflicts_present,
            evidence_gap=evidence_gap,
            evidence_gap_reason=evidence_gap_reason
        )

    @staticmethod
    def _infer_source_type(metadata: dict, text: str, source: Optional[str]) -> Optional[SourceType]:
        raw_type = (metadata.get("source_type") or metadata.get("quality") or "").lower()
        if raw_type in {"authoritative", "official"}:
            return SourceType.AUTHORITATIVE
        if raw_type in {"scientific", "peer_reviewed", "peer-reviewed", "preprint"}:
            return SourceType.SCIENTIFIC
        if raw_type in {"fact_check", "fact-check"}:
            return SourceType.FACT_CHECK
        if raw_type in {"journalism", "news", "high_credibility"}:
            return SourceType.HIGH_CREDIBILITY
        if raw_type in {"internal"}:
            return SourceType.INTERNAL
        if source:
            return SourceType.INTERNAL
        return None

    def _parse_timestamp(self, timestamp_str: Optional[str]) -> Optional[datetime]:
        """Parse timestamp string to datetime object."""
        if not timestamp_str:
            return None

        try:
            # Try ISO format first
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            try:
                # Try common formats
                return datetime.strptime(timestamp_str, "%Y-%m-%d")
            except ValueError:
                return None
