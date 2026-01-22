"""Decision Orchestrator: Coordinates agent pipeline and makes final decisions."""
from typing import Optional, Callable
from src.agents.claim_agent import ClaimAgent
from src.agents.risk_agent import RiskAgent
from src.agents.evidence_agent import EvidenceAgent
from src.agents.factuality_agent import FactualityAgent
from src.agents.policy_agent import PolicyAgent
from src.config import settings
from src.governance.system_config_store import get_threshold_value
from src.rag.external_search import ExternalSearchClient
from src.rag.vector_store import VectorStore
from src.llm.groq_client import GroqClient
from src.models.schemas import (
    Decision, DecisionAction, RiskTier, AnalysisResponse,
    Claim, RiskAssessment, Evidence, FactualityAssessment, PolicyInterpretation, EvidenceItem, SourceType,
    AgentExecutionDetail
)


class DecisionOrchestrator:
    """Orchestrates the agent pipeline and makes final decisions."""

    def __init__(self):
        """Initialize orchestrator with all agents."""
        self.claim_agent = ClaimAgent()
        self.risk_agent = RiskAgent()
        self.evidence_agent = EvidenceAgent()
        self.factuality_agent = FactualityAgent()
        self.policy_agent = PolicyAgent()

    def analyze(
        self,
        transcript: str,
        progress_callback: Optional[Callable[[str, str], None]] = None
    ) -> AnalysisResponse:
        """
        Execute full agent pipeline and return analysis result.

        Args:
            transcript: Content transcript to analyze

        Returns:
            AnalysisResponse with decision and all intermediate results
        """
        agent_executions: list[AgentExecutionDetail] = []

        def report_progress(stage: str, status: str) -> None:
            if progress_callback:
                progress_callback(stage, status)

        # Step 1: Extract claims
        report_progress("Claim extraction", "started")
        claims, claim_detail = self.claim_agent.process(transcript)
        agent_executions.append(claim_detail)
        claim_confidence = claim_detail.confidence or 0.0
        report_progress("Claim extraction", "completed")
        report_progress("Claim decomposition", "completed")

        # Step 2: Assess risk
        report_progress("Risk & policy classification", "started")
        risk_assessment, risk_detail = self.risk_agent.process(transcript, claims)
        agent_executions.append(risk_detail)

        # Step 3: Fast path for low-risk content (skip RAG)
        evidence: Optional[Evidence] = None
        factuality_assessments: list[FactualityAssessment] = []
        policy_interpretation: Optional[PolicyInterpretation] = None
        risk_threshold = get_threshold_value("risk_confidence_threshold", settings.risk_confidence_threshold)
        risk_confident = risk_assessment.confidence >= risk_threshold

        if risk_assessment.tier in [RiskTier.MEDIUM, RiskTier.HIGH] and risk_confident:
            # Step 3a: Retrieve evidence (only for medium/high risk)
            report_progress("Evidence retrieval", "started")
            evidence, evidence_detail = self.evidence_agent.process(claims)
            agent_executions.append(evidence_detail)
            report_progress("Evidence retrieval", "completed")

            # Step 3b: External search for medium/high-risk + high-novelty
            if risk_assessment.tier in [RiskTier.MEDIUM, RiskTier.HIGH]:
                # Calculate similarity to internal evidence (0.0 = no match = high novelty)
                similarity_score = self._max_claim_similarity(claims)
                # Also check if we have no internal evidence at all
                has_internal_evidence = len(evidence.supporting) > 0 or len(evidence.contradicting) > 0
                novelty_threshold = get_threshold_value("novelty_similarity_threshold", settings.novelty_similarity_threshold)
                if similarity_score < novelty_threshold or not has_internal_evidence:
                    # High novelty: similarity below threshold OR no internal evidence triggers external search
                    evidence = self._attach_external_context(evidence, claims)
                    # Mark that external search was used due to novelty
                    if evidence:
                        reason_parts = []
                        if similarity_score < novelty_threshold:
                            reason_parts.append(f"High novelty (similarity: {similarity_score:.2f} < {novelty_threshold})")
                        if not has_internal_evidence:
                            reason_parts.append("No internal evidence found")
                        if reason_parts:
                            evidence.evidence_gap_reason = ". ".join(reason_parts) + ". " + (evidence.evidence_gap_reason or "")

            # Step 4: Assess factuality
            report_progress("Claim-evidence evaluation", "started")
            factuality_assessments, factuality_detail = self.factuality_agent.process(claims, evidence)
            agent_executions.append(factuality_detail)
            report_progress("Claim-evidence evaluation", "completed")

            # Step 5: Interpret policy
            policy_interpretation, policy_detail = self.policy_agent.process(
                claims,
                factuality_assessments,
                risk_assessment
            )
            agent_executions.append(policy_detail)
            report_progress("Risk & policy classification", "completed")
        else:
            # Low risk: Skip RAG, but still do policy interpretation with limited info
            report_progress("Evidence retrieval", "skipped")
            report_progress("Claim-evidence evaluation", "skipped")
            evidence_detail = AgentExecutionDetail(
                agent_name="Evidence Agent",
                agent_type="evidence",
                system_prompt="",
                user_prompt="Skipped due to low risk or low confidence routing decision.",
                execution_time_ms=None,
                status="skipped"
            )
            factuality_detail = AgentExecutionDetail(
                agent_name="Factuality Agent",
                agent_type="factuality",
                system_prompt="",
                user_prompt="Skipped due to low risk or low confidence routing decision.",
                execution_time_ms=None,
                status="skipped"
            )
            agent_executions.extend([evidence_detail, factuality_detail])

            policy_interpretation, policy_detail = self.policy_agent.process(
                claims,
                [],  # No factuality assessments for low risk
                risk_assessment
            )
            agent_executions.append(policy_detail)
            report_progress("Risk & policy classification", "completed")

        # Step 6: Make decision
        decision = self._make_decision(
            risk_assessment,
            policy_interpretation,
            evidence
        )

        # Determine if human review is needed
        requires_human_review = self._requires_human_review(
            risk_assessment,
            policy_interpretation,
            evidence,
            decision,
            claim_confidence
        )

        return AnalysisResponse(
            decision=decision,
            claims=claims,
            risk_assessment=risk_assessment,
            evidence=evidence,
            factuality_assessments=factuality_assessments,
            policy_interpretation=policy_interpretation,
            review_request_id=None,  # Will be set if escalated
            agent_executions=agent_executions
        )

    def _make_decision(
        self,
        risk_assessment: RiskAssessment,
        policy_interpretation: Optional[PolicyInterpretation],
        evidence: Optional[Evidence]
    ) -> Decision:
        """
        Make final decision based on risk and policy confidence.

        Implements decision matrix:
        - Low risk + High policy confidence → Allow
        - Medium risk + Medium policy confidence → Label / Downrank
        - High risk + Low policy confidence → Escalate to Human
        - High risk + High policy confidence → Human Confirmation
        """
        risk_tier = risk_assessment.tier

        if policy_interpretation is None:
            # Fallback if no policy interpretation
            return Decision(
                action=DecisionAction.ESCALATE_HUMAN,
                rationale="Unable to interpret policy - requires human review",
                requires_human_review=True,
                confidence=0.0,
                escalation_reason="Policy interpretation unavailable"
            )

        policy_confidence = policy_interpretation.policy_confidence

        # Decision matrix
        if risk_tier == RiskTier.LOW:
            if policy_confidence >= 0.7:
                action = DecisionAction.ALLOW
                rationale = f"Low risk content with high policy confidence ({policy_confidence:.2f}). Content does not violate policy."
                confidence = policy_confidence
            else:
                action = DecisionAction.LABEL_DOWNRANK
                rationale = f"Low risk content but uncertain policy interpretation ({policy_confidence:.2f}). Apply label/downrank."
                confidence = policy_confidence
        elif risk_tier == RiskTier.MEDIUM:
            if policy_confidence >= 0.6:
                action = DecisionAction.LABEL_DOWNRANK
                rationale = f"Medium risk content with moderate policy confidence ({policy_confidence:.2f}). Apply label/downrank."
                confidence = policy_confidence
            else:
                action = DecisionAction.ESCALATE_HUMAN
                rationale = f"Medium risk content with low policy confidence ({policy_confidence:.2f}). Requires human review."
                confidence = policy_confidence
        else:  # HIGH risk
            if policy_confidence < 0.6:
                action = DecisionAction.ESCALATE_HUMAN
                rationale = f"High risk content with low policy confidence ({policy_confidence:.2f}). Escalate to human review."
                confidence = policy_confidence
            else:
                action = DecisionAction.HUMAN_CONFIRMATION
                rationale = f"High risk content with high policy confidence ({policy_confidence:.2f}). Requires human confirmation before action."
                confidence = policy_confidence

        return Decision(
            action=action,
            rationale=rationale,
            requires_human_review=(action in [DecisionAction.ESCALATE_HUMAN, DecisionAction.HUMAN_CONFIRMATION]),
            confidence=confidence,
            escalation_reason=None if action == DecisionAction.ALLOW else rationale
        )

    def _requires_human_review(
        self,
        risk_assessment: RiskAssessment,
        policy_interpretation: Optional[PolicyInterpretation],
        evidence: Optional[Evidence],
        decision: Decision,
        claim_confidence: float
    ) -> bool:
        """
        Determine if human review is required based on escalation criteria.

        Human review is triggered when:
        - High risk + low policy confidence
        - Conflicting evidence
        - Sensitive domains (elections, health)
        - Decision action requires it
        - Low confidence from upstream agents
        """
        # Decision already requires review
        if decision.requires_human_review:
            return True

        # Conflicting evidence
        if evidence and evidence.conflicts_present and risk_assessment.tier in [RiskTier.MEDIUM, RiskTier.HIGH]:
            return True

        # Sensitive domains (health, civic)
        sensitive_domains = ["health", "civic"]
        # This would need claims passed in, but for now we check risk assessment
        # In practice, you'd check claim domains

        # Low confidence gates (only for high-impact content)
        if risk_assessment.tier in [RiskTier.MEDIUM, RiskTier.HIGH]:
            claim_threshold = get_threshold_value("claim_confidence_threshold", settings.claim_confidence_threshold)
            if claim_confidence < claim_threshold:
                return True

            risk_threshold = get_threshold_value("risk_confidence_threshold", settings.risk_confidence_threshold)
            if risk_assessment.confidence < risk_threshold:
                return True

        # High risk + low confidence
        if (risk_assessment.tier == RiskTier.HIGH and
            policy_interpretation and
            policy_interpretation.policy_confidence < get_threshold_value("policy_confidence_threshold", settings.policy_confidence_threshold)):
            return True

        if (risk_assessment.tier in [RiskTier.MEDIUM, RiskTier.HIGH] and
            policy_interpretation and
            policy_interpretation.conflict_detected):
            return True

        return False

    @staticmethod
    def _max_claim_similarity(claims: list[Claim]) -> float:
        """Calculate max similarity of claims to internal evidence.

        Returns similarity score (0.0-1.0):
            - 0.0 = no similarity (high novelty) - triggers external search
            - 1.0 = perfect match (low novelty) - no external search needed

        Note: This is a SIMILARITY score, not novelty. Lower similarity = higher novelty.
        """
        if not claims:
            return 0.0
        vector_store = VectorStore()
        # Check if vector store has any documents
        all_docs = vector_store.get_all_documents()
        if not all_docs:
            # Empty vector store = similarity 0.0 = high novelty
            return 0.0

        max_similarity = 0.0
        for claim in claims:
            score = vector_store.max_similarity(claim.text, index_version=settings.evidence_index_version)
            if score is not None:
                max_similarity = max(max_similarity, score)
        # If no matches found for any claim, return 0.0 (similarity 0 = high novelty)
        return max_similarity

    @staticmethod
    def _classify_external_evidence(claim: str, evidence_text: str) -> str:
        """Classify external evidence as supporting, contradicting, or contextual.

        Returns: "supporting", "contradicting", or "contextual"
        """
        try:
            groq = GroqClient()
            prompt = f"""Classify whether the following evidence supports, contradicts, or is neutral/contextual to the claim.

Claim: {claim}

Evidence: {evidence_text}

Respond with ONLY one word: "supporting", "contradicting", or "contextual"."""

            response = groq.chat(
                prompt=prompt,
                system_prompt="You are a fact-checking classifier. Classify evidence as supporting, contradicting, or contextual. Respond with only one word.",
                temperature=0.1,
                max_tokens=15
            )

            classification = response["content"].strip().lower()
            # More robust classification parsing
            if any(word in classification for word in ["support", "agree", "confirm", "true", "correct"]):
                return "supporting"
            elif any(word in classification for word in ["contradict", "false", "dispute", "refute", "incorrect", "wrong"]):
                return "contradicting"
            else:
                return "contextual"
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to classify external evidence: {e}")
            # If classification fails, default to contextual
            return "contextual"

    @staticmethod
    def _attach_external_context(evidence: Optional[Evidence], claims: list[Claim]) -> Evidence:
        if evidence is None:
            evidence = Evidence(
                supporting=[],
                contradicting=[],
                contextual=[],
                evidence_confidence=0.0,
                conflicts_present=False,
                evidence_gap=True,
                evidence_gap_reason="No internal evidence available."
            )
        search_client = ExternalSearchClient()
        external_items = []
        for claim in claims[:3]:
            results = search_client.search(claim.text)
            for result in results:
                text = result.get("snippet") or result.get("title") or "Context snippet unavailable."
                external_items.append({
                    "text": text,
                    "source": result.get("source") or "external",
                    "url": result.get("url"),
                    "source_type": result.get("source_type"),
                    "claim": claim.text,
                })

        # Classify external results as supporting, contradicting, or contextual
        supporting_items = []
        contradicting_items = []
        contextual_items = []

        for item in external_items:
            classification = DecisionOrchestrator._classify_external_evidence(
                claim=item["claim"],
                evidence_text=item["text"]
            )
            source_type = item.get("source_type")
            evidence_item = EvidenceItem(
                text=item["text"],
                source=item["source"],
                source_quality=item.get("source_type") or "external",
                source_type=SourceType(source_type) if source_type else None,
                url=item.get("url"),
                timestamp=None,
                relevance_score=0.5,  # External results get moderate relevance
            )

            if classification == "supporting":
                supporting_items.append(evidence_item)
            elif classification == "contradicting":
                contradicting_items.append(evidence_item)
            else:
                contextual_items.append(evidence_item)

        # Add classified items to evidence
        evidence.supporting.extend(supporting_items)
        evidence.contradicting.extend(contradicting_items)
        evidence.contextual.extend(contextual_items)

        # Update evidence confidence if we found supporting/contradicting evidence
        if supporting_items or contradicting_items:
            total_classified = len(supporting_items) + len(contradicting_items)
            support_ratio = len(supporting_items) / total_classified if total_classified > 0 else 0.0
            evidence.evidence_confidence = max(evidence.evidence_confidence, support_ratio * 0.7)
            evidence.conflicts_present = len(contradicting_items) > 0

        if evidence.supporting == [] and evidence.contradicting == []:
            evidence.evidence_gap = True
            evidence.evidence_gap_reason = evidence.evidence_gap_reason or "No supporting or contradicting internal evidence."
        else:
            credible_types = {
                SourceType.AUTHORITATIVE,
                SourceType.HIGH_CREDIBILITY,
                SourceType.SCIENTIFIC,
                SourceType.FACT_CHECK,
                SourceType.INTERNAL,
            }
            credible_items = [
                item for item in (evidence.supporting + evidence.contradicting)
                if item.source_type in credible_types
            ]
            if credible_items:
                evidence.evidence_gap = False
                evidence.evidence_gap_reason = None

        if settings.allow_external_enrichment and (supporting_items or contradicting_items or contextual_items):
            vector_store = VectorStore()
            # Collect all external evidence for enrichment
            all_external = supporting_items + contradicting_items + contextual_items
            docs = [item.text for item in all_external]
            metadatas = [
                {
                    "source": item.source,
                    "domain": "external",
                    "quality": "external",
                    "timestamp": None,
                    "index_version": settings.evidence_index_version,
                    "origin": "external_search",
                }
                for item in all_external
            ]
            if docs:
                try:
                    vector_store.add_documents(docs, metadatas=metadatas)
                except ValueError:
                    pass
        return evidence
