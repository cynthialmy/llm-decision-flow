"""Decision Orchestrator: Coordinates agent pipeline and makes final decisions."""
from typing import Optional
from src.agents.claim_agent import ClaimAgent
from src.agents.risk_agent import RiskAgent
from src.agents.evidence_agent import EvidenceAgent
from src.agents.factuality_agent import FactualityAgent
from src.agents.policy_agent import PolicyAgent
from src.models.schemas import (
    Decision, DecisionAction, RiskTier, AnalysisResponse,
    Claim, RiskAssessment, Evidence, FactualityAssessment, PolicyInterpretation,
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

    def analyze(self, transcript: str) -> AnalysisResponse:
        """
        Execute full agent pipeline and return analysis result.

        Args:
            transcript: Content transcript to analyze

        Returns:
            AnalysisResponse with decision and all intermediate results
        """
        agent_executions: list[AgentExecutionDetail] = []

        # Step 1: Extract claims
        claims, claim_detail = self.claim_agent.process(transcript)
        agent_executions.append(claim_detail)

        # Step 2: Assess risk
        risk_assessment, risk_detail = self.risk_agent.process(transcript, claims)
        agent_executions.append(risk_detail)

        # Step 3: Fast path for low-risk content (skip RAG)
        evidence: Optional[Evidence] = None
        factuality_assessments: list[FactualityAssessment] = []
        policy_interpretation: Optional[PolicyInterpretation] = None

        if risk_assessment.tier in [RiskTier.MEDIUM, RiskTier.HIGH]:
            # Step 3a: Retrieve evidence (only for medium/high risk)
            evidence, evidence_detail = self.evidence_agent.process(claims)
            agent_executions.append(evidence_detail)

            # Step 4: Assess factuality
            factuality_assessments, factuality_detail = self.factuality_agent.process(claims, evidence)
            agent_executions.append(factuality_detail)

            # Step 5: Interpret policy
            policy_interpretation, policy_detail = self.policy_agent.process(
                claims,
                factuality_assessments,
                risk_assessment
            )
            agent_executions.append(policy_detail)
        else:
            # Low risk: Skip RAG, but still do policy interpretation with limited info
            evidence_detail = AgentExecutionDetail(
                agent_name="Evidence Agent",
                agent_type="evidence",
                system_prompt="",
                user_prompt="Skipped due to low risk routing decision.",
                execution_time_ms=None,
                status="skipped"
            )
            factuality_detail = AgentExecutionDetail(
                agent_name="Factuality Agent",
                agent_type="factuality",
                system_prompt="",
                user_prompt="Skipped due to low risk routing decision.",
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
            decision
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
        decision: Decision
    ) -> bool:
        """
        Determine if human review is required based on escalation criteria.

        Human review is triggered when:
        - High risk + low policy confidence
        - Conflicting evidence
        - Sensitive domains (elections, health)
        - Decision action requires it
        """
        # Decision already requires review
        if decision.requires_human_review:
            return True

        # Conflicting evidence
        if evidence and evidence.conflicts_present:
            return True

        # Sensitive domains (health, civic)
        sensitive_domains = ["health", "civic"]
        # This would need claims passed in, but for now we check risk assessment
        # In practice, you'd check claim domains

        # High risk + low confidence
        if (risk_assessment.tier == RiskTier.HIGH and
            policy_interpretation and
            policy_interpretation.policy_confidence < 0.6):
            return True

        return False
