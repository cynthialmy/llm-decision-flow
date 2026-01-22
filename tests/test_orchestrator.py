"""Integration tests for decision orchestrator."""
import pytest
from unittest.mock import Mock, patch
from src.orchestrator.decision_orchestrator import DecisionOrchestrator
from src.models.schemas import RiskTier, DecisionAction


class TestDecisionOrchestrator:
    """Tests for Decision Orchestrator."""

    @patch('src.orchestrator.decision_orchestrator.ClaimAgent.process')
    @patch('src.orchestrator.decision_orchestrator.RiskAgent.process')
    @patch('src.orchestrator.decision_orchestrator.EvidenceAgent.process')
    @patch('src.orchestrator.decision_orchestrator.FactualityAgent.process')
    @patch('src.orchestrator.decision_orchestrator.PolicyAgent.process')
    @patch('src.orchestrator.decision_orchestrator.DecisionOrchestrator._max_claim_similarity')
    def test_low_risk_fast_path(
        self,
        mock_similarity,
        mock_policy,
        mock_factuality,
        mock_evidence,
        mock_risk,
        mock_claim
    ):
        """Test that low-risk content skips RAG."""
        from src.models.schemas import Claim, Domain, RiskAssessment, PolicyInterpretation, ViolationStatus, AgentExecutionDetail

        # Mock agents
        mock_claim.return_value = (
            [Claim(text="Low risk claim", domain=Domain.OTHER, is_explicit=True, confidence=0.8)],
            AgentExecutionDetail(agent_name="Claim Agent", agent_type="claim", system_prompt="", user_prompt="")
        )
        mock_similarity.return_value = 1.0
        mock_risk.return_value = (
            RiskAssessment(
            tier=RiskTier.LOW,
            reasoning="Low risk",
            confidence=0.8,
            potential_harm="Minimal",
            estimated_exposure="Limited",
            vulnerable_populations=[]
            ),
            AgentExecutionDetail(agent_name="Risk Agent", agent_type="risk", system_prompt="", user_prompt="")
        )
        mock_policy.return_value = (
            PolicyInterpretation(
                violation=ViolationStatus.NO,
                policy_confidence=0.8,
                allowed_contexts=[],
                reasoning="No violation",
                conflict_detected=False
            ),
            AgentExecutionDetail(agent_name="Policy Agent", agent_type="policy", system_prompt="", user_prompt="")
        )

        orchestrator = DecisionOrchestrator()
        result = orchestrator.analyze("Low risk content")

        # Verify RAG was skipped
        mock_evidence.assert_not_called()
        mock_factuality.assert_not_called()

        # Verify decision was made
        assert result.decision is not None
        assert result.risk_assessment.tier == RiskTier.LOW

    @patch('src.orchestrator.decision_orchestrator.ClaimAgent.process')
    @patch('src.orchestrator.decision_orchestrator.RiskAgent.process')
    @patch('src.orchestrator.decision_orchestrator.EvidenceAgent.process')
    @patch('src.orchestrator.decision_orchestrator.FactualityAgent.process')
    @patch('src.orchestrator.decision_orchestrator.PolicyAgent.process')
    @patch('src.orchestrator.decision_orchestrator.DecisionOrchestrator._max_claim_similarity')
    def test_high_risk_full_pipeline(
        self,
        mock_similarity,
        mock_policy,
        mock_factuality,
        mock_evidence,
        mock_risk,
        mock_claim
    ):
        """Test that high-risk content goes through full pipeline."""
        from src.models.schemas import (
            Claim, Domain, RiskAssessment, Evidence, FactualityAssessment,
            FactualityStatus, PolicyInterpretation, ViolationStatus, AgentExecutionDetail
        )

        # Mock agents
        mock_similarity.return_value = 1.0
        mock_claim.return_value = (
            [Claim(text="High risk claim", domain=Domain.HEALTH, is_explicit=True, confidence=0.9)],
            AgentExecutionDetail(agent_name="Claim Agent", agent_type="claim", system_prompt="", user_prompt="")
        )
        mock_risk.return_value = (
            RiskAssessment(
            tier=RiskTier.HIGH,
            reasoning="High risk",
            confidence=0.8,
            potential_harm="Severe",
            estimated_exposure="Wide",
            vulnerable_populations=["elderly"]
            ),
            AgentExecutionDetail(agent_name="Risk Agent", agent_type="risk", system_prompt="", user_prompt="")
        )
        mock_evidence.return_value = (
            Evidence(
                supporting=[],
                contradicting=[],
                contextual=[],
                evidence_confidence=0.5,
                conflicts_present=False,
                evidence_gap=True,
                evidence_gap_reason="No matching internal evidence found."
            ),
            AgentExecutionDetail(agent_name="Evidence Agent", agent_type="evidence", system_prompt="", user_prompt="")
        )
        mock_factuality.return_value = (
            [
                FactualityAssessment(
                    claim_text="High risk claim",
                    status=FactualityStatus.LIKELY_FALSE,
                    confidence=0.8,
                    reasoning="False",
                    evidence_summary="Contradicted",
                    evidence_map={
                        "supports": [],
                        "contradicts": ["Evidence quote"],
                        "does_not_address": []
                    },
                    quoted_evidence=["Evidence quote"]
                )
            ],
            AgentExecutionDetail(agent_name="Factuality Agent", agent_type="factuality", system_prompt="", user_prompt="")
        )
        mock_policy.return_value = (
            PolicyInterpretation(
                violation=ViolationStatus.YES,
                violation_type="Health misinformation",
                policy_confidence=0.9,
                allowed_contexts=[],
                reasoning="Violates policy",
                conflict_detected=False
            ),
            AgentExecutionDetail(agent_name="Policy Agent", agent_type="policy", system_prompt="", user_prompt="")
        )

        orchestrator = DecisionOrchestrator()
        result = orchestrator.analyze("High risk content")

        # Verify full pipeline executed
        mock_evidence.assert_called_once()
        mock_factuality.assert_called_once()

        # Verify decision
        assert result.decision is not None
        assert result.risk_assessment.tier == RiskTier.HIGH
        assert result.evidence is not None
        assert len(result.factuality_assessments) > 0

    @patch('src.orchestrator.decision_orchestrator.ClaimAgent.process')
    @patch('src.orchestrator.decision_orchestrator.RiskAgent.process')
    @patch('src.orchestrator.decision_orchestrator.EvidenceAgent.process')
    @patch('src.orchestrator.decision_orchestrator.FactualityAgent.process')
    @patch('src.orchestrator.decision_orchestrator.PolicyAgent.process')
    def test_risk_low_confidence_skips_rag(
        self,
        mock_policy,
        mock_factuality,
        mock_evidence,
        mock_risk,
        mock_claim
    ):
        """Low risk confidence should skip RAG even for high tier."""
        from src.models.schemas import Claim, Domain, RiskAssessment, PolicyInterpretation, ViolationStatus, AgentExecutionDetail

        mock_claim.return_value = (
            [Claim(text="Claim", domain=Domain.HEALTH, is_explicit=True, confidence=0.9)],
            AgentExecutionDetail(agent_name="Claim Agent", agent_type="claim", system_prompt="", user_prompt="")
        )
        mock_risk.return_value = (
            RiskAssessment(
                tier=RiskTier.HIGH,
                reasoning="High risk but low confidence",
                confidence=0.4,
                potential_harm="Severe",
                estimated_exposure="Wide",
                vulnerable_populations=["elderly"]
            ),
            AgentExecutionDetail(agent_name="Risk Agent", agent_type="risk", system_prompt="", user_prompt="")
        )
        mock_policy.return_value = (
            PolicyInterpretation(
                violation=ViolationStatus.NO,
                policy_confidence=0.7,
                allowed_contexts=[],
                reasoning="No violation",
                conflict_detected=False
            ),
            AgentExecutionDetail(agent_name="Policy Agent", agent_type="policy", system_prompt="", user_prompt="")
        )

        orchestrator = DecisionOrchestrator()
        result = orchestrator.analyze("High risk content")

        mock_evidence.assert_not_called()
        mock_factuality.assert_not_called()
        assert result.risk_assessment.confidence < 0.6
