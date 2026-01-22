"""Unit tests for agents."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agents.claim_agent import ClaimAgent
from src.agents.risk_agent import RiskAgent
from src.models.schemas import Claim, Domain, RiskAssessment, RiskTier


class TestClaimAgent:
    """Tests for Claim Agent."""

    @patch('src.agents.claim_agent.GroqClient')
    def test_extract_claims(self, mock_groq):
        """Test claim extraction."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.claims = [
            Claim(
                text="COVID vaccines are safe",
                domain=Domain.HEALTH,
                is_explicit=True,
                confidence=0.9
            )
        ]
        mock_groq_instance = mock_groq.return_value
        mock_groq_instance.chat.return_value = {
            "content": '{"claims":[{"text":"COVID vaccines are safe","domain":"health","is_explicit":true,"confidence":0.9}]}',
            "model": "llama",
            "prompt_hash": "hash"
        }

        agent = ClaimAgent()
        claims, detail = agent.process("COVID vaccines are safe and effective.")

        assert len(claims) == 1
        assert claims[0].domain == Domain.HEALTH
        assert detail.agent_type == "claim"
        mock_groq_instance.chat.assert_called_once()


class TestRiskAgent:
    """Tests for Risk Agent."""

    @patch('src.agents.risk_agent.RiskAgent._call_llm_structured')
    @patch('src.agents.risk_agent.ZentropiClient')
    def test_assess_risk(self, mock_zentropi, mock_llm):
        """Test risk assessment."""
        # Mock LLM response
        mock_response = RiskAssessment(
            tier=RiskTier.HIGH,
            reasoning="High potential harm",
            confidence=0.7,
            potential_harm="Could cause physical harm",
            estimated_exposure="Wide exposure expected",
            vulnerable_populations=["elderly", "immunocompromised"]
        )
        mock_llm.return_value = mock_response
        mock_zentropi.return_value.is_configured.return_value = False

        agent = RiskAgent()
        claims = [Claim(text="Test claim", domain=Domain.HEALTH, is_explicit=True, confidence=0.8)]
        risk, detail = agent.process("Test transcript", claims)

        assert risk.tier == RiskTier.HIGH
        assert len(risk.vulnerable_populations) > 0
        assert detail.agent_type == "risk"
        mock_llm.assert_called_once()


class TestEvidenceAgent:
    """Tests for Evidence Agent."""

    def test_process_empty_claims(self):
        """Test evidence retrieval with empty claims."""
        from src.agents.evidence_agent import EvidenceAgent

        agent = EvidenceAgent()
        evidence, detail = agent.process([])

        assert len(evidence.supporting) == 0
        assert len(evidence.contradicting) == 0
        assert evidence.evidence_confidence == 0.0
        assert detail.status == "skipped"


class TestFactualityAgent:
    """Tests for Factuality Agent."""

    @patch('src.agents.factuality_agent.FactualityAgent._call_llm_structured_with_timing')
    def test_assess_factuality(self, mock_llm):
        """Test factuality assessment."""
        from src.agents.factuality_agent import FactualityAgent
        from src.models.schemas import FactualityAssessment, FactualityStatus, Evidence, EvidenceItem, SourceType

        # Mock LLM response
        mock_response = Mock()
        mock_response.assessments = [
            FactualityAssessment(
                claim_text="Test claim",
                status=FactualityStatus.LIKELY_FALSE,
                confidence=0.8,
                reasoning="Contradicted by evidence",
                evidence_summary="Multiple sources contradict",
                evidence_map={
                    "supports": [],
                    "contradicts": ["Vaccines are safe. (Source: who.int)"],
                    "does_not_address": []
                },
                quoted_evidence=["Vaccines are safe. (Source: who.int)"]
            )
        ]
        mock_llm.return_value = (mock_response, 5.0)

        agent = FactualityAgent()
        claims = [Claim(text="Test claim", domain=Domain.HEALTH, is_explicit=True, confidence=0.8)]
        evidence = Evidence(
            supporting=[
                EvidenceItem(
                    text="Vaccines are safe.",
                    source="who.int",
                    source_quality="authoritative",
                    source_type=SourceType.AUTHORITATIVE,
                    url="https://www.who.int/",
                    relevance_score=0.9
                )
            ],
            contradicting=[],
            contextual=[],
            evidence_confidence=0.5,
            conflicts_present=False,
            evidence_gap=False
        )

        assessments, detail = agent.process(claims, evidence)

        assert len(assessments) == 1
        assert assessments[0].status == FactualityStatus.LIKELY_FALSE
        assert detail.agent_type == "factuality"
        mock_llm.assert_called_once()


class TestPolicyAgent:
    """Tests for Policy Agent."""

    @patch('src.agents.policy_agent.PolicyAgent._load_policy')
    @patch('src.agents.policy_agent.PolicyAgent._call_llm_structured')
    @patch('src.agents.policy_agent.ZentropiClient')
    def test_interpret_policy(self, mock_zentropi, mock_llm, mock_load_policy):
        """Test policy interpretation."""
        from src.agents.policy_agent import PolicyAgent
        from src.models.schemas import PolicyInterpretation, ViolationStatus, RiskAssessment, RiskTier

        mock_load_policy.return_value = "Test policy text"

        # Mock LLM response
        mock_response = PolicyInterpretation(
            violation=ViolationStatus.YES,
            violation_type="Health misinformation",
            policy_confidence=0.9,
            allowed_contexts=[],
            reasoning="Violates health misinformation policy",
            conflict_detected=False
        )
        mock_llm.return_value = mock_response
        mock_zentropi.return_value.is_configured.return_value = False

        agent = PolicyAgent()
        claims = [Claim(text="Test claim", domain=Domain.HEALTH, is_explicit=True, confidence=0.8)]
        risk = RiskAssessment(
            tier=RiskTier.HIGH,
            reasoning="High risk",
            confidence=0.8,
            potential_harm="Harmful",
            estimated_exposure="Wide",
            vulnerable_populations=[]
        )

        interpretation, detail = agent.process(claims, [], risk)

        assert interpretation.violation == ViolationStatus.YES
        assert interpretation.policy_confidence > 0.5
        assert detail.agent_type == "policy"
        mock_llm.assert_called_once()
