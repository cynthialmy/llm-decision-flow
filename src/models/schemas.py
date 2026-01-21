"""Pydantic schemas for data validation and serialization."""
from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class Domain(str, Enum):
    """Content domain categories."""
    HEALTH = "health"
    CIVIC = "civic"
    FINANCE = "finance"
    OTHER = "other"


class RiskTier(str, Enum):
    """Risk assessment tiers."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class FactualityStatus(str, Enum):
    """Factuality assessment status."""
    LIKELY_TRUE = "Likely True"
    LIKELY_FALSE = "Likely False"
    UNCERTAIN = "Uncertain / Disputed"


class ViolationStatus(str, Enum):
    """Policy violation status."""
    YES = "Yes"
    NO = "No"
    CONTEXTUAL = "Contextual"


class DecisionAction(str, Enum):
    """Final decision actions."""
    ALLOW = "Allow"
    LABEL_DOWNRANK = "Label / Downrank"
    ESCALATE_HUMAN = "Escalate to Human"
    HUMAN_CONFIRMATION = "Human Confirmation"


class Claim(BaseModel):
    """Extracted factual claim."""
    text: str = Field(..., description="The claim text")
    domain: Domain = Field(..., description="Domain category")
    is_explicit: bool = Field(True, description="Whether claim is explicit or implicit")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Extraction confidence")


class RiskAssessment(BaseModel):
    """Risk assessment result."""
    tier: RiskTier = Field(..., description="Risk tier")
    reasoning: str = Field(..., description="Reasoning for risk assessment")
    potential_harm: str = Field(..., description="Description of potential harm")
    estimated_exposure: str = Field(..., description="Estimated exposure level")
    vulnerable_populations: List[str] = Field(default_factory=list, description="Affected vulnerable populations")


class EvidenceItem(BaseModel):
    """Single evidence item."""
    text: str = Field(..., description="Evidence text")
    source: str = Field(..., description="Source identifier")
    source_quality: str = Field(..., description="Source quality assessment")
    timestamp: Optional[datetime] = Field(None, description="Evidence timestamp")
    relevance_score: float = Field(0.0, ge=0.0, le=1.0, description="Relevance score")


class Evidence(BaseModel):
    """Evidence retrieval result."""
    supporting: List[EvidenceItem] = Field(default_factory=list, description="Supporting evidence")
    contradicting: List[EvidenceItem] = Field(default_factory=list, description="Contradicting evidence")
    evidence_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Overall evidence confidence")
    conflicts_present: bool = Field(False, description="Whether conflicting evidence exists")


class FactualityAssessment(BaseModel):
    """Factuality assessment for a claim."""
    claim_text: str = Field(..., description="The claim being assessed")
    status: FactualityStatus = Field(..., description="Factuality status")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Assessment confidence")
    reasoning: str = Field(..., description="Reasoning for assessment")
    evidence_summary: str = Field(..., description="Summary of evidence considered")


class PolicyInterpretation(BaseModel):
    """Policy interpretation result."""
    violation: ViolationStatus = Field(..., description="Violation status")
    violation_type: Optional[str] = Field(None, description="Type of violation if applicable")
    policy_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Policy interpretation confidence")
    allowed_contexts: List[str] = Field(default_factory=list, description="Allowed contexts (e.g., satire, personal experience)")
    reasoning: str = Field(..., description="Reasoning for policy interpretation")


class Decision(BaseModel):
    """Final decision result."""
    action: DecisionAction = Field(..., description="Decision action")
    rationale: str = Field(..., description="Decision rationale")
    requires_human_review: bool = Field(False, description="Whether human review is required")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Decision confidence")
    escalation_reason: Optional[str] = Field(None, description="Reason for escalation if applicable")


class ReviewRequest(BaseModel):
    """Human review request."""
    id: Optional[int] = Field(None, description="Review request ID")
    transcript: str = Field(..., description="Original transcript")
    claims: List[Claim] = Field(..., description="Extracted claims")
    risk_assessment: RiskAssessment = Field(..., description="Risk assessment")
    evidence: Optional[Evidence] = Field(None, description="Retrieved evidence")
    factuality_assessments: List[FactualityAssessment] = Field(default_factory=list, description="Factuality assessments")
    policy_interpretation: Optional[PolicyInterpretation] = Field(None, description="Policy interpretation")
    system_decision: Decision = Field(..., description="System's recommended decision")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    reviewed_at: Optional[datetime] = Field(None, description="Review timestamp")
    human_decision: Optional[Decision] = Field(None, description="Human reviewer's decision")
    human_rationale: Optional[str] = Field(None, description="Human reviewer's rationale")


class AnalysisRequest(BaseModel):
    """Request for content analysis."""
    transcript: str = Field(..., description="Content transcript to analyze")


class AnalysisResponse(BaseModel):
    """Response from content analysis."""
    decision: Decision = Field(..., description="Final decision")
    claims: List[Claim] = Field(..., description="Extracted claims")
    risk_assessment: RiskAssessment = Field(..., description="Risk assessment")
    evidence: Optional[Evidence] = Field(None, description="Retrieved evidence")
    factuality_assessments: List[FactualityAssessment] = Field(default_factory=list, description="Factuality assessments")
    policy_interpretation: Optional[PolicyInterpretation] = Field(None, description="Policy interpretation")
    review_request_id: Optional[int] = Field(None, description="Review request ID if escalated")


class HumanDecisionRequest(BaseModel):
    """Request to submit human decision."""
    decision: Decision = Field(..., description="Human decision")
    rationale: str = Field(..., description="Human reviewer's rationale")
