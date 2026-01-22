"""Pydantic schemas for data validation and serialization."""
from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


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


class SourceType(str, Enum):
    """Evidence source categories."""
    AUTHORITATIVE = "authoritative"
    HIGH_CREDIBILITY = "high_credibility"
    SCIENTIFIC = "scientific"
    FACT_CHECK = "fact_check"
    INTERNAL = "internal"
    EXTERNAL = "external"


class ReviewerAction(str, Enum):
    """Human reviewer action."""
    CONFIRM = "confirm"
    OVERRIDE = "override"
    REQUEST_EVIDENCE = "request_evidence"
    ESCALATE_POLICY = "escalate_policy"


class ChangeProposal(BaseModel):
    """Structured proposal for system updates after review."""
    model_config = ConfigDict(protected_namespaces=())
    prompt_updates: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Prompt edits by component")
    threshold_updates: Optional[Dict[str, float]] = Field(default_factory=dict, description="Threshold adjustments")
    weighting_updates: Optional[Dict[str, float]] = Field(default_factory=dict, description="Retrieval/source weight changes")
    rationale: Optional[str] = Field(None, description="Reason for change proposal")


class ReviewerFeedback(BaseModel):
    """Structured reviewer feedback and change confirmation."""
    model_config = ConfigDict(protected_namespaces=())
    action: ReviewerAction = Field(..., description="Reviewer action")
    review_time_seconds: Optional[float] = Field(None, description="Time spent reviewing")
    reviewer_notes: Optional[str] = Field(None, description="Human reviewer notes")
    proposed_change: Optional[ChangeProposal] = Field(None, description="Proposed system change package")
    accepted_change: Optional[ChangeProposal] = Field(None, description="Accepted or edited change package")
    previous_behavior: Optional[str] = Field(None, description="Summary of prior system behavior")
    updated_behavior: Optional[str] = Field(None, description="Summary of updated system behavior")


class Claim(BaseModel):
    """Extracted factual claim."""
    model_config = ConfigDict(protected_namespaces=())
    text: str = Field(..., description="The claim text")
    domain: Domain = Field(..., description="Domain category")
    is_explicit: bool = Field(True, description="Whether claim is explicit or implicit")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Extraction confidence")
    subclaims: List["Claim"] = Field(default_factory=list, description="Atomic sub-claims if decomposed")
    parent_claim: Optional[str] = Field(None, description="Parent claim text if this is a sub-claim")
    decomposition_method: Optional[str] = Field(None, description="Method or prompt version for decomposition")


class RiskAssessment(BaseModel):
    """Risk assessment result."""
    model_config = ConfigDict(protected_namespaces=())
    tier: RiskTier = Field(..., description="Risk tier")
    reasoning: str = Field(..., description="Reasoning for risk assessment")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Risk confidence")
    potential_harm: str = Field(..., description="Description of potential harm")
    estimated_exposure: str = Field(..., description="Estimated exposure level")
    vulnerable_populations: List[str] = Field(default_factory=list, description="Affected vulnerable populations")
    route_reason: Optional[str] = Field(None, description="Routing reason or fallback explanation")
    model_used: Optional[str] = Field(None, description="Model or labeler used")


class EvidenceItem(BaseModel):
    """Single evidence item."""
    model_config = ConfigDict(protected_namespaces=())
    text: str = Field(..., description="Evidence text")
    source: str = Field(..., description="Source identifier")
    source_quality: str = Field(..., description="Source quality assessment")
    source_type: Optional[SourceType] = Field(None, description="Tiered source category")
    url: Optional[str] = Field(None, description="Source URL")
    timestamp: Optional[datetime] = Field(None, description="Evidence timestamp")
    relevance_score: float = Field(0.0, ge=0.0, le=1.0, description="Relevance score")


class Evidence(BaseModel):
    """Evidence retrieval result."""
    model_config = ConfigDict(protected_namespaces=())
    supporting: List[EvidenceItem] = Field(default_factory=list, description="Supporting evidence")
    contradicting: List[EvidenceItem] = Field(default_factory=list, description="Contradicting evidence")
    contextual: List[EvidenceItem] = Field(default_factory=list, description="Context-only external evidence")
    evidence_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Overall evidence confidence")
    conflicts_present: bool = Field(False, description="Whether conflicting evidence exists")
    evidence_gap: bool = Field(False, description="Whether internal evidence is missing")
    evidence_gap_reason: Optional[str] = Field(None, description="Reason for missing evidence")


class FactualityAssessment(BaseModel):
    """Factuality assessment for a claim."""
    model_config = ConfigDict(protected_namespaces=())
    claim_text: str = Field(..., description="The claim being assessed")
    status: FactualityStatus = Field(..., description="Factuality status")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Assessment confidence")
    reasoning: str = Field(..., description="Reasoning for assessment")
    evidence_summary: str = Field(..., description="Summary of evidence considered")
    evidence_map: Optional[Dict[str, List[str]]] = Field(
        default_factory=dict,
        description="Mapping: supports/contradicts/does_not_address to evidence quotes"
    )
    quoted_evidence: List[str] = Field(default_factory=list, description="Verbatim evidence snippets")


class PolicyInterpretation(BaseModel):
    """Policy interpretation result."""
    model_config = ConfigDict(protected_namespaces=())
    violation: ViolationStatus = Field(..., description="Violation status")
    violation_type: Optional[str] = Field(None, description="Type of violation if applicable")
    policy_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Policy interpretation confidence")
    allowed_contexts: List[str] = Field(default_factory=list, description="Allowed contexts (e.g., satire, personal experience)")
    reasoning: str = Field(..., description="Reasoning for policy interpretation")
    conflict_detected: bool = Field(False, description="Whether cross-policy conflicts detected")
    model_used: Optional[str] = Field(None, description="Model or labeler used")
    route_reason: Optional[str] = Field(None, description="Routing reason or fallback explanation")


class Decision(BaseModel):
    """Final decision result."""
    model_config = ConfigDict(protected_namespaces=())
    action: DecisionAction = Field(..., description="Decision action")
    rationale: str = Field(..., description="Decision rationale")
    requires_human_review: bool = Field(False, description="Whether human review is required")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Decision confidence")
    escalation_reason: Optional[str] = Field(None, description="Reason for escalation if applicable")


class AgentExecutionDetail(BaseModel):
    """Agent execution details for UI inspection."""
    model_config = ConfigDict(protected_namespaces=())
    agent_name: str = Field(..., description="Human-readable agent name")
    agent_type: str = Field(..., description="Agent type identifier")
    system_prompt: str = Field(..., description="System prompt used")
    user_prompt: str = Field(..., description="User prompt used")
    model_name: Optional[str] = Field(None, description="Model or labeler name")
    model_provider: Optional[str] = Field(None, description="Provider name")
    prompt_hash: Optional[str] = Field(None, description="Hash of system + user prompt")
    confidence: Optional[float] = Field(None, description="Primary confidence score")
    route_reason: Optional[str] = Field(None, description="Routing reason or fallback explanation")
    fallback_used: bool = Field(False, description="Whether fallback model was used")
    policy_version: Optional[str] = Field(None, description="Policy version in effect")
    execution_time_ms: Optional[float] = Field(None, description="Execution time in ms")
    status: str = Field("completed", description="Execution status")
    error: Optional[str] = Field(None, description="Error message if failed")


class ReviewRequest(BaseModel):
    """Human review request."""
    model_config = ConfigDict(protected_namespaces=())
    id: Optional[int] = Field(None, description="Review request ID")
    decision_id: Optional[int] = Field(None, description="Decision record ID")
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
    reviewer_feedback: Optional[ReviewerFeedback] = Field(None, description="Structured reviewer feedback")


class AnalysisRequest(BaseModel):
    """Request for content analysis."""
    model_config = ConfigDict(protected_namespaces=())
    transcript: str = Field(..., description="Content transcript to analyze")


class AnalysisResponse(BaseModel):
    """Response from content analysis."""
    model_config = ConfigDict(protected_namespaces=())
    decision: Decision = Field(..., description="Final decision")
    claims: List[Claim] = Field(..., description="Extracted claims")
    risk_assessment: RiskAssessment = Field(..., description="Risk assessment")
    evidence: Optional[Evidence] = Field(None, description="Retrieved evidence")
    factuality_assessments: List[FactualityAssessment] = Field(default_factory=list, description="Factuality assessments")
    policy_interpretation: Optional[PolicyInterpretation] = Field(None, description="Policy interpretation")
    review_request_id: Optional[int] = Field(None, description="Review request ID if escalated")
    agent_executions: List[AgentExecutionDetail] = Field(default_factory=list, description="Agent execution metadata")


class HumanDecisionRequest(BaseModel):
    """Request to submit human decision."""
    model_config = ConfigDict(protected_namespaces=())
    decision: Decision = Field(..., description="Human decision")
    rationale: str = Field(..., description="Human reviewer's rationale")
    reviewer_feedback: Optional[ReviewerFeedback] = Field(None, description="Structured reviewer feedback")


Claim.model_rebuild()
