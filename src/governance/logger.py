"""Governance logger for decision versioning and rationale logging."""
from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session
from src.models.database import DecisionRecord, ReviewRecord, SessionLocal
from src.models.schemas import AnalysisResponse, Decision, ReviewRequest, ReviewerFeedback
from src.config import settings
from src.governance.system_config_store import (
    get_active_config_payload,
    create_config_version,
    has_meaningful_updates,
)
import json


class GovernanceLogger:
    """Logs decisions and tracks policy versions for governance."""

    def __init__(self, policy_version: Optional[str] = None):
        """
        Initialize governance logger.

        Args:
            policy_version: Current policy version identifier
        """
        self.policy_version = policy_version or settings.policy_version
        self.db: Session = SessionLocal()

    def log_decision(self, analysis_response: AnalysisResponse, transcript: str) -> int:
        """
        Log a decision with full rationale.

        Args:
            analysis_response: Complete analysis response
            transcript: Original transcript

        Returns:
            Decision record ID
        """
        # Convert Pydantic models to dicts for JSON storage
        claims_json = [claim.model_dump() for claim in analysis_response.claims]
        risk_json = analysis_response.risk_assessment.model_dump()
        evidence_json = analysis_response.evidence.model_dump() if analysis_response.evidence else None
        factuality_json = [fa.model_dump() for fa in analysis_response.factuality_assessments] if analysis_response.factuality_assessments else None
        policy_json = analysis_response.policy_interpretation.model_dump() if analysis_response.policy_interpretation else None

        agent_executions_json = [detail.model_dump() for detail in analysis_response.agent_executions]

        active_config = get_active_config_payload()

        decision_record = DecisionRecord(
            transcript=transcript,
            decision_action=analysis_response.decision.action.value,
            decision_rationale=analysis_response.decision.rationale,
            requires_human_review=analysis_response.decision.requires_human_review,
            confidence=analysis_response.decision.confidence,
            escalation_reason=analysis_response.decision.escalation_reason,
            claims_json=claims_json,
            risk_assessment_json=risk_json,
            evidence_json=evidence_json,
            factuality_assessments_json=factuality_json,
            policy_interpretation_json=policy_json,
            agent_executions_json=agent_executions_json,
            policy_version=self.policy_version,
            decision_version=1,
            system_config_version_id=active_config.get("version_id"),
        )

        self.db.add(decision_record)
        self.db.commit()
        self.db.refresh(decision_record)

        # Create review record if human review is required
        if analysis_response.decision.requires_human_review:
            review_record = ReviewRecord(
                decision_id=decision_record.id,
                status="pending"
            )
            self.db.add(review_record)
            self.db.commit()
            self.db.refresh(review_record)

        return decision_record.id

    def get_review_request(self, review_id: int) -> Optional[ReviewRequest]:
        """
        Get review request by ID.

        Args:
            review_id: Review record ID

        Returns:
            ReviewRequest if found, None otherwise
        """
        review_record = self.db.query(ReviewRecord).filter(ReviewRecord.id == review_id).first()
        if not review_record:
            return None

        decision_record = review_record.decision

        # Reconstruct ReviewRequest from database
        from src.models.schemas import Claim, RiskAssessment, Evidence, FactualityAssessment, PolicyInterpretation

        claims = [Claim(**c) for c in decision_record.claims_json]
        risk_assessment = RiskAssessment(**decision_record.risk_assessment_json)
        evidence = Evidence(**decision_record.evidence_json) if decision_record.evidence_json else None
        factuality_assessments = [
            FactualityAssessment(**fa) for fa in decision_record.factuality_assessments_json
        ] if decision_record.factuality_assessments_json else []
        policy_interpretation = (
            PolicyInterpretation(**decision_record.policy_interpretation_json)
            if decision_record.policy_interpretation_json else None
        )
        from src.models.schemas import DecisionAction

        system_decision = Decision(
            action=DecisionAction(decision_record.decision_action),
            rationale=decision_record.decision_rationale,
            requires_human_review=decision_record.requires_human_review,
            confidence=decision_record.confidence,
            escalation_reason=decision_record.escalation_reason
        )

        human_decision = None
        if review_record.human_decision_action:
            human_decision = Decision(
                action=DecisionAction(review_record.human_decision_action),
                rationale=review_record.human_decision_rationale or "",
                requires_human_review=False,
                confidence=1.0,
                escalation_reason=None
            )

        # Parse reviewer feedback if present
        reviewer_feedback = None
        if review_record.reviewer_feedback_json:
            try:
                if isinstance(review_record.reviewer_feedback_json, dict):
                    reviewer_feedback = ReviewerFeedback(**review_record.reviewer_feedback_json)
                else:
                    reviewer_feedback = review_record.reviewer_feedback_json
            except Exception:
                # If parsing fails, keep as dict
                reviewer_feedback = review_record.reviewer_feedback_json

        payload = {
            "id": review_record.id,
            "decision_id": decision_record.id,
            "transcript": decision_record.transcript,
            "claims": [claim.model_dump() for claim in claims],
            "risk_assessment": risk_assessment.model_dump(),
            "evidence": evidence.model_dump() if evidence else None,
            "factuality_assessments": [fa.model_dump() for fa in factuality_assessments],
            "policy_interpretation": policy_interpretation.model_dump() if policy_interpretation else None,
            "system_decision": system_decision.model_dump(),
            "created_at": review_record.created_at,
            "reviewed_at": review_record.reviewed_at,
            "human_decision": human_decision.model_dump() if human_decision else None,
            "human_rationale": review_record.human_rationale,
            "reviewer_feedback": (
                reviewer_feedback.model_dump()
                if isinstance(reviewer_feedback, ReviewerFeedback)
                else reviewer_feedback
            ),
        }

        return ReviewRequest.model_validate(payload)

    def list_pending_reviews(self) -> list[ReviewRequest]:
        """List all pending review requests."""
        pending_reviews = self.db.query(ReviewRecord).filter(
            ReviewRecord.status == "pending"
        ).all()

        return [
            self.get_review_request(review.id)
            for review in pending_reviews
            if self.get_review_request(review.id) is not None
        ]

    def list_reviewed_reviews(self, limit: int = 50) -> list[ReviewRequest]:
        """List recently reviewed requests."""
        reviewed_reviews = self.db.query(ReviewRecord).filter(
            ReviewRecord.status == "reviewed"
        ).order_by(ReviewRecord.reviewed_at.desc()).limit(limit).all()

        return [
            self.get_review_request(review.id)
            for review in reviewed_reviews
            if self.get_review_request(review.id) is not None
        ]

    def enqueue_review_for_decision(self, decision_id: int) -> str:
        """
        Enqueue a decision for human review, regardless of original routing.

        Returns:
            "created" if a new review was created,
            "reset_pending" if an existing reviewed item was reset,
            "already_pending" if already pending,
            "not_found" if the decision does not exist.
        """
        decision_record = self.db.query(DecisionRecord).filter(
            DecisionRecord.id == decision_id
        ).first()
        if not decision_record:
            return "not_found"

        review_record = decision_record.review
        if review_record is None:
            review_record = ReviewRecord(
                decision_id=decision_id,
                status="pending",
                manual_override=True
            )
            self.db.add(review_record)
            decision_record.requires_human_review = True
            self.db.commit()
            return "created"

        if review_record.status == "pending":
            return "already_pending"

        if review_record.status == "reviewed":
            review_record.status = "pending"
            review_record.reviewed_at = None
            review_record.human_decision_action = None
            review_record.human_decision_rationale = None
            review_record.human_rationale = None
            review_record.reviewer_feedback_json = None
            review_record.manual_override = True
            decision_record.requires_human_review = True
            self.db.commit()
            return "reset_pending"

        return "not_found"

    def submit_human_decision(
        self,
        review_id: int,
        human_decision: Decision,
        human_rationale: str,
        reviewer_feedback: Optional[ReviewerFeedback] = None
    ) -> bool:
        """
        Submit human decision for a review.

        Args:
            review_id: Review record ID
            human_decision: Human reviewer's decision
            human_rationale: Human reviewer's rationale

        Returns:
            True if successful, False otherwise
        """
        review_record = self.db.query(ReviewRecord).filter(ReviewRecord.id == review_id).first()
        if not review_record:
            return False

        review_record.human_decision_action = human_decision.action.value
        review_record.human_decision_rationale = human_decision.rationale
        review_record.human_rationale = human_rationale
        if reviewer_feedback is not None:
            if isinstance(reviewer_feedback, ReviewerFeedback):
                review_record.reviewer_feedback_json = reviewer_feedback.model_dump()
            else:
                review_record.reviewer_feedback_json = reviewer_feedback
        review_record.status = "reviewed"
        review_record.reviewed_at = datetime.utcnow()

        self.db.commit()

        if reviewer_feedback and reviewer_feedback.accepted_change:
            change = reviewer_feedback.accepted_change
            if has_meaningful_updates(
                change.prompt_updates,
                change.threshold_updates,
                change.weighting_updates,
                change.rationale,
            ):
                create_config_version(
                    prompt_updates=change.prompt_updates,
                    threshold_updates=change.threshold_updates,
                    weighting_updates=change.weighting_updates,
                    rationale=change.rationale,
                    source_review_id=review_id,
                    activate=True,
                )
        return True

    def reset_review_to_pending(self, review_id: int, clear_human_decision: bool = True) -> bool:
        """
        Reset a reviewed request back to pending status.

        Args:
            review_id: Review record ID
            clear_human_decision: If True, clears human decision fields for a full reset

        Returns:
            True if successful, False otherwise
        """
        review_record = self.db.query(ReviewRecord).filter(ReviewRecord.id == review_id).first()
        if not review_record:
            return False

        review_record.status = "pending"
        review_record.reviewed_at = None

        if clear_human_decision:
            # Clear human decision fields for a full reset
            review_record.human_decision_action = None
            review_record.human_decision_rationale = None
            review_record.human_rationale = None
            review_record.reviewer_feedback_json = None

        self.db.commit()
        return True

    def close(self):
        """Close database session."""
        self.db.close()
