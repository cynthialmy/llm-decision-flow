"""Governance logger for decision versioning and rationale logging."""
from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session
from src.models.database import DecisionRecord, ReviewRecord, SessionLocal
from src.models.schemas import AnalysisResponse, Decision, ReviewRequest
import json


class GovernanceLogger:
    """Logs decisions and tracks policy versions for governance."""

    def __init__(self, policy_version: Optional[str] = None):
        """
        Initialize governance logger.

        Args:
            policy_version: Current policy version identifier
        """
        self.policy_version = policy_version or "1.0"
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
        claims_json = [claim.dict() for claim in analysis_response.claims]
        risk_json = analysis_response.risk_assessment.dict()
        evidence_json = analysis_response.evidence.dict() if analysis_response.evidence else None
        factuality_json = [fa.dict() for fa in analysis_response.factuality_assessments] if analysis_response.factuality_assessments else None
        policy_json = analysis_response.policy_interpretation.dict() if analysis_response.policy_interpretation else None

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
            policy_version=self.policy_version,
            decision_version=1
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

        return ReviewRequest(
            id=review_record.id,
            transcript=decision_record.transcript,
            claims=claims,
            risk_assessment=risk_assessment,
            evidence=evidence,
            factuality_assessments=factuality_assessments,
            policy_interpretation=policy_interpretation,
            system_decision=system_decision,
            created_at=review_record.created_at,
            reviewed_at=review_record.reviewed_at,
            human_decision=human_decision,
            human_rationale=review_record.human_rationale
        )

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

    def submit_human_decision(
        self,
        review_id: int,
        human_decision: Decision,
        human_rationale: str
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
        review_record.status = "reviewed"
        review_record.reviewed_at = datetime.utcnow()

        self.db.commit()
        return True

    def close(self):
        """Close database session."""
        self.db.close()
