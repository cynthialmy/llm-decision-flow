"""Metrics calculation for trust metrics."""
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from src.models.database import DecisionRecord, ReviewRecord, MetricsSnapshot, SessionLocal
from src.config import settings
from src.models.schemas import RiskTier, DecisionAction


class MetricsCalculator:
    """Calculates core trust metrics."""

    def __init__(self):
        """Initialize metrics calculator."""
        self.db: Session = SessionLocal()

    def calculate_metrics(self, days: int = 7) -> Dict[str, Any]:
        """
        Calculate core trust metrics for the last N days.

        Args:
            days: Number of days to look back

        Returns:
            Dictionary of metrics
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Get all decisions in time period
        decisions = self.db.query(DecisionRecord).filter(
            DecisionRecord.created_at >= cutoff_date
        ).all()

        if not decisions:
            return self._empty_metrics()

        total_decisions = len(decisions)

        # Volume & coverage
        case_count_by_risk_tier: Dict[str, int] = {
            RiskTier.LOW.value: 0,
            RiskTier.MEDIUM.value: 0,
            RiskTier.HIGH.value: 0,
        }
        case_count_by_decision_action: Dict[str, int] = {}
        evidence_gap_count = 0
        evidence_gap_suggestions: List[str] = []

        for decision in decisions:
            tier = (decision.risk_assessment_json or {}).get("tier")
            if tier in case_count_by_risk_tier:
                case_count_by_risk_tier[tier] += 1
            action = decision.decision_action
            if action:
                case_count_by_decision_action[action] = case_count_by_decision_action.get(action, 0) + 1

            evidence = decision.evidence_json or {}
            if evidence.get("evidence_gap"):
                evidence_gap_count += 1
                reason = evidence.get("evidence_gap_reason", "")
                suggestion = self._suggest_enrichment_source(reason)
                if suggestion and suggestion not in evidence_gap_suggestions:
                    evidence_gap_suggestions.append(suggestion)

        # High-risk exposure rate
        high_risk_decisions = [
            d for d in decisions
            if d.risk_assessment_json.get('tier') == RiskTier.HIGH.value
            and d.decision_action == DecisionAction.ALLOW.value
        ]
        high_risk_exposure_rate = len(high_risk_decisions) / total_decisions if total_decisions > 0 else 0.0

        # Over-enforcement proxy (appeal reversal rate)
        # This is approximated by human decisions that differ from system decisions
        reviews = self.db.query(ReviewRecord).filter(
            ReviewRecord.created_at >= cutoff_date
        ).all()

        reversals = 0
        manual_override_count = 0
        disagreement_cases = set()
        total_reviews = len(reviews)
        for review in reviews:
            decision = review.decision
            if (review.human_decision_action and
                review.human_decision_action != decision.decision_action):
                reversals += 1
                disagreement_cases.add(review.id)
            if review.manual_override:
                manual_override_count += 1
                disagreement_cases.add(review.id)

        model_human_disagreement = (
            len(disagreement_cases) / total_reviews if total_reviews > 0 else 0.0
        )

        over_enforcement_proxy = reversals / total_reviews if total_reviews > 0 else 0.0

        # Human review load
        pending_reviews = self.db.query(ReviewRecord).filter(
            ReviewRecord.status == "pending"
        ).count()

        # Average time to decision for high-risk content
        high_risk_with_reviews = [
            d for d in decisions
            if d.risk_assessment_json.get('tier') == RiskTier.HIGH.value
            and d.review is not None
        ]

        avg_time_to_decision = 0.0
        if high_risk_with_reviews:
            total_time = sum([
                (d.review.reviewed_at - d.created_at).total_seconds()
                for d in high_risk_with_reviews
                if d.review.reviewed_at
            ])
            avg_time_to_decision = total_time / len(high_risk_with_reviews) if high_risk_with_reviews else 0.0

        auto_resolved_rate = (
            len([d for d in decisions if not d.requires_human_review]) / total_decisions
            if total_decisions > 0 else 0.0
        )

        return {
            "high_risk_exposure_rate": high_risk_exposure_rate,
            "over_enforcement_proxy": over_enforcement_proxy,
            "model_human_disagreement": model_human_disagreement,
            "human_review_load": pending_reviews,
            "avg_time_to_decision": avg_time_to_decision,
            "total_decisions": total_decisions,
            "total_reviews": total_reviews,
            "reversals": reversals,
            "manual_override_count": manual_override_count,
            "disagreement_count": len(disagreement_cases),
            "period_days": days,
            "case_count_by_risk_tier": case_count_by_risk_tier,
            "case_count_by_decision_action": case_count_by_decision_action,
            "evidence_gap_count": evidence_gap_count,
            "evidence_gap_suggestions": evidence_gap_suggestions,
            "claim_precision": None,
            "claim_recall": None,
            "appeal_overturn_rate": over_enforcement_proxy,
            "user_trust_survey_score": None,
            "creator_appeal_satisfaction": None,
            "misinfo_prevalence_by_views": None,
            "repeat_offender_rate": None,
            "auto_resolved_rate": auto_resolved_rate,
            "cost_per_verified_claim": None,
            "rollback_recommended": (
                model_human_disagreement >= settings.disagreement_rollback_threshold
                or avg_time_to_decision >= settings.latency_rollback_threshold_s
            )
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure."""
        return {
            "high_risk_exposure_rate": 0.0,
            "over_enforcement_proxy": 0.0,
            "model_human_disagreement": 0.0,
            "human_review_load": 0,
            "avg_time_to_decision": 0.0,
            "total_decisions": 0,
            "total_reviews": 0,
            "reversals": 0,
            "manual_override_count": 0,
            "disagreement_count": 0,
            "period_days": 7,
            "case_count_by_risk_tier": {
                RiskTier.LOW.value: 0,
                RiskTier.MEDIUM.value: 0,
                RiskTier.HIGH.value: 0,
            },
            "case_count_by_decision_action": {},
            "evidence_gap_count": 0,
            "evidence_gap_suggestions": [],
            "claim_precision": None,
            "claim_recall": None,
            "appeal_overturn_rate": 0.0,
            "user_trust_survey_score": None,
            "creator_appeal_satisfaction": None,
            "misinfo_prevalence_by_views": None,
            "repeat_offender_rate": None,
            "auto_resolved_rate": 0.0,
            "cost_per_verified_claim": None,
            "rollback_recommended": False
        }

    @staticmethod
    def _suggest_enrichment_source(reason: str) -> Optional[str]:
        if not reason:
            return None
        reason_lower = reason.lower()
        if "credibility" in reason_lower or "threshold" in reason_lower:
            return "Prioritize authoritative or peer-reviewed sources"
        if "no internal evidence" in reason_lower:
            return "Add internal knowledge base coverage"
        if "novelty" in reason_lower:
            return "Expand external allowlist for high-novelty topics"
        return "Review evidence coverage for this domain"

    def save_snapshot(self, metrics: Dict[str, Any]) -> int:
        """
        Save metrics snapshot to database.

        Args:
            metrics: Metrics dictionary

        Returns:
            Snapshot ID
        """
        snapshot = MetricsSnapshot(
            high_risk_exposure_rate=metrics.get("high_risk_exposure_rate"),
            over_enforcement_proxy=metrics.get("over_enforcement_proxy"),
            model_human_disagreement=metrics.get("model_human_disagreement"),
            human_review_load=metrics.get("human_review_load"),
            avg_time_to_decision=metrics.get("avg_time_to_decision"),
            additional_metrics={
                "total_decisions": metrics.get("total_decisions"),
                "total_reviews": metrics.get("total_reviews"),
                "reversals": metrics.get("reversals"),
                "rollback_recommended": metrics.get("rollback_recommended")
            }
        )

        self.db.add(snapshot)
        self.db.commit()
        self.db.refresh(snapshot)

        return snapshot.id

    def close(self):
        """Close database session."""
        self.db.close()
