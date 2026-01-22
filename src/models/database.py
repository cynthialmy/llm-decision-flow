"""SQLAlchemy database models for persistence."""
from datetime import datetime
from typing import Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Text, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from src.config import settings
import json

Base = declarative_base()


class DecisionRecord(Base):
    """Database model for decisions."""
    __tablename__ = "decisions"

    id = Column(Integer, primary_key=True, index=True)
    transcript = Column(Text, nullable=False)
    decision_action = Column(String, nullable=False)
    decision_rationale = Column(Text, nullable=False)
    requires_human_review = Column(Boolean, default=False)
    confidence = Column(Float, nullable=False)
    escalation_reason = Column(Text, nullable=True)

    # Full analysis data stored as JSON
    claims_json = Column(JSON, nullable=False)
    risk_assessment_json = Column(JSON, nullable=False)
    evidence_json = Column(JSON, nullable=True)
    factuality_assessments_json = Column(JSON, nullable=True)
    policy_interpretation_json = Column(JSON, nullable=True)
    agent_executions_json = Column(JSON, nullable=True)

    # Governance fields
    policy_version = Column(String, nullable=True)
    decision_version = Column(Integer, default=1)
    system_config_version_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship to reviews
    review = relationship("ReviewRecord", back_populates="decision", uselist=False)


class ReviewRecord(Base):
    """Database model for human reviews."""
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    decision_id = Column(Integer, ForeignKey("decisions.id"), nullable=False)

    # Human decision
    human_decision_action = Column(String, nullable=True)
    human_decision_rationale = Column(Text, nullable=True)
    human_rationale = Column(Text, nullable=True)
    reviewer_feedback_json = Column(JSON, nullable=True)
    manual_override = Column(Boolean, default=False)

    # Status
    status = Column(String, default="pending")  # pending, reviewed, resolved
    created_at = Column(DateTime, default=datetime.utcnow)
    reviewed_at = Column(DateTime, nullable=True)

    # Relationship
    decision = relationship("DecisionRecord", back_populates="review")


class SystemConfigVersion(Base):
    """Database model for system configuration versions."""
    __tablename__ = "system_config_versions"

    id = Column(Integer, primary_key=True, index=True)
    prompt_updates = Column(JSON, nullable=True)
    threshold_updates = Column(JSON, nullable=True)
    weighting_updates = Column(JSON, nullable=True)
    rationale = Column(Text, nullable=True)
    active = Column(Boolean, default=False)
    source_review_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class MetricsSnapshot(Base):
    """Database model for metrics snapshots."""
    __tablename__ = "metrics_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    snapshot_date = Column(DateTime, default=datetime.utcnow, index=True)

    # Core trust metrics
    high_risk_exposure_rate = Column(Float, nullable=True)
    over_enforcement_proxy = Column(Float, nullable=True)
    model_human_disagreement = Column(Float, nullable=True)
    human_review_load = Column(Integer, nullable=True)
    avg_time_to_decision = Column(Float, nullable=True)  # in seconds

    # Additional metrics stored as JSON
    additional_metrics = Column(JSON, nullable=True)


# Database setup
def get_database_url() -> str:
    """Get database URL from settings."""
    return f"sqlite:///{settings.sqlite_db_path}"


def get_engine():
    """Create SQLAlchemy engine."""
    database_url = get_database_url()
    # Ensure directory exists
    import os
    os.makedirs(os.path.dirname(settings.sqlite_db_path) or ".", exist_ok=True)
    return create_engine(database_url, connect_args={"check_same_thread": False})


def _ensure_schema(engine):
    """Ensure new columns exist for auditability."""
    import sqlalchemy

    def _has_column(table: str, column: str) -> bool:
        with engine.connect() as conn:
            result = conn.execute(sqlalchemy.text(f"PRAGMA table_info({table})"))
            columns = [row[1] for row in result.fetchall()]
            return column in columns

    with engine.connect() as conn:
        if not _has_column("decisions", "agent_executions_json"):
            conn.execute(sqlalchemy.text("ALTER TABLE decisions ADD COLUMN agent_executions_json JSON"))
        if not _has_column("decisions", "system_config_version_id"):
            conn.execute(sqlalchemy.text("ALTER TABLE decisions ADD COLUMN system_config_version_id INTEGER"))
        if not _has_column("reviews", "reviewer_feedback_json"):
            conn.execute(sqlalchemy.text("ALTER TABLE reviews ADD COLUMN reviewer_feedback_json JSON"))
        if not _has_column("reviews", "manual_override"):
            conn.execute(sqlalchemy.text("ALTER TABLE reviews ADD COLUMN manual_override BOOLEAN"))
        conn.commit()


def get_session_local():
    """Create database session factory."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    _ensure_schema(engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Global session factory
SessionLocal = get_session_local()
