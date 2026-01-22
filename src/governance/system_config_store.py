"""System configuration version store and access helpers."""
from __future__ import annotations

from typing import Any, Dict, Optional, List

from sqlalchemy.orm import Session

from src.config import settings
from src.models.database import SessionLocal, SystemConfigVersion


DEFAULT_THRESHOLDS: Dict[str, float] = {
    "claim_confidence_threshold": settings.claim_confidence_threshold,
    "risk_confidence_threshold": settings.risk_confidence_threshold,
    "policy_confidence_threshold": settings.policy_confidence_threshold,
    "novelty_similarity_threshold": settings.novelty_similarity_threshold,
    "evidence_similarity_cutoff": settings.evidence_similarity_cutoff,
}

DEFAULT_WEIGHTINGS: Dict[str, float] = {
    "authoritative": 1.2,
    "high_credibility": 1.1,
    "scientific": 1.15,
    "fact_check": 1.15,
    "internal": 1.0,
    "external": 0.9,
}


def _normalize_dict(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _get_session(session: Optional[Session] = None) -> Session:
    return session or SessionLocal()


def list_config_versions(limit: int = 50) -> List[SystemConfigVersion]:
    session = _get_session()
    try:
        return (
            session.query(SystemConfigVersion)
            .order_by(SystemConfigVersion.created_at.desc())
            .limit(limit)
            .all()
        )
    finally:
        session.close()


def get_active_config_version(session: Optional[Session] = None) -> Optional[SystemConfigVersion]:
    owns_session = session is None
    session = _get_session(session)
    try:
        return (
            session.query(SystemConfigVersion)
            .filter(SystemConfigVersion.active.is_(True))
            .order_by(SystemConfigVersion.created_at.desc())
            .first()
        )
    finally:
        if owns_session:
            session.close()


def get_active_config_payload() -> Dict[str, Any]:
    version = get_active_config_version()
    if not version:
        return {
            "version_id": None,
            "prompt_updates": {},
            "threshold_updates": {},
            "weighting_updates": {},
            "rationale": None,
        }
    return {
        "version_id": version.id,
        "prompt_updates": _normalize_dict(version.prompt_updates),
        "threshold_updates": _normalize_dict(version.threshold_updates),
        "weighting_updates": _normalize_dict(version.weighting_updates),
        "rationale": version.rationale,
    }


def get_prompt_overrides() -> Dict[str, Any]:
    return _normalize_dict(get_active_config_payload().get("prompt_updates"))


def get_threshold_overrides() -> Dict[str, float]:
    raw = _normalize_dict(get_active_config_payload().get("threshold_updates"))
    return {key: value for key, value in raw.items() if isinstance(value, (int, float))}


def get_weighting_overrides() -> Dict[str, Any]:
    return _normalize_dict(get_active_config_payload().get("weighting_updates"))


def get_weightings_with_overrides() -> Dict[str, float]:
    overrides = get_weighting_overrides()
    merged = dict(DEFAULT_WEIGHTINGS)
    merged.update({k: float(v) for k, v in overrides.items() if isinstance(v, (int, float))})
    return merged


def get_thresholds_with_overrides() -> Dict[str, float]:
    overrides = get_threshold_overrides()
    merged = dict(DEFAULT_THRESHOLDS)
    merged.update({k: float(v) for k, v in overrides.items()})
    return merged


def get_threshold_value(name: str, default: Optional[float] = None) -> float:
    thresholds = get_thresholds_with_overrides()
    if name in thresholds:
        return thresholds[name]
    if default is not None:
        return default
    return DEFAULT_THRESHOLDS.get(name, 0.0)


def activate_config_version(version_id: int, session: Optional[Session] = None) -> bool:
    owns_session = session is None
    session = _get_session(session)
    try:
        target = session.query(SystemConfigVersion).filter(SystemConfigVersion.id == version_id).first()
        if not target:
            return False
        session.query(SystemConfigVersion).update({SystemConfigVersion.active: False})
        target.active = True
        session.commit()
        return True
    finally:
        if owns_session:
            session.close()


def create_config_version(
    prompt_updates: Optional[Dict[str, Any]] = None,
    threshold_updates: Optional[Dict[str, Any]] = None,
    weighting_updates: Optional[Dict[str, Any]] = None,
    rationale: Optional[str] = None,
    source_review_id: Optional[int] = None,
    activate: bool = True,
) -> SystemConfigVersion:
    session = _get_session()
    try:
        version = SystemConfigVersion(
            prompt_updates=_normalize_dict(prompt_updates),
            threshold_updates=_normalize_dict(threshold_updates),
            weighting_updates=_normalize_dict(weighting_updates),
            rationale=rationale,
            active=False,
            source_review_id=source_review_id,
        )
        session.add(version)
        session.commit()
        session.refresh(version)
        if activate:
            activate_config_version(version.id, session=session)
            session.refresh(version)
        return version
    finally:
        session.close()


def has_meaningful_updates(
    prompt_updates: Optional[Dict[str, Any]],
    threshold_updates: Optional[Dict[str, Any]],
    weighting_updates: Optional[Dict[str, Any]],
    rationale: Optional[str],
) -> bool:
    return any([
        bool(_normalize_dict(prompt_updates)),
        bool(_normalize_dict(threshold_updates)),
        bool(_normalize_dict(weighting_updates)),
        bool(rationale and str(rationale).strip()),
    ])
