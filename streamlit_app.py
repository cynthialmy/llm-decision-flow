import os
from typing import List, Optional, Dict, Any
import tempfile
import json

import streamlit as st
import matplotlib.pyplot as plt
from pyvis.network import Network
from sqlalchemy.orm import joinedload

from src.orchestrator.decision_orchestrator import DecisionOrchestrator
from src.models.schemas import (
    AnalysisResponse, AgentExecutionDetail, RiskTier, DecisionAction, Decision,
    Claim, EvidenceItem, FactualityAssessment, SourceType, ReviewerFeedback, ChangeProposal, ReviewerAction
)
from src.config import settings
from src.governance.metrics import MetricsCalculator
from src.governance.logger import GovernanceLogger
from src.models.database import SessionLocal, DecisionRecord, ReviewRecord
from src.rag.vector_store import VectorStore


def load_policy_text() -> str:
    policy_path = settings.policy_file_path
    if not policy_path:
        return "Policy path not configured."
    if not os.path.exists(policy_path):
        return f"Policy file not found: {policy_path}"
    try:
        with open(policy_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as exc:
        return f"Failed to load policy text: {exc}"


def _status_color(status: str) -> str:
    if status == "completed":
        return "#4CAF50"
    if status == "skipped":
        return "#B0BEC5"
    if status == "error":
        return "#E53935"
    return "#D3D3D3"


def build_flow_graph(
    agent_executions: Optional[List[AgentExecutionDetail]],
    analysis: Optional[AnalysisResponse],
) -> None:
    """Build and display the decision flow graph using pyvis with improved layout and path highlighting."""
    net = Network(
        height="600px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="black"
    )

    # Define nodes with labels, levels (horizontal), and y positions (vertical) for better branch separation
    # Levels control horizontal position, y positions control vertical position for branches
    node_data = {
        "Transcript": {"label": "User Input:\nTranscript", "level": 0, "y": 0},
        "ClaimAgent": {"label": "Claim Agent", "level": 1, "y": 0},
        "ClaimsCache": {"label": "Claims Cache", "level": 2, "y": -120},
        "RiskSLM": {"label": "Risk SLM\n(Zentropi)", "level": 3, "y": 0},
        "RiskFallback": {"label": "Risk Fallback\n(Frontier)", "level": 4, "y": 120},
        "RiskAgent": {"label": "Risk Decision", "level": 4, "y": 0},
        # High/Medium risk path: positioned above center
        "EvidenceAgent": {"label": "Evidence Agent\n(RAG)", "level": 5, "y": -200},
        "ExternalSearch": {"label": "External Search\n(Allowlist)", "level": 6, "y": -200},
        "FactualityAgent": {"label": "Factuality Agent", "level": 7, "y": -200},
        # Policy Agent: can be reached from both paths, positioned at center
        "PolicySLM": {"label": "Policy SLM\n(Zentropi)", "level": 8, "y": 0},
        "PolicyFallback": {"label": "Policy Fallback\n(Frontier)", "level": 9, "y": 120},
        "PolicyAgent": {"label": "Policy Decision", "level": 9, "y": 0},
        "QualityGate": {"label": "Quality Gates", "level": 10, "y": 0},
        "DecisionOrch": {"label": "Decision\nOrchestrator", "level": 11, "y": 0},
        # Escalate path: positioned below center
        "HumanReview": {"label": "Human Review\nInterface", "level": 12, "y": 160},
        "Governance": {"label": "Governance Log", "level": 13, "y": 0},
        "Metrics": {"label": "Metrics\nDashboard", "level": 14, "y": 0},
    }

    # Determine active execution path
    active_edges = set()
    risk_tier = analysis.risk_assessment.tier if analysis else None
    requires_human_review = analysis.decision.requires_human_review if analysis else None
    risk_route = analysis.risk_assessment.route_reason if analysis else None
    policy_route = analysis.policy_interpretation.route_reason if analysis and analysis.policy_interpretation else None
    external_context_used = bool(analysis.evidence and analysis.evidence.contextual) if analysis else False

    # Always active: initial flow
    active_edges.add(("Transcript", "ClaimAgent"))
    active_edges.add(("ClaimAgent", "ClaimsCache"))
    active_edges.add(("ClaimsCache", "RiskSLM"))

    if risk_route == "fallback_frontier":
        active_edges.add(("RiskSLM", "RiskFallback"))
        active_edges.add(("RiskFallback", "RiskAgent"))
    else:
        active_edges.add(("RiskSLM", "RiskAgent"))

    # Routing based on risk tier
    if risk_tier in (RiskTier.HIGH, RiskTier.MEDIUM):
        # High/Medium risk: Full path through Evidence -> Factuality -> Policy
        active_edges.add(("RiskAgent", "EvidenceAgent"))
        if external_context_used:
            active_edges.add(("EvidenceAgent", "ExternalSearch"))
            active_edges.add(("ExternalSearch", "PolicySLM"))
        active_edges.add(("EvidenceAgent", "FactualityAgent"))
        active_edges.add(("FactualityAgent", "PolicySLM"))
        if policy_route == "fallback_frontier":
            active_edges.add(("PolicySLM", "PolicyFallback"))
            active_edges.add(("PolicyFallback", "PolicyAgent"))
        else:
            active_edges.add(("PolicySLM", "PolicyAgent"))
        active_edges.add(("PolicyAgent", "QualityGate"))
        active_edges.add(("QualityGate", "DecisionOrch"))
    elif risk_tier == RiskTier.LOW:
        # Low risk: Skip Evidence/Factuality, go directly to Policy Agent
        active_edges.add(("RiskAgent", "PolicySLM"))
        if policy_route == "fallback_frontier":
            active_edges.add(("PolicySLM", "PolicyFallback"))
            active_edges.add(("PolicyFallback", "PolicyAgent"))
        else:
            active_edges.add(("PolicySLM", "PolicyAgent"))
        active_edges.add(("PolicyAgent", "QualityGate"))
        active_edges.add(("QualityGate", "DecisionOrch"))
    else:
        # No analysis yet - show all paths as inactive
        pass

    # Always active: final flow
    active_edges.add(("DecisionOrch", "Governance"))
    if requires_human_review:
        active_edges.add(("DecisionOrch", "HumanReview"))
        active_edges.add(("HumanReview", "Governance"))
    active_edges.add(("Governance", "Metrics"))

    # Set node colors and borders based on execution status
    if agent_executions:
        status_by_type = {detail.agent_type: detail.status for detail in agent_executions}

        def get_node_style(node_id: str) -> dict:
            """Get node color and border style based on status."""
            if node_id == "Transcript":
                return {"color": "#E3F2FD", "border": "#2196F3", "borderWidth": 2}
            elif node_id == "ClaimAgent":
                status = status_by_type.get("claim", "pending")
                return _get_node_style_for_status(status)
            elif node_id == "RiskAgent":
                status = status_by_type.get("risk", "pending")
                return _get_node_style_for_status(status)
            elif node_id == "RiskSLM":
                status = status_by_type.get("risk", "pending")
                return _get_node_style_for_status(status)
            elif node_id == "RiskFallback":
                if risk_route == "fallback_frontier":
                    return {"color": "#FFCC80", "border": "#FB8C00", "borderWidth": 2}
                return {"color": "#B0BEC5", "border": "#78909C", "borderWidth": 1}
            elif node_id == "EvidenceAgent":
                status = status_by_type.get("evidence", "pending")
                return _get_node_style_for_status(status)
            elif node_id == "ExternalSearch":
                if external_context_used:
                    return {"color": "#BBDEFB", "border": "#1E88E5", "borderWidth": 2}
                return {"color": "#B0BEC5", "border": "#78909C", "borderWidth": 1}
            elif node_id == "FactualityAgent":
                status = status_by_type.get("factuality", "pending")
                return _get_node_style_for_status(status)
            elif node_id == "PolicySLM":
                status = status_by_type.get("policy", "pending")
                return _get_node_style_for_status(status)
            elif node_id == "PolicyFallback":
                if policy_route == "fallback_frontier":
                    return {"color": "#FFCC80", "border": "#FB8C00", "borderWidth": 2}
                return {"color": "#B0BEC5", "border": "#78909C", "borderWidth": 1}
            elif node_id == "PolicyAgent":
                status = status_by_type.get("policy", "pending")
                return _get_node_style_for_status(status)
            elif node_id == "QualityGate":
                return {"color": "#FFE082", "border": "#F9A825", "borderWidth": 2}
            elif node_id == "DecisionOrch":
                return {"color": "#4CAF50", "border": "#2E7D32", "borderWidth": 3}
            elif node_id == "HumanReview":
                if requires_human_review:
                    return {"color": "#FF7043", "border": "#D84315", "borderWidth": 3}
                else:
                    return {"color": "#B0BEC5", "border": "#78909C", "borderWidth": 1}
            elif node_id == "Governance":
                return {"color": "#4CAF50", "border": "#2E7D32", "borderWidth": 2}
            elif node_id == "Metrics":
                return {"color": "#2196F3", "border": "#1565C0", "borderWidth": 2}
            else:
                return {"color": "#D3D3D3", "border": "#9E9E9E", "borderWidth": 1}

        node_styles = {node_id: get_node_style(node_id) for node_id in node_data.keys()}
    else:
        # Default styles when no analysis
        node_styles = {
            node_id: {"color": "#D3D3D3", "border": "#9E9E9E", "borderWidth": 1}
            for node_id in node_data.keys()
        }
        node_styles["Transcript"] = {"color": "#E3F2FD", "border": "#2196F3", "borderWidth": 2}

    # Add nodes with improved styling and explicit positioning
    # Calculate x positions based on level, y positions for vertical branch separation
    base_x_spacing = 200
    base_y = 0

    for node_id, data in node_data.items():
        style = node_styles.get(node_id, {"color": "#D3D3D3", "border": "#9E9E9E", "borderWidth": 1})
        x_pos = data["level"] * base_x_spacing
        y_pos = base_y + data["y"]

        node_params = {
            "label": data["label"],
            "color": style["color"],
            "borderWidth": style["borderWidth"],
            "shape": "box",
            "font": {"size": 14, "face": "Arial"},
            "x": x_pos,
            "y": y_pos,
            "fixed": {"x": True, "y": True}  # Fix positions to prevent physics from moving them
        }
        net.add_node(node_id, **node_params)

    # Define all possible edges
    all_edges = [
        ("Transcript", "ClaimAgent", None),
        ("ClaimAgent", "ClaimsCache", None),
        ("ClaimsCache", "RiskSLM", None),
        ("RiskSLM", "RiskFallback", "Fallback"),
        ("RiskSLM", "RiskAgent", "SLM"),
        ("RiskFallback", "RiskAgent", None),
        ("RiskAgent", "EvidenceAgent", "High/Medium Risk"),
        ("RiskAgent", "PolicySLM", "Low Risk"),
        ("EvidenceAgent", "ExternalSearch", "High Novelty"),
        ("ExternalSearch", "PolicySLM", "Context"),
        ("EvidenceAgent", "FactualityAgent", None),
        ("FactualityAgent", "PolicySLM", None),
        ("PolicySLM", "PolicyFallback", "Fallback"),
        ("PolicySLM", "PolicyAgent", "SLM"),
        ("PolicyFallback", "PolicyAgent", None),
        ("PolicyAgent", "QualityGate", None),
        ("QualityGate", "DecisionOrch", None),
        ("DecisionOrch", "HumanReview", "Escalate"),
        ("DecisionOrch", "Governance", "Auto"),
        ("HumanReview", "Governance", None),
        ("Governance", "Metrics", None),
    ]

    # Add edges with styling based on active/inactive status
    for from_node, to_node, label in all_edges:
        is_active = (from_node, to_node) in active_edges

        if is_active:
            # Active edge: bold, solid, bright color
            if label == "Escalate" or (from_node == "RiskAgent" and to_node == "EvidenceAgent"):
                # Escalation or high-risk path: orange/red
                edge_color = "#FF7043"
                edge_width = 4
            elif label == "Auto" or (from_node == "DecisionOrch" and to_node == "Governance"):
                # Normal flow: green
                edge_color = "#4CAF50"
                edge_width = 3
            else:
                # Standard active flow: blue
                edge_color = "#2196F3"
                edge_width = 3
            net.add_edge(from_node, to_node, label=label, color=edge_color, width=edge_width)
        else:
            # Inactive edge: thin, dashed, grayed
            net.add_edge(from_node, to_node, label=label, color="#B0BEC5", width=1, dashes=True)

    # Configure layout with fixed positions - disable physics since nodes are fixed
    # This ensures branches are properly separated vertically
    net.set_options("""
    {
      "physics": {
        "enabled": false
      },
      "layout": {
        "improvedLayout": false
      },
      "edges": {
        "smooth": {
          "type": "curvedCW",
          "roundness": 0.2
        },
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 0.8
          }
        },
        "font": {
          "size": 12,
          "align": "middle"
        }
      }
    }
    """)

    # Generate HTML and display
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w") as tmp_file:
        net.save_graph(tmp_file.name)
        html_string = open(tmp_file.name, "r", encoding="utf-8").read()
        os.unlink(tmp_file.name)

    st.components.v1.html(html_string, height=620, scrolling=False)


def _get_node_style_for_status(status: str) -> dict:
    """Get node style based on execution status."""
    if status == "completed":
        return {"color": "#4CAF50", "border": "#2E7D32", "borderWidth": 3}
    elif status == "skipped":
        return {"color": "#B0BEC5", "border": "#78909C", "borderWidth": 1, "borderDashes": True}
    elif status == "error":
        return {"color": "#E53935", "border": "#C62828", "borderWidth": 3}
    else:  # pending
        return {"color": "#D3D3D3", "border": "#9E9E9E", "borderWidth": 1}


def _render_claim_with_subclaims(claim: Claim, level: int = 0) -> None:
    """Render a claim with its subclaims in a hierarchical format."""
    indent = "  " * level
    prefix = "└─ " if level > 0 else ""

    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"{indent}{prefix}**{claim.text}**")
        with col2:
            st.caption(f"Domain: {claim.domain.value} | Confidence: {claim.confidence:.2f}")

        if claim.is_explicit:
            st.caption(f"{indent}  (Explicit claim)")
        else:
            st.caption(f"{indent}  (Implicit claim)")

        if claim.decomposition_method:
            st.caption(f"{indent}  Decomposition: {claim.decomposition_method}")

        if claim.subclaims:
            st.markdown(f"{indent}  **Subclaims:**")
            for subclaim in claim.subclaims:
                _render_claim_with_subclaims(subclaim, level + 1)


def _get_source_type_badge(source_type: Optional[SourceType], source: Optional[str] = None) -> str:
    """Get a colored badge for source type."""
    if not source_type:
        return "🔷 Unknown"

    source_label = f" ({source})" if source else ""
    badges = {
        SourceType.AUTHORITATIVE: f"🟢 Authoritative{source_label}",
        SourceType.HIGH_CREDIBILITY: f"🔵 High Credibility{source_label}",
        SourceType.SCIENTIFIC: f"🔬 Scientific{source_label}",
        SourceType.FACT_CHECK: f"✅ Fact Check{source_label}",
        SourceType.INTERNAL: f"📚 Internal{source_label}",
        SourceType.EXTERNAL: f"🌐 External{source_label}",
    }
    return badges.get(source_type, f"🔷 Unknown{source_label}")


def _truncate_text(text: str, max_len: int = 240) -> str:
    if len(text) <= max_len:
        return text
    return f"{text[:max_len].rstrip()}..."


def _render_pie_chart(title: str, labels: List[str], values: List[float]) -> None:
    total = sum(values)
    if total <= 0:
        st.caption(f"{title}: No data available.")
        return

    def _autopct(pct: float) -> str:
        count = int(round(pct * total / 100.0))
        return f"{pct:.0f}% ({count})"

    fig, ax = plt.subplots(figsize=(3, 2.4))
    wedges, _, _ = ax.pie(values, labels=None, autopct=_autopct, textprops={"fontsize": 9})
    ax.set_title(title)
    ax.axis("equal")
    ax.legend(
        wedges,
        labels,
        loc="center",
        bbox_to_anchor=(0.5, 0),
        frameon=True,
        framealpha=0.5,
        fontsize=8,
    )
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def _render_bar_chart(
    title: str,
    labels: List[str],
    values: List[float],
    total_count: Optional[int] = None
) -> None:
    if not values:
        st.caption(f"{title}: No data available.")
        return
    total = sum(values)
    fig, ax = plt.subplots(figsize=(3, 2.4))
    bars = ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, max(values) * 1.25 if values else 1)
    for bar, value in zip(bars, values):
        if total_count is not None:
            count = int(round(value * total_count))
            pct = value * 100.0
        else:
            count = int(round(value))
            pct = (value / total * 100.0) if total else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{pct:.0f}% ({count})",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def _collect_atomic_claims(claims: List[Claim]) -> List[Claim]:
    atomic_claims = []
    for claim in claims:
        if claim.subclaims:
            atomic_claims.extend(_collect_atomic_claims(claim.subclaims))
        else:
            atomic_claims.append(claim)
    return atomic_claims


def _render_evidence_item(item: EvidenceItem) -> None:
    """Render a single evidence item with source type badge and URL."""
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"**{_truncate_text(item.text, 200)}**")
        if item.source:
            st.caption(f"Source: {item.source}")
        if item.url:
            st.markdown(f"Source URL: [{item.url}]({item.url})")
    with col2:
        st.markdown(_get_source_type_badge(item.source_type, item.source))
        st.caption(f"Relevance: {item.relevance_score:.2f}")


def _render_factuality_assessment(assessment: FactualityAssessment) -> None:
    """Render a factuality assessment with evidence mapping."""
    st.markdown(f"**Claim**: {assessment.claim_text}")

    col1, col2, col3 = st.columns(3)
    with col1:
        status_colors = {
            "Likely True": "🟢",
            "Likely False": "🔴",
            "Uncertain / Disputed": "🟡"
        }
        st.markdown(f"{status_colors.get(assessment.status.value, '⚪')} **{assessment.status.value}**")
    with col2:
        st.caption(f"Confidence: {assessment.confidence:.2f}")
    with col3:
        st.caption(f"Evidence items: {len(assessment.quoted_evidence)}")

    st.markdown(f"**Reasoning**: {assessment.reasoning}")
    st.markdown(f"**Evidence Summary**: {assessment.evidence_summary}")

    if assessment.evidence_map:
        with st.expander("Evidence Mapping", expanded=False):
            if assessment.evidence_map.get("supports"):
                st.markdown("**Supports:**")
                for quote in assessment.evidence_map["supports"]:
                    st.markdown(f"  • {quote}")
            if assessment.evidence_map.get("contradicts"):
                st.markdown("**Contradicts:**")
                for quote in assessment.evidence_map["contradicts"]:
                    st.markdown(f"  • {quote}")
            if assessment.evidence_map.get("does_not_address"):
                st.markdown("**Does Not Address:**")
                for quote in assessment.evidence_map["does_not_address"]:
                    st.markdown(f"  • {quote}")

    if assessment.quoted_evidence:
        with st.expander("Quoted Evidence", expanded=False):
            for i, quote in enumerate(assessment.quoted_evidence, 1):
                st.markdown(f"{i}. {quote}")


def render_agent_details(
    analysis: AnalysisResponse,
    policy_text: str,
) -> None:
    total_agents = len(analysis.agent_executions)
    for index, detail in enumerate(analysis.agent_executions):
        header = f"{detail.agent_name} ({detail.agent_type})"
        with st.expander(header, expanded=False):
            st.markdown(f"**Agent rank**: {index + 1}/{total_agents}")
            st.markdown(f"**Status**: {detail.status}")
            if detail.execution_time_ms is not None:
                st.markdown(f"**Execution time**: {detail.execution_time_ms:.0f} ms")
            if detail.model_name:
                st.markdown(f"**Model**: {detail.model_name}")
            if detail.model_provider:
                st.markdown(f"**Provider**: {detail.model_provider}")
            if detail.policy_version:
                st.markdown(f"**Policy version**: {detail.policy_version}")
            if detail.confidence is not None:
                st.markdown(f"**Confidence**: {detail.confidence:.2f}")
            if detail.route_reason:
                st.markdown(f"**Route**: {detail.route_reason}")
            if detail.fallback_used:
                st.markdown("**Fallback used**: Yes")
            if detail.prompt_hash:
                st.markdown(f"**Prompt hash**: `{detail.prompt_hash}`")
            if detail.error:
                st.error(detail.error)

            with st.expander("Prompts", expanded=False):
                if detail.system_prompt:
                    st.markdown("System prompt")
                    st.code(detail.system_prompt)
                if detail.user_prompt:
                    st.markdown("User prompt")
                    st.code(detail.user_prompt)

            st.markdown("**Result**")
            if detail.agent_type == "claim":
                st.json([claim.model_dump() for claim in analysis.claims])
            elif detail.agent_type == "risk":
                st.json(analysis.risk_assessment.model_dump())
            elif detail.agent_type == "evidence":
                st.json(analysis.evidence.model_dump() if analysis.evidence else {})
            elif detail.agent_type == "factuality":
                st.json([item.model_dump() for item in analysis.factuality_assessments])
            elif detail.agent_type == "policy":
                st.json(analysis.policy_interpretation.model_dump() if analysis.policy_interpretation else {})
                if policy_text:
                    with st.expander("Policy text", expanded=False):
                        st.code(policy_text)


def load_recent_decisions(limit: int = 20) -> List[DecisionRecord]:
    session = SessionLocal()
    try:
        return (
            session.query(DecisionRecord)
            .options(joinedload(DecisionRecord.review))
            .order_by(DecisionRecord.created_at.desc())
            .limit(limit)
            .all()
        )
    finally:
        session.close()


def load_recent_reviews(limit: int = 20) -> List[ReviewRecord]:
    session = SessionLocal()
    try:
        return session.query(ReviewRecord).order_by(ReviewRecord.created_at.desc()).limit(limit).all()
    finally:
        session.close()


def main() -> None:
    st.set_page_config(page_title="Agentic Factuality Evaluator", layout="wide")
    st.title("Agentic Factuality Evaluator")
    st.caption("Run the pipeline and inspect each agent's prompts, routing, and results.")

    policy_text = load_policy_text()

    tabs = st.tabs(["Analysis", "Dashboard", "Human Review"])

    with tabs[0]:
        transcript = st.text_area(
            "Transcript",
            placeholder="Paste content transcript here...",
            height=180,
        )

        run_analysis = st.button("Analyze Transcript", type="primary", disabled=not transcript.strip())

        if "analysis" not in st.session_state:
            st.session_state.analysis = None

        if run_analysis:
            stage_order = [
                "Claim extraction",
                "Claim decomposition",
                "Evidence retrieval",
                "Claim-evidence evaluation",
                "Risk & policy classification",
            ]
            stage_status = {stage: "pending" for stage in stage_order}
            progress_bar = st.progress(0)
            status_container = st.empty()

            def render_stage_status() -> None:
                labels = {
                    "pending": "Pending",
                    "in_progress": "In progress",
                    "done": "Done",
                    "skipped": "Skipped",
                }
                status_container.markdown(
                    "\n".join(
                        f"- {stage}: {labels[stage_status[stage]]}"
                        for stage in stage_order
                    )
                )

            def progress_callback(stage: str, status: str) -> None:
                if stage not in stage_status:
                    return
                if status == "started":
                    stage_status[stage] = "in_progress"
                elif status == "completed":
                    stage_status[stage] = "done"
                elif status == "skipped":
                    stage_status[stage] = "skipped"
                render_stage_status()
                completed = sum(
                    1 for value in stage_status.values()
                    if value in {"done", "skipped"}
                )
                progress_bar.progress(completed / len(stage_order))

            render_stage_status()
            orchestrator = DecisionOrchestrator()
            st.session_state.analysis = orchestrator.analyze(
                transcript,
                progress_callback=progress_callback
            )
            progress_bar.progress(1.0)
            # Persist governance trail for UI tabs
            governance_logger = GovernanceLogger()
            decision_id = governance_logger.log_decision(st.session_state.analysis, transcript)
            if st.session_state.analysis.decision.requires_human_review:
                pending_reviews = governance_logger.list_pending_reviews()
                if pending_reviews:
                    st.session_state.analysis.review_request_id = pending_reviews[-1].id

        analysis: Optional[AnalysisResponse] = st.session_state.analysis

        with st.expander("Provider Status", expanded=False):
            zentropi_ready = bool(settings.zentropi_api_key and settings.zentropi_labeler_id and settings.zentropi_labeler_version_id)
            st.markdown(f"**Zentropi configured**: {zentropi_ready}")
            st.markdown(f"**Groq configured**: {bool(settings.groq_api_key)}")
            st.markdown(f"**Serper configured**: {bool(settings.serper_api_key)}")

        st.subheader("Decision Flow")
        build_flow_graph(
            analysis.agent_executions if analysis else None,
            analysis,
        )

        if analysis:
            st.subheader("Routing Decision")
            route = "High/Medium risk → Evidence Agent" if analysis.risk_assessment.tier in [RiskTier.HIGH, RiskTier.MEDIUM] else "Low risk → Policy Decision"
            st.markdown(f"**Risk tier**: {analysis.risk_assessment.tier.value}")
            st.markdown(f"**Routing**: {route}")
            st.markdown(f"**Risk reasoning**: {analysis.risk_assessment.reasoning}")
            st.markdown(f"**Risk confidence**: {analysis.risk_assessment.confidence:.2f}")

            # Show novelty info for medium/high-risk cases
            if analysis.risk_assessment.tier in [RiskTier.MEDIUM, RiskTier.HIGH] and analysis.evidence:
                if analysis.evidence.contextual or analysis.evidence.supporting or analysis.evidence.contradicting:
                    external_count = len(analysis.evidence.contextual) + len(analysis.evidence.supporting) + len(analysis.evidence.contradicting)
                    st.info(f"🔍 **External search triggered**: {analysis.risk_assessment.tier.value} risk + high novelty. Found {external_count} external result(s).")
                elif analysis.evidence.evidence_gap:
                    st.warning("⚠️ **High novelty detected** (no internal evidence found), but external search may be disabled, failed, or returned no results.")

            st.subheader("Final Decision")
            st.markdown(f"**Action**: {analysis.decision.action.value}")
            st.markdown(f"**Confidence**: {analysis.decision.confidence:.2f}")
            st.markdown(f"**Rationale**: {analysis.decision.rationale}")
            if analysis.review_request_id:
                st.markdown(f"**Review request ID**: {analysis.review_request_id}")

            st.subheader("Claims")
            with st.expander("Claims (Hierarchical View)", expanded=True):
                for claim in analysis.claims:
                    _render_claim_with_subclaims(claim)
                    st.divider()
            with st.expander("Claims (JSON)", expanded=False):
                st.json([claim.model_dump() for claim in analysis.claims])

            st.subheader("Evidence & Factuality")
            if analysis.evidence:
                if analysis.evidence.evidence_gap:
                    st.warning(f"Evidence gap: {analysis.evidence.evidence_gap_reason or 'No internal evidence.'}")

                # Supporting Evidence
                if analysis.evidence.supporting:
                    st.markdown("**Supporting Evidence**")
                    for item in analysis.evidence.supporting:
                        _render_evidence_item(item)
                        st.divider()
                else:
                    st.caption("No supporting evidence found.")

                # Contradicting Evidence
                if analysis.evidence.contradicting:
                    st.markdown("**Contradicting Evidence**")
                    for item in analysis.evidence.contradicting:
                        _render_evidence_item(item)
                        st.divider()
                else:
                    st.caption("No contradicting evidence found.")

                # Contextual Evidence
                if analysis.evidence.contextual:
                    st.markdown("**Context-only Evidence**")
                    for item in analysis.evidence.contextual:
                        _render_evidence_item(item)
                        st.divider()

                # Evidence Summary
                st.info(f"**Evidence Confidence**: {analysis.evidence.evidence_confidence:.2f} | "
                       f"**Conflicts Present**: {'Yes' if analysis.evidence.conflicts_present else 'No'}")

                with st.expander("Evidence (JSON)", expanded=False):
                    st.json(analysis.evidence.model_dump())
            else:
                st.caption("No evidence retrieved (low risk or skipped).")

            if analysis.evidence and not analysis.evidence.supporting and not analysis.evidence.contradicting:
                vector_store = VectorStore()
                doc_count = len(vector_store.get_all_documents())
                if doc_count == 0:
                    st.warning("No internal evidence indexed. Run `python scripts/populate_evidence.py` to add evidence.")

            if analysis.factuality_assessments:
                st.markdown("**Factuality Assessments**")
                for assessment in analysis.factuality_assessments:
                    _render_factuality_assessment(assessment)
                    st.divider()
                with st.expander("Factuality Assessments (JSON)", expanded=False):
                    st.json([item.model_dump() for item in analysis.factuality_assessments])

            st.subheader("Policy Interpretation")
            if analysis.policy_interpretation:
                st.json(analysis.policy_interpretation.model_dump())

            st.subheader("Agent Execution Details")
            render_agent_details(analysis, policy_text)

    with tabs[1]:
        st.subheader("Dashboard")
        metrics_calculator = MetricsCalculator()
        metrics = metrics_calculator.calculate_metrics(days=7)

        risk_counts = metrics.get("case_count_by_risk_tier", {})
        decision_counts = metrics.get("case_count_by_decision_action", {})
        risk_labels = list(risk_counts.keys())
        risk_values = [risk_counts[label] for label in risk_labels]
        decision_labels = list(decision_counts.keys())
        decision_values = [decision_counts[label] for label in decision_labels]

        chart_col1, chart_col2, chart_col3, chart_col4 = st.columns(4)
        with chart_col1:
            _render_pie_chart("Risk Tier Distribution", risk_labels, risk_values)
        with chart_col2:
            _render_pie_chart("Decision Type Distribution", decision_labels, decision_values)

        total_decisions = metrics.get("total_decisions", 0)
        auto_rate = metrics.get("auto_resolved_rate", 0.0)
        auto_count = int(round(total_decisions * auto_rate)) if total_decisions else 0
        human_count = max(total_decisions - auto_count, 0)

        with chart_col3:
            _render_pie_chart(
                "Auto vs Human Review",
                ["Auto-resolved", "Human review"],
                [auto_count, human_count]
            )
        with chart_col4:
            disagreement_count = metrics.get("disagreement_count", 0)
            total_reviews = metrics.get("total_reviews", 0)
            disagreement_rate = (
                disagreement_count / total_reviews if total_reviews > 0 else 0.0
            )
            st.markdown("**Disagreement**")
            st.markdown(
                f"<div style='font-size: 40px; line-height: 1.1;'>"
                f"{disagreement_rate:.0%}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"{disagreement_count} / {total_reviews} reviews")

        st.subheader("Recent Decisions")
        decisions = load_recent_decisions()
        if decisions:
            decision_rows = [
                {
                    "id": d.id,
                    "created_at": d.created_at,
                    "action": d.decision_action,
                    "risk_tier": d.risk_assessment_json.get("tier") if d.risk_assessment_json else None,
                    "policy_version": d.policy_version,
                    "confidence": d.confidence,
                }
                for d in decisions
            ]
            st.metric("Total cases (7d)", metrics.get("total_decisions", 0))
            st.dataframe(decision_rows, width="stretch")
            st.metric("Evidence gaps (7d)", metrics.get("evidence_gap_count", 0))
            st.markdown("**Evidence gaps (targeted enrichment)**")
            gap_rows = []
            for decision in decisions:
                evidence = decision.evidence_json or {}
                if evidence.get("evidence_gap"):
                    gap_rows.append({
                        "id": decision.id,
                        "created_at": decision.created_at,
                        "risk_tier": (decision.risk_assessment_json or {}).get("tier"),
                        "reason": evidence.get("evidence_gap_reason") or "No internal evidence.",
                        "claim_sample": (decision.claims_json or [{}])[0].get("text"),
                    })
            if gap_rows:
                st.dataframe(gap_rows, width="stretch")
            else:
                st.caption("No evidence gaps found in recent decisions.")

            selected_id = st.selectbox("Inspect decision", [row["id"] for row in decision_rows])
            selected = next((d for d in decisions if d.id == selected_id), None)
            if selected:
                review_status = "Not queued"
                if selected.review:
                    if selected.review.status == "pending":
                        review_status = "Pending review"
                    elif selected.review.status == "reviewed":
                        review_status = "Reviewed"
                    else:
                        review_status = selected.review.status
                review_id = selected.review.id if selected.review else None
                review_label = f"{review_status}"
                if review_id:
                    review_label = f"{review_status} (Review ID {review_id})"
                st.caption(f"Review status: {review_label}")
                if st.button("Send to human review queue", type="secondary"):
                    governance_logger = GovernanceLogger()
                    result = governance_logger.enqueue_review_for_decision(selected.id)
                    if result == "created":
                        st.success("Sent to human review queue.")
                    elif result == "reset_pending":
                        st.success("Review reset to pending.")
                    elif result == "already_pending":
                        st.info("Review is already pending.")
                    else:
                        st.error("Failed to enqueue review.")
                    st.rerun()
                st.markdown("**Decision details**")
                st.json({
                    "decision_action": selected.decision_action,
                    "decision_rationale": selected.decision_rationale,
                    "policy_version": selected.policy_version,
                    "claims": selected.claims_json,
                    "risk_assessment": selected.risk_assessment_json,
                    "policy_interpretation": selected.policy_interpretation_json,
                    "agent_executions": selected.agent_executions_json,
                })
        else:
            st.info("No decisions logged yet.")

        st.subheader("Review Trail")
        reviews = load_recent_reviews()
        if reviews:
            review_rows = [
                {
                    "id": r.id,
                    "decision_id": r.decision_id,
                    "status": r.status,
                    "created_at": r.created_at,
                    "reviewed_at": r.reviewed_at,
                }
                for r in reviews
            ]
            st.dataframe(review_rows, width="stretch")
        else:
            st.info("No reviews found.")

    with tabs[2]:
        st.subheader("Human Review Queue")
        governance_logger = GovernanceLogger()
        pending = governance_logger.list_pending_reviews()

        # Show reviewed reviews section
        reviewed = governance_logger.list_reviewed_reviews(limit=20)
        if reviewed:
            with st.expander(f"Recently Reviewed ({len(reviewed)} reviews)", expanded=False):
                reviewed_rows = []
                reviewed_options = {}
                reviewed_decision_ids = {}
                for review_item in reviewed:
                    snippet = _truncate_text(review_item.transcript.replace("\n", " "), 100)
                    reviewed_rows.append({
                        "Decision ID": review_item.decision_id,
                        "Review ID": review_item.id,
                        "Reviewed at": review_item.reviewed_at.strftime("%Y-%m-%d %H:%M:%S") if review_item.reviewed_at else "N/A",
                        "Human Decision": review_item.human_decision.action.value if review_item.human_decision else "N/A",
                        "Transcript snippet": snippet,
                    })
                    reviewed_options[review_item.id] = review_item
                    reviewed_decision_ids[review_item.id] = review_item.decision_id
                st.dataframe(reviewed_rows, width="stretch")

                # Reset controls
                st.markdown("**Reset Reviews to Pending**")
                col1, col2 = st.columns(2)
                with col1:
                    selected_review_id = st.selectbox(
                        "Select review to reset",
                        options=list(reviewed_options.keys()),
                        format_func=lambda x: f"Review {x} (Decision {reviewed_decision_ids.get(x)}) - {_truncate_text(reviewed_options[x].transcript.replace(chr(10), ' '), 60)}"
                    )
                    reset_single = st.button("Reset Selected Review", type="secondary")

                with col2:
                    reset_all = st.button("Reset All Reviewed Reviews", type="secondary")

                if reset_single:
                    try:
                        success = governance_logger.reset_review_to_pending(selected_review_id)
                        if success:
                            st.success(f"Review {selected_review_id} reset to pending.")
                            st.rerun()
                        else:
                            st.error(f"Failed to reset review {selected_review_id}.")
                    except Exception as e:
                        st.error(f"Error resetting review: {str(e)}")
                        st.exception(e)

                if reset_all:
                    try:
                        reset_count = 0
                        failed_count = 0
                        for review_id in reviewed_options.keys():
                            if governance_logger.reset_review_to_pending(review_id):
                                reset_count += 1
                            else:
                                failed_count += 1
                        if reset_count > 0:
                            st.success(f"Reset {reset_count} review(s) to pending.")
                        if failed_count > 0:
                            st.warning(f"Failed to reset {failed_count} review(s).")
                        if reset_count > 0:
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error resetting reviews: {str(e)}")
                        st.exception(e)

        if not pending:
            st.info("No pending reviews.")
        else:
            # Initialize session state for current review index
            if "current_review_index" not in st.session_state:
                st.session_state.current_review_index = 0

            # Build review list and table
            pending_rows = []
            review_list = []
            review_id_to_decision_id = {}
            for review_item in pending:
                snippet = _truncate_text(review_item.transcript.replace("\n", " "), 140)
                pending_rows.append({
                    "Decision ID": review_item.decision_id,
                    "Review ID": review_item.id,
                    "Risk tier": review_item.risk_assessment.tier.value,
                    "Transcript snippet": snippet,
                })
                review_list.append(review_item)
                review_id_to_decision_id[review_item.id] = review_item.decision_id

            # Display review queue table at the top
            st.dataframe(pending_rows, width="stretch")

            # Navigation controls
            total_reviews = len(review_list)
            if total_reviews > 0:
                # Ensure index is within bounds
                if st.session_state.current_review_index >= total_reviews:
                    st.session_state.current_review_index = 0
                if st.session_state.current_review_index < 0:
                    st.session_state.current_review_index = total_reviews - 1

                current_review = review_list[st.session_state.current_review_index]
                case_ids = [review.id for review in review_list]
                current_case_id = current_review.id

                # Navigation buttons - Previous on far left, Next on far right
                nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
                with nav_col1:
                    if st.button("◀ Previous", disabled=(st.session_state.current_review_index == 0), use_container_width=True):
                        st.session_state.current_review_index = max(0, st.session_state.current_review_index - 1)
                        st.rerun()
                with nav_col2:
                    # Center: Clickable Case ID selector
                    # Use a dynamic key based on index to force update when index changes
                    selectbox_key = f"case_id_selector_{st.session_state.current_review_index}"
                    selected_case_id = st.selectbox(
                        "Select review to inspect",
                        options=case_ids,
                        index=st.session_state.current_review_index,
                        format_func=lambda x: f"Review {x} (Decision {review_id_to_decision_id.get(x)})",
                        key=selectbox_key,
                        label_visibility="collapsed"
                    )
                    # Update index if case ID changed via dropdown
                    selected_index = case_ids.index(selected_case_id)
                    if selected_index != st.session_state.current_review_index:
                        st.session_state.current_review_index = selected_index
                        st.rerun()
                with nav_col3:
                    if st.button("Next ▶", disabled=(st.session_state.current_review_index >= total_reviews - 1), use_container_width=True):
                        st.session_state.current_review_index = min(total_reviews - 1, st.session_state.current_review_index + 1)
                        st.rerun()

                st.divider()

                # Display current review
                review = review_list[st.session_state.current_review_index]

            st.markdown("**Transcript**")
            st.code(review.transcript)

            st.markdown("**Risk Assessment**")
            st.json(review.risk_assessment.model_dump())

            st.markdown("**Claims (Hierarchical View)**")
            with st.expander("Claims (Hierarchical View)", expanded=False):
                for claim in review.claims:
                    _render_claim_with_subclaims(claim)
                    st.divider()

            st.markdown("**Claim Review**")
            atomic_claims = _collect_atomic_claims(review.claims)
            assessments_by_claim = {
                assessment.claim_text: assessment
                for assessment in (review.factuality_assessments or [])
            }

            for claim in atomic_claims:
                assessment = assessments_by_claim.get(claim.text)
                left, right = st.columns(2)
                with left:
                    st.markdown(f"**Claim**: {claim.text}")
                    st.caption(f"Domain: {claim.domain.value} | Explicit: {claim.is_explicit}")
                    st.caption(f"Claim confidence: {claim.confidence:.2f}")
                    if claim.decomposition_method:
                        st.caption(f"Decomposition: {claim.decomposition_method}")
                    if assessment:
                        st.markdown(f"**Factuality**: {assessment.status.value}")
                        st.caption(f"Factuality confidence: {assessment.confidence:.2f}")
                        st.markdown(f"**Model summary**: {_truncate_text(assessment.reasoning, 240)}")
                    else:
                        st.caption("No factuality assessment available.")
                with right:
                    st.markdown("**Evidence mapping**")
                    if assessment and assessment.evidence_map:
                        supports = assessment.evidence_map.get("supports", [])
                        contradicts = assessment.evidence_map.get("contradicts", [])
                        does_not_address = assessment.evidence_map.get("does_not_address", [])
                        st.markdown("**Supports**")
                        if supports:
                            for quote in supports:
                                st.caption(quote)
                        else:
                            st.caption("None")
                        st.markdown("**Contradicts**")
                        if contradicts:
                            for quote in contradicts:
                                st.caption(quote)
                        else:
                            st.caption("None")
                        st.markdown("**Does not address**")
                        if does_not_address:
                            for quote in does_not_address:
                                st.caption(quote)
                        else:
                            st.caption("None")
                    else:
                        st.caption("No evidence mapping available.")
                st.divider()

            if review.evidence:
                st.markdown("**Evidence Summary**")
                if review.evidence.evidence_gap:
                    st.warning(f"Evidence gap: {review.evidence.evidence_gap_reason or 'No internal evidence.'}")

                st.info(
                    f"**Evidence Dashboard**: Confidence {review.evidence.evidence_confidence:.2f} | "
                    f"Conflicts {'Yes' if review.evidence.conflicts_present else 'No'}"
                )

                with st.expander("Evidence (All Sources)", expanded=False):
                    if review.evidence.supporting:
                        st.markdown("**Supporting Evidence**")
                        for item in review.evidence.supporting:
                            _render_evidence_item(item)
                            st.divider()
                    if review.evidence.contradicting:
                        st.markdown("**Contradicting Evidence**")
                        for item in review.evidence.contradicting:
                            _render_evidence_item(item)
                            st.divider()
                    if review.evidence.contextual:
                        st.markdown("**Context-only Evidence**")
                        for item in review.evidence.contextual:
                            _render_evidence_item(item)
                            st.divider()

                with st.expander("Evidence (JSON)", expanded=False):
                    st.json({
                        "supporting": [item.model_dump() for item in review.evidence.supporting],
                        "contradicting": [item.model_dump() for item in review.evidence.contradicting],
                        "contextual": [item.model_dump() for item in review.evidence.contextual],
                    })

            if review.factuality_assessments:
                with st.expander("Factuality Assessments (JSON)", expanded=False):
                    st.json([item.model_dump() for item in review.factuality_assessments])

            if review.policy_interpretation:
                st.markdown("**Policy Interpretation**")
                st.json(review.policy_interpretation.model_dump())

            st.markdown("**System Decision**")
            st.json(review.system_decision.model_dump())

            # Display existing reviewer feedback if available
            if review.reviewer_feedback:
                st.subheader("Previous Reviewer Feedback")
                feedback = review.reviewer_feedback
                if isinstance(feedback, dict):
                    # Handle dict format (from JSON)
                    st.json(feedback)
                else:
                    # Handle ReviewerFeedback object
                    st.markdown(f"**Action**: {feedback.action.value}")
                    if feedback.review_time_seconds:
                        st.markdown(f"**Review Time**: {feedback.review_time_seconds:.1f} seconds")
                    if feedback.reviewer_notes:
                        st.markdown(f"**Notes**: {feedback.reviewer_notes}")
                    if feedback.proposed_change:
                        st.markdown("**Proposed Change**:")
                        st.json(feedback.proposed_change.model_dump())
                    if feedback.accepted_change:
                        st.markdown("**Accepted Change**:")
                        st.json(feedback.accepted_change.model_dump())

            st.subheader("Submit Override / Feedback")

            # Decision override
            action = st.selectbox("Decision override", [a.value for a in DecisionAction])
            rationale = st.text_area("Rationale", height=120)

            # Reviewer action
            reviewer_action = st.selectbox(
                "Reviewer Action",
                [a.value for a in ReviewerAction],
                help="Select the type of action you're taking"
            )

            # Review metadata
            col1, col2 = st.columns(2)
            with col1:
                review_time_seconds = st.number_input(
                    "Review Time (seconds)",
                    min_value=0.0,
                    value=0.0,
                    step=0.1,
                    help="Time spent reviewing this case"
                )
            with col2:
                reviewer_notes = st.text_area("Reviewer Notes", height=80, help="Additional notes about this review")

            # Change proposal
            with st.expander("System Change Proposal (Optional)", expanded=False):
                st.markdown("Propose changes to system behavior based on this review.")

                prompt_updates = st.text_area(
                    "Prompt Updates (JSON)",
                    height=100,
                    help='JSON object with component names as keys and prompt edits as values, e.g. {"claim_agent": "new prompt"}',
                    value=""
                )

                threshold_updates = st.text_area(
                    "Threshold Updates (JSON)",
                    height=80,
                    help='JSON object with threshold names and values, e.g. {"risk_confidence_threshold": 0.8}',
                    value=""
                )

                weighting_updates = st.text_area(
                    "Weighting Updates (JSON)",
                    height=80,
                    help='JSON object with weighting adjustments, e.g. {"authoritative_weight": 1.2}',
                    value=""
                )

                change_rationale = st.text_area(
                    "Change Rationale",
                    height=60,
                    help="Explain why these changes are needed"
                )

                proposed_change = None
                if prompt_updates or threshold_updates or weighting_updates or change_rationale:
                    try:
                        prompt_dict = json.loads(prompt_updates) if prompt_updates.strip() else {}
                        threshold_dict = json.loads(threshold_updates) if threshold_updates.strip() else {}
                        weighting_dict = json.loads(weighting_updates) if weighting_updates.strip() else {}

                        proposed_change = ChangeProposal(
                            prompt_updates=prompt_dict if isinstance(prompt_dict, dict) else {},
                            threshold_updates=threshold_dict if isinstance(threshold_dict, dict) else {},
                            weighting_updates=weighting_dict if isinstance(weighting_dict, dict) else {},
                            rationale=change_rationale if change_rationale.strip() else None
                        )
                    except json.JSONDecodeError as e:
                        st.warning(f"Invalid JSON in change proposal: {e}")
                    except Exception as e:
                        st.warning(f"Error creating change proposal: {e}")

            # Accepted change (if editing the proposal)
            accepted_change = None
            if proposed_change:
                edit_accepted = st.checkbox("Edit the proposed change before accepting")
                if edit_accepted:
                    accepted_prompt_updates = st.text_area(
                        "Accepted Prompt Updates (JSON)",
                        height=100,
                        value=prompt_updates
                    )
                    accepted_threshold_updates = st.text_area(
                        "Accepted Threshold Updates (JSON)",
                        height=80,
                        value=threshold_updates
                    )
                    accepted_weighting_updates = st.text_area(
                        "Accepted Weighting Updates (JSON)",
                        height=80,
                        value=weighting_updates
                    )
                    accepted_rationale = st.text_area(
                        "Accepted Rationale",
                        height=60,
                        value=change_rationale
                    )

                    try:
                        accepted_prompt_dict = json.loads(accepted_prompt_updates) if accepted_prompt_updates.strip() else {}
                        accepted_threshold_dict = json.loads(accepted_threshold_updates) if accepted_threshold_updates.strip() else {}
                        accepted_weighting_dict = json.loads(accepted_weighting_updates) if accepted_weighting_updates.strip() else {}

                        accepted_change = ChangeProposal(
                            prompt_updates=accepted_prompt_dict if isinstance(accepted_prompt_dict, dict) else {},
                            threshold_updates=accepted_threshold_dict if isinstance(accepted_threshold_dict, dict) else {},
                            weighting_updates=accepted_weighting_dict if isinstance(accepted_weighting_dict, dict) else {},
                            rationale=accepted_rationale if accepted_rationale.strip() else None
                        )
                    except json.JSONDecodeError as e:
                        st.warning(f"Invalid JSON in accepted change: {e}")
                    except Exception as e:
                        st.warning(f"Error creating accepted change: {e}")
                else:
                    accepted_change = proposed_change

            if st.button("Submit human decision", type="primary"):
                try:
                    decision = Decision(
                        action=DecisionAction(action),
                        rationale=rationale or "Human override",
                        requires_human_review=False,
                        confidence=1.0,
                        escalation_reason=None
                    )

                    # Create ReviewerFeedback
                    reviewer_feedback = ReviewerFeedback(
                        action=ReviewerAction(reviewer_action),
                        review_time_seconds=review_time_seconds if review_time_seconds > 0 else None,
                        reviewer_notes=reviewer_notes.strip() if reviewer_notes.strip() else None,
                        proposed_change=proposed_change,
                        accepted_change=accepted_change
                    )

                    success = governance_logger.submit_human_decision(
                        review_id=review.id,
                        human_decision=decision,
                        human_rationale=rationale or "Human override",
                        reviewer_feedback=reviewer_feedback
                    )
                    if success:
                        st.success("Review submitted.")
                        st.rerun()
                    else:
                        st.error("Failed to submit review.")
                except Exception as e:
                    st.error(f"Error submitting review: {str(e)}")
                    import traceback
                    st.exception(e)


if __name__ == "__main__":
    main()
