import os
from typing import List, Optional, Dict, Any
import tempfile

import streamlit as st
from pyvis.network import Network

from src.orchestrator.decision_orchestrator import DecisionOrchestrator
from src.models.schemas import AnalysisResponse, AgentExecutionDetail, RiskTier, DecisionAction, Decision
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
        return session.query(DecisionRecord).order_by(DecisionRecord.created_at.desc()).limit(limit).all()
    finally:
        session.close()


def load_recent_reviews(limit: int = 20) -> List[ReviewRecord]:
    session = SessionLocal()
    try:
        return session.query(ReviewRecord).order_by(ReviewRecord.created_at.desc()).limit(limit).all()
    finally:
        session.close()


def main() -> None:
    st.set_page_config(page_title="LLM Decision Flow", layout="wide")
    st.title("LLM Decision Flow - Step-by-Step Analysis")
    st.caption("Run the pipeline and inspect each agent's prompts, routing, and results.")

    policy_text = load_policy_text()

    tabs = st.tabs(["Analysis", "Governance", "Human Review"])

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
            with st.spinner("Running agent pipeline..."):
                orchestrator = DecisionOrchestrator()
                st.session_state.analysis = orchestrator.analyze(transcript)
                # Persist governance trail for UI tabs
                governance_logger = GovernanceLogger()
                decision_id = governance_logger.log_decision(st.session_state.analysis, transcript)
                if st.session_state.analysis.decision.requires_human_review:
                    pending_reviews = governance_logger.list_pending_reviews()
                    if pending_reviews:
                        st.session_state.analysis.review_request_id = pending_reviews[-1].id

        analysis: Optional[AnalysisResponse] = st.session_state.analysis

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

            st.subheader("Final Decision")
            st.markdown(f"**Action**: {analysis.decision.action.value}")
            st.markdown(f"**Confidence**: {analysis.decision.confidence:.2f}")
            st.markdown(f"**Rationale**: {analysis.decision.rationale}")
            if analysis.review_request_id:
                st.markdown(f"**Review request ID**: {analysis.review_request_id}")

            st.subheader("Claims")
            st.json([claim.model_dump() for claim in analysis.claims])

            st.subheader("Evidence & Factuality")
            if analysis.evidence:
                st.markdown("Supporting evidence")
                st.json([item.model_dump() for item in analysis.evidence.supporting])
                st.markdown("Contradicting evidence")
                st.json([item.model_dump() for item in analysis.evidence.contradicting])
                if analysis.evidence.contextual:
                    st.markdown("Context-only evidence")
                    st.json([item.model_dump() for item in analysis.evidence.contextual])
            if analysis.evidence and not analysis.evidence.supporting and not analysis.evidence.contradicting:
                vector_store = VectorStore()
                doc_count = len(vector_store.get_all_documents())
                if doc_count == 0:
                    st.warning("No internal evidence indexed. Run `python scripts/populate_evidence.py` to add evidence.")
            if analysis.factuality_assessments:
                st.markdown("Factuality assessments")
                st.json([item.model_dump() for item in analysis.factuality_assessments])

            st.subheader("Policy Interpretation")
            if analysis.policy_interpretation:
                st.json(analysis.policy_interpretation.model_dump())

            st.subheader("Agent Execution Details")
            render_agent_details(analysis, policy_text)

    with tabs[1]:
        st.subheader("Governance & Auditability")
        metrics_calculator = MetricsCalculator()
        metrics = metrics_calculator.calculate_metrics(days=7)
        st.markdown("**Quality Gates**")
        st.json(metrics)

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
            st.dataframe(decision_rows, width="stretch")
            selected_id = st.selectbox("Inspect decision", [row["id"] for row in decision_rows])
            selected = next((d for d in decisions if d.id == selected_id), None)
            if selected:
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
        if not pending:
            st.info("No pending reviews.")
        else:
            review_options = {f"{r.id} - {r.risk_assessment.tier.value}": r for r in pending}
            selected_key = st.selectbox("Select review", list(review_options.keys()))
            review = review_options[selected_key]

            st.markdown("**Transcript**")
            st.code(review.transcript)

            st.markdown("**Claims**")
            st.json([claim.model_dump() for claim in review.claims])

            st.markdown("**Risk Assessment**")
            st.json(review.risk_assessment.model_dump())

            if review.evidence:
                st.markdown("**Evidence (supporting/contradicting/contextual)**")
                st.json({
                    "supporting": [item.model_dump() for item in review.evidence.supporting],
                    "contradicting": [item.model_dump() for item in review.evidence.contradicting],
                    "contextual": [item.model_dump() for item in review.evidence.contextual],
                })

            if review.policy_interpretation:
                st.markdown("**Policy Interpretation**")
                st.json(review.policy_interpretation.model_dump())

            st.markdown("**System Decision**")
            st.json(review.system_decision.model_dump())

            st.subheader("Submit Override / Feedback")
            action = st.selectbox("Decision override", [a.value for a in DecisionAction])
            rationale = st.text_area("Rationale", height=120)
            feedback_labels = st.text_input("Feedback labels (comma-separated)", value="")
            evidence_snippets = st.text_area("Evidence snippets (one per line)", height=120)
            feedback_notes = st.text_area("Additional notes", height=80)

            if st.button("Submit human decision", type="primary"):
                decision = Decision(
                    action=DecisionAction(action),
                    rationale=rationale or "Human override",
                    requires_human_review=False,
                    confidence=1.0,
                    escalation_reason=None
                )
                feedback = {
                    "labels": [label.strip() for label in feedback_labels.split(",") if label.strip()],
                    "evidence_snippets": [line.strip() for line in evidence_snippets.splitlines() if line.strip()],
                    "notes": feedback_notes.strip() or None,
                }
                success = governance_logger.submit_human_decision(
                    review_id=review.id,
                    human_decision=decision,
                    human_rationale=rationale or "Human override",
                    reviewer_feedback=feedback
                )
                if success:
                    st.success("Review submitted.")
                else:
                    st.error("Failed to submit review.")


if __name__ == "__main__":
    main()
