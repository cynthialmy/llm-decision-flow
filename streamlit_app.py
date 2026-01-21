import os
from typing import List, Optional
import tempfile

import streamlit as st
from pyvis.network import Network

from src.orchestrator.decision_orchestrator import DecisionOrchestrator
from src.models.schemas import AnalysisResponse, AgentExecutionDetail, RiskTier
from src.config import settings


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
    risk_tier: Optional[RiskTier],
    requires_human_review: Optional[bool],
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
        "RiskAgent": {"label": "Risk Agent", "level": 2, "y": 0},
        # High/Medium risk path: positioned above center
        "EvidenceAgent": {"label": "Evidence Agent\n(RAG)", "level": 3, "y": -150},
        "FactualityAgent": {"label": "Factuality Agent", "level": 4, "y": -150},
        # Policy Agent: can be reached from both paths, positioned at center
        "PolicyAgent": {"label": "Policy Interpretation\nAgent", "level": 5, "y": 0},
        "DecisionOrch": {"label": "Decision\nOrchestrator", "level": 6, "y": 0},
        # Escalate path: positioned below center
        "HumanReview": {"label": "Human Review\nInterface", "level": 7, "y": 150},
        "Governance": {"label": "Governance Log", "level": 8, "y": 0},
        "Metrics": {"label": "Metrics\nDashboard", "level": 9, "y": 0},
    }

    # Determine active execution path
    active_edges = set()
    
    # Always active: initial flow
    active_edges.add(("Transcript", "ClaimAgent"))
    active_edges.add(("ClaimAgent", "RiskAgent"))
    
    # Routing based on risk tier
    if risk_tier in (RiskTier.HIGH, RiskTier.MEDIUM):
        # High/Medium risk: Full path through Evidence -> Factuality -> Policy
        active_edges.add(("RiskAgent", "EvidenceAgent"))
        active_edges.add(("EvidenceAgent", "FactualityAgent"))
        active_edges.add(("FactualityAgent", "PolicyAgent"))
        active_edges.add(("PolicyAgent", "DecisionOrch"))
    elif risk_tier == RiskTier.LOW:
        # Low risk: Skip Evidence/Factuality, go directly to Policy Agent
        active_edges.add(("RiskAgent", "PolicyAgent"))
        active_edges.add(("PolicyAgent", "DecisionOrch"))
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
            elif node_id == "EvidenceAgent":
                status = status_by_type.get("evidence", "pending")
                return _get_node_style_for_status(status)
            elif node_id == "FactualityAgent":
                status = status_by_type.get("factuality", "pending")
                return _get_node_style_for_status(status)
            elif node_id == "PolicyAgent":
                status = status_by_type.get("policy", "pending")
                return _get_node_style_for_status(status)
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
        ("ClaimAgent", "RiskAgent", None),
        ("RiskAgent", "EvidenceAgent", "High/Medium Risk"),
        ("RiskAgent", "PolicyAgent", "Low Risk"),
        ("EvidenceAgent", "FactualityAgent", None),
        ("FactualityAgent", "PolicyAgent", None),
        ("PolicyAgent", "DecisionOrch", None),
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


def main() -> None:
    st.set_page_config(page_title="LLM Decision Flow", layout="wide")
    st.title("LLM Decision Flow - Step-by-Step Analysis")
    st.caption("Run the pipeline and inspect each agent's prompts, routing, and results.")

    policy_text = load_policy_text()

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

    analysis: Optional[AnalysisResponse] = st.session_state.analysis

    st.subheader("Decision Flow")
    build_flow_graph(
        analysis.agent_executions if analysis else None,
        analysis.risk_assessment.tier if analysis else None,
        analysis.decision.requires_human_review if analysis else None,
    )

    if analysis:
        st.subheader("Routing Decision")
        route = "High/Medium risk → Evidence Agent" if analysis.risk_assessment.tier in [RiskTier.HIGH, RiskTier.MEDIUM] else "Low risk → Decision Orchestrator"
        st.markdown(f"**Risk tier**: {analysis.risk_assessment.tier.value}")
        st.markdown(f"**Routing**: {route}")
        st.markdown(f"**Risk reasoning**: {analysis.risk_assessment.reasoning}")

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
        if analysis.factuality_assessments:
            st.markdown("Factuality assessments")
            st.json([item.model_dump() for item in analysis.factuality_assessments])

        st.subheader("Policy Interpretation")
        if analysis.policy_interpretation:
            st.json(analysis.policy_interpretation.model_dump())

        st.subheader("Agent Execution Details")
        render_agent_details(analysis, policy_text)


if __name__ == "__main__":
    main()
