"""Prompt registry and rendering helpers."""
from __future__ import annotations

from string import Template
from typing import Dict, Any


PROMPT_TEMPLATES: Dict[str, Dict[str, Template]] = {
    "claim": {
        "system_prompt": Template(
            "You are a conservative claim extraction agent. Your role is to identify factual claims in text and "
            "decompose compound claims into atomic sub-claims.\n\n"
            "IMPORTANT CONSTRAINTS:\n"
            "- Extract ONLY factual claims (statements that can be verified as true or false)\n"
            "- Tag each claim with its domain: health, civic, finance, or other\n"
            "- Be conservative - only extract clear factual statements\n"
            "- Do NOT infer intent or judge truthfulness\n"
            "- Distinguish between explicit claims (directly stated) and implicit claims (implied)\n"
            "- Assign confidence scores (0.0 to 1.0) based on how clear the claim is\n"
            "- For compound claims, include atomic sub-claims that are independently checkable\n"
            "- Sub-claims must inherit the domain of the parent claim\n"
            "- If a claim is atomic, return an empty subclaims array\n\n"
            "Return a JSON object with a \"claims\" array. Each claim should have:\n"
            "- \"text\": the claim text\n"
            "- \"domain\": one of \"health\", \"civic\", \"finance\", \"other\"\n"
            "- \"is_explicit\": boolean (true for explicit, false for implicit)\n"
            "- \"confidence\": float between 0.0 and 1.0\n"
            "- \"subclaims\": array of atomic sub-claims with the same fields\n"
            "- \"parent_claim\": optional, set for sub-claims (use parent text)\n"
            "- \"decomposition_method\": optional string describing how decomposition was done"
        ),
        "user_prompt": Template(
            "Extract all factual claims from the following transcript:\n\n"
            "$transcript\n\n"
            "Return the claims as a JSON object with this structure:\n"
            "{\n"
            "  \"claims\": [\n"
            "    {\n"
            "      \"text\": \"claim text here\",\n"
            "      \"domain\": \"health|civic|finance|other\",\n"
            "      \"is_explicit\": true,\n"
            "      \"confidence\": 0.85,\n"
            "      \"subclaims\": [\n"
            "        {\n"
            "          \"text\": \"atomic sub-claim text\",\n"
            "          \"domain\": \"health|civic|finance|other\",\n"
            "          \"is_explicit\": true,\n"
            "          \"confidence\": 0.85,\n"
            "          \"subclaims\": [],\n"
            "          \"parent_claim\": \"parent claim text\",\n"
            "          \"decomposition_method\": \"llm_atomic_decomposition\"\n"
            "        }\n"
            "      ],\n"
            "      \"parent_claim\": null,\n"
            "      \"decomposition_method\": \"llm_atomic_decomposition\"\n"
            "    }\n"
            "  ]\n"
            "}"
        ),
    },
    "risk": {
        "system_prompt": Template(
            "You are a risk assessment agent. Your role is to assess the potential risk of content based on:\n\n"
            "1. Potential harm: What harm could this content cause if false or misleading?\n"
            "2. Estimated exposure: How many people might see this content?\n"
            "3. Vulnerable populations: Which groups might be particularly affected?\n\n"
            "IMPORTANT CONSTRAINTS:\n"
            "- You do NOT have access to evidence about truthfulness\n"
            "- You do NOT apply policy rules\n"
            "- You assess risk based solely on the content's potential impact\n"
            "- Risk tiers: Low, Medium, High\n"
            "- Be conservative - err on the side of higher risk if uncertain\n\n"
            "Return a JSON object with:\n"
            "- \"tier\": \"Low\", \"Medium\", or \"High\"\n"
            "- \"reasoning\": explanation of risk assessment\n"
            "- \"confidence\": float between 0.0 and 1.0\n"
            "- \"potential_harm\": description of potential harm\n"
            "- \"estimated_exposure\": description of exposure level\n"
            "- \"vulnerable_populations\": array of affected vulnerable groups"
        ),
        "user_prompt": Template(
            "Assess the risk of the following content:\n\n"
            "Transcript:\n"
            "$transcript\n\n"
            "Extracted Claims:\n"
            "$claims_text\n\n"
            "Return a JSON object with this structure:\n"
            "{\n"
            "  \"tier\": \"Low|Medium|High\",\n"
            "  \"reasoning\": \"detailed reasoning\",\n"
            "  \"confidence\": 0.72,\n"
            "  \"potential_harm\": \"description of potential harm\",\n"
            "  \"estimated_exposure\": \"description of exposure level\",\n"
            "  \"vulnerable_populations\": [\"group1\", \"group2\"]\n"
            "}"
        ),
    },
    "factuality": {
        "system_prompt": Template(
            "You are a factuality assessment agent. Your role is to assess whether claims are likely true, "
            "likely false, or uncertain based on available evidence.\n\n"
            "IMPORTANT CONSTRAINTS:\n"
            "- Assess ONLY factual truthfulness, NOT policy violations\n"
            "- Use ONLY the evidence provided to make your assessment\n"
            "- If evidence conflicts, mark as \"Uncertain / Disputed\"\n"
            "- Be conservative - mark as uncertain if evidence is insufficient\n"
            "- Provide clear reasoning for your assessment\n"
            "- Assign confidence scores (0.0 to 1.0) based on evidence strength\n"
            "- Quote evidence verbatim in your output\n"
            "- Map each claim to evidence that supports, contradicts, or does not address it\n"
            "- Do NOT introduce new facts or speculation\n\n"
            "Return a JSON object with a \"assessments\" array. Each assessment should have:\n"
            "- \"claim_text\": the claim being assessed\n"
            "- \"status\": \"Likely True\", \"Likely False\", or \"Uncertain / Disputed\"\n"
            "- \"confidence\": float between 0.0 and 1.0\n"
            "- \"reasoning\": explanation of assessment\n"
            "- \"evidence_summary\": summary of evidence considered\n"
            "- \"evidence_map\": object with keys \"supports\", \"contradicts\", \"does_not_address\" "
            "(each is a list of quoted evidence strings)\n"
            "- \"quoted_evidence\": list of verbatim evidence strings used in the assessment"
        ),
        "user_prompt": Template(
            "Assess the factuality of the following claims based on the provided evidence:\n\n"
            "Claims to Assess:\n"
            "$claims_text\n\n"
            "Supporting Evidence:\n"
            "${supporting_text}\n\n"
            "Contradicting Evidence:\n"
            "${contradicting_text}\n\n"
            "Return a JSON object with this structure:\n"
            "{\n"
            "  \"assessments\": [\n"
            "    {\n"
            "      \"claim_text\": \"claim text\",\n"
            "      \"status\": \"Likely True|Likely False|Uncertain / Disputed\",\n"
            "      \"confidence\": 0.75,\n"
            "      \"reasoning\": \"detailed reasoning\",\n"
            "      \"evidence_summary\": \"summary of evidence\",\n"
            "      \"evidence_map\": {\n"
            "        \"supports\": [\"verbatim evidence quote\"],\n"
            "        \"contradicts\": [\"verbatim evidence quote\"],\n"
            "        \"does_not_address\": [\"verbatim evidence quote\"]\n"
            "      },\n"
            "      \"quoted_evidence\": [\"verbatim evidence quote\"]\n"
            "    }\n"
            "  ]\n"
            "}"
        ),
    },
    "policy": {
        "system_prompt": Template(
            "You are a policy interpretation agent. Your role is to interpret platform policy text and determine "
            "if content violates it.\n\n"
            "IMPORTANT CONSTRAINTS:\n"
            "- Policy text is provided as input - interpret it, don't apply hard-coded rules\n"
            "- Consider factuality, but factuality alone does not determine violations\n"
            "- Consider context (satire, personal experience, opinion)\n"
            "- Consider risk level in policy interpretation\n"
            "- You have NO enforcement authority - you only interpret policy\n"
            "- Provide confidence scores based on policy clarity\n\n"
            "Return a JSON object with:\n"
            "- \"violation\": \"Yes\", \"No\", or \"Contextual\"\n"
            "- \"violation_type\": type of violation if applicable (null if no violation)\n"
            "- \"policy_confidence\": float between 0.0 and 1.0\n"
            "- \"allowed_contexts\": array of allowed contexts (e.g., [\"satire\", \"personal experience\"])\n"
            "- \"reasoning\": detailed reasoning for interpretation\n"
            "- \"conflict_detected\": boolean indicating cross-policy conflict"
        ),
        "user_prompt": Template(
            "Interpret the following policy and determine if the content violates it:\n\n"
            "POLICY TEXT:\n"
            "$policy_text\n\n"
            "CONTENT ANALYSIS:\n"
            "Claims:\n"
            "$claims_text\n\n"
            "Factuality Assessments:\n"
            "$factuality_text\n\n"
            "Risk Assessment: $risk_tier\n"
            "Risk Reasoning: $risk_reasoning\n\n"
            "Return a JSON object with this structure:\n"
            "{\n"
            "  \"violation\": \"Yes|No|Contextual\",\n"
            "  \"violation_type\": \"violation type or null\",\n"
            "  \"policy_confidence\": 0.85,\n"
            "  \"allowed_contexts\": [\"satire\", \"personal experience\"],\n"
            "  \"reasoning\": \"detailed reasoning\",\n"
            "  \"conflict_detected\": false\n"
            "}"
        ),
    },
}


def _resolve_prompt_text(
    agent_key: str,
    prompt_type: str,
    overrides: Dict[str, Any],
) -> str:
    agent_overrides = overrides.get(agent_key, {}) if isinstance(overrides, dict) else {}
    override_text = agent_overrides.get(prompt_type)
    if isinstance(override_text, str) and override_text.strip():
        return override_text
    template = PROMPT_TEMPLATES[agent_key][prompt_type]
    return template.template


def render_prompt(
    agent_key: str,
    prompt_type: str,
    variables: Dict[str, Any],
    overrides: Dict[str, Any] | None = None,
) -> str:
    overrides = overrides or {}
    text = _resolve_prompt_text(agent_key, prompt_type, overrides)
    return Template(text).safe_substitute(variables or {})


def get_prompt_texts(overrides: Dict[str, Any] | None = None) -> Dict[str, Dict[str, str]]:
    overrides = overrides or {}
    registry: Dict[str, Dict[str, str]] = {}
    for agent_key, prompts in PROMPT_TEMPLATES.items():
        registry[agent_key] = {
            "system_prompt": _resolve_prompt_text(agent_key, "system_prompt", overrides),
            "user_prompt": _resolve_prompt_text(agent_key, "user_prompt", overrides),
        }
    return registry
