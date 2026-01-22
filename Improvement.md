# Policy Optimization Plan

## Approach

- Introduce explicit model routing for each agent (SLM-first for Policy/Risk, Groq for Claim, frontier fallback for low-confidence), driven by config.
- Add confidence thresholds and novelty checks in the orchestrator to gate downstream execution and external search.
- Harden RAG with similarity cutoffs and versioned indexing; prevent runtime re-indexing.
- Expand governance logging to include policy/model versions and prompt hashes; add quality-gate metrics and rollback signals.

## Key Files to Update

- [`src/config.py`](/Users/MLI114/Projects/llm-decision-flow/src/config.py) for new settings: Zentropi/Groq/Serper keys, model IDs, thresholds, SLOs, allowlist, policy artifact version, and toggles.
- [`src/agents/claim_agent.py`](/Users/MLI114/Projects/llm-decision-flow/src/agents/claim_agent.py) to route claim extraction through Groq (high-throughput only).
- [`src/agents/risk_agent.py`](/Users/MLI114/Projects/llm-decision-flow/src/agents/risk_agent.py) to front-run with Zentropi SLM and fall back to frontier LLM if low confidence.
- [`src/agents/policy_agent.py`](/Users/MLI114/Projects/llm-decision-flow/src/agents/policy_agent.py) to use Zentropi SLM and fall back on low confidence; add conflict signal.
- [`src/agents/base.py`](/Users/MLI114/Projects/llm-decision-flow/src/agents/base.py) to capture prompt hashes, enforce token budgets/timeouts, and support model version reporting.
- [`src/orchestrator/decision_orchestrator.py`](/Users/MLI114/Projects/llm-decision-flow/src/orchestrator/decision_orchestrator.py) to enforce confidence gates, novelty routing, and human-review escalation rules.
- [`src/rag/evidence_retriever.py`](/Users/MLI114/Projects/llm-decision-flow/src/rag/evidence_retriever.py) and [`src/rag/vector_store.py`](/Users/MLI114/Projects/llm-decision-flow/src/rag/vector_store.py) for similarity cutoffs, embedding caching, and indexing locks.
- [`src/models/schemas.py`](/Users/MLI114/Projects/llm-decision-flow/src/models/schemas.py) to add fields for confidence signals, conflict flags, prompt hashes, model versions, and external/context evidence.
- [`src/governance/logger.py`](/Users/MLI114/Projects/llm-decision-flow/src/governance/logger.py) and [`src/models/database.py`](/Users/MLI114/Projects/llm-decision-flow/src/models/database.py) to persist model/policy versions, prompt hashes, and reviewer feedback labels.
- [`src/api/routes/review.py`](/Users/MLI114/Projects/llm-decision-flow/src/api/routes/review.py) to accept structured rationale + evidence snippets from reviewers.
- [`streamlit_app.py`](/Users/MLI114/Projects/llm-decision-flow/streamlit_app.py) to allow human overrides and feedback submission.
- [`src/governance/metrics.py`](/Users/MLI114/Projects/llm-decision-flow/src/governance/metrics.py) to compute disagreement/quality gates and rollback signals.
- [`streamlit_app.py`](/Users/MLI114/Projects/llm-decision-flow/streamlit_app.py) to visualize new routing paths (SLM gating, fallback, external search, cache).
- [`streamlit_app.py`](/Users/MLI114/Projects/llm-decision-flow/streamlit_app.py) to surface governance/auditability views (policy/model versions, prompt hashes, review trails).
- [`src/api/routes/metrics.py`](/Users/MLI114/Projects/llm-decision-flow/src/api/routes/metrics.py) and [`src/api/routes/review.py`](/Users/MLI114/Projects/llm-decision-flow/src/api/routes/review.py) for UI data sources.
- New modules: `src/llm/zentropi_client.py`, `src/llm/groq_client.py`, and `src/rag/external_search.py` for provider integrations and domain allowlist filtering.
- Docs updates: [`API_Usage_Explanation.md`](/Users/MLI114/Projects/llm-decision-flow/API_Usage_Explanation.md) and [`SETUP.md`](/Users/MLI114/Projects/llm-decision-flow/SETUP.md) to document keys and routing.

## Implementation Steps

1. **Provider clients + settings**

- Add Zentropi, Groq, Serper settings and default thresholds (confidence gates, novelty cutoff, similarity cutoff, SLOs, token budgets) in `src/config.py`.
- Implement `ZentropiClient` wrapper using the provided label API and parse `label`/`confidence` (fallback to frontier when missing/invalid).
- Implement `GroqClient` wrapper for claim extraction (structured JSON response, low temperature, explicit model name).

2. **Agent routing and deterministic orchestration**

- ClaimAgent → Groq-only; record model version and prompt hash in `AgentExecutionDetail`.
- RiskAgent → Zentropi coarse tiering first; if confidence < threshold, call frontier LLM; return `risk_confidence` + `route_reason`.
- PolicyAgent → Zentropi policy interpretation first; on low confidence, call frontier LLM; add `conflict_detected` boolean.
- Update `DecisionOrchestrator` to enforce confidence gates and escalate low-confidence/cross-policy-conflict cases to human review.

3. **RAG hardening + novelty routing**

- Add similarity score cutoff in `EvidenceRetriever` to skip low-signal evidence.
- Add vector-store embedding caching and batch embedding generation; block runtime `add_documents` when `ALLOW_RUNTIME_INDEXING=false`.
- Introduce novelty check (max similarity vs internal evidence) and route **High risk + High novelty** to external search.

4. **External search controls**

- Implement `ExternalSearchClient` (Serper + Wikipedia), enforce domain allowlist, and tag outputs as context-only.
- Ensure factuality enforcement uses internal evidence only; contextual sources are included for rationale but not used to assert violations.

5. **Governance + auditability**

- Externalize policy artifact versioning and log `policy_version`, `model_version`, `prompt_hash` per agent.
- Extend DB schema to store agent execution metadata and reviewer feedback labels for training refresh.
- Add quality-gate metrics (model-vs-human disagreement, agent-level disagreement proxies) and rollback flags in metrics snapshots.

6. **Docs + tests**

- Update docs for new env vars and routing behavior.
- Add tests for routing thresholds, fallback paths, and external search gating.

7. **Human-in-the-loop optimization**

- Tighten escalation rules to only route high-impact + ambiguous cases to human review.
- Store structured reviewer rationales and evidence snippets with decisions.
- Capture reviewer labels as training data for periodic refresh.

8. **UI graph updates**

- Add nodes/edges to the Streamlit graph for SLM gating, fallback path, external search, cache hits, and human-review gates.
- Show per-agent confidence and model used in the graph/tooltips where feasible.

9. **Governance + auditability UI**

- Add Streamlit views for decision logs, policy/model versions, prompt hashes, and review trails.
- Surface quality-gate metrics (disagreement, rollback flags) with filters by time window.

10. **Human-in-the-loop UI**

- Add Streamlit view to list pending reviews, show structured rationale/evidence, and submit overrides/feedback.
- Persist feedback labels for training refresh.

## Defaults to Implement (per your selection)

- Confidence thresholds: claim ≥ 0.65, risk ≥ 0.6, policy ≥ 0.7; fallback when below.
- Novelty cutoff: max similarity < 0.35 → “high novelty”.
- Evidence cutoff: relevance_score < 0.4 filtered.
- Serper allowlist: `gov`, `edu`, `who.int`, `cdc.gov`, `nih.gov`, `factcheck.org`, `reuters.com`, `apnews.com` (editable in config).
- Latency SLOs: 2.5s per SLM call, 6s per frontier call; hard timeouts enforced.

## Todos

- [ ] Add provider clients + config for Zentropi/Groq/Serper.
- [ ] Update agents + orchestrator for confidence-gated routing and fallback.
- [ ] Harden RAG, add novelty routing + external search allowlisting.
- [ ] Expand governance logging, metrics, and review feedback capture.
- [ ] Update docs + tests to reflect new behavior.
- [ ] Update Streamlit graph for new routing paths.
- [ ] Add human review routing + feedback capture.
- [ ] Add governance/auditability UI views.
- [ ] Add human-in-the-loop review UI.
