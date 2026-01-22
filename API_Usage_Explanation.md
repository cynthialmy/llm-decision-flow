# API Usage in This System

This document explains how the APIs are actually used in this codebase. For system architecture and decision flow, see [README.md](README.md). For setup instructions, see [SETUP.md](SETUP.md).

---

## Overview

This system uses:
1. **Groq** (LLM) - Claim extraction and evidence classification
2. **Zentropi** (SLM) - Risk and policy classification (with fallback)
3. **Azure OpenAI/Foundry** (LLM) - Factuality assessment and fallback routes
4. **Serper** (Search API) - External evidence retrieval
5. **Wikipedia API** - Structured knowledge source

---

## 1. Groq API (Claim Extraction)

### Role in This System

Groq is used for **claim extraction** - the first step in the pipeline. It's chosen for speed and throughput, not deep reasoning.

### Implementation

```python
from src.llm.groq_client import GroqClient

groq = GroqClient()
response = groq.chat(
    prompt=user_prompt,
    system_prompt=system_prompt,
    temperature=0.2,
    max_tokens=900
)
```

The `GroqClient` wraps the OpenAI-compatible Groq API using the OpenAI SDK. See `src/llm/groq_client.py` for implementation.

### Key Advantages

- **Speed**: Ultra-low latency for high-volume claim extraction
- **Cost**: Affordable for front-of-pipeline operations
- **Model**: Uses `llama-3.3-70b-versatile` (configurable via `GROQ_MODEL`)

---

## 2. Zentropi API (Risk & Policy Classification)

### Role in This System

Zentropi SLM is used for **fast classification** with confidence-gated fallback:
1. **Risk Agent**: Primary route for risk tiering (Low/Medium/High)
2. **Policy Agent**: Primary route for policy violation classification

If confidence < threshold, falls back to Azure OpenAI/Foundry.

### Implementation

```python
from src.llm.zentropi_client import ZentropiClient

zentropi = ZentropiClient()
result = zentropi.label(
    content_text=content,
    criteria_text="Label the content risk tier as one of: Low, Medium, High."
)
# Returns ZentropiResult with label, confidence, and raw response
```

The `ZentropiClient` uses `httpx` for direct API calls. See `src/llm/zentropi_client.py` for implementation.

### Key Advantages

- **Speed**: Low latency (2.5s timeout) for high-volume classification
- **Cost**: Lower cost per decision than full LLM prompts
- **Fallback**: Automatically routes to frontier LLM if confidence < threshold

---

## 3. Serper API (External Evidence Search)

### Role in This System

Serper is used for **external evidence retrieval** when internal RAG is insufficient. Only triggered for high-novelty claims (similarity < 0.35) or when no internal evidence is found.

### Implementation

```python
from src.rag.external_search import ExternalSearchClient

search_client = ExternalSearchClient()
results = search_client.serper_search(query="claim text", max_results=5)
```

The `ExternalSearchClient` uses `httpx` for direct API calls and filters results by domain allowlist. See `src/rag/external_search.py` for implementation.

### Key Features

- **Allowlist Filtering**: Only returns results from allowed domains (gov, edu, who.int, etc.)
- **Source Type Inference**: Automatically classifies sources (AUTHORITATIVE, HIGH_CREDIBILITY, FACT_CHECK, etc.)
- **Conditional Execution**: Only runs when `ALLOW_EXTERNAL_SEARCH=true` and novelty threshold is met

---

## 4. Wikipedia API (Structured Knowledge)

### Role in This System

Wikipedia is used alongside Serper for external evidence retrieval. Provides structured, authoritative information for fact-checking.

### Implementation

```python
from src.rag.external_search import ExternalSearchClient

search_client = ExternalSearchClient()
results = search_client.wikipedia_search(query="claim text", max_results=3)
```

Uses direct Wikipedia API calls via `httpx`. Results are automatically classified by source type. See `src/rag/external_search.py` for implementation.

---

## 5. Azure OpenAI / Foundry (Factuality & Fallback)

### Role in This System

Azure OpenAI/Foundry is used for:
1. **Factuality Agent**: Evidence-based factuality assessment (highest-risk reasoning)
2. **Fallback Routes**: When Zentropi SLM confidence is below threshold for Risk or Policy

### Implementation

The system uses either:
- **Foundry Agents** (recommended): Via `client.responses.create()` with agent reference
- **Direct Deployments**: Standard Azure OpenAI API calls

See `src/agents/base.py` for the implementation that handles both approaches.

### Key Features

- **Structured Output**: Uses JSON schema for reliable parsing
- **Timeout Control**: 6.0s timeout per call
- **Token Budget**: 2000 tokens max for frontier LLM calls
- **Automatic Fallback**: Routes from SLM when confidence is low

---

## How APIs Work Together

The system uses a **confidence-gated routing** approach:

1. **Claim Extraction** (Groq) → Always runs, extracts claims
2. **Risk Assessment** (Zentropi → Azure if low confidence) → Always runs
3. **Evidence Retrieval** (RAG → External if needed) → Only for Medium/High risk
4. **Factuality Assessment** (Azure) → Only for Medium/High risk
5. **Policy Interpretation** (Zentropi → Azure if low confidence) → Always runs

For detailed decision flow and routing logic, see the [README.md](README.md) "Tool Selection & Rationale" section.

---

## Configuration

All API keys and settings are configured via environment variables. See [SETUP.md](SETUP.md) for complete configuration instructions.

Key environment variables:
- `GROQ_API_KEY` - Required for claim extraction
- `ZENTROPI_API_KEY`, `ZENTROPI_LABELER_ID`, `ZENTROPI_LABELER_VERSION_ID` - Optional for SLM routing
- `SERPER_API_KEY` - Optional for external search
- `AZURE_EXISTING_AIPROJECT_ENDPOINT`, `AZURE_EXISTING_AGENT_ID` - Required for Foundry
- `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT_NAME` - Required for direct deployments
