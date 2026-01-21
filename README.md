# LLM‑Powered Misinformation Decision System (Policy‑Aware)

> *This demo shows how a policy‑aware, agent‑orchestrated system can make responsible moderation decisions under uncertainty by combining model reasoning, policy logic, and human judgment—optimized for platform trust rather than automation.*

**Purpose of this document**
This doc describes a *demo‑grade* implementation of a Policy‑Aware, Agent‑Orchestrated misinformation decision system.

The goal is **not** to build a production system, but to demonstrate *Senior PM‑level system thinking* across:

* Agentic workflows
* Policy‑driven moderation
* Human‑in‑the‑loop governance
* Metrics‑aligned decisioning

---

## 1. What This Demo Is (and Is Not)

### ✅ This demo *is*

* A **decision system demo**, not a model demo
* Policy‑first and uncertainty‑aware
* Explicitly human‑gated
* Designed to show *how TikTok‑scale Safety systems think*

### ❌ This demo is *not*

* A fully autonomous moderation agent
* A truth‑oracle
* A multi‑agent debate experiment
* A training or self‑learning system

> **Design principle:** *Agents propose, policies constrain, humans decide.*

---

## 2. Demo User Flow (End‑to‑End)

1. User inputs content transcript (text)
2. System extracts factual claims
3. Risk is assessed before deep analysis
4. Evidence is retrieved (RAG)
5. Factuality is assessed (non‑enforcement)
6. Policy is interpreted
7. Decision is recommended
8. Human review is triggered when required
9. Rationale is logged for governance

---

## 3. High‑Level Architecture

```
[Transcript]
    ↓
[Claim Agent]
    ↓
[Risk Agent]
    ↓ (if high/medium risk)
[Evidence Agent – RAG]
    ↓
[Factuality Agent]
    ↓
[Policy Interpretation Agent]
    ↓
[Decision Orchestrator]
    ↓
[Human Review (Conditional)]
    ↓
[Governance & Metrics Log]
```

---

## 4. Agentic Workflow Design (Bounded Agents)

> **Important:** These are *specialized agents with limited authority*, not autonomous decision‑makers.

### 4.1 Claim Agent

**Responsibility**

* Extract explicit and implicit factual claims
* Tag domain (health / civic / finance)

**Constraints**

* Must be conservative
* Cannot infer intent or judge truth

---

### 4.2 Risk Agent

**Responsibility**

* Assign preliminary risk score based on:

  * Potential harm
  * Estimated exposure
  * Vulnerable populations

**Output**

* Risk tier: Low / Medium / High

**Constraints**

* Does not see evidence
* Does not apply policy

---

### 4.3 Evidence Agent (RAG)

**Responsibility**

* Retrieve supporting and contradicting evidence
* Attach timestamps and source quality

**Output**

* Supporting evidence
* Contradicting evidence
* Evidence confidence

**Constraints**

* No synthesis into “truth”
* Conflicting evidence is preserved

---

### 4.4 Factuality Agent

**Responsibility**

* Assess factual status of each claim

**Output categories**

* Likely True
* Likely False
* Uncertain / Disputed

**Important:**

> Factuality ≠ Policy Violation

---

### 4.5 Policy Interpretation Agent (Critical Layer)

**Inputs**

* Claim + factuality
* Risk tier
* Platform misinformation policy text

**Outputs**

* Policy violation: Yes / No / Contextual
* Violation type
* Allowed contexts (e.g. satire, personal experience)
* Policy confidence

**Constraints**

* Policy text is treated as *input*, not hard‑coded logic
* No enforcement authority

---

### 4.6 Decision Orchestrator

**Responsibility**

* Combine risk + policy confidence
* Decide automation vs human escalation

| Risk   | Policy Confidence | Action             |
| ------ | ----------------- | ------------------ |
| Low    | High              | Allow              |
| Medium | Medium            | Label / Downrank   |
| High   | Low               | Escalate to Human  |
| High   | High              | Human Confirmation |

> Escalation is treated as a **feature**, not a failure.

---

## 5. Human‑in‑the‑Loop Design

### When Human Review Is Triggered

* High risk + low policy confidence
* Conflicting evidence
* Model disagreement
* Sensitive domains (elections, health)

### What Humans Do

* Review system rationale
* Apply judgment
* Enter decision rationale

### What Humans Do *Not* Do

* Label training data automatically
* Retrain models directly

---

## 6. Governance & Failure Handling

### Common Failure Modes

* Evidence conflict
* Policy ambiguity
* Reviewer disagreement
* Fact drift over time

### Governance Mechanisms

* Decision versioning
* Rationale logging
* Policy version tracking
* Re‑evaluation of historical content

**Design principle:**

> Optimize for *recoverability*, not assumed correctness.

---

## 7. Metrics & Dashboard (Demo Scope)

### Core Trust Metrics

* High‑risk misinformation exposure rate
* Over‑enforcement proxy (appeal reversal)
* Model vs human disagreement
* Human review load concentration
* Time‑to‑decision for high‑risk content

### What This Demo Shows

* Metrics as *decision health*, not model accuracy

---

## 8. MVP Scope (Intentional Constraints)

**Included**

* Text‑only content
* English language
* Health + civic domains
* One policy version
* Azure AI Foundry integration (agents or direct models)

**Explicitly Excluded**

* Multilingual support
* Video / image understanding
* Real‑time enforcement
* Model self‑learning

## 9. Setup & Configuration

See [SETUP.md](SETUP.md) for installation instructions and [FOUNDRY_SETUP.md](FOUNDRY_SETUP.md) for Azure AI Foundry configuration.

**Quick Start**:
1. Install dependencies: `pip install -r requirements.txt`
2. Configure `.env` file (see FOUNDRY_SETUP.md)
3. Populate evidence: `python scripts/populate_evidence.py`
4. Run server: `python run_server.py`
5. Open `frontend/index.html` in browser

---

## 10. How to Demo This (10 Minutes)

1. Problem framing (1 min)
2. System overview (2 min)
3. Agentic workflow with guardrails (3 min)
4. Policy vs factuality distinction (2 min)
5. Human escalation & metrics (2 min)
