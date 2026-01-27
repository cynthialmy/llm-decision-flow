# Setup & Testing Guide

Complete guide for setting up, configuring, and testing the LLM-Powered Misinformation Decision System.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Testing](#testing)
5. [Running the Application](#running-the-application)
6. [Troubleshooting](#troubleshooting)
7. [Project Structure](#project-structure)

---

## Prerequisites

- **Python 3.11 or higher**
- **Azure AI Foundry account** with:
  - Project endpoint (format: `https://your-resource.services.ai.azure.com/api/projects/your-project`)
  - Agent ID (recommended) OR API key for direct model access
- **Azure CLI** installed and logged in (`az login`)

---

## Installation

### 1. Clone or Navigate to Project Directory

```bash
cd llm-decision-flow
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

**For Local Development:**
```bash
pip install -r requirements.txt
```

**For Cloud Deployment:**
```bash
pip install -r requirements-cloud.txt
```

**Optional: Install Foundry SDK (if using Foundry agents):**
```bash
# After installing base requirements, optionally install Foundry SDK
pip install --pre azure-ai-projects>=2.0.0b1 azure-identity>=1.15.0
# Or use the helper script:
./install_foundry.sh
```

This installs all required packages including:
- FastAPI and Uvicorn (web server)
- OpenAI SDK (for Azure OpenAI)
- ChromaDB (vector database)
- SQLAlchemy (database ORM)
- Pydantic (data validation)
- Azure AI Foundry SDK (`azure-ai-projects`, `azure-identity`) - **Optional**, only needed for Foundry agents

**Note:** The Foundry SDK packages are optional. The application works without them using API keys instead. Cloud platforms may not install pre-release packages by default, so use `requirements-cloud.txt` for cloud deployments.

**Python 3.13 Compatibility:** If you're using Python 3.13, make sure you're using pydantic 2.8.0 or newer (already included in requirements files).

### 4. Create Local Data Folders

The app expects local directories that are not tracked in git:

```bash
mkdir -p policies data/evidence
```

If you do not provide `policies/misinformation_policy.txt`, the app uses a small built-in fallback policy.

---

## Configuration

The application supports two configuration approaches:

### Option 1: Foundry Agent (Recommended)

Use a Foundry agent for all LLM operations. This provides better abstraction and integration.

**Environment Variables** (create `.env` file in project root):

```env
# Foundry Project Endpoint (required)
AZURE_EXISTING_AIPROJECT_ENDPOINT=https://your-resource.services.ai.azure.com/api/projects/your-project

# Foundry Agent ID (required)
AZURE_EXISTING_AGENT_ID=your-agent-name:1

# API Version (optional, default: 2024-02-15-preview)
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Embedding Deployment (required for RAG/evidence retrieval)
# This should be the name of your embedding model deployment (e.g., text-embedding-ada-002)
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# Optional: explicit embedding endpoint override
# Useful when the derived endpoint doesn't host the embedding deployment
AZURE_OPENAI_EMBEDDING_ENDPOINT=https://your-resource.openai.azure.com/

# API Key (required for embeddings - Foundry agents use Azure credentials, but embeddings need API key)
AZURE_OPENAI_API_KEY=your-api-key-here

# Optional Foundry variables (not required for basic setup)
AZURE_ENV_NAME=your-env-name
AZURE_LOCATION=your-location
AZURE_SUBSCRIPTION_ID=your-subscription-id

# SLM + Search Providers (optional but recommended for optimized routing)
GROQ_API_KEY=your-groq-key
ZENTROPI_API_KEY=your-zentropi-key
ZENTROPI_LABELER_ID=your-labeler-id
ZENTROPI_LABELER_VERSION_ID=your-labeler-version-id
SERPER_API_KEY=your-serper-key

# Optional policy + routing controls
POLICY_VERSION=1.0
CLAIM_CONFIDENCE_THRESHOLD=0.65
RISK_CONFIDENCE_THRESHOLD=0.6
POLICY_CONFIDENCE_THRESHOLD=0.7
NOVELTY_SIMILARITY_THRESHOLD=0.35
EVIDENCE_SIMILARITY_CUTOFF=0.4

# Optional runtime controls
ALLOW_EXTERNAL_SEARCH=true
ALLOW_RUNTIME_INDEXING=false
EVIDENCE_INDEX_VERSION=v1

# External search allowlist (comma-separated domains)
EXTERNAL_SEARCH_ALLOWLIST=gov,edu,who.int,cdc.gov,nih.gov,factcheck.org,reuters.com,apnews.com

# Optional external enrichment (write external context into internal index)
ALLOW_EXTERNAL_ENRICHMENT=false
```

**How It Works**:
1. Application detects `AZURE_EXISTING_AIPROJECT_ENDPOINT` and `AZURE_EXISTING_AGENT_ID`
2. Creates `AIProjectClient` with Azure credentials (`DefaultAzureCredential`)
3. Retrieves your agent from Foundry
4. Gets OpenAI client via `project_client.get_openai_client()` (includes `responses` extension)
5. Uses `client.responses.create()` with agent reference for all LLM calls

**Benefits**:
- Simplified configuration (no deployment names needed)
- Agent handles model selection
- Unified interface for all operations
- Better Foundry integration

### Option 2: Direct Model Deployments (Fallback)

If you prefer to use direct model deployments instead of agents:

**Environment Variables**:

```env
# Base endpoint (without /api/projects/)
AZURE_OPENAI_ENDPOINT=https://your-resource.services.ai.azure.com/

# API Key
AZURE_OPENAI_API_KEY=your-api-key-here

# Deployment name
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o

# API Version
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Embedding deployment (required for RAG)
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# Optional embedding endpoint override
AZURE_OPENAI_EMBEDDING_ENDPOINT=https://your-resource.openai.azure.com/

# Optional SLM + Search Providers
GROQ_API_KEY=your-groq-key
ZENTROPI_API_KEY=your-zentropi-key
ZENTROPI_LABELER_ID=your-labeler-id
ZENTROPI_LABELER_VERSION_ID=your-labeler-version-id
SERPER_API_KEY=your-serper-key

# Optional policy + routing controls
POLICY_VERSION=1.0
CLAIM_CONFIDENCE_THRESHOLD=0.65
RISK_CONFIDENCE_THRESHOLD=0.6
POLICY_CONFIDENCE_THRESHOLD=0.7
NOVELTY_SIMILARITY_THRESHOLD=0.35
EVIDENCE_SIMILARITY_CUTOFF=0.4

# Optional runtime controls
ALLOW_EXTERNAL_SEARCH=true
ALLOW_RUNTIME_INDEXING=false
EVIDENCE_INDEX_VERSION=v1
EXTERNAL_SEARCH_ALLOWLIST=gov,edu,who.int,cdc.gov,nih.gov,factcheck.org,reuters.com,apnews.com
ALLOW_EXTERNAL_ENRICHMENT=false
```

**Note**: When using API keys, use the base endpoint format (without `/api/projects/`).

### Azure Authentication Setup

For Foundry Agent approach, you need Azure CLI authentication:

```bash
# Login to Azure
az login

# Set your subscription
az account set --subscription <your-subscription-id>

# Verify authentication
az account show
az account get-access-token --resource https://ai.azure.com
```

---

## Testing

### Quick Verification Checklist

#### 1. Test Foundry Agent Setup (Recommended First Step)

```bash
python scripts/test_foundry_agent.py
```

**Expected Output**:
- ✅ Foundry project client created successfully
- ✅ Successfully retrieved agent
- ✅ Client has 'responses' attribute
- ✅ Agent response received
- ✅ ClaimAgent integration working (end-to-end workflow)

This test verifies:
- Foundry project client connection
- Agent retrieval
- Agent API calls using `responses.create()`
- Integration with ClaimAgent
- Input format correctness (with `type: "message"` fields)

#### 2. Test Embedding Setup (Required for RAG)

```bash
python scripts/test_embedding_setup.py
```

This test verifies your embedding endpoint and deployment are reachable and can generate vectors.

#### 3. Test Direct Model Setup (If Using API Key)

```bash
python scripts/test_api_key_setup.py
```

#### 4. Populate Evidence Database

```bash
python scripts/populate_evidence.py
```

This populates the ChromaDB vector store with evidence documents for RAG retrieval.

#### 5. Run Unit Tests

```bash
pytest
```

#### 6. Test API Endpoints

Start the server first (see [Running the Application](#running-the-application)), then test:

**Health Check**:
```bash
curl http://localhost:8000/health
```

**Analyze Content**:
```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"transcript": "COVID vaccines contain microchips and will alter your DNA."}'
```

**Get Metrics**:
```bash
curl http://localhost:8000/api/metrics
```

**List Reviews**:
```bash
curl http://localhost:8000/api/reviews
```

#### 7. Test Streamlit UI

```bash
streamlit run streamlit_app.py
```

Then navigate to `http://localhost:8501` in your browser to access the interactive UI.

---

## Running the Application

### Start the FastAPI Server

```bash
python run_server.py
```

Or using uvicorn directly:

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Server will start on `http://localhost:8000`

### Start the Streamlit UI

**Important**: Make sure the backend server is running first (see above).

Open a new terminal window and run:

```bash
streamlit run streamlit_app.py
```

The Streamlit UI will open automatically in your browser at `http://localhost:8501`. This provides:
- **Content Analysis Interface** - Submit transcripts and view analysis results
- **Decision Flow Visualization** - Interactive graph showing the execution path
- **Agent Execution Details** - Inspect prompts and responses for each agent
- **Review Queue** - View and handle items requiring human review
- **Metrics Dashboard** - Monitor system trust metrics

### API Endpoints

- `POST /api/analyze` - Analyze content transcript
- `GET /api/reviews` - List pending reviews
- `GET /api/reviews/{id}` - Get review details
- `POST /api/reviews/{id}/decide` - Submit human decision
- `GET /api/metrics` - Get trust metrics
- `GET /health` - Health check

### Demo Flow

1. Submit a transcript via the Streamlit UI
2. System processes through agent pipeline:
   - Claim extraction
   - Risk assessment
   - Evidence retrieval (for medium/high risk)
   - Factuality assessment
   - Policy interpretation
   - Decision making
3. If escalation needed, review appears in review queue
4. Human reviewer makes decision via review interface
5. Metrics dashboard shows system health

---

## Cloud Deployment

### Using requirements-cloud.txt

For cloud platforms (Railway, Render, Fly.io, Heroku, etc.), use `requirements-cloud.txt` instead of `requirements.txt`:

```bash
pip install -r requirements-cloud.txt
```

**Why?** Cloud platforms typically don't install pre-release packages by default. The `requirements-cloud.txt` file excludes the pre-release `azure-ai-projects` package, which is optional anyway.

### Foundry SDK on Cloud

If you need Foundry agent support on cloud:

1. **Option 1: Install after deployment**
   - Deploy with `requirements-cloud.txt`
   - After deployment, SSH into your instance and run:
     ```bash
     pip install --pre azure-ai-projects>=2.0.0b1 azure-identity>=1.15.0
     ```

2. **Option 2: Use API keys instead**
   - The application works perfectly fine without Foundry SDK
   - Just configure `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` in your environment variables
   - This is simpler and doesn't require Azure CLI authentication

### Environment Variables for Cloud

Make sure to set these in your cloud platform's environment variables:

**Required (if not using Foundry):**
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_DEPLOYMENT_NAME`
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`

**Optional:**
- `GROQ_API_KEY` (for faster claim extraction)
- `ZENTROPI_API_KEY` (for SLM routing)
- `SERPER_API_KEY` (for external search)

### Streamlit Community Cloud

**Why it works locally but not on Cloud:** Locally the app reads from your **`.env`** file. On Streamlit Cloud, **`.env` is not deployed** (it’s in `.gitignore`), so the app never sees it. Cloud config comes only from **Settings → Secrets**. If your secrets are “the same” as in `.env`, you must paste them into the app’s **Settings → Secrets** as TOML. The app copies those into the environment at startup.

When deploying to **Streamlit Community Cloud**, add your secrets in the app’s **Settings → Secrets** (or paste TOML in “Advanced settings” during deploy). The app injects these into the environment before loading config.

**Required secrets (when using Azure OpenAI with API key):**

Use the **exact deployment name** from Azure Portal → your resource → **Deployments**. Use the **base endpoint** with no extra path, e.g.:

- `https://YOUR-RESOURCE.openai.azure.com/` or  
- `https://YOUR-RESOURCE.services.ai.azure.com/`

Do not use a Foundry project URL (`.../api/projects/...`) unless you use Foundry agents.

**Option A – flat TOML (recommended):**

```toml
AZURE_OPENAI_ENDPOINT = "https://YOUR-RESOURCE.services.ai.azure.com/"
AZURE_OPENAI_API_KEY = "your-api-key"
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = "text-embedding-ada-002"
AZURE_OPENAI_API_VERSION = "2024-11-20"
```

Use your real endpoint (e.g. `https://support-8844-resource.services.ai.azure.com/`), deployment name, and API version from Azure.

**Option B – nested TOML:**

```toml
[azure]
openai_endpoint = "https://YOUR-RESOURCE.services.ai.azure.com/"
openai_api_key = "your-api-key"
openai_deployment_name = "gpt-4o"
openai_embedding_deployment = "text-embedding-ada-002"
openai_api_version = "2024-11-20"
```

Nested keys are mapped to `AZURE_OPENAI_*` env vars (e.g. `azure.openai_api_key` → `AZURE_OPENAI_API_KEY`).

**Common causes of "deployment not found" on Streamlit Cloud:**

1. **Secrets not set on Cloud** – `.env` is ignored on Cloud. Paste the same keys/values into **Settings → Secrets** (as TOML).
2. **Wrong deployment name** – It must match exactly what you see in Azure Portal → your resource → **Deployments** (e.g. `gpt-4o`, `gpt-40`, `gpt-4`—no spaces, correct spelling).
3. **Wrong endpoint** – Use the **base** URL only: `https://YOUR-RESOURCE.openai.azure.com/` (or `...services.ai.azure.com/`). **Do not** use `.../openai/v1/` — the SDK adds the path and that URL causes 404. The app will strip `/openai/v1/` if you paste it by mistake.

**Optional secrets:** `GROQ_API_KEY`, `ZENTROPI_API_KEY`, `ZENTROPI_LABELER_ID`, `ZENTROPI_LABELER_VERSION_ID`, `SERPER_API_KEY`.

**If you see `KeyError: 'src'` or `KeyError: 'src.models.schemas'` on Cloud:** The app now adds the project root to `sys.path` at the very start of `streamlit_app.py`, so `import src...` should resolve. If it still fails, set **Main file path** in the app’s Cloud settings to `streamlit_app.py` (not a path like `some_folder/streamlit_app.py`) so the working directory is the repo root where `src/` lives.

**Build / runtime:** Use `requirements-cloud.txt` and Python 3.11 or 3.12 if your platform allows (3.13 is supported with the versions in `requirements-cloud.txt`).

---

## Troubleshooting

### Authentication & Credentials

**"Not logged in" or "Credential error"**:
- Run `az login` to authenticate with Azure
- Verify: `az account show`
- Set correct subscription: `az account set --subscription <your-subscription-id>`

**"401 Unauthorized"**:
- Verify you're logged in: `az login`
- Check you have access to the Foundry project
- Verify endpoint URL is correct
- Refresh credentials: `az account get-access-token --resource https://ai.azure.com`
- Consider using API key approach as fallback

**"Embedding 404" or "Deployment not found"**:
- Run `python scripts/test_embedding_setup.py`
- Confirm `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` exists
- If using Foundry, set `AZURE_OPENAI_EMBEDDING_ENDPOINT` explicitly
- Embeddings always require `AZURE_OPENAI_API_KEY`

**"Deployment not found" or `openai.NotFoundError` on Streamlit Cloud**:
- The app will show a clear message: check **AZURE_OPENAI_DEPLOYMENT_NAME** and **AZURE_OPENAI_ENDPOINT** in **Settings → Secrets**.
- **Endpoint**: When using API key, use the **base** URL (e.g. `https://YOUR-RESOURCE.openai.azure.com/`). Do not use a Foundry project URL (`.../api/projects/...`) unless you use Foundry agents.
- **Deployment name**: Must exactly match the deployment name in Azure Portal (e.g. `gpt-4o`, `gpt-4`). Copy it from Azure Portal → your resource → Deployments.
- See [Streamlit Community Cloud](#streamlit-community-cloud) above for the exact TOML to paste in Secrets.

### SDK & Dependencies

**"Module not found: azure.ai.projects"**:
- This is expected if you're not using Foundry agents. The application works without it using API keys.
- If you need Foundry support: `pip install --pre azure-ai-projects>=2.0.0b1 azure-identity>=1.15.0`
- Or use the helper script: `./install_foundry.sh`
- For cloud deployments, use `requirements-cloud.txt` which excludes pre-release packages

**"Module not found" errors (general)**:
- Make sure virtual environment is activated
- Run: `pip install -r requirements.txt`
- Ensure you're in the project root directory

**"Client.__init__() got an unexpected keyword argument 'proxies'"**:
- This is a compatibility issue with httpx
- Solution: `pip install 'httpx<0.27.0'`
- Or the code will automatically patch httpx to handle this

### Foundry Agent Issues

**"Agent not found" or "404 Resource not found"**:
- Verify agent name in Foundry web interface
- Check `AZURE_EXISTING_AGENT_ID` matches exactly (e.g., `your-agent-name:1`)
- Agent name format: `agent-name` or `agent-name:version`

**"OpenAI client doesn't have 'responses' attribute"**:
- This indicates an SDK version issue
- Solution: Update to latest SDK version: `pip install --pre --upgrade azure-ai-projects`
- The code will try both `inference.get_azure_openai_client()` and `get_openai_client()` methods
- Verify SDK version: `pip show azure-ai-projects` (should be >= 2.0.0b3)
- Check Azure documentation for latest SDK version requirements

**"Invalid value: ''" or "Invalid input format"**:
- Foundry `responses.create()` requires input items to have a `type` field
- Format: `[{"type": "message", "role": "user", "content": "..."}]`
- System messages: `[{"type": "message", "role": "system", "content": "..."}]`
- This is automatically handled by the application - if you see this error, update your code

### Database & Storage

**"Database errors"**:
- Ensure SQLite database directory exists and is writable
- Check file permissions

**"ChromaDB errors"**:
- Ensure ChromaDB directory exists and is writable
- Try deleting the ChromaDB directory and re-running `populate_evidence.py`

**"Evidence database not found"**:
- Run: `python scripts/populate_evidence.py`

### General Issues

**"Import errors"**:
- Ensure you're in the project root and virtual environment is activated
- Check Python version: `python --version` (should be 3.11+)
- For Python 3.13, ensure pydantic>=2.8.0 is installed (already in requirements files)

**"Failed building wheel for pydantic-core" or "ForwardRef._evaluate() missing 1 required keyword-only argument"**:
- This indicates Python 3.13 compatibility issue with older pydantic versions
- Solution: Upgrade to pydantic>=2.8.0 (already updated in requirements files)
- If using Python 3.13, make sure you're using the latest requirements files

**"AssertionError" when importing SQLAlchemy or "Class directly inherits TypingOnly but has additional attributes"**:
- This indicates Python 3.13 compatibility issue with SQLAlchemy 2.0.23 or earlier
- Solution: Upgrade to sqlalchemy>=2.0.30 (already updated in requirements files)
- SQLAlchemy 2.0.30+ fixes Generic typing issues with Python 3.13

**"Port already in use"**:
- Change port in `run_server.py` or use: `uvicorn src.api.main:app --port 8001`

---

## Project Structure

```
llm-decision-flow/
├── src/
│   ├── agents/          # Agent implementations (Claim, Risk, Factuality, Policy)
│   ├── orchestrator/     # Decision orchestrator
│   ├── rag/             # RAG system (vector store, evidence retrieval)
│   ├── models/          # Data models (schemas, database)
│   ├── api/             # FastAPI application
│   │   ├── routes/      # API route handlers
│   │   └── main.py      # FastAPI app initialization
│   └── governance/      # Governance and metrics
├── streamlit_app.py    # Streamlit UI application
├── policies/            # Policy files (local, not committed)
├── data/                # Data and evidence (local, not committed)
│   └── evidence/       # Evidence documents for RAG
├── scripts/             # Utility scripts
│   ├── test_foundry_agent.py    # Test Foundry agent setup
│   ├── test_api_key_setup.py    # Test API key setup
│   ├── populate_evidence.py     # Populate evidence database
│   └── discover_foundry_settings.py  # Discover Foundry settings
├── tests/               # Unit tests
├── requirements.txt     # Python dependencies
├── run_server.py       # Server startup script
└── SETUP.md            # This file
```

---

## Migration from Direct Models to Agents

If you're currently using direct model deployments and want to switch to agents:

1. Create an agent in Foundry (or use existing one)
2. Update `.env`:
   - Add `AZURE_EXISTING_AIPROJECT_ENDPOINT`
   - Add `AZURE_EXISTING_AGENT_ID`
   - Remove `AZURE_OPENAI_API_KEY` (not needed with agents)
   - Remove `AZURE_OPENAI_DEPLOYMENT_NAME` (agent handles this)
3. Test: `python scripts/test_foundry_agent.py`

The application will automatically detect and use your agent!

---

## Next Steps

After setup and testing:

1. Review the [README.md](README.md) for system architecture and design principles
2. Explore the API endpoints using the Streamlit UI or curl commands
3. Check the metrics dashboard in Streamlit to monitor system health
4. Review pending items in the review queue via Streamlit

For more details on the system design, see [README.md](README.md).
