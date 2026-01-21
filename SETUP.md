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

```bash
pip install -r requirements.txt
```

This installs all required packages including:
- FastAPI and Uvicorn (web server)
- Azure AI Foundry SDK (`azure-ai-projects`, `azure-identity`)
- OpenAI SDK (for Azure OpenAI)
- ChromaDB (vector database)
- SQLAlchemy (database ORM)
- Pydantic (data validation)

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

# Embedding Deployment (required for RAG/evidence retrieval)
# This should be the name of your embedding model deployment (e.g., text-embedding-ada-002)
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# API Key (required for embeddings - Foundry agents use Azure credentials, but embeddings need API key)
AZURE_OPENAI_API_KEY=your-api-key-here

# Optional Foundry variables (not required for basic setup)
AZURE_ENV_NAME=your-env-name
AZURE_LOCATION=your-location
AZURE_SUBSCRIPTION_ID=your-subscription-id
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

#### 2. Test Direct Model Setup (If Using API Key)

```bash
python scripts/test_api_key_setup.py
```

#### 3. Populate Evidence Database

```bash
python scripts/populate_evidence.py
```

This populates the ChromaDB vector store with evidence documents for RAG retrieval.

#### 4. Run Unit Tests

```bash
pytest
```

#### 5. Test API Endpoints

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

#### 6. Test Streamlit UI

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

### SDK & Dependencies

**"Module not found: azure.ai.projects"**:
- Install Foundry SDK: `pip install --pre azure-ai-projects>=2.0.0b1`
- Install Azure Identity: `pip install azure-identity`
- Or reinstall all dependencies: `pip install -r requirements.txt`

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
├── policies/            # Policy files
├── data/                # Data and evidence
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
