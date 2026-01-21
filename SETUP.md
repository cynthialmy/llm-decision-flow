# Setup Guide

## Prerequisites

- Python 3.11 or higher
- Azure AI Foundry account with:
  - Project endpoint
  - Agent ID (recommended) OR API key
  - Azure CLI login (`az login`)

## Installation

1. **Clone or navigate to the project directory**

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
   - Create a `.env` file in the project root
   - Add your Azure AI Foundry configuration (see [FOUNDRY_SETUP.md](FOUNDRY_SETUP.md) for details)

5. **Populate evidence database**:
```bash
python scripts/populate_evidence.py
```

## Running the Application

1. **Start the FastAPI server**:
```bash
python run_server.py
```

Or using uvicorn directly:
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

2. **Open the frontend**:
   - Open `frontend/index.html` in your web browser
   - Or serve it with a simple HTTP server:
```bash
cd frontend
python -m http.server 8080
```
   - Then navigate to `http://localhost:8080`

## API Endpoints

- `POST /api/analyze` - Analyze content transcript
- `GET /api/reviews` - List pending reviews
- `GET /api/reviews/{id}` - Get review details
- `POST /api/reviews/{id}/decide` - Submit human decision
- `GET /api/metrics` - Get trust metrics

## Testing

Run tests with pytest:
```bash
pytest
```

Test Foundry agent setup:
```bash
python scripts/test_foundry_agent.py
```

## Project Structure

```
llm-decision-flow/
├── src/
│   ├── agents/          # Agent implementations
│   ├── orchestrator/     # Decision orchestrator
│   ├── rag/             # RAG system
│   ├── models/          # Data models
│   ├── api/             # FastAPI application
│   └── governance/      # Governance and metrics
├── frontend/            # Web UI
├── policies/            # Policy files
├── data/                # Data and evidence
├── scripts/             # Utility scripts
└── tests/               # Tests
```

## Demo Flow

1. Submit a transcript via the web UI (`frontend/index.html`)
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

## Troubleshooting

- **Foundry connection errors**: See [FOUNDRY_SETUP.md](FOUNDRY_SETUP.md)
- **Database errors**: Ensure SQLite database directory exists and is writable
- **ChromaDB errors**: Ensure ChromaDB directory exists and is writable
- **Import errors**: Ensure you're in the project root and virtual environment is activated
