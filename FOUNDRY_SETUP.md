# Azure AI Foundry Setup Guide

## Overview

This application supports Azure AI Foundry using **Foundry Agents** (recommended) or direct model deployments. Foundry agents provide a cleaner abstraction and better integration with Foundry's capabilities.

## Prerequisites

1. **Install Foundry SDK**:
```bash
pip install --pre azure-ai-projects>=2.0.0b1
pip install azure-identity
```

2. **Login to Azure**:
```bash
az login
az account set --subscription <your-subscription-id>
```

## Configuration Options

### Option 1: Foundry Agent (Recommended)

Use a Foundry agent for all LLM operations. This is the recommended approach.

**Environment Variables**:
```env
# Foundry Project Endpoint (required)
AZURE_EXISTING_AIPROJECT_ENDPOINT=https://your-resource.services.ai.azure.com/api/projects/your-project

# Foundry Agent ID (required for agent approach)
AZURE_EXISTING_AGENT_ID=your-agent-name:1

# Optional Foundry variables
AZURE_ENV_NAME=your-env-name
AZURE_LOCATION=your-location
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_EXISTING_AIPROJECT_RESOURCE_ID=/subscriptions/.../projects/your-project
AZURE_EXISTING_RESOURCE_ID=/subscriptions/.../accounts/your-resource
```

**How It Works**:
- The application automatically detects Foundry agent configuration
- All agents (Claim, Risk, Factuality, Policy) use your Foundry agent
- Agent handles model selection and configuration
- No need to specify deployment names

**Test Agent Setup**:
```bash
python scripts/test_foundry_agent.py
```

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

## How It Works

### Foundry Agent Approach

1. **Detection**: Application detects `AZURE_EXISTING_AIPROJECT_ENDPOINT` and `AZURE_EXISTING_AGENT_ID`
2. **Initialization**: Creates `AIProjectClient` with Azure credentials
3. **Agent Retrieval**: Gets your agent from Foundry
4. **API Calls**: Uses `client.responses.create()` with agent reference instead of direct model calls

### Direct Model Approach

1. **Detection**: Application detects standard Azure OpenAI endpoint format
2. **Initialization**: Creates `AzureOpenAI` client with API key
3. **API Calls**: Uses `client.chat.completions.create()` with deployment name

## Testing

### Test Foundry Agent
```bash
python scripts/test_foundry_agent.py
```

This will:
- Verify Foundry project client connection
- Retrieve your agent
- Test agent API calls
- Test integration with ClaimAgent

### Test Direct Model (if using API key)
```bash
python scripts/test_api_key_setup.py
```

## Troubleshooting

### "Not logged in" or "Credential error"
- Run `az login` to authenticate with Azure
- Verify: `az account show`
- Set correct subscription: `az account set --subscription <your-subscription-id>`

### "Module not found: azure.ai.projects"
- Install Foundry SDK: `pip install --pre azure-ai-projects>=2.0.0b1`
- Install Azure Identity: `pip install azure-identity`

### "Agent not found" or "404 Resource not found"
- Verify agent name in Foundry web interface
- Check `AZURE_EXISTING_AGENT_ID` matches exactly (e.g., `your-agent-name:1`)
- Agent name format: `agent-name` or `agent-name:version`

### "Client.__init__() got an unexpected keyword argument 'proxies'"
- This is a compatibility issue with httpx
- Solution: `pip install 'httpx<0.27.0'`
- Or the code will automatically patch httpx to handle this

### "401 Unauthorized"
- Verify you're logged in: `az login`
- Check you have access to the Foundry project
- Verify endpoint URL is correct
- Consider using API key approach as fallback

## Benefits of Foundry Agents

1. **Simplified Configuration**: No need to specify deployment names
2. **Better Abstraction**: Agent handles model selection
3. **Unified Interface**: Single agent for all operations
4. **Foundry Integration**: Better integration with Foundry's capabilities

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
