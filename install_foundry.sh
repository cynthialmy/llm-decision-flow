#!/bin/bash
# Optional script to install Foundry SDK with pre-release flag
# This is only needed if you want to use Foundry agents instead of API keys

echo "Installing Azure AI Foundry SDK (pre-release)..."
pip install --pre azure-ai-projects>=2.0.0b1 azure-identity>=1.15.0

if [ $? -eq 0 ]; then
    echo "✅ Foundry SDK installed successfully"
    echo "You can now use Foundry agents by setting AZURE_EXISTING_AIPROJECT_ENDPOINT and AZURE_EXISTING_AGENT_ID"
else
    echo "❌ Failed to install Foundry SDK"
    echo "The application will still work using API keys (AZURE_OPENAI_API_KEY)"
    exit 1
fi
