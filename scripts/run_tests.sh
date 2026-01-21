#!/bin/bash
# Test script to verify the application setup

echo "🧪 Testing LLM Decision Flow Setup"
echo "===================================="
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Virtual environment not activated"
    echo "   Run: source venv/bin/activate"
    echo ""
fi

# Check dependencies
echo "📦 Checking dependencies..."
python3 -c "import pydantic_settings" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Dependencies not installed"
    echo "   Run: pip install -r requirements.txt"
    exit 1
fi
echo "✅ Dependencies installed"
echo ""

# Test Foundry agent setup
echo "🤖 Testing Foundry Agent Setup..."
python3 scripts/test_foundry_agent.py
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Foundry agent setup is working!"
else
    echo ""
    echo "⚠️  Foundry agent test failed - check your .env configuration"
fi
echo ""

# Check if evidence is populated
echo "📚 Checking evidence database..."
if [ -d "data/chroma_db" ] && [ "$(ls -A data/chroma_db 2>/dev/null)" ]; then
    echo "✅ Evidence database exists"
else
    echo "⚠️  Evidence database not populated"
    echo "   Run: python scripts/populate_evidence.py"
fi
echo ""

echo "===================================="
echo "✅ Setup check complete!"
echo ""
echo "To start the server:"
echo "  python run_server.py"
echo ""
