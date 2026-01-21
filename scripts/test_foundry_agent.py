"""Test script to verify Foundry agent setup works."""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to load .env, but don't fail if we can't
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception as e:
    print(f"⚠️  Could not load .env file: {e}")
    print("   Make sure your environment variables are set")
    print()

from src.config import get_foundry_project_client, get_foundry_agent_name, settings
from src.agents.claim_agent import ClaimAgent

def test_foundry_agent():
    """Test Foundry agent setup."""
    print("🔍 Testing Foundry Agent Setup...")
    print()

    # Check configuration
    endpoint = settings.azure_existing_aiproject_endpoint
    agent_id = settings.azure_existing_agent_id

    print(f"Endpoint: {endpoint}")
    print(f"Agent ID: {agent_id}")
    print()

    if not endpoint:
        print("❌ AZURE_EXISTING_AIPROJECT_ENDPOINT not configured!")
        return False

    if not agent_id:
        print("❌ AZURE_EXISTING_AGENT_ID not configured!")
        return False

    # Test Foundry project client
    print("🔌 Testing Foundry project client...")
    try:
        project_client = get_foundry_project_client()
        if not project_client:
            print("❌ Could not create Foundry project client")
            return False
        print("✅ Foundry project client created successfully")
        print()
    except Exception as e:
        print(f"❌ Error creating Foundry project client: {e}")
        return False

    # Test getting agent
    print("🤖 Testing agent retrieval...")
    try:
        agent_name = get_foundry_agent_name()
        print(f"   Agent name: {agent_name}")

        agent = project_client.agents.get(agent_name=agent_name)
        print(f"✅ Successfully retrieved agent: {agent.name}")
        print(f"   Agent ID: {agent.id}")
        print()
    except Exception as e:
        print(f"❌ Error retrieving agent: {e}")
        return False

    # Test agent via OpenAI client
    print("💬 Testing agent via OpenAI client...")
    try:
        openai_client = project_client.get_openai_client()

        response = openai_client.responses.create(
            input=[{"role": "user", "content": "Say 'Hello, Foundry agent is working!'"}],
            extra_body={"agent": {"name": agent.name, "type": "agent_reference"}},
        )

        result = response.output_text
        print(f"✅ Agent response: {result}")
        print()
    except Exception as e:
        print(f"❌ Error calling agent: {e}")
        return False

    # Test via ClaimAgent (integration test)
    print("🧪 Testing ClaimAgent with Foundry agent...")
    try:
        claim_agent = ClaimAgent()

        # Check if it's using Foundry agent
        if claim_agent.use_foundry_agent:
            print(f"   ✅ ClaimAgent is using Foundry agent: {claim_agent.foundry_agent_name}")
        else:
            print("   ⚠️  ClaimAgent is NOT using Foundry agent (using direct model calls)")

        # Test claim extraction
        test_text = "COVID vaccines are safe and effective. They have been tested in clinical trials."
        print(f"   Testing with text: '{test_text}'")

        claims = claim_agent.process(test_text)
        print(f"✅ Successfully extracted {len(claims)} claims:")
        for i, claim in enumerate(claims, 1):
            print(f"   {i}. {claim.text} ({claim.domain.value})")
        print()
    except Exception as e:
        print(f"❌ Error testing ClaimAgent: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print()
    print("Your Foundry agent is configured and working correctly!")
    return True

if __name__ == "__main__":
    success = test_foundry_agent()
    sys.exit(0 if success else 1)
