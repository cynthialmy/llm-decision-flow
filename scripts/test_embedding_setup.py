"""Test script to verify embedding setup works."""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from src.config import get_azure_openai_embedding_client, get_embedding_deployment_name, settings

def test_embedding_setup():
    """Test embedding client setup."""
    print("🔍 Testing Embedding Setup...")
    print()

    # Show configuration
    endpoint = settings.azure_existing_aiproject_endpoint or settings.azure_openai_endpoint
    embedding_endpoint = settings.azure_openai_embedding_endpoint
    embedding_deployment = settings.azure_openai_embedding_deployment
    api_key_set = bool(settings.azure_openai_api_key)

    print(f"Main endpoint: {endpoint}")
    print(f"Explicit embedding endpoint: {embedding_endpoint or 'Not set (will use derived endpoint)'}")
    print(f"Embedding deployment: {embedding_deployment or 'Not set (will use main deployment)'}")
    print(f"API key configured: {'Yes' if api_key_set else 'No'}")
    print()

    if not endpoint:
        print("❌ AZURE_EXISTING_AIPROJECT_ENDPOINT or AZURE_OPENAI_ENDPOINT not configured!")
        return False

    if not api_key_set:
        print("❌ AZURE_OPENAI_API_KEY not configured!")
        print("   Embeddings require an API key even when using Foundry agents.")
        return False

    # Test creating embedding client
    print("🔌 Testing embedding client creation...")
    try:
        embedding_client = get_azure_openai_embedding_client()
        deployment_name = get_embedding_deployment_name()

        # Get endpoint for display
        endpoint_used = getattr(embedding_client, '_azure_endpoint', None) or 'unknown'
        print(f"✅ Embedding client created")
        print(f"   Endpoint: {endpoint_used}")
        print(f"   Deployment: {deployment_name}")
        print()
    except Exception as e:
        print(f"❌ Error creating embedding client: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test embedding generation
    print("🧪 Testing embedding generation...")
    try:
        test_text = "This is a test sentence for embedding."
        print(f"   Test text: '{test_text}'")

        response = embedding_client.embeddings.create(
            model=deployment_name,
            input=test_text
        )

        embedding = response.data[0].embedding
        print(f"✅ Successfully generated embedding")
        print(f"   Embedding dimension: {len(embedding)}")
        print(f"   First few values: {embedding[:5]}")
        print()
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error generating embedding: {error_msg}")
        print()

        if '404' in error_msg or 'not found' in error_msg.lower():
            print("=" * 60)
            print("TROUBLESHOOTING:")
            print("=" * 60)
            print()
            print("The embedding deployment was not found. This usually means:")
            print()
            print("1. The deployment name is incorrect")
            print(f"   Current: {deployment_name}")
            print()
            print("2. The deployment doesn't exist at this endpoint")
            print(f"   Endpoint: {endpoint_used}")
            print()
            print("3. Your API key doesn't have access to this deployment")
            print()
            print("To fix this:")
            print()
            print("Option A: Set explicit embedding endpoint and deployment")
            print("   In your .env file, add:")
            print("   AZURE_OPENAI_EMBEDDING_ENDPOINT=https://your-resource.openai.azure.com/")
            print("   AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your-embedding-deployment-name")
            print()
            print("Option B: Check your Azure portal")
            print("   1. Go to Azure Portal > Your OpenAI resource")
            print("   2. Check 'Deployments' section")
            print("   3. Find your embedding deployment name")
            print("   4. Update AZURE_OPENAI_EMBEDDING_DEPLOYMENT in .env")
            print()
            print("Option C: If using Foundry, embeddings use Cognitive Services endpoint")
            print("   The endpoint should be: https://your-resource.cognitiveservices.azure.com/")
            print("   Check that this endpoint has the embedding deployment")
            print()

        import traceback
        traceback.print_exc()
        return False

    print("=" * 60)
    print("✅ EMBEDDING SETUP WORKS!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_embedding_setup()
    sys.exit(0 if success else 1)
