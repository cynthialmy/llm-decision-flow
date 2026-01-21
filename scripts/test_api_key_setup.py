"""Test script to verify API key setup works."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.config import get_azure_openai_client, settings

def test_setup():
    """Test the API key setup."""
    print("🔍 Testing API Key Setup...")
    print()

    # Check configuration
    endpoint = settings.azure_openai_endpoint or settings.azure_existing_aiproject_endpoint
    api_key = settings.azure_openai_api_key
    deployment = settings.azure_openai_deployment_name

    print(f"Endpoint: {endpoint}")
    print(f"API Key: {api_key[:20]}...{api_key[-10:] if api_key else 'NOT SET'}")
    print(f"Deployment: {deployment}")
    print(f"API Version: {settings.azure_openai_api_version}")
    print()

    if not endpoint:
        print("❌ Endpoint not configured!")
        return False

    if not api_key:
        print("❌ API key not configured!")
        return False

    if not deployment:
        print("❌ Deployment name not configured!")
        return False

    # Test connection
    print("🔌 Testing connection...")
    try:
        client = get_azure_openai_client()

        # Try a simple completion
        print(f"📝 Testing deployment: {deployment}")
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": "Say 'Hello'"}],
            max_tokens=10
        )

        result = response.choices[0].message.content
        print(f"✅ Success! Response: {result}")
        print()
        print("=" * 60)
        print("✅ CONFIGURATION IS WORKING!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("Troubleshooting:")
        print("1. Check your endpoint URL (should be base URL, not /api/projects/)")
        print("2. Verify your API key is correct")
        print("3. Check your deployment name matches exactly")
        print("4. Ensure API version is correct")
        return False

if __name__ == "__main__":
    success = test_setup()
    sys.exit(0 if success else 1)
