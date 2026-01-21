"""Script to discover Azure AI Foundry deployment names using Foundry SDK."""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Try importing Foundry SDK
try:
    from azure.identity import DefaultAzureCredential
    from azure.ai.projects import AIProjectClient
    FOUNDRY_SDK_AVAILABLE = True
except ImportError:
    FOUNDRY_SDK_AVAILABLE = False
    print("⚠️  Foundry SDK not available. Install with:")
    print("   pip install --pre azure-ai-projects>=2.0.0b1")
    print("   pip install azure-identity")
    print()

# Load environment variables
load_dotenv()

def test_deployment_foundry(endpoint: str, deployment: str) -> tuple[bool, str]:
    """Test if a deployment works using Foundry SDK. Returns (success, error_message)."""
    if not FOUNDRY_SDK_AVAILABLE:
        return False, "Foundry SDK not available"

    try:
        # Patch httpx.Client to handle proxies parameter issue
        # Newer httpx versions don't accept 'proxies' parameter directly
        import httpx
        original_httpx_init = httpx.Client.__init__

        def patched_httpx_init(self, *args, **kwargs):
            # Convert proxies to mount if present (httpx 0.27+ uses mount instead)
            if 'proxies' in kwargs:
                proxies = kwargs.pop('proxies')
                if proxies:
                    # Convert proxies dict to httpx mount format
                    # This is a simplified conversion - may need adjustment
                    try:
                        if isinstance(proxies, dict):
                            # For now, just remove it - Foundry SDK shouldn't need proxies
                            pass
                    except:
                        pass
            return original_httpx_init(self, *args, **kwargs)

        # Apply patch
        httpx.Client.__init__ = patched_httpx_init

        try:
            # Create credential - DefaultAzureCredential should handle audience automatically
            # But if you get 401 with "audience is incorrect", you may need to configure it
            credential = DefaultAzureCredential()

            project_client = AIProjectClient(
                endpoint=endpoint,
                credential=credential,
            )
            openai_client = project_client.get_openai_client()

            # Try a simple completion
            response = openai_client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            return True, ""
        finally:
            # Restore original
            httpx.Client.__init__ = original_httpx_init
    except Exception as e:
        error_msg = str(e)
        # Extract more useful error info
        if hasattr(e, 'status_code'):
            error_msg = f"HTTP {e.status_code}: {error_msg}"
        return False, error_msg

def list_available_models(endpoint: str):
    """Try to list available models/deployments."""
    if not FOUNDRY_SDK_AVAILABLE:
        return None

    try:
        project_client = AIProjectClient(
            endpoint=endpoint,
            credential=DefaultAzureCredential(),
        )
        openai_client = project_client.get_openai_client()

        # Try to list models
        models = openai_client.models.list()
        return [model.id for model in models]
    except Exception as e:
        return None

def discover_settings():
    """Discover Foundry settings."""
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT') or os.getenv('AZURE_EXISTING_AIPROJECT_ENDPOINT')

    if not endpoint:
        print("❌ AZURE_OPENAI_ENDPOINT or AZURE_EXISTING_AIPROJECT_ENDPOINT not found in .env file")
        print("\nPlease add to your .env file:")
        print("AZURE_EXISTING_AIPROJECT_ENDPOINT=https://your-resource.services.ai.azure.com/api/projects/your-project")
        print("\nOr:")
        print("AZURE_OPENAI_ENDPOINT=https://your-resource.services.ai.azure.com/api/projects/your-project")
        return

    # Clean up endpoint (remove quotes if present)
    endpoint = endpoint.strip().strip('"').strip("'")

    if not FOUNDRY_SDK_AVAILABLE:
        print("❌ Foundry SDK not installed")
        print("\nInstall with:")
        print("  pip install --pre azure-ai-projects>=2.0.0b1")
        print("  pip install azure-identity")
        print("\nThen run: az login")
        return

    print("🔍 Discovering Azure AI Foundry Settings...")
    print(f"Endpoint: {endpoint}")
    print()
    print("ℹ️  Using Azure AI Foundry SDK (azure-ai-projects)")
    print("   Make sure you're logged in: az login")
    print()

    # Test connection first
    print("🔌 Testing connection to Foundry...")
    try:
        project_client = AIProjectClient(
            endpoint=endpoint,
            credential=DefaultAzureCredential(),
        )
        print("✅ Successfully connected to Foundry!")
        print()
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Run: az login")
        print("2. Verify endpoint URL is correct")
        print("3. Check you have access to this Foundry project")
        return

    # Try to list available models first
    print("📋 Attempting to list available models...")
    available_models = list_available_models(endpoint)
    if available_models:
        print(f"✅ Found {len(available_models)} available models:")
        for model in available_models[:10]:  # Show first 10
            print(f"   - {model}")
        print()
    else:
        print("⚠️  Could not list models automatically, will test common names")
        print()

    # Test deployment names - prioritize the ones you actually see in Foundry
    # If we found available models, use those; otherwise test common names
    if available_models:
        deployments_to_test = available_models
        print(f"🧪 Testing {len(deployments_to_test)} discovered deployments...")
    else:
        deployments_to_test = [
            "gpt-4o",  # You have this
            "gpt-5-chat",  # You have this
            "gpt-5-mini",  # You have this
            "text-embedding-ada-002",  # You have this
            # Also test common alternatives
            "gpt-4",
            "gpt-35-turbo",
            "gpt-4-turbo",
            "gpt-4o-mini",
            "gpt-4-32k",
        ]
        print(f"🧪 Testing {len(deployments_to_test)} common deployment names...")

    print("🔬 Testing deployments with Foundry SDK...")
    print("   (Showing error details for debugging)")
    print()

    working_configs = []
    errors = []

    for deployment in deployments_to_test:
        print(f"Testing deployment: {deployment}...", end=" ")
        success, error_msg = test_deployment_foundry(endpoint, deployment)
        if success:
            print("✅ Works!")
            working_configs.append({
                'deployment': deployment
            })
        else:
            print(f"❌ Failed")
            if error_msg:
                # Show first 100 chars of error
                short_error = error_msg[:100] + "..." if len(error_msg) > 100 else error_msg
                print(f"   Error: {short_error}")
            errors.append((deployment, error_msg))
    print()

    if working_configs:
        print("=" * 60)
        print("✅ FOUND WORKING DEPLOYMENTS:")
        print("=" * 60)

        deployments = [config['deployment'] for config in working_configs]
        print(f"\n📌 Working Deployments: {', '.join(deployments)}")

        # Recommend first working config
        recommended = working_configs[0]
        print("\n" + "=" * 60)
        print("💡 RECOMMENDED CONFIGURATION:")
        print("=" * 60)
        print(f"""
Add this to your .env file:

AZURE_EXISTING_AIPROJECT_ENDPOINT={endpoint}
AZURE_OPENAI_DEPLOYMENT_NAME={recommended['deployment']}

Note: With Foundry SDK, you don't need AZURE_OPENAI_API_KEY.
Make sure you're logged in: az login
""")
    else:
        print("❌ No working deployments found.")
        print("\n" + "=" * 60)
        print("ERROR DETAILS:")
        print("=" * 60)
        for deployment, error in errors[:5]:  # Show first 5 errors
            print(f"\n{deployment}:")
            print(f"  {error}")

        print("\n" + "=" * 60)
        print("TROUBLESHOOTING:")
        print("=" * 60)
        print("1. Verify you're logged in: az account show")
        print("2. Check your endpoint URL is correct")
        print("3. Verify you have access to the Foundry project")
        print("4. Try testing manually:")
        print("   python -c \"from azure.identity import DefaultAzureCredential; from azure.ai.projects import AIProjectClient; client = AIProjectClient(endpoint='{}', credential=DefaultAzureCredential()); print(client.get_openai_client().models.list())\"".format(endpoint))

if __name__ == "__main__":
    discover_settings()
