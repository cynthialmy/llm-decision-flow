"""Configuration management for Azure OpenAI and Azure AI Foundry settings."""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from openai import AzureOpenAI

# Try importing Foundry SDK (optional)
try:
    from azure.identity import DefaultAzureCredential
    from azure.ai.projects import AIProjectClient
    FOUNDRY_AVAILABLE = True
except ImportError:
    FOUNDRY_AVAILABLE = False


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Supports both Azure OpenAI Service and Azure AI Foundry endpoints.
    For Foundry endpoints, uses Foundry SDK with Azure credentials (no API key needed).
    For standard Azure OpenAI: https://{resource-name}.openai.azure.com/
    """

    # Azure OpenAI / Azure AI Foundry Configuration
    # Support both standard names and Foundry SDK names
    azure_openai_endpoint: Optional[str] = None  # Standard name
    azure_openai_api_key: Optional[str] = None  # Optional for Foundry (uses Azure credentials)
    azure_openai_deployment_name: Optional[str] = None  # Deployment name in Foundry or Azure OpenAI
    azure_openai_api_version: str = "2024-02-15-preview"
    azure_openai_embedding_deployment: Optional[str] = None

    # Foundry-specific: Use Azure credentials instead of API key
    use_foundry: bool = False  # Set to True to use Foundry SDK, or auto-detect from endpoint

    # Azure AI Foundry environment variable names (from azd/Foundry SDK)
    azure_existing_aiproject_endpoint: Optional[str] = None
    azure_existing_agent_id: Optional[str] = None
    azure_env_name: Optional[str] = None
    azure_location: Optional[str] = None
    azure_subscription_id: Optional[str] = None

    # Database Configuration
    sqlite_db_path: str = "./data/decisions.db"
    chroma_db_path: str = "./data/chroma_db"

    # Policy Configuration
    policy_file_path: str = "./policies/misinformation_policy.txt"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def _normalize_endpoint(endpoint: str) -> str:
    """
    Normalize endpoint URL for Azure OpenAI client.

    Handles different endpoint formats:
    - Foundry project endpoint: https://...services.ai.azure.com/api/projects/...
    - Foundry base endpoint: https://...services.ai.azure.com/
    - Standard Azure OpenAI: https://...openai.azure.com/
    """
    endpoint = endpoint.strip()

    # Remove trailing slash
    if endpoint.endswith('/'):
        endpoint = endpoint[:-1]

    # If it's a project endpoint, extract base endpoint
    if '/api/projects/' in endpoint:
        # Extract base: https://support-8844-resource.services.ai.azure.com
        base = endpoint.split('/api/projects/')[0]
        return base + '/'

    # Ensure ends with /
    if not endpoint.endswith('/'):
        endpoint = endpoint + '/'

    return endpoint


def _get_foundry_openai_client(endpoint: str):
    """
    Get OpenAI client from Foundry SDK with workaround for proxies parameter issue.

    The Foundry SDK's get_openai_client() may pass 'proxies' parameter to httpx.Client
    which newer httpx versions don't accept. This function patches httpx to handle it.
    """
    # Patch httpx.Client to handle proxies parameter issue
    # Newer httpx versions (0.27+) don't accept 'proxies' parameter directly
    import httpx
    original_httpx_init = httpx.Client.__init__

    def patched_httpx_init(self, *args, **kwargs):
        # Remove proxies parameter - Foundry SDK shouldn't need it
        kwargs.pop('proxies', None)
        return original_httpx_init(self, *args, **kwargs)

    # Apply patch
    httpx.Client.__init__ = patched_httpx_init

    try:
        credential = DefaultAzureCredential()

        project_client = AIProjectClient(
            endpoint=endpoint,
            credential=credential,
        )
        return project_client.get_openai_client()
    finally:
        # Restore original httpx.Client.__init__
        httpx.Client.__init__ = original_httpx_init


def get_foundry_project_client():
    """
    Get Foundry AIProjectClient for agent operations.

    Returns:
        AIProjectClient instance if Foundry is configured, None otherwise
    """
    if not FOUNDRY_AVAILABLE:
        return None

    endpoint = settings.azure_openai_endpoint or settings.azure_existing_aiproject_endpoint
    if not endpoint:
        return None

    endpoint = endpoint.strip().strip('"').strip("'")

    # Check if this is a Foundry project endpoint
    if '/api/projects/' not in endpoint:
        return None

    # Patch httpx for proxies issue
    import httpx
    original_httpx_init = httpx.Client.__init__

    def patched_httpx_init(self, *args, **kwargs):
        kwargs.pop('proxies', None)
        return original_httpx_init(self, *args, **kwargs)

    httpx.Client.__init__ = patched_httpx_init

    try:
        credential = DefaultAzureCredential()
        return AIProjectClient(
            endpoint=endpoint,
            credential=credential,
        )
    finally:
        httpx.Client.__init__ = original_httpx_init


def get_foundry_agent_name() -> Optional[str]:
    """Get Foundry agent name from settings."""
    agent_id = settings.azure_existing_agent_id
    if not agent_id:
        return None

    # Agent ID format: "agent-name:version" or just "agent-name"
    # Extract just the name part
    agent_id = agent_id.strip().strip('"').strip("'")
    if ':' in agent_id:
        return agent_id.split(':')[0]
    return agent_id


def get_azure_openai_client() -> AzureOpenAI:
    """Create and return Azure OpenAI client for chat completions.

    Works with both Azure OpenAI Service and Azure AI Foundry endpoints.
    For Foundry, uses AIProjectClient.get_openai_client() if available.

    Supports both standard env vars and Foundry SDK env vars.
    """
    # Get endpoint (support Foundry env var names)
    endpoint = settings.azure_openai_endpoint or settings.azure_existing_aiproject_endpoint
    if not endpoint:
        raise ValueError(
            "AZURE_OPENAI_ENDPOINT or AZURE_EXISTING_AIPROJECT_ENDPOINT must be set. "
            "Example: https://support-8844-resource.services.ai.azure.com/api/projects/support-8844"
        )

    # Clean up endpoint (remove quotes if present from env file)
    endpoint = endpoint.strip().strip('"').strip("'")

    # Check if this is a Foundry project endpoint
    is_foundry_endpoint = '/api/projects/' in endpoint
    use_foundry = settings.use_foundry or (is_foundry_endpoint and FOUNDRY_AVAILABLE)

    if use_foundry and FOUNDRY_AVAILABLE:
        # Use Foundry SDK with Azure credentials
        try:
            return _get_foundry_openai_client(endpoint)
        except Exception as e:
            # Fall back to standard approach if Foundry fails
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Foundry SDK failed, falling back to standard client: {e}")
            if not settings.azure_openai_api_key:
                raise ValueError("Foundry SDK failed and no API key provided. Please run 'az login' or provide AZURE_OPENAI_API_KEY")

    # Standard Azure OpenAI approach
    if not settings.azure_openai_api_key:
        raise ValueError(
            "AZURE_OPENAI_API_KEY is required for standard Azure OpenAI endpoints. "
            "For Foundry endpoints, make sure Foundry SDK is installed and you're logged in with 'az login'"
        )

    normalized_endpoint = _normalize_endpoint(endpoint)
    return AzureOpenAI(
        azure_endpoint=normalized_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
    )


def get_azure_openai_embedding_client() -> AzureOpenAI:
    """Create and return Azure OpenAI client for embeddings."""
    # Use same approach as chat client
    return get_azure_openai_client()


def get_deployment_name() -> str:
    """Get deployment name from settings."""
    deployment = settings.azure_openai_deployment_name
    if not deployment:
        raise ValueError(
            "AZURE_OPENAI_DEPLOYMENT_NAME must be set. "
            "Run 'python scripts/discover_foundry_settings.py' to find available deployments."
        )
    return deployment.strip().strip('"').strip("'")


def get_embedding_deployment_name() -> str:
    """Get the embedding deployment name, falling back to main deployment if not specified."""
    if settings.azure_openai_embedding_deployment:
        return settings.azure_openai_embedding_deployment.strip().strip('"').strip("'")
    return get_deployment_name()
