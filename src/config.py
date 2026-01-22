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
    azure_openai_api_key: Optional[str] = None  # Optional for Foundry (uses Azure credentials), but required for embeddings
    azure_openai_deployment_name: Optional[str] = None  # Deployment name in Foundry or Azure OpenAI
    azure_openai_api_version: str = "2024-02-15-preview"
    azure_openai_embedding_deployment: Optional[str] = None
    azure_openai_embedding_endpoint: Optional[str] = None  # Optional: explicit embedding endpoint (for Foundry, use cognitiveservices.azure.com)

    # Foundry-specific: Use Azure credentials instead of API key
    use_foundry: bool = False  # Set to True to use Foundry SDK, or auto-detect from endpoint

    # Azure AI Foundry environment variable names (from azd/Foundry SDK)
    azure_existing_aiproject_endpoint: Optional[str] = None
    azure_existing_agent_id: Optional[str] = None
    azure_env_name: Optional[str] = None
    azure_location: Optional[str] = None
    azure_subscription_id: Optional[str] = None
    azure_existing_aiproject_resource_id: Optional[str] = None
    azure_existing_resource_id: Optional[str] = None
    azd_allow_non_empty_folder: Optional[str] = None

    # Database Configuration
    sqlite_db_path: str = "./data/decisions.db"
    chroma_db_path: str = "./data/chroma_db"

    # Policy Configuration
    policy_file_path: str = "./policies/misinformation_policy.txt"

    # External Providers
    groq_api_key: Optional[str] = None
    groq_model: str = "llama-3.3-70b-versatile"
    zentropi_api_key: Optional[str] = None
    zentropi_labeler_id: Optional[str] = None
    zentropi_labeler_version_id: Optional[str] = None
    serper_api_key: Optional[str] = None

    # Routing + Confidence Thresholds
    claim_confidence_threshold: float = 0.65
    risk_confidence_threshold: float = 0.6
    policy_confidence_threshold: float = 0.7
    novelty_similarity_threshold: float = 0.35
    evidence_similarity_cutoff: float = 0.4

    # Token + Latency Budgets
    slm_max_tokens: int = 800
    frontier_max_tokens: int = 2000
    claim_max_tokens: int = 900
    slm_timeout_s: float = 2.5
    frontier_timeout_s: float = 6.0

    # Evidence Indexing
    allow_runtime_indexing: bool = False
    evidence_index_version: str = "v1"

    # External Search Controls
    allow_external_search: bool = True
    external_search_allowlist: str = "gov,edu,who.int,cdc.gov,nih.gov,factcheck.org,reuters.com,apnews.com"
    allow_external_enrichment: bool = False

    # Governance + Quality Gates
    policy_version: str = "1.0"
    disagreement_rollback_threshold: float = 0.2
    latency_rollback_threshold_s: float = 10.0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables


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
        # Extract base endpoint from project endpoint
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

    Also ensures DefaultAzureCredential is configured with correct audience for Foundry.
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
        # Configure credential with correct audience for Foundry
        # Foundry requires audience "https://ai.azure.com"
        credential = DefaultAzureCredential()

        # Try to configure audience if the credential supports it
        # Some credential types need explicit audience configuration
        try:
            # For ManagedIdentityCredential and similar, we might need to set audience
            # But DefaultAzureCredential should handle this automatically
            pass
        except Exception:
            pass

        project_client = AIProjectClient(
            endpoint=endpoint,
            credential=credential,
        )

        # Try to get OpenAI client - check for inference API first (newer SDK versions)
        openai_client = None
        try:
            # Method 1: Try inference.get_azure_openai_client() (newer SDK)
            if hasattr(project_client, 'inference') and hasattr(project_client.inference, 'get_azure_openai_client'):
                api_version = settings.azure_openai_api_version or "2024-02-15-preview"
                openai_client = project_client.inference.get_azure_openai_client(api_version=api_version)
            # Method 2: Fallback to get_openai_client() (older SDK)
            elif hasattr(project_client, 'get_openai_client'):
                openai_client = project_client.get_openai_client()
            else:
                raise ValueError("Foundry SDK doesn't provide a method to get OpenAI client")
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to get OpenAI client from Foundry project client: {e}")
            raise

        # Verify client has responses attribute (Foundry extension)
        if not hasattr(openai_client, 'responses'):
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                "Foundry OpenAI client doesn't have 'responses' attribute. "
                "This might indicate an SDK version mismatch or configuration issue. "
                f"Client type: {type(openai_client)}, Available attributes: {[attr for attr in dir(openai_client) if not attr.startswith('_')][:10]}"
            )

        return openai_client
    finally:
        # Restore original httpx.Client.__init__
        httpx.Client.__init__ = original_httpx_init


def get_foundry_project_client():
    """
    Get Foundry AIProjectClient for agent operations.

    Returns:
        AIProjectClient instance if Foundry is configured, None otherwise

    Raises:
        ValueError: If Foundry SDK is not available or configuration is invalid
        Exception: If client creation fails
    """
    if not FOUNDRY_AVAILABLE:
        raise ValueError(
            "Foundry SDK not available. Install with: "
            "pip install --pre azure-ai-projects>=2.0.0b1 azure-identity"
        )

    # Prefer azure_existing_aiproject_endpoint (Foundry format) over azure_openai_endpoint
    endpoint = settings.azure_existing_aiproject_endpoint or settings.azure_openai_endpoint
    if not endpoint:
        raise ValueError(
            "Foundry endpoint not configured. Set AZURE_EXISTING_AIPROJECT_ENDPOINT in .env"
        )

    endpoint = endpoint.strip().strip('"').strip("'")

    # Check if this is a Foundry project endpoint
    if '/api/projects/' not in endpoint:
        raise ValueError(
            f"Endpoint does not appear to be a Foundry project endpoint: {endpoint}. "
            "Foundry endpoints should contain '/api/projects/'. "
            f"Current endpoint from settings: azure_existing_aiproject_endpoint={settings.azure_existing_aiproject_endpoint}, "
            f"azure_openai_endpoint={settings.azure_openai_endpoint}"
        )

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
    except Exception as e:
        raise ValueError(
            f"Failed to create Foundry project client: {e}. "
            "Make sure you're logged in with 'az login' and have access to the Foundry project."
        ) from e
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
            "Example: https://your-resource.services.ai.azure.com/api/projects/your-project"
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
    """Create and return Azure OpenAI client for embeddings.

    For Foundry endpoints, embeddings need to use the base endpoint (not project endpoint).
    According to Microsoft docs, Foundry supports multiple endpoints:
    - https://<resource-name>.openai.azure.com (preferred for embeddings)
    - https://<resource-name>.services.ai.azure.com
    - https://<resource-name>.cognitiveservices.azure.com

    This function tries openai.azure.com first, then falls back to others.
    """
    # Check if we're using Foundry
    endpoint = settings.azure_existing_aiproject_endpoint or settings.azure_openai_endpoint
    if not endpoint:
        raise ValueError(
            "AZURE_OPENAI_ENDPOINT or AZURE_EXISTING_AIPROJECT_ENDPOINT must be set."
        )

    endpoint = endpoint.strip().strip('"').strip("'")
    is_foundry_endpoint = '/api/projects/' in endpoint

    # Check if explicit embedding endpoint is set (takes precedence)
    if settings.azure_openai_embedding_endpoint:
        embedding_endpoint = settings.azure_openai_embedding_endpoint.strip().strip('"').strip("'")
        if not settings.azure_openai_api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is required when using AZURE_OPENAI_EMBEDDING_ENDPOINT")

        import logging
        logger = logging.getLogger(__name__)

        # Normalize endpoint - Azure OpenAI SDK expects base URL without trailing slash
        embedding_endpoint = embedding_endpoint.rstrip('/')

        # For embeddings with cognitiveservices.azure.com endpoints, use a stable API version
        # The 2024-02-15-preview version should work, but 2023-05-15 is more stable for embeddings
        embedding_api_version = settings.azure_openai_api_version
        if '.cognitiveservices.azure.com' in embedding_endpoint:
            # Use a stable, widely-supported API version for cognitiveservices endpoints
            embedding_api_version = "2023-05-15"
            logger.info("Using stable API version 2023-05-15 for cognitiveservices.azure.com endpoint")
        elif 'preview' in embedding_api_version.lower():
            embedding_api_version = "2024-08-01-preview"

        logger.info(f"Using explicit embedding endpoint: {embedding_endpoint}")
        logger.info(f"Using embedding deployment: {get_embedding_deployment_name()}")
        logger.info(f"Using API version: {embedding_api_version}")

        return AzureOpenAI(
            azure_endpoint=embedding_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=embedding_api_version,
        )

    # For Foundry, embeddings need the base endpoint (not Foundry project endpoint)
    if is_foundry_endpoint and FOUNDRY_AVAILABLE:
        # Extract resource name from Foundry endpoint
        # e.g., https://resource.services.ai.azure.com/api/projects/project-name
        # -> resource
        resource_name = None
        if '.services.ai.azure.com' in endpoint:
            resource_name = endpoint.split('://')[1].split('.services.ai.azure.com')[0]
        elif '.openai.azure.com' in endpoint:
            resource_name = endpoint.split('://')[1].split('.openai.azure.com')[0]
        elif '.cognitiveservices.azure.com' in endpoint:
            resource_name = endpoint.split('://')[1].split('.cognitiveservices.azure.com')[0]
        else:
            # Fallback: try to extract from any endpoint format
            parts = endpoint.split('://')[1].split('/')[0].split('.')
            if len(parts) > 0:
                resource_name = parts[0]

        if not resource_name:
            raise ValueError(
                f"Could not extract resource name from endpoint: {endpoint}. "
                "Please set AZURE_OPENAI_EMBEDDING_ENDPOINT explicitly."
            )

        # Use openai.azure.com endpoint first (standard, most compatible per Microsoft docs)
        # If this doesn't work, user can set AZURE_OPENAI_EMBEDDING_ENDPOINT explicitly
        embedding_endpoint = f"https://{resource_name}.openai.azure.com"

        # Embeddings require API key (even when using Foundry agents)
        if not settings.azure_openai_api_key:
            raise ValueError(
                "AZURE_OPENAI_API_KEY is required for embeddings. "
                "Even when using Foundry agents, embeddings need an API key with the base endpoint."
            )

        import logging
        logger = logging.getLogger(__name__)

        # For embeddings, try a more stable API version if preview version is being used
        embedding_api_version = settings.azure_openai_api_version
        if 'preview' in embedding_api_version.lower():
            embedding_api_version = "2024-08-01-preview"

        logger.info(f"Using embedding endpoint: {embedding_endpoint}")
        logger.info(f"Using embedding deployment: {get_embedding_deployment_name()}")
        logger.info(f"Using API version: {embedding_api_version}")
        logger.info(
            f"If you get a 404 error, try setting AZURE_OPENAI_EMBEDDING_ENDPOINT to:\n"
            f"  - https://{resource_name}.cognitiveservices.azure.com\n"
            f"  - https://{resource_name}.services.ai.azure.com"
        )

        return AzureOpenAI(
            azure_endpoint=embedding_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=embedding_api_version,
        )

    # For standard Azure OpenAI, use the same approach as chat client
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
