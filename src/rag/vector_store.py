"""Vector store wrapper for ChromaDB with Azure OpenAI embeddings."""
import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
from src.config import settings, get_azure_openai_embedding_client, get_embedding_deployment_name
from openai import AzureOpenAI


class VectorStore:
    """ChromaDB vector store with Azure OpenAI embeddings."""

    def __init__(self):
        """Initialize ChromaDB client and collection."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(settings.chroma_db_path) or ".", exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=settings.chroma_db_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="evidence",
            metadata={"description": "Evidence documents for misinformation detection"}
        )

        # Azure OpenAI client for embeddings
        self.embedding_client: AzureOpenAI = get_azure_openai_embedding_client()
        self.embedding_deployment = get_embedding_deployment_name()

    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text using Azure OpenAI.
        Automatically retries with alternative endpoints if 404 error occurs.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        import logging
        logger = logging.getLogger(__name__)

        # Try with current client first
        try:
            logger.debug(f"Generating embedding with deployment: {self.embedding_deployment}")
            endpoint = getattr(self.embedding_client, '_azure_endpoint', None) or getattr(self.embedding_client, 'azure_endpoint', 'unknown')
            endpoint_str = str(endpoint) if endpoint != 'unknown' else 'unknown'
            logger.debug(f"Using endpoint: {endpoint_str}")

            response = self.embedding_client.embeddings.create(
                model=self.embedding_deployment,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            error_msg = str(e)
            endpoint = getattr(self.embedding_client, '_azure_endpoint', None) or getattr(self.embedding_client, 'azure_endpoint', 'unknown')
            endpoint_str = str(endpoint) if endpoint != 'unknown' else 'unknown'

            # If 404 error, try alternative endpoints automatically
            if '404' in error_msg or 'not found' in error_msg.lower():
                # Extract resource name from endpoint if possible
                resource_name = None
                if endpoint_str != 'unknown':
                    if '.openai.azure.com' in endpoint_str:
                        resource_name = endpoint_str.split('://')[1].split('.openai.azure.com')[0]
                    elif '.cognitiveservices.azure.com' in endpoint_str:
                        resource_name = endpoint_str.split('://')[1].split('.cognitiveservices.azure.com')[0]
                    elif '.services.ai.azure.com' in endpoint_str:
                        resource_name = endpoint_str.split('://')[1].split('.services.ai.azure.com')[0]

                # Try alternative endpoints if we have resource name and API key
                if resource_name and settings.azure_openai_api_key:
                    alternative_endpoints = [
                        f"https://{resource_name}.openai.azure.com",
                        f"https://{resource_name}.cognitiveservices.azure.com",
                        f"https://{resource_name}.services.ai.azure.com",
                    ]

                    # Remove current endpoint from alternatives
                    alternative_endpoints = [ep for ep in alternative_endpoints if endpoint_str not in ep]

                    logger.info(f"404 error with endpoint {endpoint_str}, trying alternative endpoints...")

                    for alt_endpoint in alternative_endpoints:
                        try:
                            logger.info(f"Trying alternative endpoint: {alt_endpoint}")
                            # Create new client with alternative endpoint
                            alt_client = AzureOpenAI(
                                azure_endpoint=alt_endpoint,
                                api_key=settings.azure_openai_api_key,
                                api_version=settings.azure_openai_api_version,
                            )

                            # Try embedding with alternative endpoint
                            response = alt_client.embeddings.create(
                                model=self.embedding_deployment,
                                input=text
                            )

                            # Success! Update the client for future use
                            logger.info(f"✅ Successfully connected to {alt_endpoint}, updating client")
                            self.embedding_client = alt_client
                            return response.data[0].embedding

                        except Exception as alt_error:
                            logger.debug(f"Alternative endpoint {alt_endpoint} also failed: {alt_error}")
                            continue

                # If all endpoints failed, provide helpful error message
                helpful_msg = (
                    f"Embedding deployment '{self.embedding_deployment}' not found at endpoint '{endpoint_str}'. "
                    f"This usually means:\n"
                    f"1. The deployment name is incorrect - check your AZURE_OPENAI_EMBEDDING_DEPLOYMENT setting\n"
                    f"2. The deployment doesn't exist at this endpoint\n"
                    f"3. Your API key doesn't have access to this deployment\n"
                    f"4. The endpoint URL might be incorrect\n\n"
                )

                # If we can extract resource name, suggest alternative endpoints
                if resource_name:
                    helpful_msg += (
                        f"For Foundry projects, try setting AZURE_OPENAI_EMBEDDING_ENDPOINT to one of:\n"
                        f"  - https://{resource_name}.openai.azure.com (recommended)\n"
                        f"  - https://{resource_name}.cognitiveservices.azure.com\n"
                        f"  - https://{resource_name}.services.ai.azure.com\n\n"
                    )

                helpful_msg += (
                    f"To fix this:\n"
                    f"- Run 'python scripts/discover_foundry_settings.py' to find available deployments\n"
                    f"- Set AZURE_OPENAI_EMBEDDING_ENDPOINT explicitly to the correct endpoint\n"
                    f"- Verify your API key has access to embeddings\n"
                    f"Original error: {error_msg}"
                )
                logger.error(helpful_msg)
                raise ValueError(helpful_msg)
            else:
                logger.error(f"Embedding error - Deployment: {self.embedding_deployment}, Endpoint: {endpoint_str}, Error: {e}")
                raise ValueError(f"Error generating embedding: {e}")

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ):
        """
        Add documents to the vector store.

        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dicts
            ids: Optional list of document IDs
        """
        if not documents:
            return

        # Generate embeddings
        embeddings = [self._get_embedding(doc) for doc in documents]

        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        # Default metadata if not provided
        if metadatas is None:
            metadatas = [{}] * len(documents)

        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documents.

        Args:
            query: Search query text
            n_results: Number of results to return
            where: Optional metadata filter

        Returns:
            List of result dictionaries with 'document', 'metadata', 'distance', 'id'
        """
        # Get query embedding
        query_embedding = self._get_embedding(query)

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )

        # Format results
        formatted_results = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None,
                    'id': results['ids'][0][i]
                })

        return formatted_results

    def get_all_documents(self) -> List[Dict]:
        """Get all documents from the collection."""
        results = self.collection.get()

        formatted_results = []
        if results['documents']:
            for i in range(len(results['documents'])):
                formatted_results.append({
                    'document': results['documents'][i],
                    'metadata': results['metadatas'][i] if 'metadatas' in results else {},
                    'id': results['ids'][i]
                })

        return formatted_results
