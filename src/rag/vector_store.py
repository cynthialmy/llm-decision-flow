"""Vector store wrapper for ChromaDB with Azure OpenAI embeddings."""
import os
import hashlib
import shelve
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

        cached = self._get_cached_embedding(text)
        if cached is not None:
            return cached

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
            embedding = response.data[0].embedding
            self._set_cached_embedding(text, embedding)
            return embedding
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
                tried_endpoints = [endpoint_str]
                endpoint_errors = {endpoint_str: error_msg}

                if not settings.azure_openai_api_key:
                    logger.warning("No API key available for trying alternative endpoints")
                elif not resource_name:
                    logger.warning(f"Could not extract resource name from endpoint: {endpoint_str}")

                if resource_name and settings.azure_openai_api_key:
                    alternative_endpoints = [
                        f"https://{resource_name}.openai.azure.com",
                        f"https://{resource_name}.cognitiveservices.azure.com",
                        f"https://{resource_name}.services.ai.azure.com",
                    ]

                    # Remove current endpoint from alternatives (normalize both for comparison)
                    current_endpoint_normalized = endpoint_str.rstrip('/').lower()
                    alternative_endpoints = [
                        ep for ep in alternative_endpoints
                        if ep.rstrip('/').lower() != current_endpoint_normalized
                    ]

                    logger.info(f"404 error with endpoint {endpoint_str}, trying {len(alternative_endpoints)} alternative endpoints...")

                    for alt_endpoint in alternative_endpoints:
                        tried_endpoints.append(alt_endpoint)
                        try:
                            logger.info(f"Trying alternative endpoint: {alt_endpoint}")

                            # Use appropriate API version based on endpoint type
                            api_version = settings.azure_openai_api_version
                            if '.cognitiveservices.azure.com' in alt_endpoint:
                                # Use stable API version for cognitiveservices endpoints
                                api_version = "2023-05-15"
                                logger.debug(f"Using stable API version 2023-05-15 for cognitiveservices endpoint")
                            elif 'preview' in api_version.lower():
                                api_version = "2024-08-01-preview"

                            # Create new client with alternative endpoint
                            alt_client = AzureOpenAI(
                                azure_endpoint=alt_endpoint.rstrip('/'),
                                api_key=settings.azure_openai_api_key,
                                api_version=api_version,
                            )

                            # Try embedding with alternative endpoint
                            response = alt_client.embeddings.create(
                                model=self.embedding_deployment,
                                input=text
                            )

                            # Success! Update the client for future use
                            logger.info(f"✅ Successfully connected to {alt_endpoint}, updating client")
                            self.embedding_client = alt_client
                            embedding = response.data[0].embedding
                            self._set_cached_embedding(text, embedding)
                            return embedding

                        except Exception as alt_error:
                            error_str = str(alt_error)
                            endpoint_errors[alt_endpoint] = error_str
                            logger.warning(f"Alternative endpoint {alt_endpoint} failed: {error_str[:200]}")
                            continue

                # If all endpoints failed, provide helpful error message
                helpful_msg = (
                    f"Embedding deployment '{self.embedding_deployment}' not found. "
                    f"Tried {len(tried_endpoints)} endpoint(s):\n"
                )

                for ep in tried_endpoints:
                    error = endpoint_errors.get(ep, "Unknown error")
                    # Extract just the key error info
                    if "404" in error or "not found" in error.lower():
                        error_summary = "404 - Resource not found"
                    elif "401" in error or "unauthorized" in error.lower():
                        error_summary = "401 - Unauthorized (check API key)"
                    elif "403" in error or "forbidden" in error.lower():
                        error_summary = "403 - Forbidden (check permissions)"
                    else:
                        error_summary = error[:100] + "..." if len(error) > 100 else error
                    helpful_msg += f"  - {ep}: {error_summary}\n"

                helpful_msg += (
                    f"\nThis usually means:\n"
                    f"1. The deployment name '{self.embedding_deployment}' is incorrect\n"
                    f"2. The deployment doesn't exist at any of these endpoints\n"
                    f"3. Your API key doesn't have access to embeddings\n"
                    f"4. The endpoint URLs might be incorrect\n\n"
                )

                # If we can extract resource name, suggest alternative endpoints
                if resource_name:
                    helpful_msg += (
                        f"Possible solutions:\n"
                        f"1. Set AZURE_OPENAI_EMBEDDING_ENDPOINT to one of:\n"
                        f"   - https://{resource_name}.openai.azure.com (recommended)\n"
                        f"   - https://{resource_name}.cognitiveservices.azure.com\n"
                        f"   - https://{resource_name}.services.ai.azure.com\n"
                        f"2. Verify the deployment name '{self.embedding_deployment}' exists\n"
                        f"3. Run 'python scripts/discover_foundry_settings.py' to find available deployments\n"
                        f"4. Check your API key has access to embeddings\n"
                    )

                logger.error(helpful_msg)
                raise ValueError(helpful_msg)
            else:
                logger.error(f"Embedding error - Deployment: {self.embedding_deployment}, Endpoint: {endpoint_str}, Error: {e}")
                raise ValueError(f"Error generating embedding: {e}")

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        missing_texts: List[str] = []
        missing_indices: List[int] = []

        for idx, text in enumerate(texts):
            cached = self._get_cached_embedding(text)
            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append([])
                missing_texts.append(text)
                missing_indices.append(idx)

        if missing_texts:
            response = self.embedding_client.embeddings.create(
                model=self.embedding_deployment,
                input=missing_texts
            )
            for i, emb in enumerate(response.data):
                index = missing_indices[i]
                embeddings[index] = emb.embedding
                self._set_cached_embedding(missing_texts[i], emb.embedding)

        return embeddings

    def _embedding_cache_path(self) -> str:
        return os.path.join(os.path.dirname(settings.chroma_db_path) or ".", "embedding_cache")

    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        cache_path = self._embedding_cache_path()
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        try:
            with shelve.open(cache_path) as cache:
                return cache.get(key)
        except Exception:
            return None

    def _set_cached_embedding(self, text: str, embedding: List[float]) -> None:
        cache_path = self._embedding_cache_path()
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        try:
            with shelve.open(cache_path) as cache:
                cache[key] = embedding
        except Exception:
            return

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

        if not settings.allow_runtime_indexing:
            raise ValueError("Runtime indexing is disabled. Set ALLOW_RUNTIME_INDEXING=true to enable.")

        # Generate embeddings (batch + cache)
        embeddings = self._get_embeddings(documents)

        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        # Default metadata if not provided
        if metadatas is None:
            metadatas = [{"index_version": settings.evidence_index_version} for _ in documents]
        else:
            for metadata in metadatas:
                metadata.setdefault("index_version", settings.evidence_index_version)

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

    def max_similarity(self, query: str, index_version: Optional[str] = None) -> Optional[float]:
        where = {"index_version": index_version} if index_version else None
        results = self.search(query, n_results=1, where=where)
        if not results:
            return None
        distance = results[0].get("distance")
        if distance is None:
            return None
        return 1.0 - distance

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
