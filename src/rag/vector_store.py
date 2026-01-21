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

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            response = self.embedding_client.embeddings.create(
                model=self.embedding_deployment,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
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
