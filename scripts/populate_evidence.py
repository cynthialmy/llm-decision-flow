"""Script to populate vector store with evidence documents."""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.vector_store import VectorStore
from src.config import settings
from datetime import datetime


def populate_evidence():
    """Populate vector store with evidence documents."""
    settings.allow_runtime_indexing = True
    vector_store = VectorStore()

    # Evidence documents directory
    evidence_dir = Path(__file__).parent.parent / "data" / "evidence"

    documents = []
    metadatas = []
    ids = []

    # Load all evidence files
    for file_path in evidence_dir.glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                documents.append(content)

                # Extract domain from filename
                filename = file_path.stem
                domain = "other"
                if "health" in filename.lower():
                    domain = "health"
                elif "civic" in filename.lower():
                    domain = "civic"
                elif "finance" in filename.lower():
                    domain = "finance"

                metadatas.append({
                    "source": file_path.name,
                    "domain": domain,
                    "quality": "high",
                    "timestamp": datetime.now().isoformat(),
                    "index_version": settings.evidence_index_version
                })
                ids.append(file_path.stem)

    if documents:
        print(f"Adding {len(documents)} evidence documents to vector store...")
        vector_store.add_documents(documents, metadatas, ids)
        print("Evidence documents added successfully!")
    else:
        print("No evidence documents found.")


if __name__ == "__main__":
    populate_evidence()
