"""
Script — Ingest Documents

Runs the full RAG ingestion pipeline:
1. Load documents from data/raw/
2. Chunk them with metadata
3. Embed and store in ChromaDB
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag.ingestion import DocumentIngestion
from rag.chunker import TextChunker
from rag.embedder import Embedder


def main():
    """Run the full ingestion pipeline."""
    print("=" * 60)
    print("  Digital Twin of Alan Turing — Document Ingestion")
    print("=" * 60)

    # Configuration
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    persist_dir = str(project_root / "storage" / "chroma_db")

    # Step 1: Load documents
    print("\n[Step 1] Loading documents...")
    ingestion = DocumentIngestion(
        raw_data_dir=str(raw_dir),
        processed_dir=str(processed_dir),
    )
    documents = ingestion.load_all_documents()

    if not documents:
        print("\n[Warning] No documents found in data/raw/")
        print("   Place source materials in the following directories:")
        print("   - data/raw/papers/    → Research papers (PDF, TXT)")
        print("   - data/raw/books/     → Books and book chapters")
        print("   - data/raw/interviews/→ Interview transcripts")
        print("   - data/raw/letters/   → Correspondence")
        print("\n   Filename convention: 'YEAR - Title.ext'")
        print("   Example: '1936 - On Computable Numbers.txt'")
        return

    # Save processed document manifest
    ingestion.save_processed(documents)

    # Step 2: Chunk documents
    print("\n[Step 2] Chunking documents...")
    chunker = TextChunker(chunk_size=512, chunk_overlap=50)
    chunks = chunker.chunk_all_documents(documents)

    # Step 3: Embed and store
    print("\n[Step 3] Embedding and storing in ChromaDB...")
    embedder = Embedder(
        model_name="all-MiniLM-L6-v2",
        persist_dir=persist_dir,
        collection_name="turing_knowledge",
    )

    # Clear existing data and re-index
    embedder.clear_collection()
    embedder.add_chunks(chunks)

    # Print stats
    stats = embedder.get_collection_stats()
    print("\n" + "=" * 60)
    print("  [Success] Ingestion Complete!")
    print(f"  Documents loaded:  {len(documents)}")
    print(f"  Chunks created:    {len(chunks)}")
    print(f"  Vectors stored:    {stats['total_chunks']}")
    print(f"  Embedding model:   {stats['embedding_model']}")
    print(f"  Embedding dim:     {stats['embedding_dim']}")
    print(f"  Storage dir:       {stats['persist_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
