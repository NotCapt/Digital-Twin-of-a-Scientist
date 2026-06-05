import pytest
from rag.chunker import TextChunker

def test_document_chunking():
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)
    doc = {
        "text": "word " * 20,
        "metadata": {"title": "Test"},
        "doc_id": "test_doc"
    }
    chunks = chunker.chunk_document(doc)
    
    assert len(chunks) > 1
    assert chunks[0]["metadata"]["title"] == "Test"

