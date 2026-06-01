import pytest
from rag.chunker import DocumentChunker

def test_document_chunking():
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    text = "A" * 200
    chunks = chunker.chunk_text(text, source_meta={"title": "Test"})
    
    assert len(chunks) > 1
    assert chunks[0]["text"] == "A" * 100
    assert chunks[1]["text"].startswith("A" * 20)
    assert chunks[0]["metadata"]["title"] == "Test"
