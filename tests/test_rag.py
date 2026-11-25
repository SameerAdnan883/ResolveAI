import pytest
from rag.knowledge_base import split_text_into_chunks, index_policy_documents, retrieve_policy_knowledge

def test_text_chunking():
    sample_text = "# Header 1\n\nFirst paragraph here.\n\n# Header 2\n\nSecond paragraph here with policy."
    chunks = split_text_into_chunks(sample_text, "test_policy.md", chunk_size=50)
    assert len(chunks) >= 2
    assert chunks[0]["source"] == "test_policy.md"

def test_policy_indexing_and_retrieval():
    total_chunks = index_policy_documents()
    assert total_chunks > 0

    results = retrieve_policy_knowledge("return window 30 days refund", top_k=2)
    assert len(results) > 0
    assert "content" in results[0]
    assert "source" in results[0]

