import os
import glob
from typing import List, Dict, Any
from rag.vector_store import get_policy_collection

POLICIES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "policies")

def split_text_into_chunks(text: str, source_doc: str, chunk_size: int = 400, overlap: int = 50) -> List[Dict[str, Any]]:
    """Splits policy markdown text into coherent logical sections/clauses."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    current_title = source_doc

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if p.startswith("#"):
            current_title = p.lstrip("#").strip()

        if len(current_chunk) + len(p) > chunk_size and current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "section": current_title,
                "source": source_doc
            })
            current_chunk = p
        else:
            current_chunk = f"{current_chunk}\n\n{p}" if current_chunk else p

    if current_chunk:
        chunks.append({
            "text": current_chunk.strip(),
            "section": current_title,
            "source": source_doc
        })

    return chunks

def index_policy_documents(force: bool = False) -> int:
    """Reads all markdown policy documents and indexes them in ChromaDB."""
    collection = get_policy_collection()
    count = collection.count()
    if count > 0 and not force:
        return count

    policy_files = glob.glob(os.path.join(POLICIES_DIR, "*.md"))

    if not policy_files:
        print(f"No policy documents found in {POLICIES_DIR}")
        return 0

    documents = []
    metadatas = []
    ids = []
    total_chunks = 0

    for file_path in policy_files:
        filename = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = split_text_into_chunks(content, filename)
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{filename}_{idx}"
            documents.append(chunk["text"])
            metadatas.append({
                "source": chunk["source"],
                "section": chunk["section"]
            })
            ids.append(chunk_id)
            total_chunks += 1

    if ids:
        # Upsert documents so re-indexing is idempotent
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Successfully indexed {total_chunks} policy chunks into ChromaDB.")

    return total_chunks

def retrieve_policy_knowledge(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Retrieves the most semantically relevant policy clauses for a given query."""
    collection = get_policy_collection()
    count = collection.count()

    if count == 0:
        index_policy_documents()

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, max(1, collection.count()))
    )

    retrieved = []
    if results and "documents" in results and results["documents"]:
        docs = results["documents"][0]
        metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
        distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(docs)

        for doc, meta, dist in zip(docs, metas, distances):
            retrieved.append({
                "content": doc,
                "source": meta.get("source", "Unknown Policy"),
                "section": meta.get("section", "General"),
                "relevance_score": round(1.0 - float(dist), 3) if dist is not None else 1.0
            })

    return retrieved

if __name__ == "__main__":
    index_policy_documents()

