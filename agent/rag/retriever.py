"""
retriever.py
Hybrid retriever: Dense (ChromaDB) + BM25, merged via Reciprocal Rank Fusion.
Returns top-k chunks with citations.
"""

import os
import json
import pickle
from sentence_transformers import SentenceTransformer
import chromadb
from rank_bm25 import BM25Okapi

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_store")
BM25_PATH = os.path.join(os.path.dirname(__file__), "bm25_index.pkl")
CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "chunks.json")
EMBED_MODEL = "all-MiniLM-L6-v2"

_model = None
_chroma_collection = None
_bm25_data = None


def _load_resources():
    global _model, _chroma_collection, _bm25_data
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)

    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _chroma_collection = client.get_collection("it_policies")

    if _bm25_data is None:
        with open(BM25_PATH, "rb") as f:
            _bm25_data = pickle.load(f)


def _dense_search(query: str, k: int = 10) -> list[dict]:
    """Semantic search via ChromaDB embeddings."""
    _load_resources()
    query_embedding = _model.encode([query]).tolist()
    results = _chroma_collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "section": results["metadatas"][0][i]["section"],
            "score": 1 - results["distances"][0][i],  # ChromaDB returns distances
        })
    return hits


def _bm25_search(query: str, k: int = 10) -> list[dict]:
    """Keyword search via BM25."""
    _load_resources()
    bm25 = _bm25_data["bm25"]
    ids = _bm25_data["ids"]
    texts = _bm25_data["texts"]
    metadatas = _bm25_data["metadatas"]

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Get top-k indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    hits = []
    for idx in top_indices:
        if scores[idx] > 0:
            hits.append({
                "id": ids[idx],
                "text": texts[idx],
                "source": metadatas[idx]["source"],
                "section": metadatas[idx]["section"],
                "score": float(scores[idx]),
            })
    return hits


def _reciprocal_rank_fusion(dense_hits: list[dict], bm25_hits: list[dict], k_rrf: int = 60) -> list[dict]:
    """
    Reciprocal Rank Fusion (RRF) to merge two ranked lists.
    RRF(d) = sum over all lists: 1 / (k + rank(d))
    """
    scores = {}
    doc_data = {}

    for rank, hit in enumerate(dense_hits):
        doc_id = hit["id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k_rrf + rank + 1)
        doc_data[doc_id] = hit

    for rank, hit in enumerate(bm25_hits):
        doc_id = hit["id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k_rrf + rank + 1)
        doc_data[doc_id] = hit

    # Sort by fused score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for doc_id, fused_score in ranked:
        entry = doc_data[doc_id].copy()
        entry["rrf_score"] = round(fused_score, 4)
        results.append(entry)

    return results


def retrieve(query: str, k: int = 5, mode: str = "hybrid") -> list[dict]:
    """
    Retrieve top-k relevant policy chunks.

    Args:
        query: Natural language query
        k: Number of results to return
        mode: "hybrid" (dense+BM25+RRF), "dense" (embeddings only), "bm25" (keywords only)

    Returns:
        List of dicts with keys: text, source, section, score, citation
    """
    if mode == "dense":
        results = _dense_search(query, k=k)
    elif mode == "bm25":
        results = _bm25_search(query, k=k)
    else:  # hybrid
        dense_hits = _dense_search(query, k=k * 2)
        bm25_hits = _bm25_search(query, k=k * 2)
        results = _reciprocal_rank_fusion(dense_hits, bm25_hits)

    results = results[:k]
    for r in results:
        section = f" section {r['section']}" if r.get("section") else ""
        r["citation"] = f"[source: {r['source']}{section}]"

    return results


def format_context(results: list[dict]) -> str:
    """Format retrieved chunks into a context string for the LLM prompt."""
    if not results:
        return "No relevant policy documents found."

    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"--- Policy Reference {i} {r['citation']} ---\n{r['text']}")
    return "\n\n".join(parts)


if __name__ == "__main__":
    test_queries = [
        "What is the minimum password length according to NIST SP 800-63B?",
        "How often should privileged account passwords be rotated?",
        "What is the required retention period for audit logs?",
        "What approval process is required before installing unapproved software?",
        "How should vulnerabilities be prioritized and remediated?",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = retrieve(q, k=3)
        for r in results:
            print(f"  [{r.get('rrf_score', r['score']):.4f}] {r['citation']}")
            print(f"    {r['text'][:120]}...")
