"""
retrieval_eval.py
Offline retrieval evaluation on the golden question set.

Computes Recall@5, Recall@10, MRR@10, and nDCG@10 at document level for
each retrieval mode. Relevance is binary: a chunk counts as relevant when
it comes from the expected source document, so Recall@K here means "the
right document appears in the top K". Runs entirely offline, no LLM calls.

Usage:
    python eval/retrieval_eval.py
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'agent')))

from rag.retriever import retrieve

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_retrieval.json")
CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "..", "agent", "rag", "chunks.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "retrieval_eval.json")
MODES = ["bm25", "dense", "hybrid"]
K = 10


def chunk_counts_by_source() -> dict:
    """Total chunks per source document, for ideal-DCG normalization."""
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    counts = {}
    for c in chunks:
        counts[c["source"]] = counts.get(c["source"], 0) + 1
    return counts


def evaluate_mode(mode: str, queries: list[dict], chunk_counts: dict) -> dict:
    per_query = []
    for q in queries:
        hits = retrieve(q["query"], k=K, mode=mode)
        rels = [1 if h["source"] == q["expected_source"] else 0 for h in hits]
        first_hit_rank = next((i + 1 for i, r in enumerate(rels) if r), None)
        dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels) if r)
        relevant_total = min(chunk_counts.get(q["expected_source"], 1), K)
        idcg = sum(1 / math.log2(i + 2) for i in range(relevant_total))
        per_query.append({
            "query": q["query"],
            "expected_source": q["expected_source"],
            "top_sources": [h["source"] for h in hits[:5]],
            "hit@5": int(any(rels[:5])),
            "hit@10": int(any(rels)),
            "reciprocal_rank": 1 / first_hit_rank if first_hit_rank else 0.0,
            "ndcg@10": round(dcg / idcg, 3) if idcg else 0.0,
        })

    n = len(per_query)
    return {
        "mode": mode,
        "queries": n,
        "recall@5": round(sum(p["hit@5"] for p in per_query) / n, 3),
        "recall@10": round(sum(p["hit@10"] for p in per_query) / n, 3),
        "mrr@10": round(sum(p["reciprocal_rank"] for p in per_query) / n, 3),
        "ndcg@10": round(sum(p["ndcg@10"] for p in per_query) / n, 3),
        "per_query": per_query,
    }


def main():
    with open(GOLDEN_PATH, "r") as f:
        queries = json.load(f)

    chunk_counts = chunk_counts_by_source()
    results = [evaluate_mode(mode, queries, chunk_counts) for mode in MODES]

    print(f"\nRETRIEVAL EVALUATION ({len(queries)} golden queries, doc-level relevance)")
    print(f"{'Mode':<10} {'Recall@5':>9} {'Recall@10':>10} {'MRR@10':>8} {'nDCG@10':>9}")
    for r in results:
        print(f"{r['mode']:<10} {r['recall@5']:>9} {r['recall@10']:>10} {r['mrr@10']:>8} {r['ndcg@10']:>9}")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPer-query details saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
