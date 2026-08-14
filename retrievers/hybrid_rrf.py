import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrievers.bm25_retriever import build_index, retrieve as bm25_retrieve
from retrievers.dense_retriever import build_index as dense_build, retrieve as dense_retrieve

def rrf_score(rank, k=60):
    return 1 / (k + rank)

def retrieve(query, bm25, chunks, collection, top_k=5, k=60):
    bm25_results = bm25_retrieve(query, bm25, chunks, top_k=len(chunks))
    dense_results = dense_retrieve(query, collection, top_k=len(chunks))

    scores = {}

    for rank, result in enumerate(bm25_results):
        text = result["text"]
        scores[text] = scores.get(text, 0) + rrf_score(rank + 1, k)

    for rank, result in enumerate(dense_results):
        text = result["text"]
        scores[text] = scores.get(text, 0) + rrf_score(rank + 1, k)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [
        {"text": text, "score": round(score, 6)}
        for text, score in ranked[:top_k]
    ]

if __name__ == "__main__":
    sample_chunks = [
        "Mitochondria produce energy for the cell.",
        "The nucleus contains DNA and controls cell activity.",
        "Ribosomes are responsible for protein synthesis.",
        "The cell membrane controls what enters and exits the cell.",
        "Mitochondria are involved in programmed cell death.",
    ]

    bm25 = build_index(sample_chunks)
    collection = dense_build(sample_chunks)

    results = retrieve("what do mitochondria do", bm25, sample_chunks, collection)
    print("\nHybrid RRF Top results:")
    for i, r in enumerate(results):
        print(f"{i+1}. {r['text']}  (score: {r['score']})")