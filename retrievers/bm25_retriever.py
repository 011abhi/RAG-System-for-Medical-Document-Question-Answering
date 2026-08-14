from rank_bm25 import BM25Okapi
import numpy as np
import pickle
import os
import re

def tokenize(text):
    return re.findall(r"\w+", text.lower())

def build_index(chunks):
    tokenized = [tokenize(c) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25

def save_index(bm25, chunks, path="indexes/bm25/index.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    print(f"BM25 index saved to {path}")

def load_index(path="indexes/bm25/index.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunks"]

def retrieve(query, bm25, chunks, top_k=5):
    if not query.strip():
        return []
    scores = bm25.get_scores(tokenize(query))
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {"text": chunks[i], "score": float(scores[i])}
        for i in top_indices
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
    save_index(bm25, sample_chunks)
    results = retrieve("what do mitochondria do", bm25, sample_chunks)
    print("\nTop results:")
    for i, r in enumerate(results):
        print(f"{i+1}. {r['text']}  (score: {r['score']:.4f})")