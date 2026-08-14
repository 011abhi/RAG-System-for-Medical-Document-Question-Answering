import chromadb
from sentence_transformers import SentenceTransformer
import os

model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

def build_index(chunks, persist_path="indexes/chroma"):
    os.makedirs(persist_path, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_path)
    
    try:
        client.delete_collection("pubmedqa")
    except:
        pass
    
    collection = client.create_collection("pubmedqa")
    embeddings = model.encode(chunks).tolist()
    ids = [str(i) for i in range(len(chunks))]
    collection.add(embeddings=embeddings, documents=chunks, ids=ids)
    print(f"Dense index built with {len(chunks)} chunks")
    return collection

def load_index(persist_path="indexes/chroma"):
    client = chromadb.PersistentClient(path=persist_path)
    collection = client.get_collection("pubmedqa")
    return collection

def retrieve(query, collection, top_k=5):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return [
        {"text": doc, "score": round(1 / (1 + dist), 4)}
        for doc, dist in zip(results["documents"][0], results["distances"][0])
    ]

if __name__ == "__main__":
    sample_chunks = [
        "Mitochondria produce energy for the cell.",
        "The nucleus contains DNA and controls cell activity.",
        "Ribosomes are responsible for protein synthesis.",
        "The cell membrane controls what enters and exits the cell.",
        "Mitochondria are involved in programmed cell death.",
    ]
    collection = build_index(sample_chunks)
    results = retrieve("what do mitochondria do", collection)
    print("\nTop results:")
    for i, r in enumerate(results):
        print(f"{i+1}. {r['text']}  (score: {r['score']:.4f})")