import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
from data.load_data import load_pubmedqa
from chunkers import fixed, sentence, semantic
from retrievers.bm25_retriever import build_index as bm25_build, retrieve as bm25_retrieve
from retrievers.dense_retriever import build_index as dense_build, retrieve as dense_retrieve
from retrievers.hybrid_rrf import retrieve as rrf_retrieve
from generation.generator import load_model, run

CHUNKERS = {
    "fixed":    fixed.chunk,
    "sentence": sentence.chunk,
    "semantic": semantic.chunk,
}

MODELS = ["mistral", "biomistral"]

def check_retrieval_hit(retrieved_docs, gold_contexts, threshold=0.5):
    retrieved_text = " ".join([d["text"] if isinstance(d, dict) else d for d in retrieved_docs]).lower()
    retrieved_words = set(retrieved_text.split())
    for gold_ctx in gold_contexts:
        gold_words = set(gold_ctx.lower().split())
        if not gold_words:
            continue
        overlap = len(retrieved_words & gold_words) / len(gold_words)
        if overlap >= threshold:
            return 1
    return 0

def run_condition(condition_id, chunker_name, retriever_name, model_name,
                  test_data, model, tokenizer, model_type):
    chunker = CHUNKERS[chunker_name]
    results = []

    all_chunks = []
    for instance in test_data:
        for ctx in instance["context"]["contexts"]:
            chunks = chunker(ctx)
            all_chunks.extend(chunks)

    bm25 = bm25_build(all_chunks)
    collection = dense_build(all_chunks)

    for instance in test_data:
        question = instance["question"]
        gold = instance["final_decision"]
        gold_contexts = instance["context"]["contexts"]

        if retriever_name == "bm25":
            docs = bm25_retrieve(question, bm25, all_chunks)
        elif retriever_name == "dense":
            docs = dense_retrieve(question, collection)
        elif retriever_name == "hybrid":
            docs = rrf_retrieve(question, bm25, all_chunks, collection)

        retrieval_hit = check_retrieval_hit(docs, gold_contexts)
        result = run(question, docs, model, tokenizer, model_type)
        correct = int(result["answer"] == gold)
        retrieved_text = " ".join([d["text"] if isinstance(d, dict) else d for d in docs])

        if retrieval_hit == 1 and correct == 1:
            failure_type = "success"
        elif retrieval_hit == 0 and correct == 0:
            failure_type = "retrieval_failure"
        elif retrieval_hit == 1 and correct == 0:
            failure_type = "generation_failure"
        else:
            failure_type = "lucky_guess"

        results.append({
            "condition_id": condition_id,
            "chunker": chunker_name,
            "retriever": retriever_name,
            "model": model_name,
            "question": question,
            "gold": gold,
            "answer": result["answer"],
            "correct": correct,
            "retrieval_hit": retrieval_hit,
            "failure_type": failure_type,
            "retrieved_text": retrieved_text,
            "raw_output": result["raw_output"]
        })

    return results

def save_results(results, path="results/runs.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = results[0].keys()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} rows to {path}")

if __name__ == "__main__":
    test_data, dev_data = load_pubmedqa()
    test_data = test_data[:300]

    model, tokenizer, model_type = load_model("biomistral")

    chunkers = ["fixed", "sentence", "semantic"]
    retrievers = ["bm25", "dense", "hybrid"]

    grid_summary = []

    for chunker_name in chunkers:
        for retriever_name in retrievers:
            print(f"\n=== Running: chunker={chunker_name}, retriever={retriever_name} ===")

            results = run_condition(
                condition_id=f"{chunker_name}_{retriever_name}",
                chunker_name=chunker_name,
                retriever_name=retriever_name,
                model_name="biomistral",
                test_data=test_data,
                model=model,
                tokenizer=tokenizer,
                model_type=model_type
            )

            save_results(results, path=f"results/grid_biomistral_{chunker_name}_{retriever_name}.csv")

            correct = sum(r["correct"] for r in results)
            retrieval_hits = sum(r["retrieval_hit"] for r in results)
            total = len(results)

            print(f"EM: {correct}/{total} ({100*correct/total:.1f}%)")
            print(f"Retrieval Hits: {retrieval_hits}/{total} ({100*retrieval_hits/total:.1f}%)")

            grid_summary.append({
                "chunker": chunker_name,
                "retriever": retriever_name,
                "em": correct,
                "em_pct": round(100*correct/total, 1),
                "retrieval_hits": retrieval_hits,
                "retrieval_pct": round(100*retrieval_hits/total, 1),
                "total": total
            })

    print("\n\n=== BIOMISTRAL GRID SUMMARY ===")
    with open("results/grid_summary_biomistral.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=grid_summary[0].keys())
        writer.writeheader()
        writer.writerows(grid_summary)

    for row in grid_summary:
        print(f"{row['chunker']:10} x {row['retriever']:8} -> EM: {row['em_pct']}%  Retrieval: {row['retrieval_pct']}%")

    print("\nSaved full BioMistral grid summary to results/grid_summary_biomistral.csv")