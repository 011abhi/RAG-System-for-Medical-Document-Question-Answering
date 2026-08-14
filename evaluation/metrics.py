import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bert_score import score as bert_score_fn

def check_retrieval_hit(retrieved_docs, gold_contexts, threshold=0.5):#this function is used here to check wheather the retrival hit mentioned to gold conetxt  or not.
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

def exact_match(prediction, gold):
    return int(prediction.strip().lower() == gold.strip().lower())

def compute_bertscore(predictions, references):
    P, R, F1 = bert_score_fn(
        predictions,
        references,
        lang="en",
        model_type="distilbert-base-uncased",
        num_layers=5,
        verbose=False
    )
    return F1.tolist()

def compute_lexical(answer, contexts, question=""):
    context_text = " ".join(contexts).lower()
    question_lower = question.lower()
    
    question_words = set(question_lower.split()) - {"do", "does", "is", "are", "the", "a", "an", "of", "in", "to", "and"}
    context_words = set(context_text.split())
    
    if not question_words:
        return 0.0
    
    overlap = len(question_words & context_words)
    return round(overlap / len(question_words), 4)

if __name__ == "__main__":
    preds = ["yes", "no", "maybe"]
    golds = ["yes", "yes", "maybe"]

    for p, g in zip(preds, golds):
        print(f"EM: {exact_match(p, g)}")

    bs = compute_bertscore(preds, golds)
    print(f"BERTScores: {[round(s, 4) for s in bs]}")

    contexts = ["Mitochondria produce energy for the cell through cellular respiration."]
    question = "Do mitochondria produce energy?"
    faith = compute_lexical("yes", contexts, question)
    print(f"Lexcial_overlap: {faith}")