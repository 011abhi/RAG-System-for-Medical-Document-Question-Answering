import re

def build_prompt(question, retrieved_docs):
    context = "\n\n".join([f"[{i+1}] {doc['text']}" for i, doc in enumerate(retrieved_docs)])

    prompt = f"""You are a biomedical expert. Answer the question based only on the provided context.
First state your answer as exactly one word: Yes, No, or Maybe.
Then add a brief one-sentence justification citing the specific evidence from the context.

Context:
{context}

Question: {question}

Answer:"""

    return prompt


def build_no_rag_prompt(question):
    prompt = f"""You are a biomedical expert. Answer the following question using your own medical knowledge.
First state your answer as exactly one word: Yes, No, or Maybe.
Then add a brief one-sentence justification for your answer.

Question: {question}

Answer:"""
    return prompt


def extract_answer(raw_output):
    text = raw_output.strip().lower()
    first_line = text.split('.')[0].split(',')[0].split('\n')[0]
    if re.search(r'\byes\b', first_line):
        return "yes"
    elif re.search(r'\bno\b', first_line):
        return "no"
    elif re.search(r'\bmaybe\b', first_line):
        return "maybe"
    elif re.search(r'\byes\b', text):
        return "yes"
    elif re.search(r'\bno\b', text):
        return "no"
    elif re.search(r'\bmaybe\b', text):
        return "maybe"
    else:
        return "unknown"


if __name__ == "__main__":
    sample_docs = [
        {"text": "Mitochondria produce energy for the cell.", "score": 0.9},
        {"text": "Mitochondria are involved in programmed cell death.", "score": 0.8},
    ]
    question = "Do mitochondria play a role in programmed cell death?"
    prompt = build_prompt(question, sample_docs)
    print(prompt)

    no_rag_prompt = build_no_rag_prompt(question)
    print("\n--- No-RAG prompt ---")
    print(no_rag_prompt)

    print("\n--- Testing answer extractor with justification text ---")
    test_cases = [
        "Yes, because mitochondria are described as initiating programmed cell death.",
        "No. The context does not support this claim, as it only discusses cellular respiration.",
        "Maybe, the evidence is unclear regarding direct causation.",
    ]
    for t in test_cases:
        print(f"Input: '{t}'")
        print(f"  -> Extracted: '{extract_answer(t)}'\n")