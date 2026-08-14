import sys
import os
from generation.prompt import build_prompt, build_no_rag_prompt, extract_answer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_model(model_name):
    if model_name == "mistral":
        from mlx_lm import load
        print("Loading Mistral-7B...")
        model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
        print("Mistral loaded.")
        return model, tokenizer, "mlx"

    elif model_name == "biomistral":
        from llama_cpp import Llama
        print("Loading BioMistral-7B...")
        model = Llama(
        model_path=os.path.expanduser("~/biomistral-gguf/BioMistral-7B.Q4_K_M.gguf"),
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False)
        
        print("BioMistral loaded.")
        return model, None, "gguf"

def run(question, retrieved_docs, model, tokenizer, model_type, max_tokens=80):
    if len(retrieved_docs) == 0:
        prompt = build_no_rag_prompt(question)
    else:
        prompt = build_prompt(question, retrieved_docs)

    if model_type == "mlx":
        from mlx_lm import generate
        raw_output = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)

    elif model_type == "gguf":
        output = model(prompt, max_tokens=max_tokens, temperature=0)
        raw_output = output["choices"][0]["text"]

    answer = extract_answer(raw_output)
    return {
        "question": question,
        "raw_output": raw_output,
        "answer": answer
    }

if __name__ == "__main__":
    sample_docs = [
        {"text": "Mitochondria produce energy for the cell.", "score": 0.9},
        {"text": "Mitochondria are involved in programmed cell death.", "score": 0.8},
    ]
    question = "Do mitochondria play a role in programmed cell death?"

    # Test Mistral
    model, tokenizer, model_type = load_model("mistral")
    result = run(question, sample_docs, model, tokenizer, model_type)
    print(f"Mistral answer: {result['answer']}")

    # Test BioMistral
    model, tokenizer, model_type = load_model("biomistral")
    result = run(question, sample_docs, model, tokenizer, model_type)
    print(f"BioMistral answer: {result['answer']}")