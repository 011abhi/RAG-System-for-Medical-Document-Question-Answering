import csv
import sys
import os

sys.path.insert(0, os.path.abspath('.'))

from data.load_data import load_pubmedqa
from generation.generator import load_model, run

# Load BioMistral
print("Loading BioMistral...")
model, tokenizer, model_type = load_model('biomistral')
print("Model loaded.")

all_data = load_pubmedqa()
if isinstance(all_data, tuple):
    for split in all_data:
        if len(split) == 800:
            test_data = split[:300]
            break
else:
    test_data = all_data['test'][:300]

print(f"Test instances: {len(test_data)}")

results = []
correct = 0
framing_failure = 0

os.makedirs('results', exist_ok=True)

for i, instance in enumerate(test_data):
    question = instance['question']
    gold = instance['final_decision']

    result = run(question, [], model, tokenizer, model_type, max_tokens=80)
    raw_output = result['raw_output']
    answer = result['answer']

    if not answer:
        answer = 'maybe'

    is_correct = int(answer.lower() == gold.lower())
    correct += is_correct

    text = raw_output.lower()
    framing = int('context' in text or 'the text' in text or 'the passage' in text)
    framing_failure += framing

    results.append({
        'question': question,
        'gold': gold,
        'answer': answer,
        'raw_output': raw_output,
        'correct': is_correct,
        'framing_failure': framing
    })

    if (i+1) % 10 == 0:
        em = 100 * correct / (i+1)
        ff = 100 * framing_failure / (i+1)
        print(f'Instance {i+1}/300 — EM: {em:.1f}% — Framing failure: {ff:.1f}%')
        sys.stdout.flush()

        with open('results/no_rag_biomistral.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

print(f'DONE — Final EM: {100*correct/300:.1f}% — Framing failure: {100*framing_failure/300:.1f}%')

with open('results/no_rag_biomistral.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print('Saved to results/no_rag_biomistral.csv')