
import csv
import re
import os

def compute_grounding(justification, context):
    if not justification or not context:
        return 0.0
    just_words = set(re.findall(r'\b\w+\b', justification.lower()))
    ctx_words = set(re.findall(r'\b\w+\b', context.lower()))
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was',
                  'were', 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with',
                  'it', 'that', 'this', 'not', 'have', 'has', 'be', 'as'}
    just_words -= stop_words
    if not just_words:
        return 0.0
    return len(just_words.intersection(ctx_words)) / len(just_words)

os.makedirs('results/grounding', exist_ok=True)

processed = 0
for filename in sorted(os.listdir('results')):
    if not filename.endswith('.csv'):
        continue
    if 'summary' in filename or 'grounding' in filename or 'pairwise' in filename:
        continue
    if 'rq4' in filename or 'bertscore' in filename or 'mcnemar' in filename:
        continue

    input_path = f'results/{filename}'
    output_path = f'results/grounding/{filename}'

    rows = []
    with open(input_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames) + ['grounding_score']
        for row in reader:
            justification = row.get('raw_output', '')
            context = row.get('retrieved_text', '')
            row['grounding_score'] = f"{compute_grounding(justification, context):.4f}"
            rows.append(row)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    avg = sum(float(r['grounding_score']) for r in rows) / len(rows)
    print(f'{filename}: avg grounding = {avg:.4f}')
    processed += 1

print(f'\nDone. {processed}  results/grounding/')
