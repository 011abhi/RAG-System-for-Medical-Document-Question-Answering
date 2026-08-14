from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

def chunk(text, threshold=0.7):
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= 1:
        return sentences

    embeddings = model.encode(sentences)
    
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_similarity(
            [embeddings[i-1]],
            [embeddings[i]]
        )[0][0]

        if sim >= threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]

    chunks.append(" ".join(current_chunk))
    return chunks

if __name__ == "__main__":
    sample = "Mitochondria are organelles found in cells. They produce energy for the cell. This process is called cellular respiration. The weather today is sunny and warm. It might rain tomorrow."
    result = chunk(sample)
    print(f"Number of chunks: {len(result)}")
    for i, c in enumerate(result):
        print(f"Chunk {i+1}: {c}")
