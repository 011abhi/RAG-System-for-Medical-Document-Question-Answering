import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def chunk(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

if __name__ == "__main__":
    sample = "Mitochondria are organelles found in cells. They produce energy for the cell. This process is called cellular respiration. The inner membrane contains many folds."
    result = chunk(sample)
    print(f"Number of chunks: {len(result)}")
    for i, s in enumerate(result):
        print(f"Chunk {i+1}: {s}")
