from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def chunk(text, chunk_size=256, stride=128):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        if end >= len(tokens):
            break
        start += stride
    return chunks

if __name__ == "__main__":
    sample = "This is a test sentence about mitochondria. " * 50
    result = chunk(sample)
    print(f"Number of chunks: {len(result)}")
    print(f"First chunk preview: {result[0][:100]}")
