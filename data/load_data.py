from datasets import load_dataset

def load_pubmedqa():
    print("Downloading PubMedQA...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", trust_remote_code=True)
    
    data = list(dataset["train"])
    
    test_data = data[:800]
    dev_data  = data[800:]
    
    print(f"Total: {len(data)}")
    print(f"Test:  {len(test_data)}")
    print(f"Dev:   {len(dev_data)}")
    
    example = data[0]
    print("\n--- Example ---")
    print("Question:", example["question"])
    print("Answer:  ", example["final_decision"])
    print("Context snippets:", len(example["context"]["contexts"]))
    
    return test_data, dev_data

if __name__ == "__main__":
    test_data, dev_data = load_pubmedqa()
