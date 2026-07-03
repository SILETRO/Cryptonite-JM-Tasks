import os
from collections import Counter
from bpe_tokenizer import BPETokenizer

def load_data(filepath, limit=None):
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return ""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
        if limit:
            text = text[:limit]
        return text

def evaluate_word_level(train_text, test_text):
    print("Word-Level Tokenization")
    # Training
    train_tokens = train_text.split()
    vocab = set(train_tokens)
    vocab_size = len(vocab)
    
    # Testing
    test_tokens = test_text.split()
    oov_count = sum(1 for token in test_tokens if token not in vocab)
    oov_rate = oov_count / len(test_tokens) if len(test_tokens) > 0 else 0
    
    # Sequence Length (assuming lines as sequences)
    lines = [line for line in test_text.split('\n') if line.strip()]
    avg_seq_len = sum(len(line.split()) for line in lines) / len(lines) if len(lines) > 0 else 0
    
    print(f"Vocabulary Size: {vocab_size}")
    print(f"OOV Rate: {oov_rate:.4%} ({oov_count}/{len(test_tokens)} tokens)")
    print(f"Average Sequence Length: {avg_seq_len:.2f} tokens/line\n")

def evaluate_char_level(train_text, test_text):
    print("Character-Level Tokenization")
    # Training
    train_chars = list(train_text)
    vocab = set(train_chars)
    vocab_size = len(vocab)
    
    # Testing
    test_chars = list(test_text)
    oov_count = sum(1 for char in test_chars if char not in vocab)
    oov_rate = oov_count / len(test_chars) if len(test_chars) > 0 else 0
    
    # Sequence Length
    lines = [line for line in test_text.split('\n') if line.strip()]
    avg_seq_len = sum(len(list(line)) for line in lines) / len(lines) if len(lines) > 0 else 0
    
    print(f"Vocabulary Size: {vocab_size}")
    print(f"OOV Rate: {oov_rate:.4%} ({oov_count}/{len(test_chars)} tokens)")
    print(f"Average Sequence Length: {avg_seq_len:.2f} tokens/line\n")

def evaluate_bpe(train_text, test_text, num_merges=1000):
    print(f"--- BPE Tokenization (merges={num_merges}) ---")
    # Use a small subset of training text for BPE to avoid excessive training time
    bpe_train_text = train_text[:50000]
    
    bpe = BPETokenizer(num_merges=num_merges)
    bpe.train(bpe_train_text)
    
    vocab_size = len(bpe.vocab)
    bpe_test_subset = test_text[:10000]
    
    encoded_test = bpe.encode(bpe_test_subset)
    lines = [line for line in bpe_test_subset.split('\n') if line.strip()]
    encoded_lines = [bpe.encode(line) for line in lines]
    
    avg_seq_len = sum(len(line) for line in encoded_lines) / len(lines) if len(lines) > 0 else 0

    train_chars = set(bpe_train_text)
    test_chars = set(bpe_test_subset)
    oov_chars = test_chars - train_chars
    oov_count = sum(1 for char in list(bpe_test_subset) if char in oov_chars)
    oov_rate = oov_count / len(encoded_test) if len(encoded_test) > 0 else 0
    
    print(f"Vocabulary Size: {vocab_size}")
    print(f"OOV Rate: {oov_rate:.4%} ")
    print(f"Average Sequence Length: {avg_seq_len:.2f} tokens/line\n")

if __name__ == "__main__":
    train_file = "wikitext-2-train.txt"
    test_file = "wikitext-2-test.txt"
    
    print("Loading data...")
    train_text = load_data(train_file)
    test_text = load_data(test_file)
    
    if not train_text or not test_text:
        exit(1)
        
    print(f"Loaded {len(train_text)} train chars, {len(test_text)} test chars\n")
    
    evaluate_word_level(train_text, test_text)
    evaluate_char_level(train_text, test_text)
    evaluate_bpe(train_text, test_text, num_merges=300)
