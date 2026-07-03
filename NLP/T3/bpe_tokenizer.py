import re
from collections import Counter, defaultdict

class BPETokenizer:
    def __init__(self, num_merges=10):
        self.num_merges = num_merges
        self.vocab = {}
        self.merges = {}
        self.stoi = {}
        self.itos = {}
    
    def get_stats(self, vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs

    def merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    def train(self, text):
        # Initial character-level vocab
        words = text.strip().split()
        self.vocab = Counter([' '.join(list(w)) + ' </w>' for w in words])
        
        for i in range(self.num_merges):
            pairs = self.get_stats(self.vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.vocab = self.merge_vocab(best, self.vocab)
            self.merges[best] = ''.join(best)
            
        self.build_vocab()
            
    def build_vocab(self):
        tokens = set()
        for word in self.vocab.keys():
            tokens.update(word.split())
        # Add special unknown token
        tokens.add('<unk>')
        self.itos = {i: t for i, t in enumerate(sorted(list(tokens)))}
        self.stoi = {t: i for i, t in self.itos.items()}
        self.merge_patterns = []
        for pair, merged in self.merges.items():
            bigram = re.escape(' '.join(pair))
            p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
            self.merge_patterns.append((p, merged))

    def encode(self, text):
        words = text.strip().split()
        encoded = []
        patterns = getattr(self, 'merge_patterns', None)
        if patterns is None:
            self.build_vocab()
            patterns = self.merge_patterns
        for word in words:
            word = ' '.join(list(word)) + ' </w>'
            for p, merged in patterns:
                word = p.sub(merged, word)
            encoded.extend(word.split())
        return encoded

    def encode_ids(self, text):
        tokens = self.encode(text)
        unk_id = self.stoi.get('<unk>', 0)
        return [self.stoi.get(t, unk_id) for t in tokens]

    def decode_ids(self, ids):
        tokens = [self.itos.get(i, '') for i in ids]
        text = "".join(tokens).replace('</w>', ' ').strip()
        
        # Clean up Wikitext-2 specific formatting tokens
        text = text.replace(' @-@ ', '-')
        text = text.replace(' @,@ ', ',')
        text = text.replace(' @.@ ', '.')
        text = text.replace('<unk>', '')
        
        text = re.sub(r'\s+', ' ', text).strip()
        return text

if __name__ == "__main__":
    import os
    import torch

    # Locate training data
    train_file = next(
        (f for f in ['wikitext-2-train.txt', 'train.txt'] if os.path.exists(f)),
        None
    )
    if train_file is None:
        print("Error: No training file found.")
        exit(1)

    text = open(train_file, 'r', encoding='utf-8').read()

    print(f"Training BPE on {len(text):,} chars with 7000 merges...")

    bpe = BPETokenizer(num_merges=7000)
    bpe.train(text)

    vocab_size = len(bpe.stoi)
    print(f"Vocab size: {vocab_size}")

    output_path = 'vocab.bin'
    torch.save({'bpe': bpe}, output_path)
    print(f"Saved tokenizer to '{output_path}'. Done!")
