import math
import urllib.request
import os
from collections import defaultdict, Counter

class NgramLM:
    def __init__(self, n, k=2):
        self.n = n
        self.k = k
        self.ngrams = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocab = set()
        self.vocab_size = 0

    def get_ngrams(self, text):
        tokens = text.split()
        tokens = ['<START>'] * (self.n - 1) + tokens + ['<END>']
        return [tuple(tokens[i:i+self.n]) for i in range(len(tokens)-self.n+1)]

    def train(self, text):
        tokens = text.split()
        self.vocab.update(tokens)
        self.vocab.add('<START>')
        self.vocab.add('<END>')
        self.vocab_size = len(self.vocab)

        ngrams = self.get_ngrams(text)
        for ngram in ngrams:
            self.ngrams[ngram] += 1
            if self.n > 1:
                context = ngram[:-1]
                self.context_counts[context] += 1
            else:
                self.context_counts[()] += 1

    def get_prob(self, ngram):
        # Add-k Smoothing
        if self.n > 1:
            context = ngram[:-1]
            count = self.ngrams.get(ngram, 0)
            context_count = self.context_counts.get(context, 0)
        else:
            count = self.ngrams.get(ngram, 0)
            context_count = sum(self.ngrams.values())
            
        return (count + self.k) / (context_count + self.k * self.vocab_size)

    def calculate_perplexity(self, text):
        ngrams = self.get_ngrams(text)
        if len(ngrams) == 0:
            return float('inf')
            
        log_prob_sum = 0
        for ngram in ngrams:
            prob = self.get_prob(ngram)
            log_prob_sum += math.log2(prob)
            
        avg_log_prob = log_prob_sum / len(ngrams)
        perplexity = math.pow(2, -avg_log_prob)
        return perplexity

def download_wikitext2():
    urls = {
        'train': 'https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt',
        'valid': 'https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt',
        'test': 'https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/test.txt'
    }
    
    data = {}
    for split, url in urls.items():
        filepath = f"wikitext-2-{split}.txt"
        if not os.path.exists(filepath):
            print(f"Downloading {split} set...")
            urllib.request.urlretrieve(url, filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            data[split] = f.read()
    return data

if __name__ == "__main__":
    print("Loading WikiText-2 dataset...")
    data = download_wikitext2()
    
    train_text = data['train']
    valid_text = data['valid']
    
    print(f"Train size: {len(train_text.split())} words")
    print(f"Valid size: {len(valid_text.split())} words")
    
    for n in [1, 2, 3]:
        model = NgramLM(n=n, k=2)
        model.train(train_text)
        print("Calculating Perplexity")
        ppl = model.calculate_perplexity(valid_text)
        print(f"Validation Perplexity: {ppl:.4f}\n")
