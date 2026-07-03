import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        return x + self.embedding(positions)

if __name__ == "__main__":
    d_model = 128
    seq_len = 50
    
    # 1. Sinusoidal
    sin_pe = SinusoidalPositionalEncoding(d_model)
    x_dummy = torch.zeros(1, seq_len, d_model)
    out_sin = sin_pe(x_dummy)
    
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(out_sin[0].numpy(), cmap='viridis')
    plt.title('Sinusoidal Positional Encoding')
    plt.xlabel('Embedding Dimensions')
    plt.ylabel('Sequence Position')
    plt.colorbar()
    plt.savefig('sinusoidal_pe.png')
    print("Saved sinusoidal_pe.png")
    
    # 2. Learned
    learned_pe = LearnedPositionalEmbedding(d_model)
    out_learned = learned_pe(x_dummy)
    
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(out_learned[0].detach().numpy(), cmap='viridis')
    plt.title('Learned Positional Embedding (Untrained)')
    plt.xlabel('Embedding Dimensions')
    plt.ylabel('Sequence Position')
    plt.colorbar()
    plt.savefig('learned_pe_untrained.png')
    print("Saved learned_pe_untrained.png")
