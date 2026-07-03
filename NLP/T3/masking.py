import torch
import matplotlib.pyplot as plt

def create_causal_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask

if __name__ == "__main__":
    seq_len = 10
    mask = create_causal_mask(seq_len)
    
    print("Causal Mask:")
    print(mask)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(mask.numpy(), cmap='gray')
    plt.title('Causal Attention Mask')
    plt.xlabel('Key Positions')
    plt.ylabel('Query Positions')
    plt.savefig('causal_mask.png')
    scores = torch.randn(seq_len, seq_len)
    masked_scores = scores.masked_fill(mask == 0, float('-inf'))
    print("\nMasked Scores Example (first 3x3):")
    print(masked_scores[:3, :3])
