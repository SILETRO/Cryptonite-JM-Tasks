import torch
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
    attention_weights = F.softmax(scores, dim=-1)
    
    # Attention * V
    output = torch.matmul(attention_weights, v)
    return output, attention_weights

if __name__ == "__main__":
    batch_size = 2
    seq_len = 4
    d_model = 8
    
    torch.manual_seed(42)
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)
    
    output, weights = scaled_dot_product_attention(q, k, v)
    print("Output shape:", output.shape)
    print("Attention weights shape:", weights.shape)
    print("\nAttention Weights (Batch 0):")
    print(weights[0])
