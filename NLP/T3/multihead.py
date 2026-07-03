import torch
import torch.nn as nn
from attention import scaled_dot_product_attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # 1. Linear projections
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        # 2. Split into heads: (batch_size, num_heads, seq_len, d_k)
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(0)
            
        output, attn_weights = scaled_dot_product_attention(q, k, v, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.out_linear(output)
        return output, attn_weights

if __name__ == "__main__":
    d_model = 128
    num_heads = 4
    seq_len = 10
    batch_size = 2
    
    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Optional causal mask
    from masking import create_causal_mask
    mask = create_causal_mask(seq_len)
    
    out, weights = mha(x, mask)
    print("MHA Output shape:", out.shape)
    print("MHA Weights shape:", weights.shape)
