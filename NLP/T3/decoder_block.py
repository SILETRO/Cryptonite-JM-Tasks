import torch
import torch.nn as nn
from multihead import MultiHeadAttention

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),          # GELU: standard for GPT-style models
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ffn = FeedForward(d_model, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 1. Masked Multi-Head Attention & Residual Connection
        attn_out, _ = self.mha(self.norm1(x), mask)
        x = x + self.dropout1(attn_out)
        
        # 2. Feed Forward Network & Residual Connection
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_out)
        
        return x

if __name__ == "__main__":
    d_model = 128
    num_heads = 4
    seq_len = 10
    batch_size = 2
    
    block = DecoderBlock(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    
    from masking import create_causal_mask
    mask = create_causal_mask(seq_len)
    
    out = block(x, mask)
    print("Decoder Block Output shape:", out.shape)
