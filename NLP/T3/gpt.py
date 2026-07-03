import torch
import torch.nn as nn
from positional_encoding import LearnedPositionalEmbedding
from decoder_block import DecoderBlock
from masking import create_causal_mask

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=6, context_length=256, dropout=0.1):
        super().__init__()
        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = LearnedPositionalEmbedding(d_model, max_len=context_length)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.shape

        # Ensure we don't exceed context length
        if seq_len > self.context_length:
            idx = idx[:, -self.context_length:]
            seq_len = self.context_length

        # Get embeddings
        x = self.token_embedding(idx)
        x = self.pos_embedding(x)
        x = self.dropout(x)

        # Create causal mask
        mask = create_causal_mask(seq_len).to(idx.device)

        # Pass through decoder blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final norm and projection
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.0):
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context to model's context window
            idx_cond = idx[:, -self.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(idx.shape[0]):
                    for token in set(idx[b].tolist()):
                        if logits[b, token] < 0:
                            logits[b, token] *= repetition_penalty
                        else:
                            logits[b, token] /= repetition_penalty

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_idx_to_remove = cumulative_probs - nn.functional.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_idx_to_remove] = float('-inf')
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            probs = nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

if __name__ == "__main__":
    vocab_size = 700
    model = MiniGPT(vocab_size)

    idx = torch.randint(0, vocab_size, (2, 256))
    targets = torch.randint(0, vocab_size, (2, 256))

    logits, loss = model(idx, targets)
    print("Logits shape:", logits.shape)
    print("Loss:", loss.item())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
