import torch
import torch.nn as nn
import torch.nn.functional as F


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return out


class SASRecBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = PointWiseFeedForward(hidden_dim, dropout)

    def forward(self, x, padding_mask, causal_mask):
        # x: [B, L, H]
        h = self.attn_norm(x)
        attn_out, _ = self.attn(
            query=h,
            key=h,
            value=h,
            key_padding_mask=padding_mask,   # True means ignore
            attn_mask=causal_mask,           # True means blocked
            need_weights=False,
        )
        x = x + self.attn_dropout(attn_out)

        h = self.ffn_norm(x)
        x = x + self.ffn(h)
        return x


class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_len: int,
        hidden_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.hidden_dim = hidden_dim

        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            SASRecBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

    def _make_causal_mask(self, seq_len: int, device):
        # True where future positions are blocked
        return torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(self, seq):
        """
        seq: [B, L]
        returns hidden states [B, L, H]
        """
        device = seq.device
        batch_size, seq_len = seq.size()

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.item_embedding(seq) + self.pos_embedding(positions)
        x = self.emb_dropout(x)

        padding_mask = (seq == 0)  # [B, L]
        causal_mask = self._make_causal_mask(seq_len, device)

        for block in self.blocks:
            x = block(x, padding_mask=padding_mask, causal_mask=causal_mask)
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        x = self.final_norm(x)
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x

    def score_items(self, seq, item_indices):
        """
        seq: [B, L]
        item_indices:
          - [B] for one candidate each
          - [B, K] for multiple candidates
        returns:
          - [B] or [B, K]
        """
        h = self.forward(seq)[:, -1, :]  # last position representation

        if item_indices.dim() == 1:
            item_emb = self.item_embedding(item_indices)  # [B, H]
            return torch.sum(h * item_emb, dim=-1)

        item_emb = self.item_embedding(item_indices)      # [B, K, H]
        scores = torch.sum(h.unsqueeze(1) * item_emb, dim=-1)
        return scores

    def training_logits(self, seq, pos, neg):
        """
        Token-level logits for BCE training.
        seq, pos, neg: [B, L]
        """
        h = self.forward(seq)                             # [B, L, H]
        pos_emb = self.item_embedding(pos)                # [B, L, H]
        neg_emb = self.item_embedding(neg)                # [B, L, H]

        pos_logits = torch.sum(h * pos_emb, dim=-1)       # [B, L]
        neg_logits = torch.sum(h * neg_emb, dim=-1)       # [B, L]
        return pos_logits, neg_logits