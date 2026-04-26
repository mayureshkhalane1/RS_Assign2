"""
model.py — SASRec: Self-Attentive Sequential Recommendation (PyTorch).

Architecture (per block)
------------------------
    x  →  LayerNorm  →  Multi-Head Causal Self-Attention  →  Dropout  →  + x
    x  →  LayerNorm  →  Point-Wise FFN (hidden → 4·hidden → hidden)  →  + x

The final sequence representation is the hidden state at the *last* position,
which under the causal mask has attended to all preceding items.

Prediction: score = h_{last} · item_embedding(candidate)   (dot product)

Reference: Kang & McAuley (2018) — https://arxiv.org/abs/1808.09781
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointWiseFeedForward(nn.Module):
    """
    Position-wise two-layer MLP with inner expansion.

    The standard Transformer FFN expands to 4 × hidden_dim then projects back.
    This gives the network more representational capacity than a square
    projection, at the cost of more parameters.

    Shape: [B, L, H] → [B, L, 4H] → [B, L, H]
    """

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        inner_dim = hidden_dim * 4          # BUG FIX: was hidden→hidden (no expansion)
        self.fc1     = nn.Linear(hidden_dim, inner_dim)
        self.fc2     = nn.Linear(inner_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act     = nn.GELU()            # GELU smoother than ReLU for transformers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class SASRecBlock(nn.Module):
    """
    One SASRec (Transformer-style) block with pre-layer-normalisation.

    Pre-LN (normalise before attention/FFN) is more stable during training
    than the original post-LN used in vanilla Transformers.

    Parameters
    ----------
    hidden_dim : int
        Embedding / model dimension H.
    num_heads : int
        Number of attention heads.  Must divide hidden_dim evenly.
    dropout : float
        Dropout probability applied after attention weights and inside FFN.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm    = nn.LayerNorm(hidden_dim)
        self.attn         = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn      = PointWiseFeedForward(hidden_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x            : [B, L, H]
        padding_mask : [B, L]   bool, True → this position is padding (ignore)
        causal_mask  : [L, L]   bool, True → this (query, key) pair is blocked

        Returns
        -------
        x            : [B, L, H]  updated hidden states
        """
        # --- Self-attention with pre-LN and residual ---
        h = self.attn_norm(x)
        attn_out, _ = self.attn(
            query=h,
            key=h,
            value=h,
            key_padding_mask=padding_mask,   # True = ignore this key position
            attn_mask=causal_mask,            # True = block this (q, k) pair
            need_weights=False,
        )
        x = x + self.attn_dropout(attn_out)

        # --- Feed-forward with pre-LN and residual ---
        h = self.ffn_norm(x)
        x = x + self.ffn(h)
        return x


class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation model.

    Parameters
    ----------
    num_items  : int    Number of unique items (padding idx = 0).
    max_len    : int    Maximum sequence length.
    hidden_dim : int    Embedding and model dimension H.
    num_blocks : int    Number of stacked SASRecBlock layers.
    num_heads  : int    Attention heads per block (must divide hidden_dim).
    dropout    : float  Dropout rate used throughout.
    """

    def __init__(
        self,
        num_items:  int,
        max_len:    int,
        hidden_dim: int   = 64,
        num_blocks: int   = 2,
        num_heads:  int   = 2,
        dropout:    float = 0.2,
    ) -> None:
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.num_items  = num_items
        self.max_len    = max_len
        self.hidden_dim = hidden_dim

        # Item embedding: 0 is the padding index → its gradient is zeroed
        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        # Learned absolute positional embeddings (one per position in [0, max_len))
        self.pos_embedding  = nn.Embedding(max_len, hidden_dim)
        self.emb_dropout    = nn.Dropout(dropout)

        self.blocks     = nn.ModuleList([
            SASRecBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Small normal init — prevents exploding values at the start."""
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight,  std=0.02)
        # Ensure padding embedding stays zero after init
        with torch.no_grad():
            self.item_embedding.weight[0].fill_(0.0)

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Upper-triangular boolean mask of shape [L, L].

        Entry [i, j] is True when j > i, meaning position i cannot attend to
        position j (future item).  This enforces the auto-regressive property:
        the representation at step t only depends on items 1 … t.
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of item sequences.

        Parameters
        ----------
        seq : [B, L]  long tensor of item IDs (0 = padding)

        Returns
        -------
        x   : [B, L, H]  contextualised hidden states
        """
        device = seq.device
        B, L   = seq.size()

        positions    = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        x            = self.item_embedding(seq) + self.pos_embedding(positions)
        x            = self.emb_dropout(x)

        padding_mask = (seq == 0)            # [B, L]  True = padding position
        causal_mask  = self._make_causal_mask(L, device)

        # BUG FIX: zero out padding positions before the first block so
        # positional embeddings don't pollute padding slots
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        for block in self.blocks:
            x = block(x, padding_mask=padding_mask, causal_mask=causal_mask)
            # Re-zero padding after each block (residuals can reintroduce values)
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        x = self.final_norm(x)
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x

    def score_items(
        self,
        seq: torch.Tensor,
        item_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score one or multiple candidate items for each sequence in the batch.

        Parameters
        ----------
        seq          : [B, L]    input sequence
        item_indices : [B]       → returns [B]
                       [B, K]    → returns [B, K]

        The score is a dot product between the last-position hidden state and
        the candidate item embedding:  score = h_{last} · e_{item}
        """
        h = self.forward(seq)[:, -1, :]          # [B, H]  last position

        if item_indices.dim() == 1:
            item_emb = self.item_embedding(item_indices)          # [B, H]
            return (h * item_emb).sum(dim=-1)                     # [B]

        item_emb = self.item_embedding(item_indices)              # [B, K, H]
        return (h.unsqueeze(1) * item_emb).sum(dim=-1)           # [B, K]

    def training_logits(
        self,
        seq: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute token-level logits for BCE training.

        At every non-padding position t, the model predicts whether the next
        item is the positive (pos[t]) or a sampled negative (neg[t]).

        Parameters
        ----------
        seq, pos, neg : [B, L]

        Returns
        -------
        pos_logits : [B, L]   logits for positive items
        neg_logits : [B, L]   logits for negative items
        """
        h = self.forward(seq)                             # [B, L, H]
        pos_emb = self.item_embedding(pos)                # [B, L, H]
        neg_emb = self.item_embedding(neg)                # [B, L, H]

        pos_logits = (h * pos_emb).sum(dim=-1)            # [B, L]
        neg_logits = (h * neg_emb).sum(dim=-1)            # [B, L]
        return pos_logits, neg_logits
