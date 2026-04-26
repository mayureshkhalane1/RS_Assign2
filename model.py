"""SASRec: Self-Attentive Sequential Recommendation (Kang & McAuley, 2018)."""

import torch
import torch.nn as nn


class PointWiseFeedForward(nn.Module):
    """Position-wise two-layer MLP applied independently at each sequence position.

    The inner dimension is 4 * hidden_dim, matching the standard Transformer FFN
    expansion. GELU is used instead of ReLU for smoother gradient flow.

    Parameters
    ----------
    hidden_dim : int
        Input and output dimension H.
    dropout : float
        Dropout probability applied after the first linear layer.
    """

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        inner_dim = hidden_dim * 4
        self.fc1 = nn.Linear(hidden_dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class SASRecBlock(nn.Module):
    """One SASRec transformer block using pre-layer normalisation.

    Pre-LN applies LayerNorm before the attention and FFN sub-layers rather than
    after, which is more stable during training than the original Transformer's
    post-LN design.

    Parameters
    ----------
    hidden_dim : int
        Model dimension H. Must be divisible by ``num_heads``.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout probability used in both sub-layers.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
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

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one self-attention block with residual connections.

        Parameters
        ----------
        x : torch.Tensor
            Shape [B, L, H].
        padding_mask : torch.Tensor
            Bool tensor of shape [B, L]. ``True`` marks a padding position.
        causal_mask : torch.Tensor
            Bool tensor of shape [L, L]. ``True`` blocks a (query, key) pair.

        Returns
        -------
        torch.Tensor
            Shape [B, L, H].
        """
        h = self.attn_norm(x)
        attn_out, _ = self.attn(
            query=h,
            key=h,
            value=h,
            key_padding_mask=padding_mask,
            attn_mask=causal_mask,
            need_weights=False,
        )
        x = x + self.attn_dropout(attn_out)

        h = self.ffn_norm(x)
        x = x + self.ffn(h)
        return x


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation model.

    Item and positional embeddings are summed, then passed through stacked
    ``SASRecBlock`` layers. The hidden state at the final sequence position is
    used as the user representation, and items are scored by dot product with
    their embeddings.

    Parameters
    ----------
    num_items : int
        Number of unique items. Item 0 is reserved for padding.
    max_len : int
        Maximum sequence length L.
    hidden_dim : int
        Embedding and model dimension H.
    num_blocks : int
        Number of stacked attention blocks.
    num_heads : int
        Attention heads per block. Must divide ``hidden_dim``.
    dropout : float
        Dropout rate applied throughout.
    """

    def __init__(
        self,
        num_items: int,
        max_len: int,
        hidden_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.num_items = num_items
        self.max_len = max_len
        self.hidden_dim = hidden_dim

        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [SASRecBlock(hidden_dim, num_heads, dropout) for _ in range(num_blocks)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialise embeddings with small normal noise (std=0.02)."""
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].fill_(0.0)

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return an upper-triangular bool mask of shape [L, L].

        Entry [i, j] is ``True`` when j > i, blocking position i from attending
        to any future position j.
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Encode a batch of item sequences.

        Parameters
        ----------
        seq : torch.Tensor
            Long tensor of shape [B, L] containing item IDs (0 = padding).

        Returns
        -------
        torch.Tensor
            Shape [B, L, H] containing contextualised hidden states.
        """
        device = seq.device
        B, L = seq.size()

        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        x = self.item_embedding(seq) + self.pos_embedding(positions)
        x = self.emb_dropout(x)

        padding_mask = seq == 0
        causal_mask = self._make_causal_mask(L, device)

        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        for block in self.blocks:
            x = block(x, padding_mask=padding_mask, causal_mask=causal_mask)
            # Residual connections can reintroduce non-zero values at padding
            # positions; zero them out explicitly after each block.
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        x = self.final_norm(x)
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x

    def score_items(
        self,
        seq: torch.Tensor,
        item_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidate items against the last-position hidden state.

        Parameters
        ----------
        seq : torch.Tensor
            Shape [B, L].
        item_indices : torch.Tensor
            Shape [B] for a single candidate per sequence, or [B, K] for K candidates.

        Returns
        -------
        torch.Tensor
            Shape [B] or [B, K] of dot-product scores.
        """
        h = self.forward(seq)[:, -1, :]

        if item_indices.dim() == 1:
            return (h * self.item_embedding(item_indices)).sum(dim=-1)

        item_emb = self.item_embedding(item_indices)
        return (h.unsqueeze(1) * item_emb).sum(dim=-1)

    def training_logits(
        self,
        seq: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute token-level positive and negative logits for BCE training.

        Parameters
        ----------
        seq : torch.Tensor
            Shape [B, L]. Input item sequences.
        pos : torch.Tensor
            Shape [B, L]. Ground-truth next items at each position.
        neg : torch.Tensor
            Shape [B, L]. Sampled negative items at each position.

        Returns
        -------
        pos_logits : torch.Tensor
            Shape [B, L].
        neg_logits : torch.Tensor
            Shape [B, L].
        """
        h = self.forward(seq)
        pos_logits = (h * self.item_embedding(pos)).sum(dim=-1)
        neg_logits = (h * self.item_embedding(neg)).sum(dim=-1)
        return pos_logits, neg_logits
