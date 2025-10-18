"""Minimal implementation of the AlphaFold MSA transformer stack.

The goal of this module is to expose the main building blocks used in the
original paper while keeping the code compact enough for educational purposes.
The implementation supports batched inference and provides a clean API for
stacking multiple attention blocks operating on the MSA and pair
representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass
class AttentionMask:
    """Container representing which residues and sequences are valid."""

    msa: Tensor  # (batch, msa, residues)
    pair: Optional[Tensor] = None  # (batch, residues, residues)


class MSAColumnAttention(nn.Module):
    """Perform attention across the MSA dimension for each residue column."""

    def __init__(self, c_m: int, num_heads: int) -> None:
        super().__init__()
        if c_m % num_heads != 0:
            raise ValueError("c_m must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = c_m // num_heads

        self.q = nn.Linear(c_m, num_heads * self.head_dim)
        self.k = nn.Linear(c_m, num_heads * self.head_dim)
        self.v = nn.Linear(c_m, num_heads * self.head_dim)
        self.out = nn.Linear(num_heads * self.head_dim, c_m)

    def forward(self, msa: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        b, n_seq, n_res, _ = msa.shape
        q = self.q(msa).view(b, n_seq, n_res, self.num_heads, self.head_dim).permute(0, 3, 2, 1, 4)
        k = self.k(msa).view(b, n_seq, n_res, self.num_heads, self.head_dim).permute(0, 3, 2, 1, 4)
        v = self.v(msa).view(b, n_seq, n_res, self.num_heads, self.head_dim).permute(0, 3, 2, 1, 4)

        logits = torch.einsum("bhnic,bhnjc->bhnij", q, k) / (self.head_dim ** 0.5)
        if mask is not None:
            key_mask = mask.permute(0, 2, 1).unsqueeze(1).unsqueeze(-2)  # (b,1,n_res,1,n_seq)
            logits = logits.masked_fill(key_mask == 0, float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        out = torch.einsum("bhnij,bhnjc->bhnic", weights, v)
        out = out.permute(0, 3, 2, 1, 4).reshape(b, n_seq, n_res, self.num_heads * self.head_dim)
        return self.out(out)


class MSARowAttentionWithPairBias(nn.Module):
    """Attention along the residue axis with pair bias injection."""

    def __init__(self, c_m: int, c_z: int, num_heads: int) -> None:
        super().__init__()
        if c_m % num_heads != 0:
            raise ValueError("c_m must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = c_m // num_heads

        self.q = nn.Linear(c_m, num_heads * self.head_dim)
        self.k = nn.Linear(c_m, num_heads * self.head_dim)
        self.v = nn.Linear(c_m, num_heads * self.head_dim)
        self.out = nn.Linear(num_heads * self.head_dim, c_m)
        self.pair_bias = nn.Linear(c_z, num_heads)

    def forward(self, msa: Tensor, pair: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        b, n_seq, n_res, _ = msa.shape
        q = self.q(msa).view(b, n_seq, n_res, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        k = self.k(msa).view(b, n_seq, n_res, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        v = self.v(msa).view(b, n_seq, n_res, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        logits = torch.einsum("bhqic,bhqjc->bhqij", q, k) / (self.head_dim ** 0.5)
        logits = logits + self.pair_bias(pair).permute(0, 3, 1, 2).unsqueeze(2)

        if mask is not None:
            key_mask = mask.unsqueeze(1).unsqueeze(-2)  # (b,1,n_seq,1,n_res)
            logits = logits.masked_fill(key_mask == 0, float("-inf"))

        weights = torch.softmax(logits, dim=-1)
        out = torch.einsum("bhqij,bhqjc->bhqic", weights, v)
        out = out.permute(0, 2, 3, 1, 4).reshape(b, n_seq, n_res, self.num_heads * self.head_dim)
        return self.out(out)


class Transition(nn.Module):
    """Feed-forward network used after attention blocks."""

    def __init__(self, c: int, multiplier: int = 4) -> None:
        super().__init__()
        self.linear1 = nn.Linear(c, multiplier * c)
        self.linear2 = nn.Linear(multiplier * c, c)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.act(self.linear1(x)))


class MSATransformerBlock(nn.Module):
    """Single transformer block operating on the MSA and pair representations."""

    def __init__(self, c_m: int, c_z: int, num_heads: int) -> None:
        super().__init__()
        self.col_attention = MSAColumnAttention(c_m, num_heads)
        self.row_attention = MSARowAttentionWithPairBias(c_m, c_z, num_heads)
        self.transition = Transition(c_m)
        self.norm_msa = nn.LayerNorm(c_m)
        self.norm_pair = nn.LayerNorm(c_z)

    def forward(self, msa: Tensor, pair: Tensor, mask: AttentionMask) -> Tuple[Tensor, Tensor]:
        msa = msa + self.col_attention(self.norm_msa(msa), mask.msa)
        msa = msa + self.row_attention(self.norm_msa(msa), self.norm_pair(pair), mask.msa)
        msa = msa + self.transition(self.norm_msa(msa))
        return msa, pair


class MSATransformer(nn.Module):
    """Stack of transformer blocks processing the MSA/pair features."""

    def __init__(self, c_m: int, c_z: int, num_heads: int, num_blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [MSATransformerBlock(c_m=c_m, c_z=c_z, num_heads=num_heads) for _ in range(num_blocks)]
        )

    def forward(self, msa: Tensor, pair: Tensor, mask: Optional[AttentionMask] = None) -> Tuple[Tensor, Tensor]:
        if mask is None:
            mask = AttentionMask(msa=torch.ones(msa.shape[:3], device=msa.device, dtype=torch.bool))

        for block in self.blocks:
            msa, pair = block(msa, pair, mask)
        return msa, pair


__all__ = [
    "AttentionMask",
    "MSAColumnAttention",
    "MSARowAttentionWithPairBias",
    "Transition",
    "MSATransformerBlock",
    "MSATransformer",
]
