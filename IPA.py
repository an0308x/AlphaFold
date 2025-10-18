"""Invariant point attention module used in AlphaFold.

This file contains a lightweight PyTorch re-implementation of the module
introduced in the AlphaFold model.  The goal of this implementation is to
provide a well documented reference that is easy to read and adapt for
educational experiments.  The code intentionally mirrors the notation used in
`Jumper et al. (2021)` and keeps the shapes explicit throughout the forward
pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass
class Rigid:
    """Simple rigid body transform consisting of a rotation matrix and translation.

    The original AlphaFold implementation uses quaternions to represent
    rotations.  For clarity we use a rotation matrix here.  The class contains a
    handful of helper utilities that mimic the behaviour of the AlphaFold
    `geometry` module while keeping the API compact.
    """

    rotation: Tensor  # (..., 3, 3)
    translation: Tensor  # (..., 3)

    def compose(self, other: "Rigid") -> "Rigid":
        """Compose this transform with another one."""

        rotation = self.rotation @ other.rotation
        translation = self.apply(other.translation)
        return Rigid(rotation=rotation, translation=translation)

    def apply(self, points: Tensor) -> Tensor:
        """Apply the rigid transform to a set of 3D points."""

        return torch.einsum("...ij,...j->...i", self.rotation, points) + self.translation

    def inverse(self) -> "Rigid":
        """Return the inverse transform."""

        rotation = self.rotation.transpose(-1, -2)
        translation = -torch.einsum("...ij,...j->...i", rotation, self.translation)
        return Rigid(rotation=rotation, translation=translation)


class InvariantPointAttention(nn.Module):
    """Implementation of the Invariant Point Attention (IPA) block.

    The module attends over both scalar features and 3D point features while
    staying equivariant to rigid-body transformations of the input coordinates.
    This is achieved by constructing queries and keys for the scalar part and
    by comparing relative pairwise distances for the point part.

    Parameters
    ----------
    c_q : int
        Dimensionality of the query representation.
    c_kv : int
        Dimensionality of the key/value representations.
    num_heads : int
        Number of attention heads.
    num_scalar_qk : int
        Number of scalar query/key channels per head.
    num_scalar_v : int
        Number of scalar value channels per head.
    num_point_qk : int
        Number of query/key points per head (each point is a 3-vector).
    num_point_v : int
        Number of value points per head.
    pair_bias : bool, optional
        If ``True`` the forward pass accepts a pair representation that will be
        added as an attention bias.
    """

    def __init__(
        self,
        c_q: int,
        c_kv: int,
        num_heads: int,
        num_scalar_qk: int,
        num_scalar_v: int,
        num_point_qk: int,
        num_point_v: int,
        pair_bias: bool = True,
    ) -> None:
        super().__init__()
        self.c_q = c_q
        self.c_kv = c_kv
        self.num_heads = num_heads
        self.num_scalar_qk = num_scalar_qk
        self.num_scalar_v = num_scalar_v
        self.num_point_qk = num_point_qk
        self.num_point_v = num_point_v
        self.pair_bias = pair_bias

        def linear(in_dim: int, out_dim: int) -> nn.Linear:
            layer = nn.Linear(in_dim, out_dim)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            return layer

        self.query_proj = linear(c_q, num_heads * num_scalar_qk)
        self.key_proj = linear(c_kv, num_heads * num_scalar_qk)
        self.value_proj = linear(c_kv, num_heads * num_scalar_v)

        # Projections for the point attention component.
        self.query_points_proj = linear(c_q, num_heads * num_point_qk * 3)
        self.key_points_proj = linear(c_kv, num_heads * num_point_qk * 3)
        self.value_points_proj = linear(c_kv, num_heads * num_point_v * 3)

        self.out_proj = linear(num_heads * (num_scalar_v + num_point_v * 3), c_q)

        if pair_bias:
            self.pair_bias_proj = linear(c_kv, num_heads)

    def _reshape_heads(self, x: Tensor, num_channels: int) -> Tensor:
        """Reshape ``(batch, length, num_heads * num_channels)`` -> ``(batch, num_heads, length, num_channels)``."""

        batch, length, _ = x.shape
        x = x.view(batch, length, self.num_heads, num_channels)
        return x.permute(0, 2, 1, 3)

    def _point_distribution(self, x: Tensor, num_points: int) -> Tensor:
        """Reshape a flattened point tensor into ``(..., num_points, 3)``."""

        batch, length, _ = x.shape
        x = x.view(batch, length, self.num_heads, num_points, 3)
        return x.permute(0, 2, 1, 3, 4)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_frame: Rigid,
        value_frame: Optional[Rigid] = None,
        pair_bias: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Run a single IPA attention operation.

        Parameters
        ----------
        query, key, value : Tensor
            Input representations with shape ``(batch, length, channels)``.
        query_frame : Rigid
            Reference frame for the query residues.
        value_frame : Rigid, optional
            Frame describing the coordinates of the values.  If omitted the
            query frame is used for both queries and values.  This matches the
            behaviour of the structure module in AlphaFold.
        pair_bias : Tensor, optional
            Pair embedding with shape ``(batch, length, length, c_pair)`` that
            will be projected to attention logits when ``pair_bias=True``.
        mask : Tensor, optional
            Binary mask of shape ``(batch, length)`` indicating valid residues.

        Returns
        -------
        updated_repr : Tensor
            Updated representation with shape ``(batch, length, c_q)``.
        updated_points : Tensor
            Updated point representation with shape
            ``(batch, length, num_heads * num_point_v, 3)`` expressed in the
            query frame.
        """

        if value_frame is None:
            value_frame = query_frame

        q = self._reshape_heads(self.query_proj(query), self.num_scalar_qk)
        k = self._reshape_heads(self.key_proj(key), self.num_scalar_qk)
        v = self._reshape_heads(self.value_proj(value), self.num_scalar_v)

        q_points = self._point_distribution(self.query_points_proj(query), self.num_point_qk)
        k_points = self._point_distribution(self.key_points_proj(key), self.num_point_qk)
        v_points = self._point_distribution(self.value_points_proj(value), self.num_point_v)

        # Transform points into global frame before computing distances.
        global_q_points = query_frame.apply(q_points)
        global_k_points = value_frame.apply(k_points)

        # Compute squared distances between query and key points.  The average
        # over the point dimension is used to build the invariant attention bias.
        point_dist = (global_q_points.unsqueeze(-3) - global_k_points.unsqueeze(-4)).pow(2).sum(-1)
        point_bias = -point_dist.mean(dim=-1)  # negative distance promotes closer matches

        logits = torch.einsum("bhqd,bhkd->bhqk", q, k) / q.size(-1) ** 0.5
        logits = logits + point_bias

        if self.pair_bias and pair_bias is not None:
            logits = logits + self.pair_bias_proj(pair_bias).permute(0, 3, 1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            logits = logits.masked_fill(~mask, float("-inf"))

        weights = torch.softmax(logits, dim=-1)

        scalar_out = torch.einsum("bhqk,bhkd->bhqd", weights, v)

        # Rotate value points into global frame, apply attention, then transform back.
        global_v_points = value_frame.apply(v_points)
        weighted_points = torch.einsum("bhqk,bhkpd->bhqpd", weights, global_v_points)
        local_points = query_frame.inverse().apply(weighted_points)

        scalar_out = scalar_out.permute(0, 2, 1, 3).reshape(query.shape[0], query.shape[1], -1)
        point_out = local_points.permute(0, 2, 1, 3, 4).reshape(query.shape[0], query.shape[1], -1)

        out = torch.cat([scalar_out, point_out], dim=-1)
        updated_repr = self.out_proj(out)
        updated_points = point_out.view(query.shape[0], query.shape[1], self.num_heads * self.num_point_v, 3)
        return updated_repr, updated_points


__all__ = ["InvariantPointAttention", "Rigid"]
