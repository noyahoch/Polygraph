"""Attention -> sparse graph. One rule: keep edge j->i iff max_h A[h,i,j] > tau.

Edges are stored sorted by descending strength (max over heads). Threshold edge sets are
nested in tau and a fixed top-K is the first K of the sorted order, so one extraction at a
low tau serves the whole threshold sweep and every top-K comparison as prefix slices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import List

import torch
from torch import Tensor


@dataclass(frozen=True)
class LayerGraph:
    edge_index: Tensor  # [2, E] int16; row 0 = key j (source), row 1 = query i (target)
    edge_attr: Tensor  # [E, H] float16; raw per-head attention
    strength: Tensor  # [E] float16, descending
    source_tau: float  # the threshold this edge set is complete down to

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def at_threshold(self, tau: float) -> "LayerGraph":
        """Prefix slice to a stricter threshold. Refuses a looser one: those edges were
        never captured, and returning the denser graph instead would be silently wrong."""
        if tau < self.source_tau:
            raise ValueError(f"tau={tau} is below this graph's threshold {self.source_tau}")
        keep = int(torch.searchsorted(-self.strength.float(), torch.tensor(-float(tau))))
        # The result's threshold is the NEW tau, else chained calls (0.02->0.05->0.03)
        # would silently return the 0.05 graph while claiming 0.03.
        return LayerGraph(self.edge_index[:, :keep], self.edge_attr[:keep], self.strength[:keep], float(tau))


@lru_cache(maxsize=64)
def node_coordinates(num_tokens: int, layer: int, layer_count: int) -> Tensor:
    """[T, 4] = patch row, patch col, CLS flag, layer position. Identical for every image,
    so regenerated on read and never stored."""
    coords = torch.zeros(num_tokens, 4)
    coords[0, 2] = 1.0  # token 0 is CLS
    coords[:, 3] = layer / max(layer_count - 1, 1)
    side = int(math.sqrt(num_tokens - 1))
    if side * side == num_tokens - 1:
        axis = torch.arange(side, dtype=torch.float32) / max(side - 1, 1)
        coords[1:, 0], coords[1:, 1] = axis.repeat_interleave(side), axis.repeat(side)
    return coords


class ThresholdGraphBuilder:
    def __init__(self, tau: float):
        assert tau > 0, f"tau must be positive, got {tau}"
        self.tau = float(tau)

    @property
    def name(self) -> str:
        return f"threshold_tau{self.tau}"

    def build(self, attention: Tensor) -> List[LayerGraph]:
        """attention: [B, H, T, T] indexed [batch, head, query, key]."""
        batch, heads, tokens, _ = attention.shape
        attention = attention.float()
        strength = attention.amax(1).masked_fill(
            torch.eye(tokens, dtype=torch.bool, device=attention.device), -torch.inf  # no self-edges
        )
        sorted_strength, positions = strength.reshape(batch, -1).sort(-1, descending=True)
        counts = (sorted_strength > self.tau).sum(-1)
        flat = attention.permute(0, 2, 3, 1).reshape(batch, tokens * tokens, heads)

        graphs = []
        for item in range(batch):
            keep = int(counts[item])
            pos = positions[item, :keep]
            edge_index = torch.stack([pos.remainder(tokens), pos.div(tokens, rounding_mode="floor")], 0)
            graphs.append(LayerGraph(edge_index.to("cpu", torch.int16),
                                     flat[item, pos].to("cpu", torch.float16),
                                     sorted_strength[item, :keep].to("cpu", torch.float16), self.tau))
        return graphs

    @staticmethod
    def diagonals(attention: Tensor) -> Tensor:
        """[B, T, H] per-token self-attention, used as node features."""
        return attention.diagonal(dim1=-2, dim2=-1).transpose(1, 2)
