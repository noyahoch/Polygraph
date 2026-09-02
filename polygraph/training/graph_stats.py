"""Cheap hand-designed final-layer statistics from the existing sparse attention graph."""

from __future__ import annotations

from typing import List

import torch
from torch import Tensor

from ..data.graphs import node_coordinates


def final_layer_graph_statistics(edge_index: Tensor, edge_attr: Tensor, diagonal: Tensor,
                                 num_tokens: int, tau: float) -> Tensor:
    """Return 13 per-head and 5 cross-head features without deployment labels/metadata."""
    source, target = edge_index.long()
    heads = edge_attr.shape[1]
    coords = node_coordinates(num_tokens, 0, 1)[:, :2].to(edge_attr)
    patch_edge = (source > 0) & (target > 0)
    distance = (coords[source] - coords[target]).square().sum(-1).sqrt()
    per_head: List[Tensor] = []
    vectors = []
    for head in range(heads):
        weight = edge_attr[:, head].float()
        active = weight > tau
        values = weight[active]
        src, dst = source[active], target[active]
        mass = torch.zeros(num_tokens).scatter_add_(0, dst.cpu(), values.cpu())
        squared = torch.zeros(num_tokens).scatter_add_(0, dst.cpu(), values.square().cpu())
        out_degree = torch.bincount(src.cpu(), minlength=num_tokens).float()
        in_degree = torch.bincount(dst.cpu(), minlength=num_tokens).float()
        spatial = active & patch_edge
        spatial_weight = weight[spatial]
        weighted_distance = ((spatial_weight * distance[spatial]).sum() /
                             spatial_weight.sum().clamp_min(1e-12))
        per_head.extend([
            active.float().mean(), values.max() if len(values) else weight.new_tensor(0),
            values.mean() if len(values) else weight.new_tensor(0), mass.mean(), mass.std(unbiased=False),
            mass[0], weight[active & (source == 0)].sum(), squared.mean(),
            out_degree.std(unbiased=False), in_degree.std(unbiased=False), weighted_distance,
            diagonal[:, head].float().mean(), diagonal[0, head].float(),
        ])
        vectors.append(weight.masked_fill(~active, 0))
    matrix = torch.stack(vectors, 0)
    normalized = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cosine = normalized @ normalized.T
    upper = cosine[torch.triu_indices(heads, heads, offset=1).unbind()]
    active_heads = (matrix > tau).sum(0).float()
    dominance = matrix.max(0).values / matrix.sum(0).clamp_min(1e-12)
    cross = [upper.mean(), upper.std(unbiased=False), active_heads.mean(),
             active_heads.std(unbiased=False), dominance.mean()]
    return torch.stack([x.cpu() for x in per_head + cross]).float()
