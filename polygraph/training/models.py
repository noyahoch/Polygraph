"""Detector architectures.

GNNEncoder, ReadoutModel, and SequenceConcatModel are verbatim from the POC
(legacy/lightweight_attention_experiments.py) — copied rather than imported so the package
is self-contained; the legacy file is frozen in git (commit 8036fd9), so the reference
cannot drift. ReadoutModel is the single-layer detector (Stage 3 winner:
readout="cls_gated"); SequenceConcatModel encodes several layer graphs with one shared GNN
and concatenates the per-layer embeddings. EdgeSetModel is a later addition (not POC): the
structure-blind Deep-Sets control, selected with readout="edge_set".
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import (GATv2Conv, GINEConv, TransformerConv, global_add_pool,
                                 global_max_pool, global_mean_pool)
from torch_geometric.utils import softmax


class GNNEncoder(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * layers
        self.layers = nn.ModuleList(
            [TransformerConv(dims[i], dims[i + 1], heads=2, concat=False, edge_dim=edge_dim) for i in range(layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        for layer, norm in zip(self.layers, self.norms):
            x = self.dropout(torch.relu(norm(layer(x, edge_index, edge_attr))))
        return x


class ReadoutModel(nn.Module):
    """Single-layer detector: GNN encoder + a choice of graph readout + MLP head."""

    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int, layers: int, dropout: float, readout: str):
        super().__init__()
        if readout not in {"mean", "cls", "cls_mean", "cls_mean_max", "cls_gated"}:
            raise ValueError(f"Unknown readout: {readout}")
        self.readout = readout
        self.encoder = GNNEncoder(in_dim, edge_dim, hidden_dim, layers, dropout)
        decoder_dim = hidden_dim * {"mean": 1, "cls": 1, "cls_mean": 2, "cls_gated": 2, "cls_mean_max": 3}[readout]
        if readout == "cls_gated":
            self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.decoder = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(decoder_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: Data) -> Tuple[Tensor, Tensor]:
        raw_x = batch.x
        x = self.encoder(raw_x, batch.edge_index, batch.edge_attr)
        graph_count = int(batch.batch.max().item()) + 1 if batch.batch.numel() else 0
        cls_nodes = x[raw_x[:, 2] > 0.5]  # column 2 is the CLS indicator
        if cls_nodes.shape[0] != graph_count:
            raise RuntimeError(f"Expected one CLS node per graph, got {cls_nodes.shape[0]} for {graph_count}.")
        mean_nodes = global_mean_pool(x, batch.batch)
        if self.readout == "mean":
            embedding = mean_nodes
        elif self.readout == "cls":
            embedding = cls_nodes
        elif self.readout == "cls_mean":
            embedding = torch.cat([cls_nodes, mean_nodes], dim=1)
        elif self.readout == "cls_mean_max":
            embedding = torch.cat([cls_nodes, mean_nodes, global_max_pool(x, batch.batch)], dim=1)
        else:  # cls_gated: learned per-node scores, graph-wise softmax, weighted sum
            weights = softmax(self.gate(x).view(-1), batch.batch)
            embedding = torch.cat([cls_nodes, global_add_pool(x * weights.unsqueeze(1), batch.batch)], dim=1)
        return self.decoder(embedding).view(-1), embedding


class SequenceConcatModel(nn.Module):
    """Multi-layer detector: shared GNN over per-layer graphs, embeddings concatenated in order."""

    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int, layers: int, dropout: float, layer_count: int):
        super().__init__()
        self.layer_count = layer_count
        self.encoder = GNNEncoder(in_dim, edge_dim, hidden_dim, layers, dropout)
        self.decoder = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_dim * layer_count, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: Data) -> Tuple[Tensor, Tensor]:
        x = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        graph_count = int(batch.batch.max().item()) + 1 if batch.batch.numel() else 0
        layer_graph_id = batch.batch * self.layer_count + batch.layer_id
        layer_embeddings = global_mean_pool(x, layer_graph_id, size=graph_count * self.layer_count)
        ordered = layer_embeddings.view(graph_count, self.layer_count, -1).reshape(graph_count, -1)
        return self.decoder(ordered).view(-1), ordered


class EdgeSetModel(nn.Module):
    """Deep-Sets control (NOT from the POC): the GNN's full edge-feature set with no
    structure. Each edge contributes only its per-head attention vector; per-graph
    mean+max pooling ignores which tokens an edge connects, so the model is invariant
    to rewiring by construction — the airtight version of the flat-attention baseline
    (attn_mlp sees only the top-100 edges; this sees everything the GNN sees)."""

    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim))
        self.rho = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, data: Data) -> Tuple[Tensor, None]:
        from torch_geometric.utils import scatter

        graph_of_edge = data.batch[data.edge_index[0]]
        h = self.phi(data.edge_attr)
        pooled = torch.cat([scatter(h, graph_of_edge, dim=0, dim_size=data.num_graphs, reduce="mean"),
                            scatter(h, graph_of_edge, dim=0, dim_size=data.num_graphs, reduce="max")], dim=-1)
        return self.rho(pooled).view(-1), None


class NodeEdgeSetModel(nn.Module):
    """Matched multiset control: all node/edge values, but no edge endpoints."""

    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.node_phi = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim))
        self.edge_phi = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim))
        self.rho = nn.Sequential(nn.Linear(4 * hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, data: Data) -> Tuple[Tensor, None]:
        from torch_geometric.utils import scatter

        nodes = self.node_phi(data.x)
        edges = self.edge_phi(data.edge_attr)
        graph_of_edge = data.batch[data.edge_index[0]]
        pooled = torch.cat([
            scatter(nodes, data.batch, dim=0, dim_size=data.num_graphs, reduce="mean"),
            scatter(nodes, data.batch, dim=0, dim_size=data.num_graphs, reduce="max"),
            scatter(edges, graph_of_edge, dim=0, dim_size=data.num_graphs, reduce="mean"),
            scatter(edges, graph_of_edge, dim=0, dim_size=data.num_graphs, reduce="max"),
        ], dim=-1)
        return self.rho(pooled).view(-1), None


class EndpointSetModel(nn.Module):
    """Non-message-passing set of [source node, target node, edge] records."""

    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(2 * in_dim + edge_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim))
        self.rho = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, data: Data) -> Tuple[Tensor, None]:
        from torch_geometric.utils import scatter

        source, target = data.edge_index
        records = self.phi(torch.cat([data.x[source], data.x[target], data.edge_attr], dim=-1))
        graph_of_edge = data.batch[source]
        pooled = torch.cat([
            scatter(records, graph_of_edge, dim=0, dim_size=data.num_graphs, reduce="mean"),
            scatter(records, graph_of_edge, dim=0, dim_size=data.num_graphs, reduce="max"),
        ], dim=-1)
        return self.rho(pooled).view(-1), None


def _cls_and_pool(x: Tensor, raw_x: Tensor, batch: Tensor, gate: nn.Module) -> Tensor:
    """Direct CLS access plus learned permutation-invariant global pooling."""
    graph_count = int(batch.max().item()) + 1 if batch.numel() else 0
    cls = x[raw_x[:, 2] > 0.5]
    if cls.shape[0] != graph_count:
        raise RuntimeError(f"Expected one CLS node per graph, got {cls.shape[0]} for {graph_count}")
    weights = softmax(gate(x).view(-1), batch)
    return torch.cat([cls, global_add_pool(x * weights.unsqueeze(1), batch)], dim=1)


class HiddenTokenSetModel(nn.Module):
    """S0: all M5 node values, no edges or endpoint association."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.node_phi = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.rho = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        h = self.node_phi(data.x)
        direct = _cls_and_pool(h, data.x, data.batch, self.gate)
        embedding = torch.cat([direct, global_mean_pool(h, data.batch),
                               global_max_pool(h, data.batch)], dim=1)
        return self.rho(embedding).view(-1), embedding


class M5NodeEdgeSetModel(nn.Module):
    """S1: full node/edge multisets with direct CLS access, but no endpoints."""

    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.node_phi = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.edge_phi = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        # CLS, gated nodes, node mean/max, edge mean/max, log edge count.
        self.rho = nn.Sequential(nn.Linear(6 * hidden_dim + 1, hidden_dim), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        from torch_geometric.utils import scatter

        nodes, edges = self.node_phi(data.x), self.edge_phi(data.edge_attr)
        graph_of_edge = data.batch[data.edge_index[0]]
        count = scatter(torch.ones_like(graph_of_edge, dtype=nodes.dtype), graph_of_edge,
                        dim=0, dim_size=data.num_graphs, reduce="sum").log1p().unsqueeze(1)
        embedding = torch.cat([
            _cls_and_pool(nodes, data.x, data.batch, self.gate),
            global_mean_pool(nodes, data.batch), global_max_pool(nodes, data.batch),
            scatter(edges, graph_of_edge, dim=0, dim_size=data.num_graphs, reduce="mean"),
            scatter(edges, graph_of_edge, dim=0, dim_size=data.num_graphs, reduce="max"), count], dim=1)
        return self.rho(embedding).view(-1), embedding


class FactoredEndpointAffine(nn.Module):
    """Affine([x_source,x_target,e]) without materialising repeated endpoint tensors."""

    def __init__(self, node_dim: int, edge_dim: int, out_dim: int):
        super().__init__()
        self.source = nn.Linear(node_dim, out_dim, bias=False)
        self.target = nn.Linear(node_dim, out_dim, bias=False)
        self.edge = nn.Linear(edge_dim, out_dim, bias=True)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        source, target = edge_index
        # Project once per node; only compact width-dimensional values are gathered.
        return self.source(x)[source] + self.target(x)[target] + self.edge(edge_attr)


class M5EndpointSetModel(nn.Module):
    """S2: endpoint-local records plus an independent node branch; no message passing."""

    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.node_phi = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.record_affine = FactoredEndpointAffine(in_dim, edge_dim, hidden_dim)
        self.record_tail = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        # CLS, gated nodes, node mean/max, record mean/max, log edge count.
        self.rho = nn.Sequential(nn.Linear(6 * hidden_dim + 1, hidden_dim), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        from torch_geometric.utils import scatter

        nodes = self.node_phi(data.x)
        records = self.record_tail(self.record_affine(data.x, data.edge_index, data.edge_attr))
        graph_of_edge = data.batch[data.edge_index[0]]
        count = scatter(torch.ones_like(graph_of_edge, dtype=nodes.dtype), graph_of_edge,
                        dim=0, dim_size=data.num_graphs, reduce="sum").log1p().unsqueeze(1)
        embedding = torch.cat([
            _cls_and_pool(nodes, data.x, data.batch, self.gate),
            global_mean_pool(nodes, data.batch), global_max_pool(nodes, data.batch),
            scatter(records, graph_of_edge, dim=0, dim_size=data.num_graphs, reduce="mean"),
            scatter(records, graph_of_edge, dim=0, dim_size=data.num_graphs, reduce="max"), count], dim=1)
        return self.rho(embedding).view(-1), embedding


class EdgeGatedMeanBlock(nn.Module):
    """Efficient edge-aware gated mean block with an explicit residual/root path."""

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float):
        super().__init__()
        self.msg_source, self.msg_edge = nn.Linear(hidden_dim, hidden_dim), nn.Linear(edge_dim, hidden_dim)
        self.gate_source = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate_target = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate_edge = nn.Linear(edge_dim, hidden_dim)
        self.update = nn.Sequential(nn.Linear(2 * hidden_dim + 1, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim))
        self.norm, self.dropout = nn.LayerNorm(hidden_dim), nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        from torch_geometric.utils import scatter

        source, target = edge_index
        msg = self.msg_source(x)[source] + self.msg_edge(edge_attr)
        gate = torch.sigmoid(self.gate_source(x)[source] + self.gate_target(x)[target]
                             + self.gate_edge(edge_attr))
        numerator = scatter(gate * msg, target, dim=0, dim_size=x.shape[0], reduce="sum")
        denominator = scatter(gate, target, dim=0, dim_size=x.shape[0], reduce="sum").clamp_min(1e-8)
        aggregate = numerator / denominator
        degree = scatter(torch.ones_like(target, dtype=x.dtype), target, dim=0,
                         dim_size=x.shape[0], reduce="sum").log1p().unsqueeze(1)
        return torch.relu(self.norm(x + self.dropout(self.update(torch.cat([x, aggregate, degree], 1)))))


class ResidualGraphModel(nn.Module):
    """Common residual scaffold for the edge-aware architecture comparison."""

    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int, layers: int,
                 dropout: float, family: str, jumping_knowledge: bool = False):
        super().__init__()
        self.family, self.jumping_knowledge = family, jumping_knowledge
        self.input = nn.Linear(in_dim, hidden_dim)
        self.edge_input = nn.Linear(edge_dim, hidden_dim)
        blocks = []
        for _ in range(layers):
            if family == "transformerconv_residual":
                blocks.append(TransformerConv(hidden_dim, hidden_dim, heads=2, concat=False,
                                              edge_dim=hidden_dim, root_weight=False))
            elif family == "gine":
                mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim))
                blocks.append(GINEConv(mlp, eps=0.0, train_eps=True, edge_dim=hidden_dim))
            elif family == "edge_gated_mean":
                blocks.append(EdgeGatedMeanBlock(hidden_dim, hidden_dim, dropout))
            elif family == "gatv2":
                blocks.append(GATv2Conv(hidden_dim, hidden_dim, heads=2, concat=False,
                                       edge_dim=hidden_dim, add_self_loops=False, residual=False))
            else:
                raise ValueError(f"unknown residual graph family: {family}")
        self.blocks = nn.ModuleList(blocks)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in blocks])
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.decoder = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def encode(self, data: Data) -> Tensor:
        x, edge = self.input(data.x), self.edge_input(data.edge_attr)
        states = []
        for block, norm in zip(self.blocks, self.norms):
            if self.family == "edge_gated_mean":
                x = block(x, data.edge_index, edge)
            else:
                x = torch.relu(norm(x + self.dropout(block(x, data.edge_index, edge))))
            states.append(x)
        return torch.stack(states).amax(0) if self.jumping_knowledge and len(states) > 1 else x

    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        x = self.encode(data)
        embedding = _cls_and_pool(x, data.x, data.batch, self.gate)
        return self.decoder(embedding).view(-1), embedding


class SimpleMPNN(nn.Module):
    """Two-layer explicit mean-message MPNN with no learned attention coefficient."""

    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int, layers: int,
                 dropout: float, readout: str = "cls_gated"):
        super().__init__()
        if readout != "cls_gated":
            raise ValueError("SimpleMPNN currently supports cls_gated readout only")
        self.input = nn.Linear(in_dim, hidden_dim)
        self.messages = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden_dim + edge_dim, hidden_dim), nn.ReLU(),
                          nn.Linear(hidden_dim, hidden_dim)) for _ in range(layers)])
        self.updates = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden_dim + 1, hidden_dim), nn.ReLU(),
                          nn.Linear(hidden_dim, hidden_dim)) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.decoder = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden_dim, hidden_dim),
                                     nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def encode(self, data: Data) -> Tensor:
        from torch_geometric.utils import scatter

        x = self.input(data.x)
        source, target = data.edge_index
        for message, update, norm in zip(self.messages, self.updates, self.norms):
            m = message(torch.cat([x[source], x[target], data.edge_attr], dim=-1))
            mean = scatter(m, target, dim=0, dim_size=x.shape[0], reduce="mean")
            degree = scatter(torch.ones_like(target, dtype=x.dtype), target, dim=0,
                             dim_size=x.shape[0], reduce="sum").log1p().unsqueeze(1)
            x = self.dropout(torch.relu(norm(x + update(torch.cat([x, mean, degree], dim=-1)))))
        return x

    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        raw_x, x = data.x, self.encode(data)
        graph_count = data.num_graphs
        cls = x[raw_x[:, 2] > 0.5]
        if cls.shape[0] != graph_count:
            raise RuntimeError(f"Expected one CLS node per graph, got {cls.shape[0]}")
        weights = softmax(self.gate(x).view(-1), data.batch)
        embedding = torch.cat([cls, global_add_pool(x * weights.unsqueeze(1), data.batch)], dim=1)
        return self.decoder(embedding).view(-1), embedding


class TemporalMPNN(SimpleMPNN):
    """SimpleMPNN over layer-token nodes; reads final-layer CLS plus all-node gated pool."""

    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        raw_x, x = data.x, self.encode(data)
        final_layer = int(data.layer_id.max())
        cls = x[(raw_x[:, 2] > 0.5) & (data.layer_id == final_layer)]
        if cls.shape[0] != data.num_graphs:
            raise RuntimeError(f"Expected one final-layer CLS per graph, got {cls.shape[0]}")
        weights = softmax(self.gate(x).view(-1), data.batch)
        embedding = torch.cat([cls, global_add_pool(x * weights.unsqueeze(1), data.batch)], dim=1)
        return self.decoder(embedding).view(-1), embedding


class TCPMultiTaskModel(nn.Module):
    """Adds a training-only TCP regression head without changing inference output."""

    def __init__(self, base: nn.Module, embedding_dim: int):
        super().__init__()
        self.base = base
        self.tcp_head = nn.Linear(embedding_dim, 1)

    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        return self.base(data)

    def forward_multitask(self, data: Data) -> Tuple[Tensor, Tensor, Tensor]:
        error_logit, embedding = self.base(data)
        if embedding is None:
            raise RuntimeError("TCP multitask requires a model with a graph embedding")
        return error_logit, self.tcp_head(embedding).view(-1), embedding
