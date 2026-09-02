"""Mutation-oriented tests for matched set controls and rewiring."""

from __future__ import annotations

import tempfile
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polygraph.data.storage import append_temporal_identity_edges, rewire_graph
from polygraph.training.models import (EdgeSetModel, EndpointSetModel, NodeEdgeSetModel,
                                       SimpleMPNN, TemporalMPNN)
from polygraph.training.train import TrainConfig, build_model, load_checkpoint


def graph(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(8, 16, generator=g)
    x[:, 2] = 0
    x[0, 2] = 1
    return Data(x=x,
                edge_index=torch.randint(0, 8, (2, 35), generator=g),
                edge_attr=torch.randn(35, 12, generator=g), y=torch.tensor([seed % 2.0]))


def batched(items):
    return Batch.from_data_list(items)


def test_set_control_invariances_and_endpoint_sensitivity():
    torch.manual_seed(3)
    items = [graph(i) for i in range(3)]
    batch = batched(items)
    rewired = []
    reordered = []
    for i, item in enumerate(items):
        ei, ea = rewire_graph(item.edge_index, item.edge_attr, "target_permute", i)
        rewired.append(Data(x=item.x, edge_index=ei, edge_attr=ea, y=item.y))
        p = torch.randperm(item.edge_attr.shape[0])
        reordered.append(Data(x=item.x, edge_index=item.edge_index[:, p],
                              edge_attr=item.edge_attr[p], y=item.y))
    edge = EdgeSetModel(16, 12, 32, 0).eval()
    node_edge = NodeEdgeSetModel(16, 12, 32, 0).eval()
    endpoint = EndpointSetModel(16, 12, 32, 0).eval()
    assert torch.allclose(edge(batch)[0], edge(batched(rewired))[0], atol=1e-6)
    assert torch.allclose(node_edge(batch)[0], node_edge(batched(rewired))[0], atol=1e-6)
    assert torch.allclose(endpoint(batch)[0], endpoint(batched(reordered))[0], atol=1e-6)
    assert not torch.allclose(endpoint(batch)[0], endpoint(batched(rewired))[0], atol=1e-7)


def test_all_controls_one_logit_and_backpropagate():
    batch = batched([graph(i) for i in range(4)])
    for model in (EdgeSetModel(16, 12, 32, 0), NodeEdgeSetModel(16, 12, 32, 0),
                  EndpointSetModel(16, 12, 32, 0), SimpleMPNN(16, 12, 32, 2, 0)):
        out, _ = model(batch)
        assert out.shape == (4,)
        out.square().mean().backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_rewiring_is_deterministic_and_preserves_required_multisets():
    item = graph(9)
    a_i, a_e = rewire_graph(item.edge_index, item.edge_attr, "target_permute", 41, 7)
    b_i, b_e = rewire_graph(item.edge_index, item.edge_attr, "target_permute", 41, 7)
    assert torch.equal(a_i, b_i) and torch.equal(a_e, b_e)
    assert torch.equal(a_i[0], item.edge_index[0])
    assert torch.equal(a_i[1].sort().values, item.edge_index[1].sort().values)
    assert torch.equal(a_e, item.edge_attr)
    s_i, s_e = rewire_graph(item.edge_index, item.edge_attr, "shuffle_attr", 41, 7)
    assert torch.equal(s_i, item.edge_index)
    # Lexicographic row multiset fingerprint; catches accidental feature modification.
    assert sorted(map(tuple, s_e.tolist())) == sorted(map(tuple, item.edge_attr.tolist()))


def test_old_checkpoint_loads_with_new_config_defaults():
    old = {"layers": [11], "readout": "cls_gated", "tau": None, "top_k": None,
           "hidden_dim": 32, "gnn_layers": 2, "dropout": .15, "lr": .002,
           "weight_decay": .0001, "batch_size": 64, "epochs": 60, "patience": 8,
           "min_delta": .002, "seed": 7, "shuffle_labels": False, "charm": False,
           "hidden": False, "epochs_per_process": None}
    config = TrainConfig(**old)
    model = build_model(config, 16, 12)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "old.pt"
        torch.save({"config": old, "in_dim": 16, "edge_dim": 12,
                    "state_dict": model.state_dict()}, path)
        loaded, loaded_config = load_checkpoint(path, torch.device("cpu"))
    assert loaded_config.architecture == "transformerconv"
    assert type(loaded) is type(model)


def test_temporal_edge_construction_and_model():
    edges = [torch.tensor([[0, 1], [1, 2]]), torch.tensor([[3, 4], [4, 5]])]
    attrs = [torch.ones(2, 2), torch.ones(2, 2) * 2]
    edge_index, edge_attr = append_temporal_identity_edges(edges, attrs, tokens=3)
    assert torch.equal(edge_index[:, -3:], torch.tensor([[0, 1, 2], [3, 4, 5]]))
    assert torch.equal(edge_attr[:4, -1], torch.zeros(4))
    assert torch.equal(edge_attr[-3:, -1], torch.ones(3))
    x = torch.randn(6, 4); x[:, 2] = 0; x[0, 2] = x[3, 2] = 1
    data = Batch.from_data_list([Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                                      layer_id=torch.tensor([0, 0, 0, 1, 1, 1]))])
    output, _ = TemporalMPNN(4, 3, 8, 2, 0)(data)
    assert output.shape == (1,)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"All {len(tests)} tests passed.")
