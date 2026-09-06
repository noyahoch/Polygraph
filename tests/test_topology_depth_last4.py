"""Mutation-oriented tests for the topology/depth follow-up models."""

import copy
import tempfile
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

from polygraph.training.models import (FactoredEndpointAffine, HiddenTokenSetModel,
                                       LastFourGraphSequenceModel,
                                       LastFourTokenSetModel, LastFourUnionEndpointSetModel,
                                       LastFourUnionGraphModel,
                                       M5EndpointSetModel, M5NodeEdgeSetModel,
                                       ResidualGraphModel)
from polygraph.data.storage import build_layer_union
from polygraph.data.pipeline import LAST4_BLOCK_IDS, LAST4_HIDDEN_INDICES
from scripts.run_topology_depth_last4 import Suite


def graph():
    x = torch.randn(5, 10)
    x[:, 2] = 0
    x[0, 2] = 1
    # Multiple incoming records per target are required to exercise GATv2's
    # edge-conditioned attention softmax (a singleton neighborhood is constant).
    return Data(x=x, edge_index=torch.tensor([[0, 1, 2, 3, 4, 0], [2, 2, 3, 3, 4, 4]]),
                edge_attr=torch.randn(6, 6), y=torch.tensor([0.]))


def test_factored_endpoint_affine_matches_explicit_and_gradients():
    torch.manual_seed(3)
    x = torch.randn(7, 5, requires_grad=True)
    edge = torch.tensor([[0, 1, 4, 6], [2, 3, 0, 5]])
    attr = torch.randn(4, 3, requires_grad=True)
    layer = FactoredEndpointAffine(5, 3, 8)
    combined = torch.cat([layer.source.weight, layer.target.weight, layer.edge.weight], 1)
    explicit = torch.nn.functional.linear(torch.cat([x[edge[0]], x[edge[1]], attr], 1),
                                          combined, layer.edge.bias)
    actual = layer(x, edge, attr)
    torch.testing.assert_close(actual, explicit)
    gx1, ge1 = torch.autograd.grad(actual.square().sum(), (x, attr), retain_graph=True)
    gx2, ge2 = torch.autograd.grad(explicit.square().sum(), (x, attr))
    torch.testing.assert_close(gx1, gx2)
    torch.testing.assert_close(ge1, ge2)


def test_m5_node_edge_set_rewiring_invariant():
    data = Batch.from_data_list([graph(), graph()])
    changed = copy.copy(data)
    changed.edge_index = data.edge_index.clone()
    changed.edge_index[1] = changed.edge_index[1].roll(1)
    model = M5NodeEdgeSetModel(10, 6, 16, 0).eval()
    torch.testing.assert_close(model(data)[0], model(changed)[0])


def test_m5_endpoint_set_record_order_invariant():
    data = Batch.from_data_list([graph()])
    permutation = torch.tensor([2, 0, 5, 3, 1, 4])
    changed = copy.copy(data)
    changed.edge_index = data.edge_index[:, permutation]
    changed.edge_attr = data.edge_attr[permutation]
    model = M5EndpointSetModel(10, 6, 16, 0).eval()
    torch.testing.assert_close(model(data)[0], model(changed)[0])


def test_hidden_token_set_ignores_all_edge_mutations():
    data = Batch.from_data_list([graph()])
    changed = copy.copy(data)
    changed.edge_index = data.edge_index.flip(0)
    changed.edge_attr = data.edge_attr * 999
    model = HiddenTokenSetModel(10, 16, 0).eval()
    torch.testing.assert_close(model(data)[0], model(changed)[0])


def test_edge_aware_families_consume_signed_edge_attributes_and_backpropagate():
    data = Batch.from_data_list([graph()])
    for family in ("transformerconv_residual", "gine", "edge_gated_mean", "gatv2"):
        model = ResidualGraphModel(10, 6, 16, 2, 0, family)
        attr = data.edge_attr.clone().requires_grad_(True)
        altered = copy.copy(data)
        altered.edge_attr = attr
        loss = model(altered)[0].sum()
        gradient = torch.autograd.grad(loss, attr)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0, family
        with torch.no_grad():
            opposite = copy.copy(data)
            opposite.edge_attr = -data.edge_attr
            assert not torch.allclose(model(data)[0], model(opposite)[0]), family


def test_no_family_adds_implicit_self_loops():
    for family in ("transformerconv_residual", "gine", "edge_gated_mean", "gatv2"):
        model = ResidualGraphModel(10, 6, 16, 2, 0, family)
        for block in model.blocks:
            if hasattr(block, "add_self_loops"):
                assert block.add_self_loops is False


def test_consistent_node_relabeling_leaves_graph_prediction_unchanged():
    torch.manual_seed(19)
    original = graph()
    permutation = torch.tensor([3, 0, 4, 1, 2])  # new index -> old index
    inverse = torch.empty_like(permutation); inverse[permutation] = torch.arange(5)
    relabeled = Data(x=original.x[permutation],
                     edge_index=inverse[original.edge_index],
                     edge_attr=original.edge_attr.clone(), y=original.y)
    for family in ("transformerconv_residual", "gine", "edge_gated_mean", "gatv2"):
        model = ResidualGraphModel(10, 6, 16, 2, 0, family).eval()
        a = model(Batch.from_data_list([original]))[0]
        b = model(Batch.from_data_list([relabeled]))[0]
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)


def test_budget_limited_state_is_promoted_without_using_last_epoch_weights():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); out = root / "candidate"; out.mkdir()
        best = {"weight": torch.tensor([2.0])}
        torch.save({"epoch": 5, "model": {"weight": torch.tensor([99.0])},
                    "best_state": best, "best_epoch": 3, "best_val": .8,
                    "history": [{"epoch": 3, "val_auroc": .8}], "config": {"seed": 7},
                    "in_dim": 4, "edge_dim": 6}, out / "state_seed7.pt")
        suite = object.__new__(Suite)
        command = ["python", "-m", "polygraph.training", "train", "--plan", "plan.json",
                   "--out-dir", str(out), "--seeds", "7"]
        assert suite._promote_budget_state(command)
        payload = torch.load(out / "model_seed7.pt", map_location="cpu", weights_only=False)
        torch.testing.assert_close(payload["state_dict"]["weight"], best["weight"])
        assert payload["budget_limited"] is True and payload["last_completed_epoch"] == 5


def test_last_four_union_alignment_and_missing_mask():
    edges = [torch.tensor([[0, 1], [1, 2]]), torch.tensor([[0, 2], [1, 1]])]
    attrs = [torch.tensor([[1., 2.], [3., 4.]]), torch.tensor([[5., 6.], [7., 8.]])]
    union, features = build_layer_union(edges, attrs, tokens=3)
    assert union.tolist() == [[0, 1, 2], [1, 2, 1]]
    # 0->1 occurs in both layers; other edges have an unobserved/zero-filled slot.
    torch.testing.assert_close(features[0], torch.tensor([1., 2., 5., 6., 1., 1.]))
    torch.testing.assert_close(features[1], torch.tensor([3., 4., 0., 0., 1., 0.]))
    torch.testing.assert_close(features[2], torch.tensor([0., 0., 7., 8., 0., 1.]))


def test_last_four_models_preserve_ordered_token_identity_and_backpropagate():
    x = torch.randn(5, 4, 10, requires_grad=True)
    cls = torch.tensor([True, False, False, False, False])
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 0], [2, 2, 3, 3, 4, 4]])
    edge_attr = torch.randn(6, 12, requires_grad=True)
    data = Batch.from_data_list([Data(x=x, cls_mask=cls, edge_index=edge_index,
                                           edge_attr=edge_attr)])
    models = [LastFourTokenSetModel(10, 16, 0),
              LastFourUnionGraphModel(10, 12, 16, 2, 0, "edge_gated_mean"),
              LastFourUnionEndpointSetModel(10, 12, 16, 0)]
    for model in models:
        score = model(data)[0]
        assert score.shape == (1,)
        score.sum().backward(retain_graph=True)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert edge_attr.grad is not None and edge_attr.grad.abs().sum() > 0
    # Mutating layer order changes the ordered history representation.
    reversed_data = copy.copy(data); reversed_data.x = data.x.flip(1)
    assert not torch.allclose(models[0](data)[0], models[0](reversed_data)[0])


def test_ordered_graph_sequence_has_no_cross_layer_edges_and_uses_order():
    torch.manual_seed(29)
    tokens, width = 5, 10
    x = torch.randn(4 * tokens, width)
    x[:, 2] = 0; x[torch.arange(4) * tokens, 2] = 1
    base_edges = torch.tensor([[0, 1, 2, 3, 4, 0], [2, 2, 3, 3, 4, 4]])
    edge_index = torch.cat([base_edges + layer * tokens for layer in range(4)], 1)
    assert torch.equal(edge_index[0] // tokens, edge_index[1] // tokens)
    data = Batch.from_data_list([Data(x=x, edge_index=edge_index,
        edge_attr=torch.randn(edge_index.shape[1], 6),
        cls_mask=(torch.arange(tokens) == 0).repeat(4),
        layer_id=torch.arange(4).repeat_interleave(tokens))])
    model = LastFourGraphSequenceModel(width, 6, 16, 1, 0, "edge_gated_mean").eval()
    original = model(data)[0]
    changed = copy.copy(data)
    changed.x = data.x.view(4, tokens, width).flip(0).reshape_as(data.x)
    changed.edge_attr = data.edge_attr.view(4, -1, 6).flip(0).reshape_as(data.edge_attr)
    assert not torch.allclose(original, model(changed)[0])


def test_block_output_hidden_state_correspondence_is_not_final_state_reuse():
    assert LAST4_BLOCK_IDS == (8, 9, 10, 11)
    assert LAST4_HIDDEN_INDICES == (9, 10, 11, 12)
    assert len(set(LAST4_HIDDEN_INDICES)) == 4
