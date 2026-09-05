"""Mutation-oriented tests for the topology/depth follow-up models."""

import copy

import torch
from torch_geometric.data import Batch, Data

from polygraph.training.models import (FactoredEndpointAffine, HiddenTokenSetModel,
                                       M5EndpointSetModel, M5NodeEdgeSetModel,
                                       ResidualGraphModel)


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
