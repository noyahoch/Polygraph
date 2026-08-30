"""Tests for the Deep-Sets structure control (EdgeSetModel).

Separate file by project convention: tests/test_polygraph.py is not modified without
explicit sign-off. Run: python3 tests/test_deepsets.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch_geometric.data import Batch, Data

from polygraph.training.models import EdgeSetModel
from polygraph.training.train import TrainConfig, build_model


def make_graph(seed: int, nodes: int = 9, edges: int = 30) -> Data:
    g = torch.Generator().manual_seed(seed)
    return Data(x=torch.rand(nodes, 4, generator=g),
                edge_index=torch.randint(0, nodes, (2, edges), generator=g),
                edge_attr=torch.rand(edges, 12, generator=g),
                y=torch.tensor([float(seed % 2)]))


def batch_of(graphs) -> Batch:
    return Batch.from_data_list(graphs)


PASS = []


def check(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    PASS.append(bool(ok))


def main():
    torch.manual_seed(0)
    model = EdgeSetModel(in_dim=4, edge_dim=12, hidden_dim=16, dropout=0.0).eval()
    graphs = [make_graph(s) for s in range(4)]
    logits, aux = model(batch_of(graphs))

    check("one logit per graph, aux is None", logits.shape == (4,) and aux is None)

    # Permutation invariance: shuffling edge ORDER must not change any logit.
    shuffled = []
    for g in graphs:
        perm = torch.randperm(g.edge_attr.shape[0])
        shuffled.append(Data(x=g.x, edge_index=g.edge_index[:, perm],
                             edge_attr=g.edge_attr[perm], y=g.y))
    check("edge-order invariant",
          torch.allclose(logits, model(batch_of(shuffled))[0], atol=1e-6))

    # Structure blindness: REWIRING edges (same features, different endpoints within the
    # same graph) must not change any logit — the property that makes it a control.
    rewired = []
    for s, g in enumerate(graphs):
        gen = torch.Generator().manual_seed(100 + s)
        new_index = torch.randint(0, g.x.shape[0], g.edge_index.shape, generator=gen)
        rewired.append(Data(x=g.x, edge_index=new_index, edge_attr=g.edge_attr, y=g.y))
    check("rewiring invariant (uses no structure)",
          torch.allclose(logits, model(batch_of(rewired))[0], atol=1e-6))

    # Changing one edge's features in one graph changes that graph's logit only.
    poked = [Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr.clone(), y=g.y)
             for g in graphs]
    poked[2].edge_attr[0] += 1.0
    delta = (model(batch_of(poked))[0] - logits).abs()
    check("feature-sensitive and per-graph isolated",
          delta[2] > 1e-4 and delta[[0, 1, 3]].max() < 1e-6)

    # build_model dispatch: readout="edge_set" must construct this model.
    config = TrainConfig(readout="edge_set", hidden_dim=16)
    check("build_model dispatch", isinstance(build_model(config, 4, 12), EdgeSetModel))

    # It must train end-to-end on a separable synthetic set task: label = 1 iff the
    # mean of edge feature 0 is high (a set-level statistic, no structure needed).
    gen = torch.Generator().manual_seed(7)
    data = []
    for i in range(200):
        y = i % 2
        e = torch.rand(25, 12, generator=gen)
        e[:, 0] = e[:, 0] * 0.3 + (0.6 if y else 0.1)
        data.append(Data(x=torch.rand(6, 4, generator=gen),
                         edge_index=torch.randint(0, 6, (2, 25), generator=gen),
                         edge_attr=e, y=torch.tensor([float(y)])))
    torch.manual_seed(1)
    net = EdgeSetModel(4, 12, 16, 0.0)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    for _ in range(60):
        opt.zero_grad()
        out, _ = net(batch_of(data))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out, torch.tensor([float(g.y) for g in data]))
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        out, _ = net(batch_of(data))
    acc = ((out > 0).float() == torch.tensor([float(g.y) for g in data])).float().mean()
    check(f"learns a set-level statistic (train acc {acc:.2f})", acc > 0.95)

    print(("All %d tests passed." if all(PASS) else "FAILURES among %d tests.") % len(PASS))
    sys.exit(0 if all(PASS) else 1)


if __name__ == "__main__":
    main()
