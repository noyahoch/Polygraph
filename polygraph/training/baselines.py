"""Trained non-graph baselines — the proposal's controls for isolating graph structure.

    output_lr  logistic on [logit(msp), margin]: what training adds over raw output stats
    cls_mlp    MLP on the final class-token embedding (what the softmax head reads)
    cls_seq    GRU over the class-token embeddings of ALL layers: the proposal's
               sequence baseline — layer evolution without token-to-token structure
    attn_mlp   MLP on the same attention values the GNN sees (per-head vectors of the
               top-100 strongest edges per layer, flattened, strength-ordered): same
               data, no graph — the direct test of whether connectivity carries signal

All are fit with the detector's protocol: same splits, same seed, class-weighted BCE,
early stopping on val AUROC, best state restored.
"""

from __future__ import annotations

import random
from typing import Callable, Dict

import numpy as np
import torch
from torch import nn


def collect_features(dataset) -> Dict[str, np.ndarray]:
    """Meta + CLS features straight from the store; no model involved."""
    from torch_geometric.loader import DataLoader

    fields = ("y", "confidence", "margin", "source_id", "severity")
    out: Dict[str, list] = {f: [] for f in fields}
    cls: list = []
    for batch in DataLoader(dataset, batch_size=256, shuffle=False):
        for f in fields:
            out[f] += getattr(batch, f).view(-1).tolist()
        if hasattr(batch, "cls_layers"):
            cls.append(batch.cls_layers)
    result = {f: np.asarray(v) for f, v in out.items()}
    if cls:
        stacked = torch.cat(cls).numpy()  # [N, L, D]
        result["cls_all"] = stacked
        result["cls"] = stacked[:, -1, :]
    return result


def collect_flat_attention(dataset) -> np.ndarray:
    """[N, layers*K*heads]: the GNN's edge features without the edge structure. Pass a
    top-K view so every record has a fixed shape; edges arrive strength-sorted, which is
    the canonical order a set model is allowed to see."""
    from torch_geometric.loader import DataLoader

    rows = []
    for batch in DataLoader(dataset, batch_size=256, shuffle=False):
        rows.append(batch.edge_attr.reshape(batch.num_graphs, -1))
    return torch.cat(rows).numpy()


def output_scores(train: Dict[str, np.ndarray], test: Dict[str, np.ndarray], seed: int) -> np.ndarray:
    """Logistic on [logit(msp), margin]. Standardized — the Stage-5 lesson."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def features(pred):
        conf = np.clip(pred["confidence"], 1e-6, 1 - 1e-6)
        return np.stack([np.log((1 - conf) / conf), pred["margin"]], axis=1)

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    model.fit(features(train), train["y"])
    return model.predict_proba(features(test))[:, 1]


def output_mlp_scores(train, val, test, seed: int, device, hidden: int = 32) -> np.ndarray:
    """Nonlinear model on the same two output statistics as output_lr. Exists because the
    margin residual at fixed confidence FLIPS SIGN around conf~0.8 (anti-textbook below,
    textbook above) — structure a linear model cancels out and a small MLP can harvest.
    output_lr vs output_mlp therefore isolates 'value of nonlinearity' on outputs."""

    def features(pred):
        conf = np.clip(pred["confidence"], 1e-6, 1 - 1e-6)
        return np.stack([np.log((1 - conf) / conf), pred["margin"]], axis=1)

    x_tr, x_va, x_te = _standardize(features(train), features(val), features(test))
    build = lambda: nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                                  nn.ReLU(), nn.Dropout(0.15), nn.Linear(hidden, 1))
    return _fit_tabular(build, x_tr, train["y"], x_va, val["y"], x_te, seed, device)


def _fit_tabular(build: Callable[[], nn.Module],
                 x_train: torch.Tensor, y_train: np.ndarray,
                 x_val: torch.Tensor, y_val: np.ndarray,
                 x_test: torch.Tensor, seed: int, device,
                 epochs: int = 60, patience: int = 8, batch: int = 256) -> np.ndarray:
    """The detector's training protocol, for tensor-input baselines."""
    from sklearn.metrics import roc_auc_score

    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    model = build().to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    positives = float(y_tr.sum())
    pos_weight = torch.tensor([(len(y_tr) - positives) / max(positives, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    best_state, best_val, stale = None, -np.inf, 0
    for _ in range(epochs):
        model.train()
        for start in range(0, len(y_tr), batch):
            window = slice(start, start + batch)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_train[window].to(device)).view(-1), y_tr[window].to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_auroc = roc_auc_score(y_val, model(x_val.to(device)).view(-1).cpu().numpy())
        if val_auroc > best_val + 0.002:
            best_val, stale = val_auroc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(x_test.to(device)).view(-1).cpu().numpy()


def _standardize(train: np.ndarray, *rest: np.ndarray):
    flat = train.reshape(len(train), -1)
    mean, std = flat.mean(0), flat.std(0) + 1e-6

    def prep(x):
        normalised = (x.reshape(len(x), -1) - mean) / std
        return torch.tensor(normalised.reshape(x.shape), dtype=torch.float32)

    return [prep(x) for x in (train, *rest)]


def cls_mlp_scores(train, val, test, seed: int, device, hidden: int = 128) -> np.ndarray:
    x_tr, x_va, x_te = _standardize(train["cls"], val["cls"], test["cls"])
    build = lambda: nn.Sequential(nn.Linear(x_tr.shape[1], hidden), nn.ReLU(),
                                  nn.Dropout(0.15), nn.Linear(hidden, 1))
    return _fit_tabular(build, x_tr, train["y"], x_va, val["y"], x_te, seed, device)


class _GRUHead(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gru = nn.GRU(dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.15), nn.Linear(hidden, 1))

    def forward(self, x):
        _, state = self.gru(x)
        return self.head(state[-1]).view(-1)


def cls_seq_scores(train, val, test, seed: int, device, hidden: int = 128) -> np.ndarray:
    """The proposal's sequence baseline: layer-wise CLS evolution, no attention graph."""
    x_tr, x_va, x_te = _standardize(train["cls_all"], val["cls_all"], test["cls_all"])
    build = lambda: _GRUHead(x_tr.shape[-1], hidden)
    return _fit_tabular(build, x_tr, train["y"], x_va, val["y"], x_te, seed, device)


def attn_mlp_scores(train_x, train_y, val_x, val_y, test_x, seed: int, device,
                    hidden: int = 128) -> np.ndarray:
    """The proposal's flat-attention baseline: same values as the GNN, no structure."""
    x_tr, x_va, x_te = _standardize(train_x, val_x, test_x)
    build = lambda: nn.Sequential(nn.Linear(x_tr.shape[1], hidden), nn.ReLU(),
                                  nn.Dropout(0.15), nn.Linear(hidden, 1))
    return _fit_tabular(build, x_tr, train_y, x_va, val_y, x_te, seed, device)
