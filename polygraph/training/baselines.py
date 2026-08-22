"""Trained non-graph baselines, so the detector's edge cannot be 'it was trained'.

The proposal requires comparing against output signals and hidden representations given
the same training budget. Untrained MSP/margin live in evaluate.py; here are the trained
counterparts, fit with the same splits and seeds as the detector:

    output_lr  logistic on [logit(msp), margin] — training on output statistics alone
               cannot beat MSP's ranking by much, and showing that isolates what
               training contributes from what attention contributes
    cls_mlp    MLP on the final-layer class-token embedding — the representation-side
               baseline; the softmax head reads exactly this vector
"""

from __future__ import annotations

import random
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn


def collect_features(dataset, device=None) -> Dict[str, np.ndarray]:
    """Meta + CLS features straight from the store; no model involved."""
    from torch_geometric.loader import DataLoader

    fields = ("y", "confidence", "margin", "source_id", "severity")
    out: Dict[str, list] = {f: [] for f in fields}
    cls: list = []
    for batch in DataLoader(dataset, batch_size=256, shuffle=False):
        for f in fields:
            out[f] += getattr(batch, f).view(-1).tolist()
        if hasattr(batch, "cls_layers"):
            cls.append(batch.cls_layers[:, -1, :])  # final ViT layer's CLS token
    result = {f: np.asarray(v) for f, v in out.items()}
    if cls:
        result["cls"] = torch.cat(cls).numpy()
    return result


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


def cls_mlp_scores(train: Dict[str, np.ndarray], val: Dict[str, np.ndarray],
                   test: Dict[str, np.ndarray], seed: int, device,
                   hidden: int = 128, epochs: int = 60, patience: int = 8) -> np.ndarray:
    """MLP on the final CLS embedding, trained with the detector's protocol
    (class-weighted BCE, early stopping on val AUROC, best state restored)."""
    from sklearn.metrics import roc_auc_score

    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    mean, std = train["cls"].mean(0, keepdims=True), train["cls"].std(0, keepdims=True) + 1e-6

    def tensors(pred):
        return (torch.tensor((pred["cls"] - mean) / std, dtype=torch.float32),
                torch.tensor(pred["y"], dtype=torch.float32))

    x_train, y_train = tensors(train)
    x_val, y_val = tensors(val)
    x_test, _ = tensors(test)
    model = nn.Sequential(nn.Linear(x_train.shape[1], hidden), nn.ReLU(), nn.Dropout(0.15),
                          nn.Linear(hidden, 1)).to(device)
    positives = float(y_train.sum())
    pos_weight = torch.tensor([(len(y_train) - positives) / max(positives, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    best_state, best_val, stale = None, -np.inf, 0
    for _ in range(epochs):
        model.train()
        for start in range(0, len(y_train), 256):
            batch = slice(start, start + 256)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_train[batch].to(device)).view(-1), y_train[batch].to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_auroc = roc_auc_score(val["y"], model(x_val.to(device)).view(-1).cpu().numpy())
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
