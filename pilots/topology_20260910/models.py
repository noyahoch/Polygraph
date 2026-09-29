"""Reuse Ishi's exact detector implementations with frozen capacity matching."""
from __future__ import annotations

import torch
from torch import nn
from polygraph.training.train import TrainConfig, build_model as legacy_build_model
from .protocol import ARMS, protocol


def build_model(arm):
    spec = ARMS[arm]
    if arm == "logit":
        return nn.Sequential(nn.Linear(100, 64), nn.ReLU(), nn.Dropout(0.15),
                             nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.15), nn.Linear(32, 1))
    cfg = TrainConfig(architecture=spec["architecture"], hidden_dim=spec["width"],
                      gnn_layers=2, dropout=0.15, readout="cls_gated")
    return legacy_build_model(cfg, 784 if spec["full"] else 16, 36 if spec["full"] else 12)


def parameter_count(arm):
    return sum(parameter.numel() for parameter in build_model(arm).parameters())


EXPECTED_PARAMETERS = {"full_graph": 131970, "full_rewired": 131970, "full_set": 133142,
                       "full_endpoint": 132542, "raw_graph": 81282, "raw_set": 81284,
                       "logit": 8577}
