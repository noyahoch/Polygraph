"""The four fixed architectures; no architecture or capacity search."""
from __future__ import annotations

import torch
from polygraph.training.models import HiddenTokenSetModel, M5NodeEdgeSetModel
from pilots.layer_screen_20260913.models import build_model as graph_model
from pilots.topology_20260910.models import build_model as topology_model

from .protocol import ARMS, FAMILIES, MODEL_SPECS, require_slurm

EXPECTED_PARAMETERS = {"G": 130434, "H": 129986, "S": 131126, "O": 8577}
HIDDEN_INDEX = dict(zip(ARMS, (3, 6, 9, 12)))
GRAPH_INDEX = dict(zip(ARMS, (2, 5, 8, 11)))


def validate_arm(family, arm):
    if family not in FAMILIES or arm not in (("logits",) if family == "O" else ARMS):
        raise ValueError("Unknown fixed family/arm: " + str((family, arm)))


def build_model(family, arm):
    require_slurm()
    validate_arm(family, arm)
    if family == "G":
        model = graph_model(arm)
    elif family == "H":
        model = HiddenTokenSetModel(772, 96, 0.15)
    elif family == "S":
        model = M5NodeEdgeSetModel(784, 12, 84, 0.15)
    else:
        model = topology_model("logit")
    actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected = EXPECTED_PARAMETERS[family]
    if actual != expected or MODEL_SPECS[family]["parameters"] != expected:
        raise RuntimeError(
            f"Parameter-count mismatch for {family}/{arm}: actual={actual}, "
            f"approved={expected}; stop for review, do not change the architecture"
        )
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError("Every fixed neural architecture must initialize in FP32")
    return model


def parameter_count(model_or_family, arm=None):
    require_slurm()
    model = (build_model(model_or_family, arm)
             if isinstance(model_or_family, str) else model_or_family)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
