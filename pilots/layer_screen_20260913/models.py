"""Same width64/two-layer scaffold; input projection sizes reflect exposure."""
from polygraph.training.train import TrainConfig, build_model as build_legacy
from .protocol import ARMS

def build_model(arm):
    layers = len(ARMS[arm]["layers"])
    cfg = TrainConfig(architecture="edge_gated_mean", hidden_dim=64, gnn_layers=2,
                      dropout=0.15, readout="cls_gated")
    return build_legacy(cfg, 4 + 12 * layers + 768, 12 * layers)

def parameter_count(arm):
    return sum(p.numel() for p in build_model(arm).parameters())
