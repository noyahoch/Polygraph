"""`python3 -m polygraph.training <command>` — the detector (run many times).

    train     one checkpoint per seed, early stopping on val AUROC
    evaluate  saved checkpoints against any plan: slices, baselines, risk-coverage
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="polygraph.training", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train the detector; one checkpoint per seed")
    train.add_argument("--plan", default=None)
    train.add_argument("--out-dir", default="runs/detector")
    train.add_argument("--layers", default="last", help="last | all | final4 | comma-separated ints")
    train.add_argument("--readout", default="cls_gated",
                       choices=["mean", "cls", "cls_mean", "cls_mean_max", "cls_gated", "edge_set"])
    train.add_argument("--tau", type=float, default=None, help="load-time re-threshold ablation")
    train.add_argument("--top-k", type=int, default=None, help="load-time fixed-edge-count ablation")
    train.add_argument("--epochs-per-process", type=int, default=8,
                       help="exit and resume every N epochs (resets Metal's leaky kernel "
                            "cache); the caller loops until the checkpoint exists")
    train.add_argument("--batch-size", type=int, default=64,
                       help="reduce for multi-layer runs (12 graphs per sample)")
    train.add_argument("--hidden-dim", type=int, default=32,
                       help="GNN width; raise for the capacity-matched comparisons")
    train.add_argument("--gnn-layers", type=int, default=2,
                       help="message-passing depth; raise for the capacity-matched comparisons")
    train.add_argument("--hidden", action="store_true",
                       help="variant 2: append per-token hidden states to node features "
                            "(requires 'polygraph.data hidden' capture first)")
    train.add_argument("--charm", action="store_true",
                       help="CHARM-lite: one union graph per image, all-layer edge features")
    train.add_argument("--shuffle-labels", action="store_true",
                       help="negative control: permuted train labels, expect ~0.5 test AUROC")
    train.add_argument("--seeds", type=int, nargs="+", default=[7],
                       help="POC default: one seed; use 3+ for any claimed result "
                            "(single-seed noise measured at ~±0.02 AUROC)")

    evaluate = sub.add_parser("evaluate", help="evaluate saved checkpoints")
    evaluate.add_argument("--run-dir", default="runs/detector")
    evaluate.add_argument("--plan", default=None,
                          help="any plan whose records are in the store")
    evaluate.add_argument("--no-baselines", action="store_true",
                          help="skip retraining the non-graph baselines (ablation runs: "
                               "baselines depend on the plan, not the graph config)")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = Path.cwd()
    os.environ.setdefault("HF_HOME", str(root / ".cache" / "huggingface"))

    from ..config import PLAN_FILE, STORE_DIR
    from ..data.pipeline import choose_device

    plan_path = root / (args.plan or PLAN_FILE)
    device = choose_device()

    if args.command == "train":
        from .train import TrainConfig, train_run

        named = {"last": [11], "all": list(range(12)), "final4": [8, 9, 10, 11]}
        layers = named.get(args.layers) or [int(x) for x in args.layers.split(",")]
        config = TrainConfig(layers=layers, readout=args.readout, tau=args.tau, top_k=args.top_k,
                             hidden_dim=args.hidden_dim, gnn_layers=args.gnn_layers,
                             batch_size=args.batch_size, shuffle_labels=args.shuffle_labels,
                             charm=args.charm, hidden=args.hidden,
                             epochs_per_process=args.epochs_per_process or None)
        print(f"device: {device} | layers: {layers} | readout: {args.readout} "
              f"| hidden_dim: {args.hidden_dim} | gnn_layers: {args.gnn_layers}", flush=True)
        train_run(root / STORE_DIR, plan_path, root / args.out_dir, config, args.seeds, device)

    elif args.command == "evaluate":
        from .evaluate import evaluate_run

        summary = evaluate_run(root / args.run_dir, root / STORE_DIR, plan_path, device,
                               include_baselines=not args.no_baselines)
        print(json.dumps(summary["slices"], indent=2))


if __name__ == "__main__":
    main()
