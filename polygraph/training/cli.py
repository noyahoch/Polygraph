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
                       choices=["mean", "cls", "cls_mean", "cls_mean_max", "cls_gated"])
    train.add_argument("--tau", type=float, default=None, help="load-time re-threshold ablation")
    train.add_argument("--top-k", type=int, default=None, help="load-time fixed-edge-count ablation")
    train.add_argument("--seeds", type=int, nargs="+", default=[7, 1, 2])

    evaluate = sub.add_parser("evaluate", help="evaluate saved checkpoints")
    evaluate.add_argument("--run-dir", default="runs/detector")
    evaluate.add_argument("--plan", default=None,
                          help="any plan whose records are in the store")
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
        config = TrainConfig(layers=layers, readout=args.readout, tau=args.tau, top_k=args.top_k)
        print(f"device: {device} | layers: {layers} | readout: {args.readout}", flush=True)
        train_run(root / STORE_DIR, plan_path, root / args.out_dir, config, args.seeds, device)

    elif args.command == "evaluate":
        from .evaluate import evaluate_run

        summary = evaluate_run(root / args.run_dir, root / STORE_DIR, plan_path, device)
        print(json.dumps(summary["slices"], indent=2))


if __name__ == "__main__":
    main()
