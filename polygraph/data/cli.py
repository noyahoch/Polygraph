"""`python3 -m polygraph.data <command>` — dataset creation (run once).

    scan     frozen ViT verdicts over clean test + all corruptions, all severities
    split    group-disjoint stratified train/val/test plan from the scan
    extract  attention graphs for the plan into the shared key-indexed store
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Sequence


def resolve_sources(names: Sequence[str]) -> List[str]:
    """Expand shorthands: a family name, 'main', or 'all'."""
    from ..config import ALL_CORRUPTIONS, CORRUPTION_FAMILIES, MAIN_CORRUPTIONS

    table = {"main": MAIN_CORRUPTIONS, "all": ALL_CORRUPTIONS, **CORRUPTION_FAMILIES}
    seen: set = set()
    return [s for n in names for s in table.get(n, (n,)) if not (s in seen or seen.add(s))]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="polygraph.data", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("scan", help="ViT verdicts over the full corruption grid (resumable)")

    split = sub.add_parser("split", help="group-disjoint stratified split plan")
    split.add_argument("--plan", default=None)
    split.add_argument("--held-out", nargs="*", default=["extra"],
                       help="sources or families excluded from train+val, kept in test")
    for name in ("train", "val", "test"):
        split.add_argument(f"--{name}-cap", type=int, default=0, help="pairs per class; 0 = all")

    extract = sub.add_parser("extract", help="graphs for every record in the plan (resumable)")
    extract.add_argument("--plan", default=None)

    hidden = sub.add_parser("hidden", help="capture per-token hidden states of one ViT layer "
                                           "for every stored record (variant-2 node features)")
    hidden.add_argument("--layer", type=int, default=12, help="ViT block output, 12 = final")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = Path.cwd()
    os.environ.setdefault("HF_HOME", str(root / ".cache" / "huggingface"))

    from ..config import (ALL_CORRUPTIONS, CLEAN_TEST, CLEAN_TRAIN, DEFAULT_TAU,
                          PLAN_FILE, SCAN_FILE, STORE_DIR)
    from ..records import read_scan_records
    from .sources import ensure_downloaded

    data_root = root / "data"
    scan_path = root / SCAN_FILE
    plan_path = root / (getattr(args, "plan", None) or PLAN_FILE)

    if args.command == "scan":
        from .pipeline import FrozenClassifier, scan

        pairs = [(CLEAN_TEST, 0), (CLEAN_TRAIN, 0)] + [(c, s) for c in ALL_CORRUPTIONS
                                                       for s in (1, 2, 3, 4, 5)]
        ensure_downloaded(pairs, data_root)
        classifier = FrozenClassifier()
        print(f"device: {classifier.device}", flush=True)
        scan(classifier, data_root, scan_path, pairs)

    elif args.command == "split":
        from .splits import build_plan

        # clean_train is excluded: the ViT was fine-tuned on it (measured 99.45% accuracy,
        # confidence inflated by memorization), so its records are not comparable.
        records = [r for r in read_scan_records(scan_path) if r.key.source != CLEAN_TRAIN]
        plan = build_plan(records, caps={n: getattr(args, f"{n}_cap") for n in ("train", "val", "test")},
                          held_out=resolve_sources(args.held_out))
        plan.save(plan_path)
        print(json.dumps(plan.stats, indent=2))

    elif args.command == "hidden":
        from .pipeline import FrozenClassifier, capture_hidden

        classifier = FrozenClassifier()
        print(f"device: {classifier.device}", flush=True)
        n = capture_hidden(classifier, data_root, root / STORE_DIR,
                           root / f"data/graph_dataset/hidden{args.layer}", layer=args.layer)
        print(f"hidden states captured for {n} records")

    elif args.command == "extract":
        from .graphs import ThresholdGraphBuilder
        from .pipeline import FrozenClassifier, extract
        from .splits import SplitPlan
        from .storage import GraphStoreWriter

        plan = SplitPlan.load(plan_path)
        keys = [k for name in ("test", "val", "train") for k in plan.splits[name]]
        needed = {k.as_tuple() for k in keys}
        lookup = {r.key.as_tuple(): r for r in read_scan_records(scan_path) if r.key.as_tuple() in needed}
        builder = ThresholdGraphBuilder(DEFAULT_TAU)
        classifier = FrozenClassifier()
        print(f"device: {classifier.device} | {builder.name}", flush=True)
        writer = GraphStoreWriter(root / STORE_DIR, shard_size=2000, tau=builder.tau)
        result = extract(classifier, data_root, builder, keys, lookup, writer)
        writer.write_manifest({"model_id": classifier.model_id,
                               "prediction_drift": result["prediction_drift"]})
        print(f"store holds {result['written']} records")


if __name__ == "__main__":
    main()
