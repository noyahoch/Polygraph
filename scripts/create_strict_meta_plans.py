#!/usr/bin/env python3
"""Create deterministic group-disjoint plans for strict combiner confirmation."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from polygraph.data.splits import SplitPlan


def stats(keys):
    groups = {key.group_id for key in keys}
    return {"records": len(keys), "base_images": len(groups)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    original = SplitPlan.load(args.plan)
    val_groups = sorted({key.group_id for key in original.splits["val"]})
    random.Random(args.seed).shuffle(val_groups)
    cut = len(val_groups) // 2
    base_groups, meta_groups = set(val_groups[:cut]), set(val_groups[cut:])
    base_val = [key for key in original.splits["val"] if key.group_id in base_groups]
    meta_val = [key for key in original.splits["val"] if key.group_id in meta_groups]
    assert base_groups.isdisjoint(meta_groups)
    assert len(base_val) + len(meta_val) == len(original.splits["val"])

    common = {**original.config, "strict_meta_seed": args.seed,
              "source_plan": str(args.plan), "base_val_fraction": 0.5}
    train_plan = SplitPlan(
        {"train": original.splits["train"], "val": base_val, "test": original.splits["test"]},
        {"train": stats(original.splits["train"]), "val": stats(base_val),
         "test": stats(original.splits["test"])},
        {**common, "purpose": "strict_detector_training"})
    # Evaluation deliberately uses base_val as its nominal train split and meta_val
    # as val. This keeps all three splits group-disjoint while allowing the standard
    # evaluator to emit aligned meta-validation and untouched-test predictions.
    eval_plan = SplitPlan(
        {"train": base_val, "val": meta_val, "test": original.splits["test"]},
        {"train": stats(base_val), "val": stats(meta_val),
         "test": stats(original.splits["test"])},
        {**common, "purpose": "strict_combiner_scoring"})
    train_plan.validate(); eval_plan.validate()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "detector_train_plan.json"
    eval_path = args.out_dir / "combiner_eval_plan.json"
    train_plan.save(train_path); eval_plan.save(eval_path)
    print(f"base_val: {len(base_val)} records / {len(base_groups)} groups")
    print(f"meta_val: {len(meta_val)} records / {len(meta_groups)} groups")
    print(train_path); print(eval_path)


if __name__ == "__main__":
    main()
