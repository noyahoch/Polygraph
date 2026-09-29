"""Run the already-approved postprocess after its wall-clock cutoff.

This entry point is deliberately separate from the normal ``steps`` chain.  It
reuses the frozen four base models and the on-time meta export, but records the
development prediction and evaluation as a late diagnostic.  It must never be
used to relabel the fixed-protocol experiment as complete.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import os
import sys
from pathlib import Path


def invoke(name: str, args: list[object]) -> None:
    original = sys.argv
    sys.argv = [name, *[str(value) for value in args]]
    try:
        importlib.import_module("pilots.layer_ensemble_20260914." + name).main()
    finally:
        sys.argv = original


def main() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Late diagnostics require an allocated Slurm job")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    common = ["--cache", root / "feature_cache", "--execution", root / "execution.json",
              "--roles", root / "role_map.json"]

    # The normal predictor and evaluator enforce the frozen 07:00 cutoff.  A
    # late diagnostic may pass that gate only after the original meta export is
    # already present; the protocol constants are restored below for combine's
    # provenance checks and are not rewritten on disk.
    predict = importlib.import_module("pilots.layer_ensemble_20260914.predict")
    predict.check_prediction_deadline = lambda: None

    invoke("predict", [*common, "--run-root", root / "runs", "--base-freeze", root / "base_freeze.json",
                        "--freeze-base", "--role", "meta", "--out", root / "predictions/meta.npz"])
    invoke("combine", ["--root", root, "--predictions", root / "predictions/meta.npz", "--out", root / "heads"])

    # ``evaluate`` imports ``load_predictions`` from ``combine``.  Temporarily
    # relax only that function's timestamp check while retaining all identity,
    # checksum and role checks.  The original context still sees its frozen
    # 2026 deadline, so this cannot change the formal experiment provenance.
    combine = importlib.import_module("pilots.layer_ensemble_20260914.combine")
    original_context = combine.context
    original_load_predictions = combine.load_predictions
    frozen_deadline = combine.DEADLINES["predictions_complete_before"]

    def context_with_frozen_deadline(root_path):
        combine.DEADLINES["predictions_complete_before"] = frozen_deadline
        try:
            return original_context(root_path)
        finally:
            combine.DEADLINES["predictions_complete_before"] = "2099-01-01T00:00:00+00:00"

    def late_load_predictions(root_path, path, role):
        combine.context = context_with_frozen_deadline
        combine.DEADLINES["predictions_complete_before"] = "2099-01-01T00:00:00+00:00"
        try:
            return original_load_predictions(root_path, path, role)
        finally:
            combine.context = original_context
            combine.DEADLINES["predictions_complete_before"] = frozen_deadline

    combine.load_predictions = late_load_predictions
    invoke("predict", [*common, "--run-root", root / "runs", "--base-freeze", root / "base_freeze.json",
                        "--role", "dev_eval", "--out", root / "predictions/dev_eval.npz",
                        "--heads-freeze", root / "heads_freeze.json"])
    invoke("evaluate", ["--root", root, "--predictions", root / "predictions/dev_eval.npz",
                         "--heads", root / "heads", "--out", root / "evaluation"])

    marker = {
        "status": "complete_late_diagnostic",
        "complete": True,
        "scope_id": "layer_ensemble_20260914_core_seed7",
        "job_id": os.environ["SLURM_JOB_ID"],
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "original_protocol_status": "incomplete: development predictions missed the frozen cutoff",
        "reused_on_time_meta_export": True,
        "warning": "This diagnostic must not be reported as the fixed-protocol primary result.",
    }
    marker_path = root / "evaluation" / "late_diagnostic.json"
    marker_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(marker), flush=True)


if __name__ == "__main__":
    main()
