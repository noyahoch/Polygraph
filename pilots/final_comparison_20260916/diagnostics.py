"""History-only fixed20 diagnostics; descriptive warnings never change fit eligibility."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import statistics
import time

from . import protocol
from .heads import (ARMS, SEEDS, SCOPE, _before_new_readouts, _context, _lock,
                    _match, _verify_files, _write_bytes, _write_json)

COUNTS = protocol.COUNTS
ARTIFACTS = {"training_diagnostics.json", "training_diagnostics.csv"}
POLICY = {
    "epochs": 20,
    "selection": "earliest epoch attaining greatest actual checkpoint-role AUROC",
    "auc": "actual checkpoint_auroc, never running best_checkpoint_auroc",
    "loss_decrease_absolute": "training_loss[15] - training_loss[20], signed in loss units",
    "loss_decrease_percent": "100 * (training_loss[15] - training_loss[20]) / training_loss[15]",
    "zero_loss_policy": "percentage null with reason; retain absolute change and curve",
    "tail_slope": "ordinary least-squares loss slope across epochs 16,17,18,19,20",
    "warning": "selected_epoch == 20 AND AUROC[20] > AUROC[15] AND loss[15] > loss[20]",
    "warning_is_failure": False,
    "warning_interpretation": "possible fixed-budget sensitivity, not proven undertraining",
    "absence_of_warning": "not proof of convergence",
    "response": "no extra epochs, retries, exclusions, checkpoint changes, substitution, or retuning"}
NUMERIC_FIELDS = ("checkpoint_auroc_15", "checkpoint_auroc_20", "late_auc_change",
                  "training_loss_15", "training_loss_20", "loss_decrease_absolute",
                  "loss_decrease_percent", "loss_slope_16_20")
CSV_FIELDS = ("model_key", "family", "arm", "seed", "imported", "original_scope_id",
              "original_status", "selected_epoch", "selected_epoch_20",
              *NUMERIC_FIELDS, "loss_decrease_percent_undefined_reason",
              "budget_sensitivity_warning", "history_sha256", "complete_sha256")


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Missing/non-finite history value: " + name)
    return float(value)


def diagnose_history(history, best_epoch):
    if (not isinstance(history, list) or len(history) != 20
            or any(not isinstance(row, dict) for row in history)
            or [row.get("epoch") for row in history] != list(range(1, 21))
            or any(type(row["epoch"]) is not int for row in history)
            or type(best_epoch) is not int or not 1 <= best_epoch <= 20):
        raise ValueError("A diagnostic requires exactly actual epochs 1..20 and a selected epoch")
    aucs, losses = [], []
    selected = 0
    for index, row in enumerate(history):
        auc = _finite_number(row.get("checkpoint_auroc"), "checkpoint_auroc")
        loss = _finite_number(row.get("training_loss"), "training_loss")
        if not 0 <= auc <= 1 or loss < 0:
            raise ValueError("Invalid AUROC range or negative training loss")
        aucs.append(auc)
        losses.append(loss)
        if auc > aucs[selected]:
            selected = index
        if row.get("best_epoch") != selected + 1 or type(row["best_epoch"]) is not int:
            raise ValueError("History violates checkpoint-only selection or earliest-tie retention")
        if "best_checkpoint_auroc" in row:
            running = _finite_number(row["best_checkpoint_auroc"], "best_checkpoint_auroc")
            if running != aucs[selected]:
                raise ValueError("Recorded running maximum is inconsistent with actual AUROCs")
    if best_epoch != selected + 1:
        raise ValueError("Selected completion checkpoint differs from the immutable full history")
    loss_drop = losses[14] - losses[19]
    auc_delta = aucs[19] - aucs[14]
    boundary = best_epoch == 20
    return {"selected_epoch": best_epoch, "selected_epoch_20": boundary,
            "checkpoint_auroc_15": aucs[14], "checkpoint_auroc_20": aucs[19],
            "late_auc_change": auc_delta,
            "training_loss_15": losses[14], "training_loss_20": losses[19],
            "loss_decrease_absolute": loss_drop,
            "loss_decrease_percent": None if losses[14] == 0 else 100.0 * loss_drop / losses[14],
            "loss_decrease_percent_undefined_reason":
                "training loss at epoch 15 is exactly zero" if losses[14] == 0 else None,
            "loss_slope_16_20": sum((epoch - 18) * losses[epoch - 1]
                                   for epoch in range(16, 21)) / 10.0,
            "budget_sensitivity_warning": boundary and auc_delta > 0 and loss_drop > 0}


def summarize(rows):
    """Count every supplied fit; build() separately requires the complete registered neural matrix."""
    if not rows:
        raise ValueError("No histories may be silently omitted from a diagnostic summary")
    identities = [(row["family"], row["arm"], row["seed"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate fit in diagnostic denominators")
    for family, arm, seed in identities:
        if (family not in ("G", "H", "S", "O") or seed not in SEEDS
                or arm not in (("logits",) if family == "O" else ARMS)):
            raise ValueError("A diagnostic row belongs to a foreign neural fit")

    def group(items):
        result = {"denominator": len(items),
                  "seeds": sorted({row["seed"] for row in items}),
                  "selected_epoch_20_count": sum(row["selected_epoch_20"] for row in items),
                  "late_auc_increase_count": sum(row["late_auc_change"] > 0 for row in items),
                  "late_loss_decrease_count": sum(row["loss_decrease_absolute"] > 0 for row in items),
                  "budget_sensitivity_warning_count":
                      sum(row["budget_sensitivity_warning"] for row in items)}
        for field in NUMERIC_FIELDS:
            values = [row[field] for row in items if row[field] is not None]
            result[field] = {"mean": statistics.mean(values) if values else None,
                             "minimum": min(values) if values else None,
                             "maximum": max(values) if values else None,
                             "defined_count": len(values), "undefined_count": len(items) - len(values)}
        return result

    families = list(dict.fromkeys(row["family"] for row in rows))
    layers = list(dict.fromkeys((row["family"], row["arm"]) for row in rows))
    return {"all": group(rows),
            "by_family": {family: group([row for row in rows if row["family"] == family])
                          for family in families},
            "by_family_layer": {f"{family}/{arm}":
                group([row for row in rows if (row["family"], row["arm"]) == (family, arm)])
                for family, arm in layers}}


def _collect(root):
    from .train import verify_complete

    root, campaign, roles, bindings = _context(root, with_base=True)
    matrix = protocol.neural_matrix()
    expected = {(family, arm, seed) for family in ("G", "H", "S", "O")
                for arm in (("logits",) if family == "O" else ARMS) for seed in SEEDS}
    actual = [(row["family"], row["arm"], row["seed"]) for row in matrix]
    if len(actual) != COUNTS["neural_fits"] or set(actual) != expected:
        raise RuntimeError("Diagnostics require all registered neural histories, not a subset")
    expected_imports = {protocol.model_key("G", arm, seed)
                        for arm in ARMS for seed in protocol.IMPORTED_SEEDS}
    if set(campaign["reuse"]) != expected_imports:
        raise RuntimeError("The historical G fits may not be dropped or replaced")
    rows, runs = [], {}
    for family, arm, seed in actual:
        key = protocol.model_key(family, arm, seed)
        directory = Path(protocol.resolve_run(root, family, arm, seed)).resolve()
        complete, config = verify_complete(root, family, arm, seed)
        if (complete.get("complete") is not True or complete.get("completed_epochs") != 20
                or complete.get("selection_role") != "checkpoint"):
            raise RuntimeError("A missing or shorter-budget history blocks diagnostic completion")
        history = protocol.read(directory / "history.json")
        diagnosis = diagnose_history(history, complete["best_epoch"])
        files = {name: protocol.sha256(directory / name)
                 for name in ("config.json", "history.json", "complete.json", "best.safetensors")}
        imported = key in campaign["reuse"]
        status = campaign["reuse_groups"][str(seed)]["status"] if imported else "complete"
        if imported and status != ("complete_late_diagnostic" if seed == 7 else "complete"):
            raise RuntimeError("Imported seed7 lateness or replication status changed")
        runs[key] = {"path": str(directory), "files": files, "imported": imported,
                     "original_scope_id": config["scope_id"], "original_status": status}
        rows.append({"model_key": key, "family": family, "arm": arm, "seed": seed,
                     "imported": imported, "original_scope_id": config["scope_id"],
                     "original_status": status, **diagnosis, "history": history,
                     "history_sha256": files["history.json"],
                     "complete_sha256": files["complete.json"],
                     "completion_record": complete})
    return root, {**bindings, "runs": runs}, rows


def _validate_existing(root, inputs, rows):
    directory = root / "diagnostics"
    complete = protocol.read(directory / "complete.json")
    _match(complete, {"scope_id": SCOPE, "inputs": inputs, "neural_fit_count": COUNTS["neural_fits"],
                       "imported_fit_count": COUNTS["imported_neural_fits"],
                       "fresh_fit_count": COUNTS["new_neural_fits"], "policy": POLICY},
           "Diagnostic completion")
    _verify_files(directory, complete, ARTIFACTS)
    artifact = protocol.read(directory / "training_diagnostics.json")
    _match(artifact, {"scope_id": SCOPE, "complete": True, "inputs": inputs,
                     "policy": POLICY, "fits": rows, "summary": summarize(rows),
                     "neural_fit_count": COUNTS["neural_fits"]}, "Diagnostic table")
    return artifact, complete


def validate(root):
    root, inputs, rows = _collect(root)
    return _validate_existing(root, inputs, rows)


def build(root):
    protocol.require_slurm()
    root = Path(root).resolve()
    directory = root / "diagnostics"
    with _lock(directory, ".build.lock"):
        root, inputs, rows = _collect(root)
        if (directory / "complete.json").exists():
            return _validate_existing(root, inputs, rows)[1]
        _before_new_readouts(root)
        protocol.check_cutoff(root, "predictions")
        if any((directory / name).exists() for name in ARTIFACTS):
            raise RuntimeError("Incomplete diagnostic artifacts are preserved, not silently regenerated")
        artifact = {"schema_version": 1, "scope_id": SCOPE, "complete": True,
                    "inputs": inputs, "neural_fit_count": COUNTS["neural_fits"], "policy": POLICY,
                    "fits": rows, "summary": summarize(rows),
                    "non_neural_models": f"{COUNTS['meta_heads']} logistic heads and one L have solver audits, "
                                         "not epoch histories"}
        _write_json(directory / "training_diagnostics.json", artifact)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in CSV_FIELDS})
        _write_bytes(directory / "training_diagnostics.csv", buffer.getvalue().encode("utf-8"))
        protocol.check_cutoff(root, "predictions")
        _before_new_readouts(root)
        complete = {"schema_version": 1, "scope_id": SCOPE, "complete": True,
                    "inputs": inputs, **{key: value for key, value in inputs.items() if key != "runs"},
                    "neural_fit_count": COUNTS["neural_fits"],
                    "imported_fit_count": COUNTS["imported_neural_fits"],
                    "fresh_fit_count": COUNTS["new_neural_fits"],
                    "policy": POLICY, "files": {name: protocol.sha256(directory / name)
                                                for name in sorted(ARTIFACTS)},
                    "completed_unix": time.time(), "job_id": os.environ["SLURM_JOB_ID"]}
        _write_json(directory / "complete.json", complete)
        return _validate_existing(root, inputs, rows)[1]


def main():
    protocol.require_slurm()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.root)
    print(json.dumps({"complete": result["complete"], "neural_fit_count": COUNTS["neural_fits"],
                      "warnings_block_completion": False}), flush=True)


if __name__ == "__main__":
    main()
