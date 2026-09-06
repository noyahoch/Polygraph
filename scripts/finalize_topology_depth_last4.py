#!/usr/bin/env python3
"""Create aligned final tables, severity slices, complementarity, and group bootstraps."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from polygraph.training.evaluate import detector_metrics
from polygraph.training.research_eval import paired_group_bootstrap, verify_score_registry


def bootstrap_comparison(item):
    """Evaluate one predeclared comparison; safe to run in a worker process."""
    name, candidate_path, reference_path = item
    candidate = load(candidate_path)
    reference_score = (1 - candidate["confidence"] if reference_path is None
                       else load(reference_path)["score"])
    result = paired_group_bootstrap(candidate["y"], candidate["score"], reference_score,
                                    candidate["image_id"], repetitions=2000, seed=20260905)
    return name, result

RUN = ROOT / "runs/topology_depth_last4_20260905"
PRIOR = ROOT / "runs/research_20260830/combiners/strict"


def load(path):
    return {key: value for key, value in np.load(path, allow_pickle=False).items()}


def flagged(score):
    mask = np.zeros(len(score), bool); mask[np.argsort(score, kind="stable")[-len(score)//2:]] = True
    return mask


def blind(y, score, reference):
    own, ref = flagged(score), flagged(reference)
    errors = y == 1; blind_spot = errors & ~ref
    return {"errors_caught": int((own & errors).sum()),
            "errors_reference_misses": int((own & errors & ~ref).sum()),
            "errors_reference_catches_method_misses": int((ref & errors & ~own).sum()),
            "blind_recovery": float((own & blind_spot).sum() / max(blind_spot.sum(), 1)),
            "flag_overlap": int((own & ref).sum())}


def runtime_for(path):
    total = 0.0
    ledger = RUN / "ledger.jsonl"
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            row = json.loads(line)
            if path.parent.name in row.get("experiment_id", ""):
                total += float(row.get("runtime_seconds", 0))
    return total


def parameter_count(path):
    checkpoint = path.parent / (path.name.replace("scores_test_", "model_").replace(".npz", ".pt"))
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload["state_dict"]
        history = payload.get("history", [])
        peak = max((row.get("peak_gpu_memory_mb") or 0 for row in history), default=None)
        selected, best_epoch = -float("inf"), None
        min_delta = float(payload.get("config", {}).get("min_delta", 0.0))
        for row in history:
            if float(row["val_auroc"]) > selected + min_delta:
                selected, best_epoch = float(row["val_auroc"]), row["epoch"]
        return (sum(value.numel() for value in state.values()), str(checkpoint.relative_to(ROOT)),
                peak, best_epoch)
    return None, None, None, None


def references(plan):
    root = PRIOR / plan
    return {"output": root / "output", "M5": root / "M5", "gate": root / "gate"}


def gate_directories(plan):
    directories = [references(plan)["gate"]]
    for directory in RUN.joinpath("gates").iterdir() if RUN.joinpath("gates").exists() else []:
        is_weather = directory.name.startswith("weather_")
        if (plan == "weather") == is_weather:
            directories.append(directory)
    return directories


def all_score_paths():
    paths = list(RUN.glob("**/scores_test_seed*.npz"))
    for plan in ("main", "weather"):
        for directory in references(plan).values(): paths += list(directory.glob("scores_test_seed*.npz"))
    return sorted(set(paths))


def plan_of(path):
    text = str(path)
    return "weather" if "/weather/" in text or "/weather_" in text else "main"


def output_path(plan, seed):
    path = references(plan)["output"] / f"scores_test_seed{seed}.npz"
    if not path.exists(): path = references(plan)["output"] / "scores_test_seed7.npz"
    return path


def main():
    final = RUN / "final"; final.mkdir(parents=True, exist_ok=True)
    rows, severity, artifacts = [], {}, []
    for path in all_score_paths():
        data, plan = load(path), plan_of(path)
        if not {"score", "y", "confidence", "image_id", "store_index"} <= set(data): continue
        seed = int(data.get("seed", np.asarray(0)).item())
        method = str(data.get("method_name", np.asarray(path.parent.name)).item())
        output = load(output_path(plan, seed))
        verify_score_registry([output_path(plan, seed), path])
        y, score, msp = data["y"], data["score"], 1 - data["confidence"]
        metric = detector_metrics(y, score)
        confident = data["confidence"] >= .9
        params, checkpoint, peak, best_epoch = parameter_count(path)
        msp_blind, output_blind = blind(y, score, msp), blind(y, score, output["score"])
        row = {"method": method, "artifact_name": path.parent.name, "plan": plan, "seed": seed,
               "AUROC": metric.get("auroc"), "AUPRC": metric.get("auprc"), "AURC": metric.get("aurc"),
               "risk05": metric.get("risk@0.5"), "risk08": metric.get("risk@0.8"),
               "risk09": metric.get("risk@0.9"),
               "confident_AUROC": detector_metrics(y[confident], score[confident]).get("auroc"),
               "blindspot_MSP": msp_blind["blind_recovery"],
               "blindspot_best_output": output_blind["blind_recovery"],
               "spearman_MSP": float(spearmanr(score, msp).statistic),
               "spearman_best_output": float(spearmanr(score, output["score"]).statistic),
               "runtime_seconds": runtime_for(path), "params": params, "checkpoint": checkpoint,
               "peak_gpu_memory_mb": peak, "best_epoch": best_epoch,
               "score_file": str(path.relative_to(ROOT))}
        rows.append(row); artifacts.append(row["score_file"])
        severity[row["score_file"]] = {
            "all": metric,
            "confidence_ge_0.9": detector_metrics(y[confident], score[confident]),
            **{f"severity_{level}": detector_metrics(y[data["severity"] == level],
                                                       score[data["severity"] == level])
               for level in range(1, 6) if (data["severity"] == level).any()}}
    fields = list(rows[0]) if rows else []
    for filename in ("model_comparison.csv", "experiment_table.csv"):
        with (final / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fields); writer.writeheader(); writer.writerows(rows)
    (final / "severity_metrics.json").write_text(json.dumps(severity, indent=2) + "\n")

    # Choose gates by OOF meta-validation only; test metrics cannot enter this decision.
    gate_candidates = {}
    for directory in gate_directories("main"):
        for summary in directory.glob("summary_seed*.json"):
            report = json.loads(summary.read_text())
            name = "GATE_M5" if directory == references("main")["gate"] else directory.name
            gate_candidates.setdefault(name, []).append(
                report["oof_meta_validation"]["auroc"])
    gate_selection = {"criterion": "mean OOF meta_val AUROC only", "test_metrics_used": False,
                      "candidates": {k: float(np.mean(v)) for k, v in gate_candidates.items()}}
    gate_selection["selected"] = max(gate_selection["candidates"], key=gate_selection["candidates"].get) \
        if gate_selection["candidates"] else None
    (final / "gate_selection.json").write_text(json.dumps(gate_selection, indent=2) + "\n")
    for plan in ("main", "weather"):
        candidates = {}
        for directory in gate_directories(plan):
            summaries = list(directory.glob("summary_seed*.json"))
            if summaries:
                candidates[directory] = np.mean([json.loads(p.read_text())["oof_meta_validation"]["auroc"]
                                                 for p in summaries])
        if candidates:
            chosen = max(candidates, key=candidates.get)
            payload = {"selected_by": np.asarray("mean OOF meta_val AUROC"),
                       "method_name": np.asarray(chosen.name)}
            for score_file in chosen.glob("scores_test_seed*.npz"):
                seed = int(score_file.stem.rsplit("seed", 1)[1]); data = load(score_file)
                payload[f"score_seed{seed}"] = data["score"]
                for key in ("y", "confidence", "image_id", "source_id", "severity", "store_index"):
                    payload.setdefault(key, data[key])
            np.savez_compressed(final / f"best_scores_{plan}.npz", **payload)

    bootstraps = {}
    selection = json.loads((final / "selection_manifest.json").read_text())
    comparisons = []
    s1_name, s2_name = selection["controls"]["S1"], selection["controls"]["S2"]
    best_gnn_name = selection["selected_gnn"]
    best_multi_name = (selection.get("multilayer") or {}).get("best")
    for seed in (7, 1, 2):
        output_file = output_path("main", seed); output = load(output_file)
        m5 = references("main")["M5"] / f"scores_test_seed{seed}.npz"
        s1 = RUN / "controls" / s1_name / f"scores_test_seed{seed}.npz"
        s2 = RUN / "controls" / s2_name / f"scores_test_seed{seed}.npz"
        comparisons.append((f"P0_output_seed{seed}_minus_MSP", output_file, None))
        if m5.exists() and s1.exists(): comparisons.append((f"S1_seed{seed}_minus_M5", s1, m5))
        if m5.exists() and s2.exists(): comparisons.append((f"P1_M5_seed{seed}_minus_S2", m5, s2))
        gate_m5 = references("main")["gate"] / f"scores_test_seed{seed}.npz"
        gate_s2 = RUN / "gates" / s2_name / f"scores_test_seed{seed}.npz"
        if gate_m5.exists() and gate_s2.exists():
            comparisons.append((f"P2_GATE_M5_seed{seed}_minus_GATE_S2", gate_m5, gate_s2))
        new_gnn = RUN / "architectures" / best_gnn_name / f"scores_test_seed{seed}.npz"
        if new_gnn.exists() and m5.exists():
            comparisons.append((f"P3_{best_gnn_name}_seed{seed}_minus_M5", new_gnn, m5))
        if best_multi_name:
            multi = RUN / "multilayer" / best_multi_name / f"scores_test_seed{seed}.npz"
            if multi.exists() and new_gnn.exists():
                comparisons.append((f"P4_{best_multi_name}_seed{seed}_minus_{best_gnn_name}",
                                    multi, new_gnn))
    l1 = RUN / "multilayer/L1_union_graph/scores_test_seed7.npz"
    l1set = RUN / "multilayer/L1_SET_union_endpoint/scores_test_seed7.npz"
    if l1.exists() and l1set.exists():
        comparisons.append(("P5_L1_union_graph_seed7_minus_L1_SET", l1, l1set))
    if gate_selection["selected"]:
        for path in (RUN / "gates" / gate_selection["selected"]).glob("scores_test_seed*.npz"):
            seed = int(path.stem.rsplit("seed", 1)[1])
            comparisons.append((f"{gate_selection['selected']}_seed{seed}_vs_output",
                                path, output_path("main", seed)))
            comparisons.append((f"P7_GATE_{gate_selection['selected']}_seed{seed}_minus_MSP",
                                path, None))
        selected_internal = RUN / "architectures" / best_gnn_name
        for path in selected_internal.glob("scores_test_seed*.npz"):
            seed = int(path.stem.rsplit("seed", 1)[1])
            comparisons.append((f"P6_{best_gnn_name}_seed{seed}_minus_output",
                                path, output_path("main", seed)))
    # Each comparison is independent.  Parallelizing at this level preserves
    # the exact group-resampling procedure and seed while avoiding a long
    # single-core finalization tail.
    completed = Parallel(n_jobs=min(8, os.cpu_count() or 1), prefer="processes")(
        delayed(bootstrap_comparison)(item) for item in comparisons)
    bootstraps.update(completed)
    primary = {name: result for name, result in bootstraps.items() if name.startswith("P")}
    raw_p = {name: min(1.0, 2 * min(result["auroc_delta"]["fraction_gt_zero"],
                                    1 - result["auroc_delta"]["fraction_gt_zero"]))
             for name, result in primary.items()}
    ordered = sorted(raw_p, key=raw_p.get)
    adjusted, running, count = {}, 0.0, len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, raw_p[name] * (count - rank)))
        adjusted[name] = running
    bootstraps["multiplicity"] = {"method": "Holm adjustment of two-sided bootstrap sign-tail fractions",
                                   "scope": "predeclared P comparisons", "raw_p": raw_p,
                                   "holm_adjusted_p": adjusted}
    (final / "paired_bootstrap.json").write_text(json.dumps(bootstraps, indent=2) + "\n")
    # Keep the cross-study machine-readable filename expected by downstream
    # report tooling while retaining the more explicit follow-up filename.
    (final / "bootstrap_deltas.json").write_text(json.dumps(bootstraps, indent=2) + "\n")
    aggregate = {}
    for key in sorted({(r["method"], r["artifact_name"], r["plan"]) for r in rows}):
        subset = [r for r in rows if (r["method"], r["artifact_name"], r["plan"]) == key]
        values = np.asarray([r["AUROC"] for r in subset], dtype=float)
        aggregate["|".join(key)] = {"seeds": [r["seed"] for r in subset],
                                    "auroc_mean": float(values.mean()),
                                    "auroc_std": float(values.std(ddof=0))}
    summary = {"rows": len(rows), "selection": selection, "gate_selection": gate_selection,
               "three_seed_aggregates": aggregate,
               "severity_metrics": "severity_metrics.json", "bootstrap": "paired_bootstrap.json"}
    (final / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (final / "artifact_map.json").write_text(json.dumps({"artifacts": sorted(artifacts)}, indent=2) + "\n")
    mandatory = {name: any(name in str(p) for p in all_score_paths())
                 for name in (selection["controls"]["S1"], selection["controls"]["S2"],
                              "L0_token_trajectory", "L1_union_graph", "L1_SET_union_endpoint")}
    (final / "mandatory_completion.json").write_text(json.dumps(mandatory, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
