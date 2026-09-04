#!/usr/bin/env python3
"""Build final machine-readable tables from immutable completed score artifacts.

This script does no fitting or selection. It only verifies row alignment, calculates
predeclared metrics/slices/complementarity, and performs base-photograph bootstraps.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from polygraph.config import ALL_SOURCES
from polygraph.data.splits import SplitPlan
from polygraph.training.evaluate import detector_metrics
from polygraph.training.research_eval import paired_group_bootstrap


RUN = ROOT / "runs/research_20260830"
FINAL = RUN / "final"
SEEDS = (7, 1, 2)


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def aligned(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> bool:
    return all(np.array_equal(a[key], b[key]) for key in
               ("y", "image_id", "source_id", "severity"))


def scalar(value) -> str:
    return str(np.asarray(value).item())


def references(path: Path, data: dict[str, np.ndarray]):
    plan = "weather" if "/weather/" in str(path) else "main"
    seed = int(np.asarray(data.get("seed", 7)).item())
    if "/strict/" in str(path):
        base = RUN / f"combiners/strict/{plan}/output/scores_test_seed{seed}.npz"
    else:
        base = RUN / f"output/{plan}/raw_logit_mlp/scores_test_seed{seed}.npz"
        if not base.exists():
            base = RUN / f"output/{plan}/raw_logit_mlp/scores_test_seed7.npz"
    output = load(base) if base.exists() else None
    if output is not None and not aligned(data, output):
        output = None
    return plan, output


def flag_stats(y, candidate, reference, budget=.5):
    n = int(np.floor(len(y) * budget))
    a = np.zeros(len(y), bool); b = np.zeros(len(y), bool)
    a[np.argsort(candidate, kind="stable")[-n:]] = True
    b[np.argsort(reference, kind="stable")[-n:]] = True
    blind = (y == 1) & ~b
    return {
        "errors_caught": int((a & (y == 1)).sum()),
        "errors_reference_misses": int((a & ~b & (y == 1)).sum()),
        "errors_reference_catches_method_misses": int((b & ~a & (y == 1)).sum()),
        "flag_overlap": int((a & b).sum()),
        "blind_recovery": float((a & blind).sum() / max(int(blind.sum()), 1)),
        "spearman": float(spearmanr(candidate, reference).statistic),
    }


def score_rows():
    rows, details = [], {}
    for path in sorted(RUN.glob("**/scores_test_seed*.npz")):
        # Exclude feature sidecars and the exported aggregate files themselves.
        if FINAL in path.parents or "sidecars" in path.parts:
            continue
        try:
            data = load(path)
        except Exception:
            continue
        required = {"score", "y", "confidence", "image_id", "source_id", "severity"}
        if not required.issubset(data):
            continue
        method = scalar(data.get("method_name", path.parent.name))
        if "/strict/" in str(path):
            method = "strict_" + method
        seed = int(np.asarray(data.get("seed", 7)).item())
        plan, output = references(path, data)
        split_plan = SplitPlan.load(ROOT / f"data/graph_dataset/split_plan_{plan}.json")
        seen_sources = {key.source for key in split_plan.splits["train"]}
        names = np.asarray([ALL_SOURCES[int(index)] for index in data["source_id"]])
        unseen = np.asarray([(name not in seen_sources and name != "clean_test") for name in names])
        clean = names == "clean_test"
        seen = np.asarray([(name in seen_sources and name != "clean_test") for name in names])
        metrics = detector_metrics(data["y"], data["score"])
        confident = data["confidence"] >= .9
        confident_metrics = detector_metrics(data["y"][confident], data["score"][confident])
        msp = 1 - data["confidence"]
        msp_comp = flag_stats(data["y"], data["score"], msp)
        output_comp = flag_stats(data["y"], data["score"], output["score"]) if output else None
        row = {
            "method": method, "plan": plan, "seed": seed,
            "AUROC": metrics.get("auroc"), "AUPRC": metrics.get("auprc"),
            "AURC": metrics.get("aurc"), "risk05": metrics.get("risk@0.5"),
            "risk08": metrics.get("risk@0.8"), "risk09": metrics.get("risk@0.9"),
            "confident_AUROC": confident_metrics.get("auroc"),
            "blindspot_MSP": msp_comp["blind_recovery"],
            "blindspot_best_output": output_comp["blind_recovery"] if output_comp else None,
            "runtime": None, "params": None,
            "checkpoint": str(path.parent / f"model_seed{seed}.pt")
                          if (path.parent / f"model_seed{seed}.pt").exists() else "",
            "score_file": str(path.relative_to(ROOT)),
        }
        rows.append(row)
        details[str(path.relative_to(ROOT))] = {
            "all": metrics, "confident": confident_metrics,
            "severity": {str(level): detector_metrics(
                data["y"][data["severity"] == level], data["score"][data["severity"] == level])
                for level in sorted(set(map(int, data["severity"])))},
            "source_slices": {
                "clean": detector_metrics(data["y"][clean], data["score"][clean]),
                "seen_corruptions": detector_metrics(data["y"][seen], data["score"][seen]),
                "unseen_sources": detector_metrics(data["y"][unseen], data["score"][unseen]),
            },
            "msp_complementarity": msp_comp,
            "output_complementarity": output_comp,
        }
    return rows, details


def export_best(plan: str):
    root = RUN / f"combiners/strict/{plan}"
    payload = {}
    for seed in SEEDS:
        for name, path in {
            "gate": root / f"gate/scores_test_seed{seed}.npz",
            "internal": root / f"M5/scores_test_seed{seed}.npz",
            "output": root / f"output/scores_test_seed{seed}.npz",
        }.items():
            if path.exists():
                data = load(path)
                payload[f"{name}_score_seed{seed}"] = data["score"]
                if "y" not in payload:
                    for key in ("y", "confidence", "margin", "image_id", "source_id",
                                "severity", "store_index"):
                        payload[key] = data[key]
    if payload:
        # Explicitly retain MSP alongside the trained candidates.
        payload["msp_score"] = 1 - payload["confidence"]
        np.savez_compressed(FINAL / f"best_scores_{plan}.npz", **payload)
    return payload


def bootstraps(plan: str):
    root = RUN / f"combiners/strict/{plan}"
    results = {}
    for seed in SEEDS:
        paths = {name: root / f"{sub}/scores_test_seed{seed}.npz" for name, sub in
                 (("gate", "gate"), ("internal", "M5"), ("output", "output"))}
        if not all(path.exists() for path in paths.values()):
            continue
        data = {name: load(path) for name, path in paths.items()}
        if not aligned(data["gate"], data["output"]) or not aligned(data["internal"], data["output"]):
            raise ValueError(f"strict {plan} seed {seed} score misalignment")
        y, groups = data["gate"]["y"], data["gate"]["image_id"]
        msp = 1 - data["gate"]["confidence"]
        results[str(seed)] = {
            "gate_vs_output": paired_group_bootstrap(
                y, data["gate"]["score"], data["output"]["score"], groups),
            "gate_vs_msp": paired_group_bootstrap(y, data["gate"]["score"], msp, groups),
            "internal_vs_output": paired_group_bootstrap(
                y, data["internal"]["score"], data["output"]["score"], groups),
        }
    return results


def ledger_csv():
    ledger_path = RUN / "ledger.jsonl"
    entries = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    existing = {entry["experiment_id"] for entry in entries}
    now = datetime.now(timezone.utc).isoformat()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    final_skips = [
        ("cls_seq_tcp_main_seeds1_2", None,
         "seed-7 cls_seq_tcp AUROC 0.86474 was below final-CLS TCP 0.86892, ordinary CLS sequence 0.88291, and MSP 0.86510; confirmation triage failed"),
        ("cls_seq_tcp_weather", None,
         "main seed-7 cls_seq_tcp failed the predeclared promotion triage, so a cross-plan repeat would not affect the selected finalist"),
        ("M5_tcp_multitask_confirmation", None,
         "seed-7 multitask error head 0.88342 was below single-task M5 0.88625; TCP head and fixed average were weaker, so confirmation triage failed"),
    ]
    appended = []
    for experiment_id, seed, reason in final_skips:
        if experiment_id in existing:
            continue
        appended.append({"experiment_id": experiment_id, "status": "skipped", "command": [],
                         "start_time_utc": now, "end_time_utc": now, "runtime_seconds": 0.0,
                         "git_commit": commit, "plan_path": "", "seed": seed,
                         "configuration": {}, "parameter_count": None,
                         "peak_gpu_memory_mb": None, "result_files": [],
                         "skip_or_failure_reason": reason, "notes": "post-triage decision"})
    if appended:
        with ledger_path.open("a") as handle:
            for entry in appended:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
        entries.extend(appended)
    fields = ["experiment_id", "status", "command", "start_time_utc", "end_time_utc",
              "runtime_seconds", "git_commit", "plan_path", "seed", "configuration",
              "parameter_count", "peak_gpu_memory_mb", "result_files",
              "skip_or_failure_reason", "notes"]
    with (FINAL / "experiment_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            writer.writerow({key: json.dumps(entry.get(key)) if isinstance(entry.get(key), (dict, list))
                             else entry.get(key) for key in fields})
    return entries


def main():
    FINAL.mkdir(parents=True, exist_ok=True)
    rows, details = score_rows()
    fields = ["method", "plan", "seed", "AUROC", "AUPRC", "AURC", "risk05",
              "risk08", "risk09", "confident_AUROC", "blindspot_MSP",
              "blindspot_best_output", "runtime", "params", "checkpoint", "score_file"]
    with (FINAL / "model_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    ledgers = ledger_csv()
    best = {plan: export_best(plan) for plan in ("main", "weather")}
    existing_main = FINAL / "bootstrap_strict_main.json"
    bootstrap = {
        "main": json.loads(existing_main.read_text()) if existing_main.exists() else bootstraps("main"),
        "weather": bootstraps("weather"),
        "output_vs_msp": json.loads((FINAL / "bootstrap_output_vs_msp.json").read_text())
                         if (FINAL / "bootstrap_output_vs_msp.json").exists() else None,
    }
    (FINAL / "bootstrap_deltas.json").write_text(json.dumps(bootstrap, indent=2) + "\n")

    def strict_aurocs(plan, family):
        root = RUN / f"combiners/strict/{plan}/{family}"
        result = []
        for seed in SEEDS:
            path = root / f"scores_test_seed{seed}.npz"
            if path.exists():
                data = load(path); result.append(float(detector_metrics(data["y"], data["score"])["auroc"]))
        return result

    main_msp = float(detector_metrics(best["main"]["y"], best["main"]["msp_score"])["auroc"])
    weather_msp = float(detector_metrics(best["weather"]["y"], best["weather"]["msp_score"])["auroc"])
    summary = {
        "dataset_records": 75000,
        "scientific_baseline": "strict_raw_logit_mlp",
        "main": {"msp": main_msp,
                 "strict_output": strict_aurocs("main", "output"),
                 "strict_M5": strict_aurocs("main", "M5"),
                 "strict_gate": strict_aurocs("main", "gate")},
        "weather": {"msp": weather_msp,
                    "strict_output": strict_aurocs("weather", "output"),
                    "strict_M5": strict_aurocs("weather", "M5"),
                    "strict_gate": strict_aurocs("weather", "gate")},
        "model_score_details": details,
        "ledger_records": len(ledgers),
    }
    (FINAL / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "ledger": len(ledgers),
                      "main_gate": summary["main"]["strict_gate"],
                      "weather_gate": summary["weather"]["strict_gate"]}, indent=2))


if __name__ == "__main__":
    main()
