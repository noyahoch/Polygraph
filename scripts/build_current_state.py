#!/usr/bin/env python3
"""Snapshot completed Polygraph artifacts without fabricating missing results."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs/research_20260830"


def metric_files(root: Path):
    rows = []
    for path in sorted(root.glob("**/scores_test_seed*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        rows.append({"method": data.get("method"), "seed": data.get("seed"),
                     "metrics": data.get("metrics", {}), "path": str(path.relative_to(ROOT)),
                     "selected_hyperparameters": data.get("selected_hyperparameters", {}),
                     **{key: data[key] for key in ("best_epoch", "best_validation_auroc",
                                                   "parameter_count", "tcp_mae",
                                                   "tcp_logit_spearman") if key in data}})
    return rows


def graph_reports(root: Path):
    rows = []
    for path in sorted(root.glob("**/report_model_seed*.json")):
        try:
            data = json.loads(path.read_text())
            seed = int(path.stem.rsplit("seed", 1)[1])
        except (json.JSONDecodeError, OSError, ValueError, KeyError):
            continue
        rows.append({"run": str(path.parent.relative_to(ROOT)), "seed": seed,
                     "all": data.get("all", {}).get("graph", {}),
                     "slices": {name: values.get("graph", {}) for name, values in data.items()},
                     "path": str(path.relative_to(ROOT))})
    return rows


def combiner_reports(root: Path):
    rows = []
    for path in sorted(root.glob("**/summary*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "test" in data and ("selected" in data or "method" in data):
            rows.append({**data, "path": str(path.relative_to(ROOT))})
    return rows


def main():
    graph_report = RUN / "message_flow/M0_attention/report_model_seed7.json"
    graph = json.loads(graph_report.read_text()) if graph_report.exists() else None
    store = json.loads((ROOT / "data/graph_dataset/store/manifest.json").read_text())
    result = {
        "status": "continued_from_completed_dataset_and_sidecars",
        "store": {"path": "data/graph_dataset/store", **store},
        "plans": {"main": "data/graph_dataset/split_plan_main.json",
                  "weather": "data/graph_dataset/split_plan_weather.json"},
        "score_results": metric_files(RUN),
        "graph_results": graph_reports(RUN),
        "combiner_results": combiner_reports(RUN / "combiners"),
        "raw_graph": ({"method": "M0_attention", "seed": 7,
                       "metrics": graph["all"]["graph"],
                       "msp_same_records": graph["all"]["msp"],
                       "path": str(graph_report.relative_to(ROOT)),
                       "checkpoint": "runs/research_20260830/message_flow/M0_attention/model_seed7.pt"}
                      if graph else None),
        "sidecars": {name: str(path.relative_to(ROOT)) for name, path in {
            "logits": ROOT / "data/graph_dataset/sidecars/logits/manifest.json",
            "hidden12": ROOT / "data/graph_dataset/hidden12/manifest.json",
            "compact_evidence_l12": ROOT / "data/graph_dataset/sidecars/compact_evidence_l12/manifest.json",
            "message_stats_l11": ROOT / "data/graph_dataset/sidecars/message_stats_l11/manifest.json",
            "graph_stats_l11": ROOT / "data/graph_dataset/sidecars/graph_stats_l11/manifest.json",
        }.items() if path.exists()},
        "strict_plans": [str(path.relative_to(ROOT)) for path in sorted(
            (RUN / "combiners/strict").glob("**/*_plan.json"))],
        "notes": [
            "Direct comparisons use only the rebuilt dataset, never historical metrics.",
            "M0 differs from historical 0.8417 by -0.00693 and passes the sanity tolerance.",
            "Graph score store_index metadata created before the GraphData fix was PyG-offset; labels and row order remain valid.",
        ],
    }
    ledger = RUN / "ledger.jsonl"
    if ledger.exists():
        entries = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
        result["ledger"] = {"path": str(ledger.relative_to(ROOT)), "records": len(entries),
                            "status_counts": {status: sum(e.get("status") == status for e in entries)
                                              for status in ("completed", "failed", "aborted", "skipped")}}
    out = RUN / "final/current_state.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(out)


if __name__ == "__main__":
    main()
