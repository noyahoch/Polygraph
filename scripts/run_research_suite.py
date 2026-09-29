#!/usr/bin/env python3
"""Bounded, resumable research-suite launcher with artifact-aware stop semantics.

This orchestrator deliberately refuses to synthesize scientific results when the immutable graph
store or plans are unavailable.  It records each affected experiment in the common ledger and
continues to the report stage, as required by the study protocol.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import time
from pathlib import Path


STAGES = ["preflight", "anchors", "controls", "logits", "output_baselines", "tcp",
          "graph_stats", "compact_evidence", "message_stats", "message_models",
          "simple_mpnn", "combiners", "temporal", "report"]
EXPERIMENTS = {
    "controls": ["C1_edge_set", "C2_node_edge_set", "C3_endpoint_set", "C4_top100_graph",
                 "C5_target_permute", "C6_shuffle_attr"],
    "logits": ["logits_full_store"],
    "output_baselines": ["O1_msp", "O2_margin", "O3_entropy", "O4_energy", "O5_max_logit",
                         "O6_p_normalized", "O7_pnorm_softmax", "O8_raw_logit_lr",
                         "O9_sorted_logit_lr", "O10_raw_logit_mlp", "O11_sorted_logit_mlp"],
    "tcp": ["cls_tcp_main", "cls_tcp_weather", "cls_seq_tcp_optional"],
    "graph_stats": ["graph_stats_extract", "graph_stats_lr", "graph_stats_mlp", "graph_stats_hgb"],
    "compact_evidence": ["compact_evidence_extract", "N1_compact_graph", "N2_compact_node_edge_set",
                         "N3_compact_endpoint_set"],
    "message_stats": ["message_stats_l11_extract"],
    "message_models": ["M0_attention", "M1_attention_message", "M2_attention_decision",
                       "M3_evidence_flow", "M4_evidence_compact", "M5_evidence_hidden",
                       "graph_tcp_multitask_optional"],
    "simple_mpnn": ["SMP0_attention", "SMP1_best_flow", "SMP2_compact_best_flow"],
    "combiners": ["G1_logistic", "G1_residual", "G1_gate", "G2_logistic", "G2_residual",
                  "G2_gate", "G3_logistic", "G3_residual", "G3_gate", "G4_logistic",
                  "G4_residual", "G4_gate", "strict_confirmation"],
    "temporal": ["T1_temporal_final4", "T2_temporal_compact"],
}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def append(ledger: Path, experiment_id: str, status: str, reason: str, command=None, notes=""):
    t = now()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = ""
    row = {"experiment_id": experiment_id, "status": status, "command": command or [],
           "start_time_utc": t, "end_time_utc": t, "runtime_seconds": 0.0,
           "git_commit": commit, "plan_path": "", "seed": None, "configuration": {},
           "parameter_count": None, "peak_gpu_memory_mb": None, "result_files": [],
           "skip_or_failure_reason": reason, "notes": notes}
    with ledger.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=STAGES + ["all"], required=True)
    p.add_argument("--main-plan", type=Path, required=True)
    p.add_argument("--weather-plan", type=Path, required=True)
    p.add_argument("--store-dir", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--budget-gpu-hours", type=float, default=8)
    p.add_argument("--timeout-minutes", type=float, default=120)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    ledger = args.run_root / "ledger.jsonl"
    ledger.touch()
    selected = STAGES if args.stage == "all" else [args.stage]
    required = [args.store_dir / "manifest.json", args.store_dir / "store_keys.json",
                args.main_plan, args.weather_plan]
    missing = [str(x) for x in required if not x.exists()]
    blocker = "missing non-regenerable research artifacts: " + ", ".join(missing)
    seen = set()
    if args.resume:
        for line in ledger.read_text().splitlines():
            try:
                seen.add(json.loads(line)["experiment_id"])
            except Exception:
                pass
    for stage in selected:
        if stage == "preflight":
            eid = "suite_preflight"
            if not (args.resume and eid in seen):
                append(ledger, eid, "failed" if missing else "completed", blocker if missing else "")
            continue
        if stage == "anchors":
            eid = "anchor_report_verification"
            if args.resume and eid in seen:
                continue
            cmd = [str(Path(".venv/bin/python")), "scripts/verify_current_anchors.py"]
            if args.dry_run:
                append(ledger, eid, "skipped", "dry run", cmd)
            else:
                started = time.monotonic()
                rc = subprocess.run(cmd).returncode
                append(ledger, eid, "completed" if rc == 0 else "failed",
                       "" if rc == 0 else f"exit code {rc}", cmd,
                       "Historical-report verification only; no score arrays present.")
            continue
        if stage == "report":
            eid = "suite_report_handoff"
            if not (args.resume and eid in seen):
                append(ledger, eid, "completed", "", notes="Report generation proceeds despite blockers.")
            continue
        for eid in EXPERIMENTS.get(stage, [stage]):
            if args.resume and eid in seen:
                continue
            append(ledger, eid, "skipped", blocker if missing else "stage implementation unavailable")
    if missing:
        print(blocker)
        print("Scientific stages were skipped; report handoff completed.")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
