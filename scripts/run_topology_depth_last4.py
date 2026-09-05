#!/usr/bin/env python3
"""Bounded, resumable orchestrator for the topology/depth/last-four-block study.

Selection stages only inspect checkpoint validation histories. Test score collection is
isolated in the evaluate stage and refuses to run before selection_manifest.json exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv/bin/python"
DEFAULT_RUN = ROOT / "runs/topology_depth_last4_20260905"
PRIOR = ROOT / "runs/research_20260830"
STORE = ROOT / "data/graph_dataset/store"
MAIN_TRAIN = PRIOR / "combiners/strict/main/detector_train_plan.json"
MAIN_EVAL = PRIOR / "combiners/strict/main/combiner_eval_plan.json"
WEATHER_TRAIN = PRIOR / "combiners/strict/weather/detector_train_plan.json"
WEATHER_EVAL = PRIOR / "combiners/strict/weather/combiner_eval_plan.json"


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def planned_matrix():
    common = dict(blocks=[11], node_features="hidden", edge_features="evidence_flow",
                  protocol="train/original train; select/base_val; gate/meta_val; frozen test",
                  normalization="none beyond stored feature transforms", sparsity="store tau")
    rows = [
        dict(id="S0_h64", family="hidden_token_set", depth=0, width=64, seeds=[7], mandatory=True,
             expected="full-hidden tokens without edges", **common),
        dict(id="S1_h96", family="m5_node_edge_set", depth=0, width=96, seeds=[7], mandatory=True,
             expected="M5 values without endpoints", **common),
        dict(id="S1_h128", family="m5_node_edge_set", depth=0, width=128, seeds=[7], mandatory=True,
             expected="larger M5 values without endpoints", **common),
        dict(id="S2_h64", family="m5_endpoint_set", depth=0, width=64, seeds=[7], mandatory=True,
             expected="M5 endpoint records without message passing", **common),
        dict(id="S2_h96", family="m5_endpoint_set", depth=0, width=96, seeds=[7], mandatory=True,
             expected="larger M5 endpoint records", **common),
    ]
    for family in ("transformerconv_residual", "gine", "edge_gated_mean", "gatv2"):
        rows.append(dict(id="A_" + family, family=family, depth=2, width=64, seeds=[7],
                         mandatory=False, expected="edge-aware common-scaffold screen", **common))
    rows += [dict(id=f"LAST4_{name}", family=name, depth=depth, width=64, seeds=[7], mandatory=True,
                  blocks=[8, 9, 10, 11], node_features="hidden_last4", edge_features="evidence_flow_last4",
                  protocol=common["protocol"], normalization=common["normalization"], sparsity=common["sparsity"],
                  expected=description)
             for name, depth, description in [
                 ("token_trajectory_set", 0, "ordered per-token four-block trajectory without graph"),
                 ("union_graph", 2, "four-block union attributed graph"),
                 ("union_endpoint_set", 0, "matched four-block endpoint records")]]
    return rows


class Suite:
    def __init__(self, args):
        self.args, self.run = args, args.run_root.resolve()
        for name in ("inventory", "logs", "configs", "controls", "architectures", "multilayer",
                     "gates", "statistics", "final"):
            (self.run / name).mkdir(parents=True, exist_ok=True)
        self.ledger = self.run / "ledger.jsonl"

    def record(self, row):
        with self.ledger.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def command(self, experiment_id, command, mandatory=False, timeout_minutes=None):
        log = self.run / "logs" / f"{experiment_id}.log"
        started, t0 = utc(), time.monotonic()
        if self.args.dry_run:
            print("DRY", *command)
            return 0
        status, reason = "completed", None
        try:
            with log.open("a") as output:
                output.write(f"\n[{started}] {' '.join(map(str, command))}\n")
                output.flush()
                result = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=output,
                                        stderr=subprocess.STDOUT,
                                        timeout=60 * (timeout_minutes or self.args.timeout_minutes))
            if result.returncode:
                status, reason = "failed", f"exit code {result.returncode}"
        except subprocess.TimeoutExpired:
            status, reason = "aborted", "bounded runtime cutoff; resumable state retained"
        row = dict(experiment_id=experiment_id, status=status, command=list(map(str, command)),
                   start_time_utc=started, end_time_utc=utc(), runtime_seconds=time.monotonic() - t0,
                   mandatory=mandatory, log=str(log.relative_to(ROOT)), reason=reason)
        self.record(row); self.update_progress()
        if status != "completed" and mandatory:
            raise RuntimeError(f"mandatory {experiment_id} {status}: {reason}; see {log}")
        return 0 if status == "completed" else 1

    def inventory(self):
        manifests = [STORE / "manifest.json", ROOT / "data/graph_dataset/hidden12/manifest.json",
                     ROOT / "data/graph_dataset/sidecars/message_stats_l11/manifest.json"]
        plans = [MAIN_TRAIN, MAIN_EVAL, WEATHER_TRAIN, WEATHER_EVAL]
        resolved = {
            "created_utc": utc(), "store": str(STORE),
            "manifests": {str(p.relative_to(ROOT)): {"sha256": sha(p), "content": json.loads(p.read_text())}
                          for p in manifests},
            "plans": {str(p.relative_to(ROOT)): {"sha256": sha(p),
                      "sizes": {k: len(v) for k, v in json.loads(p.read_text())["splits"].items()}}
                      for p in plans},
            "prior_strict": {"main_M5": str(PRIOR / "combiners/strict/main/M5"),
                             "main_output": str(PRIOR / "combiners/strict/main/output"),
                             "weather_M5": str(PRIOR / "combiners/strict/weather/M5")},
            "environment": {"python": sys.version, "torch": torch.__version__,
                            "torch_cuda": torch.version.cuda,
                            "cuda_available": torch.cuda.is_available(),
                            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        }
        # Strict partitions are checked by immutable (source,severity,base) keys.
        for plan_name in ("main", "weather"):
            detector = json.loads((MAIN_TRAIN if plan_name == "main" else WEATHER_TRAIN).read_text())["splits"]
            combiner = json.loads((MAIN_EVAL if plan_name == "main" else WEATHER_EVAL).read_text())["splits"]
            a, b = set(map(tuple, detector["val"])), set(map(tuple, combiner["val"]))
            # Every corruption/source variant with the same CIFAR base index is one
            # photograph group; source must not weaken this disjointness check.
            ga, gb = {x[2] for x in a}, {x[2] for x in b}
            resolved.setdefault("strict_checks", {})[plan_name] = {
                "base_val_records": len(a), "meta_val_records": len(b),
                "record_overlap": len(a & b), "base_photo_overlap": len(ga & gb)}
            if a & b or ga & gb:
                raise RuntimeError(f"strict {plan_name} base_val/meta_val overlap")
        atomic_json(self.run / "inventory/resolved_artifacts.json", resolved)
        atomic_json(self.run / "configs/planned_matrix.json", planned_matrix())

    def train(self, experiment_id, family, width, seed, category="controls", depth=2,
              batch=128, mandatory=False, jk=False):
        out = self.run / category / experiment_id
        checkpoint = out / f"model_seed{seed}.pt"
        if checkpoint.exists() and self.args.resume:
            return
        cmd = [PY, "-m", "polygraph.training", "train", "--plan", MAIN_TRAIN,
               "--out-dir", out, "--layers", "last", "--architecture", family,
               "--node-features", "hidden", "--edge-features", "evidence_flow",
               "--message-stats-dir", "data/graph_dataset/sidecars/message_stats_l11",
               "--hidden-dim", width, "--gnn-layers", depth, "--batch-size", batch,
               "--seeds", seed, "--epochs-per-process", 0]
        if jk: cmd.append("--jumping-knowledge")
        self.command(f"{experiment_id}_seed{seed}", cmd, mandatory=mandatory,
                     timeout_minutes=150 if mandatory else 90)

    @staticmethod
    def best_val(path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return max(float(row["val_auroc"]) for row in payload["history"])

    def controls(self):
        self.train("S0_h64", "hidden_token_set", 64, 7, mandatory=True, batch=192)
        for name, family, width in (("S1_h96", "m5_node_edge_set", 96),
                                    ("S1_h128", "m5_node_edge_set", 128),
                                    ("S2_h64", "m5_endpoint_set", 64),
                                    ("S2_h96", "m5_endpoint_set", 96)):
            self.train(name, family, width, 7, mandatory=True, batch=96)
        selected = {}
        for prefix in ("S1", "S2"):
            candidates = sorted((self.run / "controls").glob(prefix + "_*"))
            scored = {p.name: self.best_val(p / "model_seed7.pt") for p in candidates}
            selected[prefix] = max(scored, key=scored.get)
            selected[prefix + "_validation"] = scored
        atomic_json(self.run / "configs/control_selection.json", selected)
        for prefix in ("S1", "S2"):
            name = selected[prefix]; ckpt = torch.load(self.run / "controls" / name / "model_seed7.pt",
                                                       map_location="cpu", weights_only=False)
            family, width = ckpt["config"]["architecture"], ckpt["config"]["hidden_dim"]
            for seed in (1, 2): self.train(name, family, width, seed, mandatory=True, batch=96)

    def architectures(self):
        for family in ("transformerconv_residual", "gine", "edge_gated_mean", "gatv2"):
            self.train("A_" + family, family, 64, 7, category="architectures", batch=48)
        scored = {p.name: self.best_val(p / "model_seed7.pt")
                  for p in (self.run / "architectures").glob("A_*") if (p / "model_seed7.pt").exists()}
        if scored: atomic_json(self.run / "configs/architecture_selection.json",
                               {"selected": max(scored, key=scored.get), "base_val": scored})

    def depth(self):
        selected = json.loads((self.run / "configs/architecture_selection.json").read_text())["selected"]
        family = selected.removeprefix("A_")
        # The second family is selected on base_val only; completed depth-2 scores are reused.
        scores = json.loads((self.run / "configs/architecture_selection.json").read_text())["base_val"]
        families = [x.removeprefix("A_") for x in sorted(scores, key=scores.get, reverse=True)[:2]]
        for fam in families:
            for depth in (1, 4):
                self.train(f"D_{fam}_d{depth}", fam, 64, 7, category="architectures", depth=depth,
                           batch=48, jk=depth >= 4)

    def select(self):
        controls = json.loads((self.run / "configs/control_selection.json").read_text())
        architectures = json.loads((self.run / "configs/architecture_selection.json").read_text())
        candidates = {}
        for p in (self.run / "architectures").glob("*/model_seed7.pt"):
            candidates[p.parent.name] = self.best_val(p)
        manifest = {"frozen_utc": utc(), "selection_data": "strict base_val only",
                    "test_metrics_consulted": False, "controls": controls,
                    "architectures": architectures,
                    "depth_base_val": candidates, "selected_gnn": max(candidates, key=candidates.get),
                    "planned_matrix_sha256": sha(self.run / "configs/planned_matrix.json")}
        atomic_json(self.run / "final/selection_manifest.json", manifest)

    def evaluate_one(self, category, name, plan=MAIN_EVAL):
        out = self.run / category / name
        cmd = [PY, "-m", "polygraph.training", "evaluate", "--run-dir", out,
               "--plan", plan, "--no-baselines"]
        self.command(f"evaluate_{name}", cmd, timeout_minutes=150)

    def evaluate(self):
        if not (self.run / "final/selection_manifest.json").exists():
            raise RuntimeError("selection manifest must be frozen before test evaluation")
        selected = json.loads((self.run / "configs/control_selection.json").read_text())
        for name in ("S0_h64", selected["S1"], selected["S2"]): self.evaluate_one("controls", name)
        best = json.loads((self.run / "final/selection_manifest.json").read_text())["selected_gnn"]
        self.evaluate_one("architectures", best)

    def update_progress(self):
        lines = ["# Topology/depth/last-four study — live durable record", "",
                 f"Updated: {utc()}", "", "Test evaluation is withheld until selection is frozen.", "",
                 "| experiment | seed | best base-val AUROC | epochs |", "|---|---:|---:|---:|"]
        for checkpoint in sorted(self.run.glob("**/model_seed*.pt")):
            try:
                p = torch.load(checkpoint, map_location="cpu", weights_only=False)
                history = p.get("history", [])
                lines.append(f"| {checkpoint.parent.name} | {p['config']['seed']} | "
                             f"{max(x['val_auroc'] for x in history):.5f} | {len(history)} |")
            except Exception:
                pass
        (self.run / "final/PROGRESS.md").write_text("\n".join(lines) + "\n")

    def run_stage(self, stage):
        getattr(self, stage)()


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["inventory", "profile", "controls", "rewiring", "architectures",
                    "depth", "capture_last4", "multilayer", "select", "confirm", "gates", "evaluate",
                    "report", "all"], required=True)
    p.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    p.add_argument("--main-strict-plan", type=Path, default=MAIN_TRAIN)
    p.add_argument("--weather-strict-plan", type=Path, default=WEATHER_TRAIN)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 1, 2])
    p.add_argument("--gpu-budget-hours", type=float, default=18)
    p.add_argument("--timeout-minutes", type=int, default=120)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse(); suite = Suite(args)
    if args.stage == "all":
        for stage in ("inventory", "controls", "architectures", "depth", "select", "evaluate"):
            suite.run_stage(stage)
    elif hasattr(suite, args.stage):
        suite.run_stage(args.stage)
    else:
        raise SystemExit(f"stage {args.stage} is declared but not implemented yet")


if __name__ == "__main__":
    main()
