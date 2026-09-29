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
                 ("union_endpoint_set", 0, "matched four-block endpoint records"),
                 ("graph_sequence", 2, "ordered shared-GNN graph observations without token recurrence")]]
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
        command_text = " ".join(map(str, command))
        gpu_job = ("polygraph.training train" in command_text or "last4-sidecars" in command_text
                   or "profile_topology_models" in command_text)
        used = 0.0
        if self.ledger.exists():
            for line in self.ledger.read_text().splitlines():
                try:
                    row = json.loads(line)
                    prior_command = " ".join(map(str, row.get("command", [])))
                    if row.get("gpu_job") or "polygraph.training train" in prior_command:
                        used += float(row.get("runtime_seconds", 0))
                except Exception: pass
        if gpu_job and not mandatory and used >= self.args.gpu_budget_hours * 3600:
            self.record(dict(experiment_id=experiment_id, status="skipped", command=list(map(str, command)),
                             start_time_utc=started, end_time_utc=started, runtime_seconds=0,
                             mandatory=False, gpu_job=True,
                             reason=f"target GPU budget already used ({used / 3600:.2f} h)"))
            return 1
        if gpu_job and mandatory and used >= 24 * 3600:
            raise RuntimeError(f"24 GPU-hour hard limit reached before mandatory {experiment_id}")
        if self.args.dry_run:
            print("DRY", *command)
            return 0
        status, reason = "completed", None
        try:
            with log.open("a") as output:
                output.write(f"\n[{started}] {command_text}\n")
                output.flush()
                result = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=output,
                                        stderr=subprocess.STDOUT,
                                        timeout=60 * (timeout_minutes or self.args.timeout_minutes))
            if result.returncode:
                status, reason = "failed", f"exit code {result.returncode}"
        except subprocess.TimeoutExpired:
            status, reason = "aborted", "bounded runtime cutoff; resumable state retained"
            if self._promote_budget_state(command):
                reason = "bounded runtime cutoff; validation-selected checkpoint retained"
        row = dict(experiment_id=experiment_id, status=status, command=list(map(str, command)),
                   start_time_utc=started, end_time_utc=utc(), runtime_seconds=time.monotonic() - t0,
                   mandatory=mandatory, log=str(log.relative_to(ROOT)), reason=reason)
        row["gpu_job"] = gpu_job
        self.record(row); self.update_progress()
        if status != "completed" and mandatory:
            raise RuntimeError(f"mandatory {experiment_id} {status}: {reason}; see {log}")
        return 0 if status == "completed" else 1

    def _promote_budget_state(self, command):
        """Turn a timed-out trainer's durable best state into an evaluable checkpoint.

        New state files carry the exact model dimensions and configuration. Older state
        files remain resumable and are deliberately not guessed at.
        """
        argv = list(map(str, command))
        if "polygraph.training" not in argv or "train" not in argv or "--out-dir" not in argv:
            return False
        out = Path(argv[argv.index("--out-dir") + 1])
        if not out.is_absolute(): out = ROOT / out
        seeds = [int(argv[argv.index("--seeds") + 1])] if "--seeds" in argv else []
        promoted = False
        for seed in seeds:
            state_path, final_path = out / f"state_seed{seed}.pt", out / f"model_seed{seed}.pt"
            if final_path.exists() or not state_path.exists(): continue
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            required = {"best_state", "config", "in_dim", "edge_dim", "history"}
            if not required <= set(state) or state["best_state"] is None: continue
            payload = {"state_dict": state["best_state"], "config": state["config"],
                       "in_dim": state["in_dim"], "edge_dim": state["edge_dim"],
                       "plan": argv[argv.index("--plan") + 1] if "--plan" in argv else None,
                       "history": state["history"], "budget_limited": True,
                       "last_completed_epoch": state.get("epoch")}
            temporary = final_path.with_suffix(".pt.tmp")
            torch.save(payload, temporary); temporary.replace(final_path)
            atomic_json(out / f"budget_limited_seed{seed}.json",
                        {"last_completed_epoch": state.get("epoch"),
                         "selected_epoch": state.get("best_epoch"),
                         "selected_base_val_auroc": state.get("best_val")})
            promoted = True
        return promoted

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
        selected, delta = -float("inf"), float(payload["config"].get("min_delta", 0.0))
        for row in payload["history"]:
            value = float(row["val_auroc"])
            if value > selected + delta:
                selected = value
        return selected

    def controls(self):
        self.train("S0_h64", "hidden_token_set", 64, 7, mandatory=True, batch=192)
        for name, family, width, batch in (("S1_h96", "m5_node_edge_set", 96, 96),
                                           ("S1_h128", "m5_node_edge_set", 128, 24),
                                           ("S2_h64", "m5_endpoint_set", 64, 64),
                                           ("S2_h96", "m5_endpoint_set", 96, 48)):
            self.train(name, family, width, 7, mandatory=True, batch=batch)
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

    def profile(self):
        self.command("profile_legacy_M5", [PY, "scripts/profile_topology_models.py",
                     "--plan", MAIN_TRAIN, "--store", STORE,
                     "--out", self.run / "inventory/profile_legacy_M5.json",
                     "--family", "transformerconv", "--width", 32, "--batch-size", 32],
                     timeout_minutes=30)

    def architectures(self):
        for family in ("transformerconv_residual", "gine", "edge_gated_mean", "gatv2"):
            self.train("A_" + family, family, 64, 7, category="architectures", batch=48)
        scored = {p.name: self.best_val(p / "model_seed7.pt")
                  for p in (self.run / "architectures").glob("A_*") if (p / "model_seed7.pt").exists()}
        if scored: atomic_json(self.run / "configs/architecture_selection.json",
                               {"selected": max(scored, key=scored.get), "base_val": scored})

    def rewiring(self):
        cache = ROOT / "data/graph_dataset/sidecars/rewire_target_l11"
        self.command("R1_cache", [PY, "-m", "polygraph.data", "rewire-cache",
                                  "--out-dir", cache, "--seed", 20260905, "--layer", 11],
                     mandatory=True, timeout_minutes=90)
        out = self.run / "controls/R1_target_permuted_M5"
        if not ((out / "model_seed7.pt").exists() and self.args.resume):
            command = [PY, "-m", "polygraph.training", "train", "--plan", MAIN_TRAIN,
                       "--out-dir", out, "--layers", "last", "--architecture", "transformerconv",
                       "--node-features", "hidden", "--edge-features", "evidence_flow",
                       "--message-stats-dir", "data/graph_dataset/sidecars/message_stats_l11",
                       "--rewire-mode", "target_permute", "--rewire-cache-dir", cache,
                       "--hidden-dim", 32, "--gnn-layers", 2, "--batch-size", 64,
                       "--seeds", 7, "--epochs-per-process", 0]
            self.command("R1_target_permuted_M5_seed7", command, mandatory=True, timeout_minutes=150)
        # R2 is lower priority but uses the same cached permutations and unchanged edge budget.
        r2 = self.run / "controls/R2_attribute_shuffled_M5"
        if not ((r2 / "model_seed7.pt").exists() and self.args.resume):
            command = [PY, "-m", "polygraph.training", "train", "--plan", MAIN_TRAIN,
                       "--out-dir", r2, "--layers", "last", "--architecture", "transformerconv",
                       "--node-features", "hidden", "--edge-features", "evidence_flow",
                       "--message-stats-dir", "data/graph_dataset/sidecars/message_stats_l11",
                       "--rewire-mode", "shuffle_attr", "--rewire-cache-dir", cache,
                       "--hidden-dim", 32, "--gnn-layers", 2, "--batch-size", 64,
                       "--seeds", 7, "--epochs-per-process", 0]
            self.command("R2_attribute_shuffled_M5_seed7", command, timeout_minutes=90)

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

    def capture_last4(self):
        records = json.loads((STORE / "manifest.json").read_text())["records"]
        required = int(records) * 3 * 197 * 768 * 2
        stat_required = int(records) * 3 * 197 * 12 * 3 * 2
        disk = os.statvfs(ROOT)
        estimate = {"records": records, "hidden_bytes": required,
                    "message_bytes": stat_required, "estimated_total_gib": (required + stat_required) / 1024**3,
                    "free_before_gib": disk.f_bavail * disk.f_frsize / 1024**3,
                    "required_reserve_gib": 20, "representation": "full float16 768d"}
        atomic_json(self.run / "multilayer/storage_estimate.json", estimate)
        # Prior full-hidden and message captures both completed at batch 512 on this
        # 24 GiB A5000 (batch 1024 OOMed). Reuse the measured largest safe batch.
        cmd = [PY, "-m", "polygraph.data", "last4-sidecars", "--batch-size", 512]
        self.command("capture_last4_missing", cmd, mandatory=False, timeout_minutes=120)

    def train_multilayer(self, name, architecture, mode, family, width=64, seed=7, batch=32,
                         mandatory=True):
        out = self.run / "multilayer" / name
        if (out / f"model_seed{seed}.pt").exists() and self.args.resume: return
        command = [PY, "-m", "polygraph.training", "train", "--plan", MAIN_TRAIN,
                   "--out-dir", out, "--architecture", architecture, "--multilayer-mode", mode,
                   "--multilayer-family", family, "--hidden-dim", width, "--gnn-layers", 2,
                   "--batch-size", batch, "--seeds", seed, "--epochs-per-process", 0]
        self.command(f"{name}_seed{seed}", command, mandatory=mandatory,
                     timeout_minutes=150 if mandatory else 90)

    def multilayer(self):
        required = [ROOT / "data/graph_dataset/sidecars/hidden_last4_missing/manifest.json",
                    ROOT / "data/graph_dataset/sidecars/message_stats_last4_missing/manifest.json"]
        if not all(p.exists() for p in required):
            raise RuntimeError("last-four sidecars are incomplete; resume capture_last4 first")
        architecture = json.loads((self.run / "configs/architecture_selection.json").read_text())["selected"]
        family = architecture.removeprefix("A_")
        self.train_multilayer("L0_token_trajectory", "last4_token_set", "trajectory", family, batch=96)
        # The initial L1 batch-24 window peaked at only 1.23 GiB allocated CUDA
        # memory. Resume its durable epoch-4 state at batch 96 to use the A5000.
        self.train_multilayer("L1_union_graph", "last4_union_graph", "union", family, batch=96)
        self.train_multilayer("L1_SET_union_endpoint", "last4_union_endpoint_set", "union", family, batch=96)
        self.train_multilayer("L2_graph_sequence", "last4_graph_sequence", "sequence", family,
                              batch=96, mandatory=False)
        scored = {p.parent.name: self.best_val(p) for p in (self.run / "multilayer").glob("*/model_seed7.pt")}
        atomic_json(self.run / "configs/multilayer_selection.json", {"base_val": scored,
                    "best": max(scored, key=scored.get) if scored else None})

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
        multi_path = self.run / "configs/multilayer_selection.json"
        manifest["multilayer"] = json.loads(multi_path.read_text()) if multi_path.exists() else None
        atomic_json(self.run / "final/selection_manifest.json", manifest)

    def train_from_checkpoint_config(self, checkpoint, seeds, plan=MAIN_TRAIN, out=None,
                                     mandatory=False, timeout=120, batch_override=None):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        c = payload["config"]
        out = out or checkpoint.parent
        for seed in seeds:
            if (out / f"model_seed{seed}.pt").exists() and self.args.resume: continue
            command = [PY, "-m", "polygraph.training", "train", "--plan", plan,
                       "--out-dir", out, "--layers", ",".join(map(str, c["layers"])),
                       "--architecture", c["architecture"], "--hidden-dim", c["hidden_dim"],
                       "--gnn-layers", c["gnn_layers"], "--batch-size",
                       batch_override or c["batch_size"],
                       "--seeds", seed, "--epochs-per-process", 0, "--epochs", c["epochs"],
                       "--patience", c["patience"], "--min-delta", c["min_delta"],
                       "--lr", c["lr"], "--weight-decay", c["weight_decay"]]
            if c.get("node_features") and c["node_features"] != "base":
                command += ["--node-features", c["node_features"]]
            if c.get("edge_features") and c["edge_features"] != "attention":
                command += ["--edge-features", c["edge_features"], "--message-stats-dir",
                            c.get("message_stats_dir") or "data/graph_dataset/sidecars/message_stats_l11"]
            if c.get("jumping_knowledge"): command += ["--jumping-knowledge"]
            if c.get("multilayer_mode", "none") != "none":
                command += ["--multilayer-mode", c["multilayer_mode"],
                            "--multilayer-family", c["multilayer_family"]]
            self.command(f"confirm_{out.name}_seed{seed}", command, mandatory=mandatory,
                         timeout_minutes=timeout)

    def confirm(self):
        selection = json.loads((self.run / "final/selection_manifest.json").read_text())
        best_gnn = self.run / "architectures" / selection["selected_gnn"] / "model_seed7.pt"
        legacy = self.best_val(PRIOR / "combiners/strict/main/M5/model_seed7.pt")
        if self.best_val(best_gnn) >= legacy + .005:
            self.train_from_checkpoint_config(best_gnn, (1, 2), timeout=150)
        multi = selection.get("multilayer") or {}
        if multi.get("best"):
            best_multi = self.run / "multilayer" / multi["best"] / "model_seed7.pt"
            if self.best_val(best_multi) >= legacy + .005:
                self.train_from_checkpoint_config(best_multi, (1, 2), timeout=150)
        # Weather uses the main-selected architecture/hyperparameters unchanged, with
        # weather train/base_val for fitting and early stopping.
        controls = selection["controls"]
        best_control = max((controls["S1"], controls["S2"]),
                           key=lambda name: controls[name.split("_")[0] + "_validation"][name])
        source = self.run / "controls" / best_control / "model_seed7.pt"
        weather_out = self.run / "controls" / ("weather_" + best_control)
        self.train_from_checkpoint_config(source, (7, 1, 2), plan=WEATHER_TRAIN,
                                          out=weather_out, mandatory=True, timeout=150,
                                          batch_override=96)
        if self.best_val(best_gnn) >= legacy + .005:
            self.train_from_checkpoint_config(best_gnn, (7, 1, 2), plan=WEATHER_TRAIN,
                out=self.run / "architectures" / ("weather_" + best_gnn.parent.name),
                timeout=150, batch_override=96)
        if multi.get("best"):
            best_multi = self.run / "multilayer" / multi["best"] / "model_seed7.pt"
            if self.best_val(best_multi) >= legacy + .005:
                self.train_from_checkpoint_config(best_multi, (7, 1, 2), plan=WEATHER_TRAIN,
                    out=self.run / "multilayer" / ("weather_" + best_multi.parent.name), timeout=150)

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
        for name in ("L0_token_trajectory", "L1_union_graph", "L1_SET_union_endpoint",
                     "L2_graph_sequence"):
            if (self.run / "multilayer" / name / "model_seed7.pt").exists():
                self.evaluate_one("multilayer", name)
        controls = json.loads((self.run / "configs/control_selection.json").read_text())
        best_control = max((controls["S1"], controls["S2"]),
                           key=lambda name: controls[name.split("_")[0] + "_validation"][name])
        weather_control = "weather_" + best_control
        if (self.run / "controls" / weather_control / "model_seed7.pt").exists():
            self.evaluate_one("controls", weather_control, WEATHER_EVAL)
        for category in ("architectures", "multilayer"):
            for directory in (self.run / category).glob("weather_*"):
                if directory.joinpath("model_seed7.pt").exists():
                    self.evaluate_one(category, directory.name, WEATHER_EVAL)
        for name in ("R1_target_permuted_M5", "R2_attribute_shuffled_M5"):
            if (self.run / "controls" / name / "model_seed7.pt").exists():
                self.evaluate_one("controls", name)
        cache = ROOT / "data/graph_dataset/sidecars/rewire_target_l11"
        if cache.joinpath("manifest.json").exists():
            checkpoint = PRIOR / "combiners/strict/main/M5/model_seed7.pt"
            for mode in ("target_permute", "shuffle_attr"):
                self.command(f"fixed_M5_inference_{mode}",
                             [PY, "scripts/evaluate_checkpoint_perturbation.py",
                              "--checkpoint", checkpoint, "--plan", MAIN_EVAL,
                              "--store", STORE, "--cache", cache, "--mode", mode,
                              "--out-dir", self.run / f"controls/fixed_M5_inference_{mode}"],
                             timeout_minutes=90)

    def gates(self):
        selected = json.loads((self.run / "configs/control_selection.json").read_text())
        for control in (selected["S1"], selected["S2"]):
            for seed in (7, 1, 2):
                out = self.run / "gates" / control
                target = out / f"scores_test_seed{seed}.npz"
                if target.exists() and self.args.resume: continue
                command = [PY, "scripts/evaluate_strict_gate.py",
                           "--output-val", PRIOR / f"combiners/strict/main/output/scores_val_seed{seed}.npz",
                           "--output-test", PRIOR / f"combiners/strict/main/output/scores_test_seed{seed}.npz",
                           "--internal-val", self.run / f"controls/{control}/scores_val_seed{seed}.npz",
                           "--internal-test", self.run / f"controls/{control}/scores_test_seed{seed}.npz",
                           "--out-dir", out, "--detector-seed", seed,
                           "--method-name", f"strict_output_{control}_gate"]
                self.command(f"gate_{control}_seed{seed}", command, mandatory=True, timeout_minutes=30)
        selection = json.loads((self.run / "final/selection_manifest.json").read_text())
        promoted = [("architectures", selection["selected_gnn"])]
        if (selection.get("multilayer") or {}).get("best"):
            promoted.append(("multilayer", selection["multilayer"]["best"]))
        for category, name in promoted:
            directory = self.run / category / name
            for seed in (7, 1, 2):
                if not directory.joinpath(f"scores_val_seed{seed}.npz").exists(): continue
                out = self.run / "gates" / name
                command = [PY, "scripts/evaluate_strict_gate.py",
                           "--output-val", PRIOR / f"combiners/strict/main/output/scores_val_seed{seed}.npz",
                           "--output-test", PRIOR / f"combiners/strict/main/output/scores_test_seed{seed}.npz",
                           "--internal-val", directory / f"scores_val_seed{seed}.npz",
                           "--internal-test", directory / f"scores_test_seed{seed}.npz",
                           "--out-dir", out, "--detector-seed", seed,
                           "--method-name", f"strict_output_{name}_gate"]
                self.command(f"gate_{name}_seed{seed}", command, timeout_minutes=30)
        # Weather matched-control gate (minimum weather confirmation).
        controls = selection["controls"]
        best_control = max((controls["S1"], controls["S2"]),
                           key=lambda name: controls[name.split("_")[0] + "_validation"][name])
        internal = self.run / "controls" / ("weather_" + best_control)
        for seed in (7, 1, 2):
            if not internal.joinpath(f"scores_val_seed{seed}.npz").exists(): continue
            out = self.run / "gates" / ("weather_" + best_control)
            self.command(f"gate_weather_{best_control}_seed{seed}",
                [PY, "scripts/evaluate_strict_gate.py",
                 "--output-val", PRIOR / f"combiners/strict/weather/output/scores_val_seed{seed}.npz",
                 "--output-test", PRIOR / f"combiners/strict/weather/output/scores_test_seed{seed}.npz",
                 "--internal-val", internal / f"scores_val_seed{seed}.npz",
                 "--internal-test", internal / f"scores_test_seed{seed}.npz",
                 "--out-dir", out, "--detector-seed", seed,
                 "--method-name", f"strict_weather_output_{best_control}_gate"],
                mandatory=True, timeout_minutes=30)

        # Weather gate for the validation-selected graph architecture, when its
        # strict weather confirmation scores exist.  This mirrors the main-plan
        # gate and avoids selecting a weather architecture from test results.
        weather_gnn = self.run / "architectures" / ("weather_" + selection["selected_gnn"])
        for seed in (7, 1, 2):
            if not weather_gnn.joinpath(f"scores_val_seed{seed}.npz").exists():
                continue
            out = self.run / "gates" / weather_gnn.name
            self.command(f"gate_{weather_gnn.name}_seed{seed}",
                [PY, "scripts/evaluate_strict_gate.py",
                 "--output-val", PRIOR / f"combiners/strict/weather/output/scores_val_seed{seed}.npz",
                 "--output-test", PRIOR / f"combiners/strict/weather/output/scores_test_seed{seed}.npz",
                 "--internal-val", weather_gnn / f"scores_val_seed{seed}.npz",
                 "--internal-test", weather_gnn / f"scores_test_seed{seed}.npz",
                 "--out-dir", out, "--detector-seed", seed,
                 "--method-name", f"strict_weather_output_{selection['selected_gnn']}_gate"],
                timeout_minutes=30)

    def update_progress(self):
        lines = ["# Topology/depth/last-four study — live durable record", "",
                 f"Updated: {utc()}", "", "Test evaluation is withheld until selection is frozen.", "",
                 "| experiment | seed | best base-val AUROC | epochs | status |", "|---|---:|---:|---:|---|"]
        for checkpoint in sorted(self.run.glob("**/model_seed*.pt")):
            try:
                p = torch.load(checkpoint, map_location="cpu", weights_only=False)
                history = p.get("history", [])
                selected = self.best_val(checkpoint)
                lines.append(f"| {checkpoint.parent.name} | {p['config']['seed']} | "
                             f"{selected:.5f} | {len(history)} | completed |")
            except Exception:
                pass
        if self.ledger.exists():
            for raw in self.ledger.read_text().splitlines():
                row = json.loads(raw)
                if row.get("status") in {"failed", "aborted", "skipped"}:
                    lines.append(f"| {row['experiment_id']} | — | — | — | {row['status']}: "
                                 f"{row.get('reason') or 'recorded in ledger'} |")
        (self.run / "final/PROGRESS.md").write_text("\n".join(lines) + "\n")
        report = ROOT / "docs/results/POLYGRAPH_TOPOLOGY_DEPTH_LAST4_REPORT_2026-09-05.md"
        if report.exists():
            text = report.read_text()
            start, end = "<!-- AUTO_RESULTS_START -->", "<!-- AUTO_RESULTS_END -->"
            if start in text and end in text:
                table = [start, "", "| experiment | seed | selected base-val AUROC | epochs | status |",
                         "|---|---:|---:|---:|---|"]
                table.extend(lines[8:])
                table += ["", f"Last durable update: {utc()}.", end]
                before, rest = text.split(start, 1)
                _, after = rest.split(end, 1)
                report.write_text(before + "\n".join(table) + after)

    def report(self):
        self.command("finalize", [PY, "scripts/finalize_topology_depth_last4.py"],
                     mandatory=True, timeout_minutes=60)

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
        for stage in ("inventory", "profile", "controls", "rewiring", "architectures", "depth",
                      "capture_last4", "multilayer", "select", "confirm", "evaluate", "gates", "report"):
            suite.run_stage(stage)
    elif hasattr(suite, args.stage):
        suite.run_stage(args.stage)
    else:
        raise SystemExit(f"stage {args.stage} is declared but not implemented yet")


if __name__ == "__main__":
    main()
