"""Prepare an exact dependency plan; this command does not submit any jobs."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--release", type=Path, required=True)
p.add_argument("--source-manifest", type=Path, required=True)
p.add_argument("--import-bundle-receipt", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--preflight-job", required=True)
p.add_argument("--evaluator-check-job", required=True)
p.add_argument("--capture-minutes", type=int, required=True)
p.add_argument("--rewire-minutes", type=int, required=True)
p.add_argument("--train-minutes", type=int, required=True)
p.add_argument("--score-minutes", type=int, required=True)
p.add_argument("--concurrency", type=int, choices=range(1, 9), default=8)
a = p.parse_args()
root, release = str(a.root), str(a.release)
python = root + "/env/bin/python"
module = "pilots.topology_20260910."
decision_relative = "docs/experiments/september10/REWIRING_DECISION.md"
decision_sha = "0453bb052c668642e42ec1fbd0b6d3359df33f2a5288fb4a6c51de0c47381099"
decision_path = release + "/" + decision_relative
cache, runs, freeze, results = [root + "/" + part for part in ("feature_cache", "runs", "freeze.json", "results")]
fits = [{"arm": arm, "seed": seed} for arm in
        ("full_graph", "full_rewired", "full_set", "full_endpoint", "raw_graph", "raw_set", "logit")
        for seed in (1, 2, 7, 17, 27)]

def command(name, *args):
    return [python, "-u", "-m", module + name, *map(str, args)]

stages = {
    "capture": {"gpus": 1, "minutes": a.capture_minutes,
                "afterok": ["external:" + a.preflight_job, "external:" + a.evaluator_check_job],
                "requires": [f"preflight/{a.preflight_job}/status.json", f"checks/{a.evaluator_check_job}/status.json",
                             "data/download_complete.json", "manifests/environment/complete.json"],
                "command": command("extract", "capture", "--data-root", root + "/data", "--cache", cache,
                                   "--device", "cuda", "--batch-size", "18", "--shard-size", "256", "--resume")},
    "rewire": {"gpus": 0, "cpus": 4, "memory_mb": 32000, "minutes": a.rewire_minutes,
               "afterok": ["capture"], "requires": ["feature_cache/manifest.json"],
               "command": command("slurm.rewire_stage", "--cache", cache, "--workers", "2",
                                  "--decision", decision_path)},
    "validate": {"gpus": 1, "minutes": 120, "afterok": ["rewire"],
                 "requires": ["feature_cache/manifest.json", "feature_cache/rewire/admission.json"],
                 "command": command("validate", "--cache", cache, "--device", "cuda", "--require-rewire")},
    "io_one": {"gpus": 1, "minutes": 20, "afterok": ["validate"],
               "requires": ["feature_cache/validation.json"],
               "command": command("slurm.benchmark", "train", "--cache", cache, "--development-only",
                                  "--worker-index", "0", "--out", root + "/io/one_worker.json")},
    "io_two": {"gpus": 1, "minutes": 20, "concurrency": 2, "afterok": ["io_one"],
               "requires": ["feature_cache/validation.json"],
               "commands": [command("slurm.benchmark", "train", "--cache", cache, "--development-only",
                                    "--worker-index", str(n), "--barrier-dir", root + "/io/paired_start",
                                    "--out", root + f"/io/two_worker_{n}.json") for n in range(2)]},
    "train": {"gpus": 1, "minutes": a.train_minutes, "concurrency": a.concurrency,
              "afterok": ["io_two"], "requires": ["feature_cache/validation.json", "manifests/io_admission.json"],
              "commands": [command("train", "--cache", cache, "--run-root", runs,
                                   "--arm", fit["arm"], "--seed", fit["seed"], "--device", "cuda") for fit in fits]},
    "freeze": {"gpus": 0, "minutes": 60, "afterok": ["train"],
               "command": command("evaluate", "freeze", "--cache", cache, "--run-root", runs, "--out", freeze)},
    "score": {"gpus": 1, "minutes": a.score_minutes, "concurrency": a.concurrency,
              "afterok": ["freeze"], "requires": ["freeze.json"], "final_detector_test": True,
              "commands": [command("evaluate", "score", "--cache", cache, "--freeze", freeze, "--out", results,
                                   "--arm", fit["arm"], "--seed", fit["seed"], "--device", "cuda") for fit in fits]},
    "analyze": {"gpus": 0, "cpus": 4, "memory_mb": 32000, "minutes": 180, "afterok": ["score"],
                "command": command("evaluate", "analyze", "--cache", cache, "--freeze", freeze, "--out", results)},
}
publication = command("publish", "--repo-id", "omrifahn/polygraph-experiments", "--run-root", runs,
                      "--receipt-dir", root + "/publication", "--code-root", release,
                      "--cache-metadata-root", cache)
stages["preparation_summary"] = {"gpus": 0, "minutes": 15, "memory_mb": 2000,
                                 "afterany": ["capture", "rewire", "validate", "io_one", "io_two"],
                                 "command": command("slurm.administrative", "--root", root, "--phase", "preparation")}
stages["publisher"] = {"gpus": 0, "minutes": 2880, "memory_mb": 16000,
                       "afterok": ["io_two"],
                       "requires": ["feature_cache/validation.json", "manifests/io_admission.json"],
                       "command": publication + ["--interval", "3600", "--max-cycles", "48",
                                                  "--terminal-marker", root + "/status/terminal.json"]}
stages["remote_summary"] = {"gpus": 0, "minutes": 15, "memory_mb": 2000,
                            "afterany": ["capture", "rewire", "validate", "io_one", "io_two", "train", "freeze", "score", "analyze"],
                            "command": command("slurm.administrative", "--root", root)}
stages["final_backup"] = {"gpus": 0, "minutes": 60, "memory_mb": 16000,
                          "afterany": ["remote_summary", "publisher"],
                          "requires": ["status/terminal.json"],
                          "command": command("slurm.final_backup", "--root", root, "--release", release)}
manifest = json.loads(a.source_manifest.read_text())
if manifest["files"].get(decision_relative) != decision_sha:
    raise SystemExit("The immutable release does not bind the explicit approved rewiring decision")
bundle = json.loads(a.import_bundle_receipt.read_text())
if bundle.get("status") != "complete":
    raise SystemExit("Import bundle preparation has not completed")
plan = {"schema": 1, "started_date": "2026-09-11", "code_root": release,
        "import_bundle": {"sha256": bundle["sha256"], "packages": bundle["packages"]},
        "source_base_commit": manifest["base_commit"], "source_sha256": manifest["files"],
        "rewiring_decision": {"path": decision_path, "sha256": decision_sha,
                              "contrast": "descriptive_only_non_diagnostic",
                              "minimum_changed_fraction_unchanged": 0.8},
        "planned_fits": fits, "max_concurrent_gpus": 8,
        "prior_job_ids": ["877491", "877530", "877532", "877539", "877551", "877580", "877588",
                          "877593", "877598", "877703", a.preflight_job, a.evaluator_check_job],
        "stages": stages,
        "resource_admission": "Root records io_admission.json only after one/two-worker timing; train is not submitted before approval",
        "submission_phases": {"preparation": ["capture", "rewire", "validate", "io_one", "io_two", "preparation_summary"],
                              "experiments": ["train", "freeze", "score", "analyze", "publisher", "remote_summary", "final_backup"]},
        "publication_admission": "No publisher runs during preparation; phase2 publication requires verified cache/I/O and available authentication",
        "blinding": "Raw labels/test feature rows physically exist; final detector scoring is allowed only by frozen evaluate score",
        "automatic_repair": False, "scientific_config_changes": False}
a.out.parent.mkdir(parents=True, exist_ok=True)
if a.out.exists():
    raise SystemExit("Refusing to replace an existing workflow; review a distinct candidate")
a.out.write_text(json.dumps(plan, indent=2) + "\n")
print(a.out)
