"""Synthetic scheduler/receipt tests. Run only in an allocated CPU Slurm job."""
from __future__ import annotations

import contextlib
from concurrent.futures import ThreadPoolExecutor
import copy
import datetime as dt
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import unittest
from unittest.mock import patch
import uuid

from . import replicate as ops


class FakeScheduler:
    def __init__(self):
        self.submissions = []
        self.reconciliations = []
        self.matches = []
        self.cancellations = []
        self.failure = None
        self.snapshot = {}

    def submit(self, command):
        self.submissions.append(command)
        if self.failure is not None:
            raise self.failure
        return str(980000 + len(self.submissions))

    def reconcile(self, intent):
        self.reconciliations.append(copy.deepcopy(intent))
        return list(self.matches)

    def states(self, ids):
        return {key: row for key, row in self.snapshot.items() if key.split("_")[0] in ids}

    def cancel(self, ids):
        self.cancellations.append(list(ids))


class ReplicationSlurmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ops.require_slurm()

    def setUp(self):
        parent = Path(os.environ.get("POLYGRAPH_TEST_ARTIFACT_ROOT", "test_artifacts")).resolve()
        self.directory = parent / ("replication-slurm-" + uuid.uuid4().hex)
        self.directory.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.directory)
        self.now = dt.datetime.now(dt.timezone.utc)
        control = self.directory / "control"
        release = control / "releases/synthetic-source"
        inventory = {}
        for name in sorted(ops.REQUIRED_SOURCE):
            path = release / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# synthetic source identity only\n")
            inventory[name] = ops.file_sha256(path)
        ops.atomic_json(release / "source_manifest.json",
                        {"release_id": release.name, "source_sha256": inventory})
        self.config = {
            "run_id": "synthetic-fixed-17-27", "root": str(self.directory / "replication"),
            "source_root": str(self.directory / "historical"), "control_root": str(control),
            "code_root": str(release), "python": str(self.directory / "historical/env/bin/python"),
            "source_sha256": inventory,
            "environment": {"import_bundle_sha256": "1" * 64, "dependency_manifest_sha256": "2" * 64},
            "deadlines": {name: (self.now + dt.timedelta(hours=hours)).isoformat()
                          for name, hours in (("base", 24), ("predictions", 25), ("evaluation", 26))},
            "resources": {"account": "gpu-students", "gpu_partition": "studentbatch",
                          "cpu_partition": "cpu-killable", "gpu_constraint": "geforce_rtx_2080",
                          "gpu_cpus": 6, "cpu_cpus": 2, "gpu_memory_mb": 32000, "cpu_memory_mb": 8000},
            "caps": {"guardian": 1590, "prepare": 15, "fits": 90, "meta_heads": 30,
                     "joint_heads": 5, "dev_eval": 45, "evaluate": 20, "aggregate": 20},
            "max_concurrent_gpus": 8, "external_reserved_gpus": 0,
            "gpu_budget_minutes": 1800, "external_gpu_minutes": 50,
            "submission_grace_minutes": 5, "guardian_grace_minutes": 10, "readiness_max_seconds": 60,
        }
        self.plan = ops.build_plan(self.config, now=self.now)
        self.authorization = ops.authorization_template(self.plan)
        self.authorization.update(approved=True, authorized_at=(self.now - dt.timedelta(minutes=1)).isoformat(),
                                  expires_at=self.plan["config"]["deadlines"]["evaluation"])
        self.scheduler = FakeScheduler()

    def write(self, relative, value):
        path = self.directory / relative
        ops.atomic_json(path, value)
        return path

    def submit(self):
        return ops.submit_plan(self.plan, self.authorization, self.scheduler, now=self.now)

    def test_default_cli_never_submits_even_with_approved_json(self):
        config = self.write("config.json", self.config)
        authorization = self.write("approved.json", self.authorization)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(ops.main(["--config", str(config), "--authorization", str(authorization)],
                                     scheduler=self.scheduler, now=self.now), 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "validated_not_submitted")
        self.assertEqual(self.scheduler.submissions, [])
        self.assertFalse((ops._control(self.plan) / "manifests").exists())
        self.assertFalse(Path(self.config["root"]).exists())
        self.assertFalse(Path(self.config["source_root"]).exists())

    def test_validate_cli_is_inert(self):
        workflow = self.write("workflow.json", self.plan)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ops.main(["validate", "--workflow", str(workflow),
                                      "--sha256", ops.digest(self.plan)],
                                     scheduler=self.scheduler, now=self.now), 0)
        self.assertEqual(self.scheduler.submissions, [])

    def test_submit_flag_requires_separate_explicit_approval(self):
        config = self.write("config.json", self.config)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            ops.main(["--config", str(config), "--submit"], scheduler=self.scheduler, now=self.now)
        for approval in (None, {}, ops.authorization_template(self.plan)):
            with self.subTest(approval=approval), self.assertRaises(ops.PlanError):
                ops.submit_plan(self.plan, approval, self.scheduler, now=self.now)
        self.assertEqual(self.scheduler.submissions, [])

    def test_exact_eight_fresh_fits_and_canonical_seed_namespaces(self):
        self.assertEqual(self.plan["matrix"], [[arm, seed] for seed in (17, 27) for arm in ops.ARMS])
        tasks = self.plan["stages"]["fits"]["tasks"]
        self.assertEqual([(t["arm"], t["seed"]) for t in tasks], [tuple(row) for row in ops.MATRIX])
        self.assertEqual((self.plan["fresh_fits"], self.plan["epochs"], self.plan["automatic_retries"]), (8, 20, 0))
        for task in tasks:
            command = task["commands"][0]
            seed = task["seed"]
            self.assertEqual(command[command.index("--seed") + 1], str(seed))
            self.assertEqual(command[command.index("--run-root") + 1], self.config["root"] + f"/seed{seed}/runs")
            self.assertEqual(command[command.index("--roles") + 1], self.config["root"] + f"/seed{seed}/role_map.json")
            self.assertNotIn(self.config["source_root"] + "/runs", command)
        self.assertFalse(self.plan["original_test_access"])

    def test_both_seeds_both_heads_gate_all_development_predictions(self):
        stages = self.plan["stages"]
        self.assertEqual(stages["meta_heads"]["dependency"], "fits")
        self.assertEqual([task["seed"] for task in stages["meta_heads"]["tasks"]], [17, 27])
        for task in stages["meta_heads"]["tasks"]:
            self.assertEqual(len(task["commands"]), 2)
            self.assertIn("--freeze-base", task["commands"][0])
            self.assertIn("meta", task["commands"][0])
            self.assertIn(ops.MODULE + "combine", task["commands"][1])
        self.assertEqual(stages["joint_heads"]["dependency"], "meta_heads")
        self.assertIn("freeze-heads", stages["joint_heads"]["tasks"][0]["commands"][0])
        self.assertEqual(stages["dev_eval"]["dependency"], "joint_heads")
        for task in stages["dev_eval"]["tasks"]:
            command = task["commands"][0]
            self.assertEqual(command[command.index("--heads-freeze") + 1],
                             self.config["root"] + f"/seed{task['seed']}/heads_freeze.json")
        altered = copy.deepcopy(self.plan)
        altered["stages"]["dev_eval"]["dependency"] = "fits"
        with self.assertRaises(ops.PlanError):
            ops.validate_plan(altered, now=self.now)

    def test_both_readiness_checks_precede_all_fits_with_explicit_gpu_cap(self):
        prepare = self.plan["stages"]["prepare"]
        self.assertTrue(prepare["gpu"])
        self.assertEqual(prepare["cpus"], 6)
        commands = prepare["tasks"][0]["commands"]
        self.assertEqual(len(commands), 3)
        self.assertIn(ops.MODULE + "replication", commands[0])
        for seed, command in zip(ops.SEEDS, commands[1:]):
            self.assertIn(ops.MODULE + "smoke", command)
            self.assertEqual(command[command.index("--execution") + 1],
                             self.config["root"] + f"/seed{seed}/execution.json")
            self.assertEqual(command[command.index("--out") + 1],
                             self.config["root"] + f"/seed{seed}/manifests/readiness.json")
            self.assertEqual(command[command.index("--max-seconds") + 1], "60")
            self.assertNotIn("--role", command)
        self.assertEqual(self.plan["stages"]["fits"]["requires"],
                         [f"seed{seed}/manifests/readiness.json" for seed in ops.SEEDS])
        for seconds in (59, 1501, 500):
            config = copy.deepcopy(self.config)
            config["readiness_max_seconds"] = seconds
            with self.subTest(seconds=seconds), self.assertRaises(ops.PlanError):
                ops.build_plan(config, now=self.now)

    def test_all_array_tasks_are_reserved_not_just_eight_sbatch_ids(self):
        self.assertEqual(self.plan["submission_count"], 8)
        self.assertEqual(self.plan["allocated_task_count"], 18)
        claims = self.plan["resource_claims"]
        self.assertEqual(claims["workflow_gpu_minutes"], 15 + 8 * 90 + 2 * 30 + 2 * 45)
        self.assertEqual(claims["reserved_gpu_minutes"], claims["workflow_gpu_minutes"] + 50)
        self.assertEqual(claims["gpu_ceiling_including_external"], 8)
        config = copy.deepcopy(self.config)
        config.update(max_concurrent_gpus=4, external_reserved_gpus=4)
        plan = ops.build_plan(config, now=self.now)
        self.assertEqual(plan["stages"]["fits"]["throttle"], 4)
        self.assertEqual(plan["critical_path_cap_minutes"]["base"], 15 + 2 * 90)
        self.assertEqual(plan["stages"]["meta_heads"]["throttle"], 2)

    def test_gpu_concurrency_and_reservation_errors(self):
        for changed in ({"max_concurrent_gpus": 9}, {"max_concurrent_gpus": 0},
                        {"max_concurrent_gpus": True}, {"external_reserved_gpus": 1},
                        {"gpu_budget_minutes": 934}, {"external_gpu_minutes": -1}):
            config = copy.deepcopy(self.config)
            config.update(changed)
            with self.subTest(changed=changed), self.assertRaises(ops.PlanError):
                ops.build_plan(config, now=self.now)
        for field, value in (("gpu_cpus", 7), ("cpu_cpus", 0), ("gpu_memory_mb", 64000)):
            config = copy.deepcopy(self.config)
            config["resources"][field] = value
            with self.subTest(field=field), self.assertRaises(ops.PlanError):
                ops.build_plan(config, now=self.now)

    def test_deadlines_are_future_ordered_timezone_aware_and_fit_full_caps(self):
        for key, value in (("base", "2026-09-16T01:00:00"),
                           ("base", (self.now - dt.timedelta(hours=1)).isoformat()),
                           ("base", (self.now + dt.timedelta(minutes=30)).isoformat()),
                           ("predictions", self.config["deadlines"]["base"]),
                           ("evaluation", self.config["deadlines"]["predictions"])):
            config = copy.deepcopy(self.config)
            config["deadlines"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ops.PlanError):
                ops.build_plan(config, now=self.now)
        config = copy.deepcopy(self.config)
        config["caps"]["guardian"] = 10
        with self.assertRaises(ops.PlanError):
            ops.build_plan(config, now=self.now)

    def test_authorization_binds_identity_hashes_caps_deadlines_and_expiry(self):
        changes = [
            ("approved", False), ("approved", 1), ("run_id", "other-run"),
            ("workflow_sha256", "0" * 64), ("release_sha256", "0" * 64),
            ("root", self.directory.as_posix()), ("matrix", [["block2", 7]]),
            ("resource_claims", {}), ("deadlines", {}), ("automatic_retries", 1),
            ("authorized_at", (self.now + dt.timedelta(minutes=1)).isoformat()),
            ("expires_at", (self.now - dt.timedelta(seconds=1)).isoformat()),
            ("expires_at", (self.now + dt.timedelta(days=10)).isoformat()),
        ]
        for key, value in changes:
            authorization = copy.deepcopy(self.authorization)
            authorization[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ops.PlanError):
                ops.validate_authorization(self.plan, authorization, now=self.now)
        authorization = copy.deepcopy(self.authorization)
        authorization["expires_at"] = (self.now + dt.timedelta(minutes=1)).isoformat()
        with self.assertRaises(ops.PlanError):
            ops.validate_plan(self.plan, authorization, now=self.now)

    def test_historical_and_control_namespaces_must_remain_separate(self):
        for key, value in (("root", self.config["source_root"]),
                           ("root", self.config["source_root"] + "/replication"),
                           ("root", self.config["control_root"]),
                           ("control_root", self.config["root"] + "/ops"),
                           ("code_root", self.config["source_root"] + "/releases/new")):
            config = copy.deepcopy(self.config)
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ops.PlanError):
                ops.build_plan(config, now=self.now)

    def test_eight_server_durable_submissions_and_idempotent_receipt_reuse(self):
        result = self.submit()
        self.assertEqual(set(result["job_ids"]), set(ops.ORDER))
        self.assertEqual(len(self.scheduler.submissions), 8)
        for stage, command in zip(ops.ORDER, self.scheduler.submissions):
            self.assertIn("--no-requeue", command)
            self.assertIn("--kill-on-invalid-dep=yes", command)
            self.assertIn("--comment=" + ops._comment(self.plan, stage), command)
            self.assertIn("--authorization-sha256 " + ops.digest(self.authorization), command[-1])
            if not self.plan["stages"][stage]["gpu"]:
                self.assertFalse(any(arg.startswith("--gpus") for arg in command))
            self.assertTrue(ops._intent_path(self.plan, stage).is_file())
        self.assertIn("--array=0-7%8", self.scheduler.submissions[2])
        self.assertIn("--dependency=after:" + result["job_ids"]["guardian"], self.scheduler.submissions[1])
        self.assertIn("--dependency=afterok:" + result["job_ids"]["meta_heads"], self.scheduler.submissions[4])
        self.assertIn("--dependency=afterok:" + result["job_ids"]["joint_heads"], self.scheduler.submissions[5])
        self.assertEqual(self.submit()["job_ids"], result["job_ids"])
        self.assertEqual(len(self.scheduler.submissions), 8)

    def test_source_or_workflow_change_refuses_submission(self):
        changed = Path(self.config["code_root"]) / ops.SELF
        changed.write_text("# modified after release\n")
        with self.assertRaises(ops.PlanError):
            self.submit()
        self.assertEqual(self.scheduler.submissions, [])
        workflow = self.write("workflow.json", self.plan)
        expected = ops.digest(self.plan)
        workflow.write_text("{}\n")
        with self.assertRaises(ops.PlanError):
            ops._bound_json(workflow, expected)

    def test_lost_sbatch_receipt_refuses_zero_and_multiple_matches_without_retry(self):
        self.scheduler.failure = subprocess.TimeoutExpired("sbatch", 30)
        for matches in ([], ["991001", "991002"]):
            self.scheduler.matches = matches
            with self.subTest(matches=matches), self.assertRaises(ops.AmbiguousSubmission):
                ops.submit_one(self.plan, "guardian", {}, ops.digest(self.authorization),
                               self.scheduler, now=self.now)
        self.assertEqual(len(self.scheduler.submissions), 1)
        intent = json.loads(ops._intent_path(self.plan, "guardian").read_text())
        self.assertEqual(intent["status"], "ambiguous")
        self.assertNotIn("job_id", intent)

    def test_unique_reconciliation_records_old_id_without_sbatch(self):
        self.scheduler.failure = subprocess.TimeoutExpired("sbatch", 30)
        with self.assertRaises(ops.AmbiguousSubmission):
            ops.submit_one(self.plan, "guardian", {}, ops.digest(self.authorization),
                           self.scheduler, now=self.now)
        self.scheduler.matches = ["991123"]
        result = ops.reconcile_intents(self.plan, self.scheduler, ops.digest(self.authorization))
        self.assertEqual(result["job_ids"], {"guardian": "991123"})
        self.assertEqual(result["unresolved"], [])
        self.assertFalse(result["submitted_new_jobs"])
        self.assertEqual(len(self.scheduler.submissions), 1)
        self.assertEqual(ops.recorded_jobs(self.plan), {"guardian": "991123"})

    def test_squeue_and_sacct_reconcile_exact_token_and_array_parent(self):
        intent = {"created_utc": self.now.isoformat(), "job_name": "bound-name", "comment": "bound-token"}
        outputs = [
            "991100|bound-name|bound-token\n991999|bound-name|other-token\n",
            "991100_0|bound-name|bound-token\n991100_1|bound-name|bound-token\n",
        ]
        with patch.object(ops.Scheduler, "_run", side_effect=outputs) as run:
            self.assertEqual(ops.Scheduler().reconcile(intent), ["991100"])
            self.assertEqual([call.args[0][0] for call in run.call_args_list], ["squeue", "sacct"])
        with patch.object(ops.Scheduler, "_run", side_effect=[outputs[0], subprocess.TimeoutExpired("sacct", 30)]):
            with self.assertRaises(subprocess.TimeoutExpired):
                ops.Scheduler().reconcile(intent)

    def test_invalid_sbatch_receipt_is_ambiguous_not_retryable(self):
        with patch.object(self.scheduler, "submit", return_value="accepted but ID missing") as submit:
            with self.assertRaises(ops.AmbiguousSubmission):
                ops.submit_one(self.plan, "guardian", {}, ops.digest(self.authorization), self.scheduler)
            with self.assertRaises(ops.AmbiguousSubmission):
                ops.submit_one(self.plan, "guardian", {}, ops.digest(self.authorization), self.scheduler)
            self.assertEqual(submit.call_count, 1)

    def test_altered_intent_and_terminal_workflow_refuse_relaunch(self):
        self.submit()
        path = ops._intent_path(self.plan, "fits")
        intent = json.loads(path.read_text())
        intent["command"].append("--gpus=8")
        ops.atomic_json(path, intent)
        with self.assertRaises(ops.PlanError):
            ops.recorded_jobs(self.plan)
        with self.assertRaises(ops.AmbiguousSubmission):
            ops.submit_one(self.plan, "fits", {"prepare": "980002"}, ops.digest(self.authorization), self.scheduler)
        ops.atomic_json(ops._control(self.plan) / "manifests/terminal.json", {"status": "incomplete"})
        with self.assertRaises(ops.PlanError):
            self.submit()
        self.assertEqual(len(self.scheduler.submissions), 8)

    def test_scientific_stage_waits_for_complete_static_submission(self):
        result = self.submit()
        self.assertEqual(ops._await_committed_launch(self.plan, self.authorization, ops.time.time() + 1),
                         result["job_ids"])
        launch_path = ops._control(self.plan) / "manifests/launch.json"
        launch = json.loads(launch_path.read_text())
        launch["job_ids"].pop("aggregate")
        ops.atomic_json(launch_path, launch)
        with self.assertRaises(ops.PlanError):
            ops._await_committed_launch(self.plan, self.authorization, ops.time.time() + 1)

    def populate_evaluations(self):
        root = Path(self.config["root"])
        created = self.now.timestamp() - 120
        ops.atomic_json(root / "role_map.json", {"synthetic": True, "original_test_access": False})
        roles_sha = ops.file_sha256(root / "role_map.json")
        source = Path(self.config["source_root"])
        for name in ops.HISTORY_FILES:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic historical fixture, never a loaded array\n")
        shutil.copyfile(root / "role_map.json", source / "role_map.json")
        marker = {"status": "complete_late_diagnostic", "complete": True,
                  "scope_id": "layer_ensemble_20260914_core_seed7", "reused_on_time_meta_export": True,
                  "original_protocol_status": "incomplete: synthetic late marker",
                  "warning": "Synthetic fixture, never scientific output."}
        ops.atomic_json(source / "evaluation/late_diagnostic.json", marker)
        historical_inputs = {"implementation": {"combine.py": "7" * 64, "evaluate.py": "7" * 64}}
        ops.atomic_json(source / "evaluation/complete.json", {
            "complete": True, "inputs": historical_inputs,
            "files": {name: ops.file_sha256(source / "evaluation" / name) for name in ops.EVALUATION_FILES}})
        historical_complete = json.loads((source / "evaluation/complete.json").read_text())
        manifest = {
            "scope_id": ops.SCOPE, "run_id": self.plan["run_id"], "root": str(root),
            "source_root": self.config["source_root"], "seeds": list(ops.SEEDS), "matrix": ops.MATRIX,
            "deadlines": {ops.DEADLINE_KEYS[key]: value for key, value in self.plan["config"]["deadlines"].items()},
            "original_test_access": False, "created_unix": created, "cache_manifest_sha256": "c" * 64,
            "roles_sha256": roles_sha,
            "source_files": {name: ops.file_sha256(source / name) for name in ops.HISTORY_FILES},
        }
        ops.atomic_json(root / "replication.json", manifest)
        bindings = {}
        for seed in ops.SEEDS:
            prefix = root / f"seed{seed}"
            for name in ("execution.json", "base_freeze.json", "heads_freeze.json",
                         "heads/stack.json", "heads/last_only.json"):
                ops.atomic_json(prefix / name, {"synthetic": True, "seed": seed, "file": name})
            shutil.copyfile(root / "role_map.json", prefix / "role_map.json")
            checks = ("finite_gradient", "immediate_state_exact", "deterministic_resume_original_tolerance_passed",
                      "record_and_hidden_parity", "sampler_epoch_resume")
            ops.atomic_json(prefix / "manifests/readiness.json", {
                "scope_id": ops.SCOPE, "seed": seed, "passed": True, "diagnostic_only": True,
                "scientific_fits_created": False, "meta_or_dev_predictions": False,
                "source_sha256": self.config["source_sha256"]["pilots/layer_ensemble_20260914/smoke.py"],
                "execution_sha256": ops.file_sha256(prefix / "execution.json"), "roles_sha256": roles_sha,
                "completed_unix": self.now.timestamp() - 90, "elapsed_seconds": 0.01, "job_id": "980002",
                "two_loader_probes": [{"role": "base_train"}, {"role": "checkpoint"}],
                "arms": {arm: dict.fromkeys(checks, True) for arm in ops.ARMS},
            })
            bindings[str(seed)] = {name: ops.file_sha256(prefix / name) for name in
                                   ("execution.json", "base_freeze.json", "heads_freeze.json",
                                    "heads/stack.json", "heads/last_only.json")}
        joint = {"scope_id": ops.SCOPE, "complete": True, "seeds": list(ops.SEEDS), "original_test_access": False,
                 "replication_manifest_sha256": ops.file_sha256(root / "replication.json"),
                 "files": bindings, "frozen_unix": self.now.timestamp() - 60}
        ops.atomic_json(root / "joint_heads_freeze.json", joint)
        for seed in ops.SEEDS:
            prefix = root / f"seed{seed}"
            (prefix / "predictions").mkdir()
            (prefix / "predictions/dev_eval.npz").write_bytes(b"synthetic bytes, never loaded as an array\n")
            ops.atomic_json(prefix / "predictions/dev_eval.json", {
                "seed": seed, "role": "dev_eval", "completed_unix": self.now.timestamp() - 50,
                "joint_heads_freeze_sha256": ops.file_sha256(root / "joint_heads_freeze.json"),
                "npz_sha256": ops.file_sha256(prefix / "predictions/dev_eval.npz"),
            })
            inputs = {"replication_manifest_sha256": ops.file_sha256(root / "replication.json"),
                      "joint_heads_freeze_sha256": ops.file_sha256(root / "joint_heads_freeze.json"),
                      "cache_manifest_sha256": manifest["cache_manifest_sha256"],
                      "implementation": {name: self.config["source_sha256"]["pilots/layer_ensemble_20260914/" + name]
                                         for name in ("combine.py", "evaluate.py", "protocol.py", "replication.py")}}
            for key, name in (("execution_sha256", "execution.json"), ("roles_sha256", "role_map.json"),
                              ("base_freeze_sha256", "base_freeze.json"), ("heads_freeze_sha256", "heads_freeze.json"),
                              ("dev_prediction_sidecar_sha256", "predictions/dev_eval.json"),
                              ("dev_prediction_npz_sha256", "predictions/dev_eval.npz")):
                inputs[key] = ops.file_sha256(prefix / name)
            ops.atomic_json(prefix / "evaluation/report.json", {
                "scope_id": ops.SCOPE, "seed": seed, "complete": True, "test_evaluated": False,
                "base_epochs": 20, "records": 7200, "inputs": inputs})
            for name in ops.EVALUATION_FILES - {"report.json"}:
                (prefix / "evaluation" / name).write_bytes(b"synthetic bound evaluation artifact\n")
            ops.atomic_json(prefix / "evaluation/complete.json", {
                "scope_id": ops.SCOPE, "complete": True, "seed": seed, "inputs": inputs,
                "completed_unix": self.now.timestamp() - 40,
                "files": {name: ops.file_sha256(prefix / "evaluation" / name) for name in ops.EVALUATION_FILES}})
        inputs = {"replication_manifest_sha256": ops.file_sha256(root / "replication.json"),
                  "roles_sha256": roles_sha, "primary_seeds": list(ops.SEEDS),
                  "joint_heads_freeze_sha256": ops.file_sha256(root / "joint_heads_freeze.json"),
                  "late_seed7_root": self.config["source_root"], "late_seed7_source_files": manifest["source_files"],
                  "seed_evaluations": {},
                  "implementation": {name: self.config["source_sha256"]["pilots/layer_ensemble_20260914/" + name]
                                     for name in ("aggregate_replications.py", "evaluate.py", "combine.py",
                                                  "replication.py", "protocol.py")}}
        for seed in ops.SEEDS:
            prefix = root / f"seed{seed}"
            complete = json.loads((prefix / "evaluation/complete.json").read_text())
            inputs["seed_evaluations"][str(seed)] = {
                "root": str(prefix), "scope_id": ops.SCOPE, "seed": seed,
                "complete_sha256": ops.file_sha256(prefix / "evaluation/complete.json"),
                "files": complete["files"], "inputs": complete["inputs"],
            }
        inputs["seed_evaluations"]["7"] = {
            "root": self.config["source_root"], "scope_id": "layer_ensemble_20260914_core_seed7", "seed": 7,
            "complete_sha256": manifest["source_files"]["evaluation/complete.json"],
            "files": historical_complete["files"], "inputs": historical_complete["inputs"],
        }
        report = {
            "scope_id": ops.SCOPE, "complete": True, "inputs": inputs, "primary_seeds": list(ops.SEEDS),
            "status": "complete", "included_seeds": [17, 27, 7],
            "primary": {"seeds": list(ops.SEEDS)}, "test_evaluated": False,
            "records_per_seed": 7200, "source_photographs": 800, "base_epochs": 20,
            "all3_descriptive_complete": True, "all3_descriptive": {"status": "complete", "seeds": [7, 17, 27]},
            "per_seed": {"7": {"status": "complete_late_diagnostic"},
                         "17": {"status": "complete"}, "27": {"status": "complete"}},
            "late_seed7_marker": marker,
        }
        ops.atomic_json(root / "evaluation/report.json", report)
        ops.atomic_json(root / "evaluation/bootstrap.json", {
            key: report[key] for key in ("primary", "all3_descriptive", "per_seed")})
        for name in ops.AGGREGATE_FILES - {"report.json", "bootstrap.json"}:
            (root / "evaluation" / name).write_bytes(b"synthetic bound aggregate artifact\n")
        ops.atomic_json(root / "evaluation/complete.json", {
            "scope_id": ops.SCOPE, "complete": True, "inputs": inputs, "primary_seeds": list(ops.SEEDS),
            "completed_unix": self.now.timestamp() - 30,
            "files": {name: ops.file_sha256(root / "evaluation" / name) for name in ops.AGGREGATE_FILES}})
        return root

    def test_readiness_requires_both_seed_reports_and_only_base_checkpoint_roles(self):
        root = self.populate_evaluations()
        self.assertEqual(len(ops.verify_readiness(self.plan, "980002")), 2)
        path = root / "seed27/manifests/readiness.json"
        original = json.loads(path.read_text())
        for field, value in (("passed", False), ("seed", 7), ("job_id", "other-job"),
                             ("source_sha256", "0" * 64), ("meta_or_dev_predictions", True),
                             ("two_loader_probes", [{"role": "base_train"}, {"role": "dev_eval"}])):
            changed = copy.deepcopy(original)
            changed[field] = value
            ops.atomic_json(path, changed)
            with self.subTest(field=field), self.assertRaises(ops.PlanError):
                ops.verify_readiness(self.plan, "980002")
        ops.atomic_json(path, original)

    def test_guardian_success_requires_every_bound_evaluation_artifact(self):
        root = self.populate_evaluations()
        states = {name: "COMPLETED" for name in ops.ORDER if name != "guardian"}
        evidence = ops.completion_evidence(self.plan)
        self.assertTrue(evidence["complete"], evidence)
        self.assertEqual(ops.guardian_decision(self.plan, states, evidence, now=self.now,
                                             launch={"status": "submitted"})["status"], "complete")
        for name in sorted(ops.required_evidence_paths()):
            path = root / name
            saved = path.with_name(path.name + ".saved")
            path.rename(saved)
            try:
                with self.subTest(missing=name):
                    evidence = ops.completion_evidence(self.plan)
                    self.assertFalse(evidence["complete"])
                    decision = ops.guardian_decision(self.plan, states, evidence, now=self.now,
                                                     launch={"status": "submitted"})
                    self.assertEqual(decision["status"], "incomplete")
            finally:
                saved.rename(path)

    def test_guardian_does_not_trust_bare_complete_marker_or_changed_scores(self):
        root = self.populate_evaluations()
        states = {name: "COMPLETED" for name in ops.ORDER if name != "guardian"}
        for evidence in (None, {"complete": True}, {"complete": True, "workflow_sha256": ops.digest(self.plan), "files": {}}):
            decision = ops.guardian_decision(self.plan, states, evidence, now=self.now, launch={"status": "submitted"})
            self.assertEqual(decision["status"], "incomplete")
        (root / "seed27/evaluation/scores.npz").write_bytes(b"changed scores; not recomputed\n")
        self.assertFalse(ops.completion_evidence(self.plan)["complete"])

    def test_aggregate_cannot_omit_one_seed_binding(self):
        root = self.populate_evaluations()
        complete = json.loads((root / "evaluation/complete.json").read_text())
        report = json.loads((root / "evaluation/report.json").read_text())
        complete["inputs"]["seed_evaluations"].pop("27")
        report["inputs"] = complete["inputs"]
        ops.atomic_json(root / "evaluation/report.json", report)
        complete["files"]["report.json"] = ops.file_sha256(root / "evaluation/report.json")
        ops.atomic_json(root / "evaluation/complete.json", complete)
        self.assertFalse(ops.completion_evidence(self.plan)["complete"])

    def test_guardian_rejects_changed_late_provenance_and_bootstrap_metadata(self):
        root = self.populate_evaluations()
        path = root / "evaluation/bootstrap.json"
        bootstrap = json.loads(path.read_text())
        bootstrap["primary"]["estimate"] = -0.1
        ops.atomic_json(path, bootstrap)
        complete_path = root / "evaluation/complete.json"
        complete = json.loads(complete_path.read_text())
        complete["files"]["bootstrap.json"] = ops.file_sha256(path)
        ops.atomic_json(complete_path, complete)
        self.assertFalse(ops.completion_evidence(self.plan)["complete"])
        report = json.loads((root / "evaluation/report.json").read_text())
        bootstrap["primary"] = report["primary"]
        ops.atomic_json(path, bootstrap)
        complete["files"]["bootstrap.json"] = ops.file_sha256(path)
        ops.atomic_json(complete_path, complete)
        self.assertTrue(ops.completion_evidence(self.plan)["complete"])
        (Path(self.config["source_root"]) / "evaluation/scores.npz").write_bytes(b"changed historical fixture\n")
        self.assertFalse(ops.completion_evidence(self.plan)["complete"])

    def test_guardian_never_requires_positive_effect_or_interval(self):
        root = self.populate_evaluations()
        for name in ("report.json", "bootstrap.json"):
            path = root / "evaluation" / name
            value = json.loads(path.read_text())
            value["primary"].update(estimate=-0.01, interval_95=None)
            ops.atomic_json(path, value)
        path = root / "evaluation/complete.json"
        complete = json.loads(path.read_text())
        for name in ("report.json", "bootstrap.json"):
            complete["files"][name] = ops.file_sha256(root / "evaluation" / name)
        ops.atomic_json(path, complete)
        self.assertTrue(ops.completion_evidence(self.plan)["complete"])

    def test_guardian_deadlines_failure_and_submission_grace(self):
        states = {}
        launch = {"status": "submitted"}
        decision = ops.guardian_decision(self.plan, states, now=self.now, launch=launch)
        self.assertEqual(decision["status"], "watching")
        for phase, stage in (("base", "fits"), ("predictions", "dev_eval"), ("evaluation", "aggregate")):
            clock = ops.timestamp(self.plan["config"]["deadlines"][phase])
            decision = ops.guardian_decision(self.plan, states, now=clock, launch=launch)
            self.assertEqual(decision["reason"], phase + "_deadline")
            states[stage] = "COMPLETED"
        decision = ops.guardian_decision(self.plan, {"fits": "FAILED"}, now=self.now, launch=launch)
        self.assertEqual(decision["reason"], "required_stage_failed")
        launch = {"status": "submitting", "created_utc": (self.now - dt.timedelta(minutes=6)).isoformat()}
        self.assertEqual(ops.guardian_decision(self.plan, {}, now=self.now, launch=launch)["reason"],
                         "submission_not_committed")

    def test_all_eight_array_elements_must_finish_successfully(self):
        result = self.submit()
        job = result["job_ids"]["fits"]
        states = {f"{job}_{index}": {"state": "COMPLETED", "exit_code": "0:0"} for index in range(7)}
        self.assertEqual(ops.stage_states(self.plan, result["job_ids"], states)["fits"], "UNKNOWN")
        states[job + "_7"] = {"state": "FAILED", "exit_code": "1:0"}
        self.assertEqual(ops.stage_states(self.plan, result["job_ids"], states)["fits"], "FAILED")
        states[job + "_7"] = {"state": "COMPLETED", "exit_code": "0:0"}
        self.assertEqual(ops.stage_states(self.plan, result["job_ids"], states)["fits"], "COMPLETED")

    def test_guardian_authorization_change_cannot_report_success(self):
        result = self.submit()
        ops.atomic_json(ops._control(self.plan) / "authorization.json", {"approved": False})
        with patch.dict(os.environ, {"SLURM_JOB_ID": result["job_ids"]["guardian"]}):
            self.assertEqual(ops.guardian(self.plan, self.authorization, self.scheduler), 2)
        receipt = json.loads((ops._control(self.plan) / "manifests/terminal.json").read_text())
        self.assertFalse(receipt["complete"])
        self.assertEqual(receipt["reason"], "guardian_observation_failed")
        self.assertEqual(set(self.scheduler.cancellations[-1]),
                         set(result["job_ids"].values()) - {result["job_ids"]["guardian"]})

    def test_guardian_dispatch_rejects_wrong_allocation_before_any_action(self):
        self.submit()
        with patch.dict(os.environ, {"SLURM_JOB_ID": "999999"}):
            with self.assertRaisesRegex(ops.PlanError, "recorded Slurm job"):
                ops.guardian(self.plan, self.authorization, self.scheduler)
            with patch.object(ops, "verify_release"), self.assertRaisesRegex(ops.PlanError, "recorded Slurm job"):
                ops.run_stage(self.plan, self.authorization, "guardian")
        self.assertFalse((ops._control(self.plan) / "manifests/terminal.json").exists())
        self.assertFalse((ops._control(self.plan) / "manifests/guardian.json").exists())
        self.assertEqual(self.scheduler.cancellations, [])
        self.assertEqual(self.scheduler.reconciliations, [])

    def test_guardian_rejects_array_allocation(self):
        result = self.submit()
        with patch.dict(os.environ, {"SLURM_JOB_ID": result["job_ids"]["guardian"], "SLURM_ARRAY_TASK_ID": "0"}):
            with self.assertRaisesRegex(ops.PlanError, "singleton"):
                ops.guardian(self.plan, self.authorization, self.scheduler)
        self.assertEqual(self.scheduler.cancellations, [])

    def test_guardian_accepts_bounded_initial_ack_race(self):
        result = self.submit()
        path = ops._intent_path(self.plan, "guardian")
        recorded = json.loads(path.read_text())
        submitting = copy.deepcopy(recorded)
        submitting["status"] = "submitting"
        submitting.pop("job_id")
        ops.atomic_json(path, submitting)

        def record_ack(_seconds):
            ops.atomic_json(path, recorded)

        with (patch.dict(os.environ, {"SLURM_JOB_ID": result["job_ids"]["guardian"]}),
              patch.object(ops.time, "sleep", side_effect=record_ack) as sleep,
              patch.object(ops, "guardian_decision", return_value={"status": "incomplete", "reason": "synthetic_stop"})):
            self.assertEqual(ops.guardian(self.plan, self.authorization, self.scheduler), 2)
            sleep.assert_called_once_with(5)
        receipt = json.loads((ops._control(self.plan) / "manifests/terminal.json").read_text())
        self.assertEqual(receipt["guardian_job_id"], result["job_ids"]["guardian"])
        self.assertEqual(receipt["reason"], "synthetic_stop")
        self.assertEqual(len(self.scheduler.submissions), 8)

    def test_guardian_ack_wait_is_bounded(self):
        result = self.submit()
        path = ops._intent_path(self.plan, "guardian")
        intent = json.loads(path.read_text())
        intent.update(status="submitting")
        intent.pop("job_id")
        ops.atomic_json(path, intent)
        clock = [self.now.timestamp()]

        def advance(seconds):
            clock[0] += seconds

        with patch.dict(os.environ, {"SLURM_JOB_ID": result["job_ids"]["guardian"]}):
            with self.assertRaisesRegex(ops.PlanError, "acknowledgement wait expired"):
                ops._guardian_job(self.plan, ops.digest(self.authorization), wait_for_ack=True,
                                  clock=lambda: clock[0], sleep=advance)
        self.assertLessEqual(clock[0] - self.now.timestamp(), self.config["submission_grace_minutes"] * 60)
        self.assertFalse((ops._control(self.plan) / "manifests/terminal.json").exists())

    def test_guardian_reconciles_lost_ack_without_new_submission(self):
        result = self.submit()
        path = ops._intent_path(self.plan, "guardian")
        intent = json.loads(path.read_text())
        intent.update(status="ambiguous")
        intent.pop("job_id")
        ops.atomic_json(path, intent)
        self.scheduler.matches = [result["job_ids"]["guardian"]]
        with patch.dict(os.environ, {"SLURM_JOB_ID": result["job_ids"]["guardian"]}):
            self.assertEqual(ops._guardian_job(self.plan, ops.digest(self.authorization), self.scheduler,
                                              wait_for_ack=True), result["job_ids"]["guardian"])
        self.assertTrue(json.loads(path.read_text())["reconciled"])
        self.assertEqual(len(self.scheduler.submissions), 8)

    def test_terminal_incomplete_cannot_be_promoted_after_failure(self):
        result = self.submit()
        terminal = ops._control(self.plan) / "manifests/terminal.json"
        with patch.dict(os.environ, {"SLURM_JOB_ID": result["job_ids"]["guardian"]}):
            self.assertEqual(ops._finish_guardian(
                self.plan, {"status": "incomplete", "reason": "guardian_observation_failed"},
                result["job_ids"], self.scheduler, ops.digest(self.authorization)), 2)
            original = terminal.read_bytes()
            with patch.object(ops, "completion_evidence", side_effect=AssertionError("Terminal work must not be re-evaluated")):
                self.assertEqual(ops.guardian(self.plan, self.authorization, self.scheduler), 2)
            self.assertEqual(ops._finish_guardian(
                self.plan, {"status": "complete", "reason": "later_accounting_recovery"},
                result["job_ids"], self.scheduler, ops.digest(self.authorization)), 2)
        with patch.dict(os.environ, {"SLURM_JOB_ID": "999999"}):
            with self.assertRaises(ops.PlanError):
                ops._finish_guardian(self.plan, {"status": "complete"}, result["job_ids"],
                                     self.scheduler, ops.digest(self.authorization))
        self.assertEqual(terminal.read_bytes(), original)
        self.assertEqual(len(self.scheduler.cancellations), 1)

    def test_identical_terminal_repeat_is_idempotent(self):
        result = self.submit()
        terminal = ops._control(self.plan) / "manifests/terminal.json"
        decision = {"status": "incomplete", "reason": "synthetic_failure"}
        with patch.dict(os.environ, {"SLURM_JOB_ID": result["job_ids"]["guardian"]}):
            self.assertEqual(ops._finish_guardian(self.plan, decision, result["job_ids"],
                                                self.scheduler, ops.digest(self.authorization)), 2)
            original, modified = terminal.read_bytes(), terminal.stat().st_mtime_ns
            self.assertEqual(ops._finish_guardian(self.plan, decision, result["job_ids"],
                                                self.scheduler, ops.digest(self.authorization)), 2)
        self.assertEqual(terminal.read_bytes(), original)
        self.assertEqual(terminal.stat().st_mtime_ns, modified)
        self.assertEqual(len(self.scheduler.cancellations), 1)

    def test_completed_terminal_repeat_preserves_success_without_overwrite(self):
        result = self.submit()
        self.populate_evaluations()
        states = {name: "COMPLETED" for name in ops.ORDER if name != "guardian"}
        decision = ops.guardian_decision(self.plan, states, ops.completion_evidence(self.plan),
                                         now=self.now, launch={"status": "submitted"})
        self.assertEqual(decision["status"], "complete")
        terminal = ops._control(self.plan) / "manifests/terminal.json"
        with patch.dict(os.environ, {"SLURM_JOB_ID": result["job_ids"]["guardian"]}):
            self.assertEqual(ops._finish_guardian(self.plan, decision, result["job_ids"],
                                                self.scheduler, ops.digest(self.authorization)), 0)
            original = terminal.read_bytes()
            self.assertEqual(ops._finish_guardian(self.plan, decision, result["job_ids"],
                                                self.scheduler, ops.digest(self.authorization)), 0)
            self.assertEqual(ops._finish_guardian(
                self.plan, {"status": "incomplete", "reason": "late_observation"},
                result["job_ids"], self.scheduler, ops.digest(self.authorization)), 0)
        self.assertEqual(terminal.read_bytes(), original)
        self.assertEqual(self.scheduler.cancellations, [])

    def test_terminal_with_other_identity_cannot_be_replaced(self):
        result = self.submit()
        terminal = ops._control(self.plan) / "manifests/terminal.json"
        with patch.dict(os.environ, {"SLURM_JOB_ID": result["job_ids"]["guardian"]}):
            ops._finish_guardian(self.plan, {"status": "incomplete"}, result["job_ids"],
                                 self.scheduler, ops.digest(self.authorization))
            foreign = json.loads(terminal.read_text())
            foreign["authorization_sha256"] = "0" * 64
            ops.atomic_json(terminal, foreign)
            original = terminal.read_bytes()
            with self.assertRaisesRegex(ops.PlanError, "terminal receipt"):
                ops._finish_guardian(self.plan, {"status": "complete"}, result["job_ids"],
                                     self.scheduler, ops.digest(self.authorization))
        self.assertEqual(terminal.read_bytes(), original)
        self.assertEqual(len(self.scheduler.cancellations), 1)

    def test_failed_guardian_accounting_never_becomes_success(self):
        result = self.submit()
        self.populate_evaluations()
        for stage, row in self.plan["stages"].items():
            job = result["job_ids"][stage]
            keys = [job] if len(row["tasks"]) == 1 else [f"{job}_{index}" for index in range(len(row["tasks"]))]
            for key in keys:
                self.scheduler.snapshot[key] = {"state": "COMPLETED", "exit_code": "0:0"}
        self.scheduler.snapshot[result["job_ids"]["guardian"]] = {"state": "FAILED", "exit_code": "1:0"}
        with patch.dict(os.environ, {"SLURM_JOB_ID": result["job_ids"]["guardian"]}):
            self.assertEqual(ops.guardian(self.plan, self.authorization, self.scheduler), 2)
        receipt = json.loads((ops._control(self.plan) / "manifests/terminal.json").read_text())
        self.assertEqual(receipt["reason"], "guardian_allocation_failed")
        self.assertFalse(receipt["complete"])

    def assert_concurrent_terminal_publishers(self, second_decision):
        result = self.submit()
        terminal = ops._control(self.plan) / "manifests/terminal.json"
        entered, release, contended = threading.Event(), threading.Event(), threading.Event()
        first_scheduler, second_scheduler = FakeScheduler(), FakeScheduler()
        original_cancel = first_scheduler.cancel
        original_flock, original_atomic = ops.fcntl.flock, ops.atomic_json
        writes = []

        def blocking_cancel(ids):
            original_cancel(ids)
            entered.set()
            if not release.wait(20):
                raise RuntimeError("Synthetic publisher coordination timed out")

        def observed_flock(fd, operation):
            try:
                return original_flock(fd, operation)
            except BlockingIOError:
                contended.set()
                raise

        def observed_atomic(path, value):
            original_atomic(path, value)
            if Path(path) == terminal:
                writes.append(terminal.read_bytes())

        first_scheduler.cancel = blocking_cancel
        with (patch.dict(os.environ, {"SLURM_JOB_ID": result["job_ids"]["guardian"]}),
              patch.object(ops.fcntl, "flock", side_effect=observed_flock),
              patch.object(ops, "atomic_json", side_effect=observed_atomic),
              ThreadPoolExecutor(max_workers=2) as pool):
            first = pool.submit(ops._finish_guardian, self.plan,
                                {"status": "incomplete", "reason": "immutable_failure"},
                                result["job_ids"], first_scheduler, ops.digest(self.authorization))
            try:
                self.assertTrue(entered.wait(20), "First publisher did not acquire the terminal lock")
                second = pool.submit(ops._finish_guardian, self.plan, second_decision,
                                     result["job_ids"], second_scheduler, ops.digest(self.authorization))
                self.assertTrue(contended.wait(20), "Second publisher did not contend for the lock")
            finally:
                release.set()
            self.assertEqual(first.result(timeout=20), 2)
            self.assertEqual(second.result(timeout=20), 2)
        self.assertEqual(len(writes), 1)
        self.assertEqual(terminal.read_bytes(), writes[0])
        self.assertEqual(json.loads(writes[0])["reason"], "immutable_failure")
        self.assertEqual(len(first_scheduler.cancellations), 1)
        self.assertEqual(second_scheduler.cancellations, [])

    def test_concurrent_identical_terminal_publishers_are_idempotent(self):
        self.assert_concurrent_terminal_publishers({"status": "incomplete", "reason": "immutable_failure"})

    def test_concurrent_conflicting_publisher_cannot_upgrade_incomplete(self):
        self.assert_concurrent_terminal_publishers({"status": "complete", "reason": "attempted_upgrade"})

    def test_terminal_publisher_lock_wait_is_bounded(self):
        self.submit()
        clock = [0.0]

        def advance(seconds):
            clock[0] += seconds

        with patch.object(ops.fcntl, "flock", side_effect=BlockingIOError):
            with self.assertRaisesRegex(ops.PlanError, "terminal publisher lock wait expired"):
                with ops._terminal_lock(ops._control(self.plan) / "manifests/terminal.json",
                                        clock=lambda: clock[0], sleep=advance):
                    self.fail("An unavailable terminal lock must never permit publication")
        self.assertLessEqual(clock[0], 35)


if __name__ == "__main__":
    unittest.main()
