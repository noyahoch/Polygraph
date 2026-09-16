"""Stdlib-only planner/receipt tests. Scheduler and scientific commands are mocked.

These tests never read datasets, restore models or call Slurm. Numerical package
tests belong exclusively in the separately approved allocated CPU validation.
Scratch files are confined to a named working-directory artifact root.
"""
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
import threading
import unittest
from unittest.mock import patch
import uuid

from . import slurm as ops


class MockScheduler:
    def __init__(self):
        self.submissions, self.reconciliations, self.cancellations = [], [], []
        self.matches, self.snapshot = [], {}
        self.external = 0
        self.failure_at = None

    def submit(self, command):
        self.submissions.append(command)
        if self.failure_at == len(self.submissions):
            raise TimeoutError("Synthetic lost ACK after possible acceptance")
        return str(990000 + len(self.submissions))

    def reconcile(self, intent):
        self.reconciliations.append(copy.deepcopy(intent))
        return list(self.matches)

    def states(self, ids):
        return {key: row for key, row in self.snapshot.items() if key.split("_")[0] in ids}

    def cancel(self, ids):
        self.cancellations.append(list(ids))

    def external_gpu_count(self, owned):
        return self.external


class FinalSlurmTests(unittest.TestCase):
    def setUp(self):
        parent = Path(os.environ.get("POLYGRAPH_TEST_ARTIFACT_ROOT", "work/final-comparison-ops-tests")).resolve()
        self.directory = parent / ("planner-" + uuid.uuid4().hex)
        self.directory.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.directory)
        self.now = dt.datetime.now(dt.timezone.utc)
        control = self.directory / "control"
        release = control / "releases/synthetic"
        inventory = {}
        for name in sorted(ops.REQUIRED_SOURCE):
            path = release / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# synthetic source identity; never executed\n")
            inventory[name] = ops.file_sha256(path)
        ops.atomic_json(release / "source_manifest.json", {"release_id": release.name, "source_sha256": inventory})
        self.config = {
            "mode": "benchmark", "approval_id": "synthetic-full-ready", "run_id": "final-synthetic",
            "root": str(self.directory / "scientific"), "control_root": str(control), "code_root": str(release),
            "pilot_root": str(self.directory / "old_pilot"), "replication_root": str(self.directory / "old_replication"),
            "cache": str(self.directory / "old_cache"), "data_root": str(self.directory / "old_raw"),
            "python": str(self.directory / "old_pilot/env/bin/python"), "source_sha256": inventory,
            "environment": {"import_bundle_sha256": "1" * 64, "dependency_manifest_sha256": "2" * 64,
                            "hf_hub_cache": str(self.directory / "old_hf_hub")},
            "resources": {**ops.RESOURCE_CLASS, "cpu_cpus": 2, "cpu_memory_mb": 8000},
            "caps": {name: 130 if name == "guardian" else 5 if name == "failure_guard" else 1 for name in ops.ORDER},
            "deadlines": {name: (self.now + dt.timedelta(minutes=minutes)).isoformat()
                          for name, minutes in (("base", 90), ("predictions", 100), ("evaluation", 110))},
            "max_concurrent_gpus": 8, "external_reserved_gpus": 0,
            "gpu_budget_minutes": 1000, "cpu_budget_minutes": 10000,
            "external_gpu_minutes": 0, "external_cpu_minutes": 0,
            "submission_grace_minutes": 5, "guardian_grace_minutes": 5,
            "test_selectors": [], "evaluation_files": [
                "complete.json", "report.json", "scores.npz", "bootstrap.json", "REPORT.md", "REPORT.he.md",
            ], "campaign_binding": None, "preflight": None,
        }
        root = Path(self.config["root"])
        ops.atomic_json(root / "role_map.json", {"original_test_access": False, "roles": "synthetic identity only"})
        campaign = {
            "scope_id": ops.SCOPE, "run_id": self.config["run_id"], "root": str(root),
            "neural_matrix": ops.neural_matrix(),
            "missing_matrix": [row for row in ops.neural_matrix() if not ops.is_import(row)],
            "reuse": {f"G/{arm}/seed{seed}": {"status": "late" if seed == 7 else "complete"}
                      for seed in ops.IMPORTED_SEEDS for arm in ops.ARMS},
            "reuse_groups": {str(seed): {"status": "late" if seed == 7 else "complete"} for seed in ops.IMPORTED_SEEDS},
            "source_identity": {name: sha for name, sha in inventory.items()
                                if name.startswith("pilots/final_comparison_20260916/")
                                and not name.endswith(("test_slurm.py", "__init__.py"))},
            "original_test_access": False,
        }
        ops.atomic_json(root / "campaign.json", campaign)
        self.config["campaign_binding"] = ops.campaign_binding(root)
        prior_path = self.directory / "prior_preflight/terminal.json"
        prior = {
            "complete": True, "mode": "preflight", "workflow_sha256": "e" * 64,
            "release_sha256": ops.digest(inventory), "run_id": self.config["run_id"],
            "evidence": {"campaign_binding": self.config["campaign_binding"]},
            "resource_claims": {"reserved_gpu_minutes": 20, "allocated_cpu_minutes": 200},
        }
        ops.atomic_json(prior_path, prior)
        self.config["preflight"] = {
            "terminal_path": str(prior_path), "terminal_sha256": ops.file_sha256(prior_path),
            "workflow_sha256": prior["workflow_sha256"],
            "reserved_gpu_minutes": 20, "allocated_cpu_minutes": 200,
        }
        self.plan = ops.build_plan(self.config, now=self.now)
        self.authorization = self.approve(self.plan)
        self.scheduler = MockScheduler()

    def approve(self, plan):
        value = ops.authorization_template(plan)
        value.update(approved=True, source_ready_at=(self.now - dt.timedelta(minutes=3)).isoformat(),
                     authorized_at=(self.now - dt.timedelta(minutes=1)).isoformat(),
                     expires_at=max(plan["config"]["deadlines"].values()))
        if plan["mode"] == "benchmark":
            value.update(full_ready_at=(self.now - dt.timedelta(minutes=2)).isoformat(),
                         measured_resource_approval=True)
        return value

    def preflight(self):
        config = copy.deepcopy(self.config)
        config.update(mode="preflight", approval_id="synthetic-source-ready", campaign_binding=None, preflight=None,
                      test_selectors=[ops.MODULE + "test_slurm"], evaluation_files=[], max_concurrent_gpus=1,
                      caps={"guardian": 65, "failure_guard": 5, "cpu_validation": 5, "gpu_preflight": 10},
                      deadlines={"validation": (self.now + dt.timedelta(minutes=45)).isoformat()})
        return ops.build_plan(config, now=self.now)

    def write(self, relative, value):
        path = self.directory / relative
        ops.atomic_json(path, value)
        return path

    def submit(self):
        return ops.submit_plan(self.plan, self.authorization, self.scheduler, now=self.now)

    def env(self, job, **extra):
        result = {"SLURM_JOB_ID": job, **extra}
        return patch.dict(os.environ, result, clear=True)

    def successful_states(self, jobs):
        states = {}
        for name, row in self.plan["stages"].items():
            job = jobs[name]
            keys = [job] if len(row["tasks"]) == 1 else [f"{job}_{i}" for i in range(len(row["tasks"]))]
            for key in keys:
                states[key] = {"state": "COMPLETED", "exit_code": "0:0"}
        return states

    def evidence(self, plan=None):
        plan = plan or self.plan
        result = {"complete": True, "workflow_sha256": ops.digest(plan), "release_sha256": plan["release_sha256"],
                  "files": {name: "a" * 64 for name in ops.required_evidence_paths(plan)},
                  "campaign_binding": self.config["campaign_binding"]}
        if plan["mode"] == "preflight":
            result.update(scientific_fits=0, dev_eval_scoring=False)
        else:
            result.update(neural_fits=39, new_neural_fits=27, imported_neural_fits=12, meta_heads=18,
                          linear_fits=1, diagnostic_histories=39, method_names=list(ops.METHODS),
                          primary_contrasts=[list(row) for row in ops.CONTRASTS],
                          bootstrap_draws=10000, source_files_verified=True)
        return result

    def populate_task_receipts(self, jobs):
        for name, row in self.plan["stages"].items():
            if name in ops.GUARDS:
                continue
            for index, task in enumerate(row["tasks"]):
                ops.atomic_json(ops._receipt_path(self.plan, name, jobs[name], index), {
                    "status": "completed", "exit_code": 0, "stage": name, "task": task["key"],
                    "workflow_sha256": ops.digest(self.plan), "authorization_sha256": ops.digest(self.authorization),
                    "job_id": jobs[name], "array_index": index, "commands": task["commands"],
                    "completed_utc": self.now.isoformat(),
                })

    def test_default_cli_is_inert_even_with_explicit_approved_json(self):
        config = self.write("config.json", self.config)
        approval = self.write("approval.json", self.authorization)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(ops.main(["--config", str(config), "--authorization", str(approval)],
                                     scheduler=self.scheduler, now=self.now), 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "validated_not_submitted")
        self.assertEqual(self.scheduler.submissions, [])
        self.assertFalse((ops._control(self.plan) / "manifests").exists())
        self.assertFalse((Path(self.config["root"]) / "execution.json").exists())

    def test_validate_and_dry_run_do_not_touch_scheduler(self):
        workflow = self.write("workflow.json", self.plan)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ops.main(["validate", "--workflow", str(workflow), "--sha256", ops.digest(self.plan)],
                                     scheduler=self.scheduler, now=self.now), 0)
        self.assertEqual(self.scheduler.submissions, [])

    def test_submit_requires_all_new_authorization_gates(self):
        for changes in ({"approved": False}, {"source_ready_at": None}, {"full_ready_at": None},
                        {"measured_resource_approval": False}, {"resource_limits": {}}, {"campaign_binding": None}):
            approval = {**self.authorization, **changes}
            with self.subTest(changes=changes), self.assertRaises((ops.PlanError, TypeError)):
                ops.submit_plan(self.plan, approval, self.scheduler, now=self.now)
        self.assertEqual(self.scheduler.submissions, [])

    def test_exact_27_new_neural_fits_and_twelve_imports(self):
        self.assertEqual(ops.SEEDS, (7, 17, 27))
        self.assertEqual(len(self.plan["matrix"]), 39)
        self.assertEqual(len(self.plan["new_matrix"]), 27)
        self.assertEqual(len(self.plan["imports"]), 12)
        self.assertEqual((self.plan["neural_fits"], self.plan["meta_heads"], self.plan["new_meta_heads"]),
                         (39, 18, 12))
        rows = self.plan["stages"]["gpu_fits"]["tasks"]
        self.assertEqual(len(rows), 24)
        self.assertEqual({family: sum(row["family"] == family for row in rows) for family in ("G", "H", "S")},
                         {"G": 0, "H": 12, "S": 12})
        self.assertFalse(any(ops.is_import(row) for row in rows))
        cpu = self.plan["stages"]["cpu_fits"]["tasks"]
        self.assertEqual([(row["family"], row["seed"]) for row in cpu], [("O", seed) for seed in ops.SEEDS])
        self.assertEqual(self.plan["automatic_retries"], 0)
        self.assertEqual(self.plan["epochs"], 20)

    def test_linear_fit_is_one_cpu_task_not_per_seed_copies(self):
        row = self.plan["stages"]["linear_fit"]
        self.assertFalse(row["gpu"])
        self.assertEqual(len(row["tasks"]), 1)
        self.assertNotIn("--seed", row["tasks"][0]["commands"][0])
        self.assertEqual(self.plan["linear_fits"], 1)

    def test_output_linear_heads_statistics_and_imports_never_reserve_gpus(self):
        for name in ("cpu_fits", "linear_fit", "meta_import", "heads", "dev_import",
                     "dev_o", "static_export", "analysis", *ops.GUARDS):
            self.assertFalse(self.plan["stages"][name]["gpu"], name)
        self.assertEqual(len(self.plan["stages"]["heads"]["tasks"]), 9)
        self.assertEqual(sum(t["imported"] for t in self.plan["stages"]["heads"]["tasks"]), 3)
        self.assertEqual(len(self.plan["stages"]["dev_o"]["tasks"]), 3)

    def test_fixed_global_dependencies_and_six_locked_phases(self):
        stages = self.plan["stages"]
        self.assertEqual(self.plan["phases"], {k: list(v) for k, v in ops.PHASES.items()})
        self.assertEqual(len(self.plan["phases"]), 6)
        for name in ("gpu_fits", "cpu_fits", "linear_fit"):
            self.assertEqual(stages[name]["dependencies"], ["hidden_full"])
        self.assertEqual(stages["base_freeze"]["dependencies"], ["gpu_fits", "cpu_fits", "linear_fit"])
        self.assertEqual(stages["heads"]["dependencies"], ["meta_gpu", "meta_import"])
        for name in ("dev_gpu", "dev_import", "dev_o", "static_export"):
            self.assertEqual(stages[name]["dependencies"], ["evaluation_gate"])
        gate = stages["evaluation_gate"]["tasks"][0]["commands"]
        self.assertEqual([command[command.index("-m") + 2] for command in gate], ["heads", "evaluation"])
        self.assertEqual(stages["failure_guard"]["dependency_type"], "afterany")

    def test_three_imported_G_prediction_groups_stay_cpu_and_preserve_all_seeds(self):
        for name in ("meta_import", "dev_import"):
            tasks = self.plan["stages"][name]["tasks"]
            self.assertEqual([(t["family"], t["seed"]) for t in tasks], [("G", n) for n in ops.IMPORTED_SEEDS])
            self.assertTrue(all(t["imported"] for t in tasks))
        for name in ("meta_gpu", "dev_gpu"):
            tasks = self.plan["stages"][name]["tasks"]
            self.assertEqual([(t["family"], t["seed"]) for t in tasks],
                             [(family, seed) for family in ("H", "S") for seed in ops.SEEDS])
            self.assertFalse(any(t["imported"] for t in tasks))

    def test_an_empty_stage_fails_planning_instead_of_submitting_an_empty_array(self):
        with patch.object(ops, "IMPORTED_SEEDS", ()), self.assertRaisesRegex(ops.PlanError, "no tasks"):
            ops.build_plan(self.config, now=self.now)

    def test_all_array_tasks_and_preflight_are_in_the_total_reservation(self):
        claims = self.plan["resource_claims"]
        self.assertEqual(claims["workflow_gpu_minutes"], 1 + 24 + 6 + 6)
        self.assertEqual(claims["reserved_gpu_minutes"], 57)
        expected_cpu = sum(row["minutes"] * row["cpus"] * len(row["tasks"]) for row in self.plan["stages"].values())
        self.assertEqual(claims["allocated_cpu_minutes"], expected_cpu + 200)
        self.assertEqual(claims["submission_count"], len(ops.ORDER))
        config = copy.deepcopy(self.config)
        config["gpu_budget_minutes"] = 56
        with self.assertRaisesRegex(ops.PlanError, "NEW budgets"):
            ops.build_plan(config, now=self.now)
        config = copy.deepcopy(self.config)
        config["cpu_budget_minutes"] = expected_cpu + 199
        with self.assertRaisesRegex(ops.PlanError, "NEW budgets"):
            ops.build_plan(config, now=self.now)

    def test_no_inherited_or_default_full_gpu_budget(self):
        config = copy.deepcopy(self.config)
        del config["gpu_budget_minutes"]
        with self.assertRaises(ops.PlanError):
            ops.build_plan(config, now=self.now)

    def test_existing_gpu_class_and_combined_concurrency_ceiling(self):
        for change in ({"max_concurrent_gpus": 9}, {"external_reserved_gpus": 1},
                       {"resources": {**self.config["resources"], "gpu_constraint": "a100"}},
                       {"resources": {**self.config["resources"], "gpu_memory_mb": 64000}}):
            with self.subTest(change=change), self.assertRaises(ops.PlanError):
                ops.build_plan({**self.config, **change}, now=self.now)
        config = {**self.config, "max_concurrent_gpus": 6, "external_reserved_gpus": 2}
        plan = ops.build_plan(config, now=self.now)
        self.assertEqual(plan["resource_claims"]["gpu_ceiling_including_external"], 8)
        self.assertEqual(plan["stages"]["gpu_fits"]["throttle"], 6)

    def test_live_external_reservations_stop_before_any_submission(self):
        self.scheduler.external = 1
        with self.assertRaisesRegex(ops.PlanError, "Live external"):
            self.submit()
        self.assertEqual(self.scheduler.submissions, [])

    def test_original_roots_and_pinned_python_cannot_be_redirected(self):
        for changes in ({"root": self.config["cache"]}, {"control_root": self.config["pilot_root"]},
                        {"python": "/usr/bin/python3"}):
            with self.subTest(changes=changes), self.assertRaises(ops.PlanError):
                ops.build_plan({**self.config, **changes}, now=self.now)

    def test_changed_source_roles_reuse_or_preflight_blocks_launch(self):
        source = Path(self.config["code_root"]) / ops.SELF
        before = source.read_text()
        source.write_text(before + "# changed\n")
        with self.assertRaisesRegex(ops.PlanError, "source"):
            self.submit()
        source.write_text(before)
        roles = Path(self.config["root"]) / "role_map.json"
        roles.write_text("{}\n")
        with self.assertRaisesRegex(ops.PlanError, "role/reuse"):
            self.submit()
        self.assertEqual(self.scheduler.submissions, [])

    def test_expired_or_nonprospective_cutoffs_and_short_guardian_are_rejected(self):
        config = copy.deepcopy(self.config)
        config["deadlines"]["base"] = (self.now + dt.timedelta(minutes=5)).isoformat()
        with self.assertRaisesRegex(ops.PlanError, "prospective"):
            ops.build_plan(config, now=self.now)
        config = copy.deepcopy(self.config)
        config["caps"]["guardian"] = 20
        with self.assertRaisesRegex(ops.PlanError, "guardian"):
            ops.build_plan(config, now=self.now)
        approval = {**self.authorization, "expires_at": (self.now - dt.timedelta(minutes=1)).isoformat()}
        with self.assertRaises(ops.PlanError):
            ops.validate_authorization(self.plan, approval, now=self.now)

    def test_preflight_is_a_bounded_non_scientific_DAG(self):
        plan = self.preflight()
        self.assertEqual(plan["order"], list(ops.PREFLIGHT_ORDER))
        self.assertEqual(plan["resource_claims"]["workflow_gpu_minutes"], 10)
        modules = [command[command.index("-m") + 1] for row in plan["stages"].values()
                   for task in row["tasks"] for command in task["commands"]]
        self.assertEqual(modules, ["unittest", ops.MODULE + "protocol", ops.MODULE + "preflight"])
        self.assertNotIn(ops.MODULE + "train", modules)
        self.assertNotIn(ops.MODULE + "evaluate", modules)
        self.assertEqual(plan["stages"]["cpu_validation"]["gpu"], False)
        config = copy.deepcopy(plan["config"])
        config["caps"]["gpu_preflight"] = 31
        with self.assertRaisesRegex(ops.PlanError, "bounded"):
            ops.build_plan(config, now=self.now)

    def test_sbatch_guardian_first_and_complete_static_dependency_IDs(self):
        result = self.submit()
        self.assertEqual(list(result["job_ids"]), list(ops.ORDER))
        self.assertEqual(len(self.scheduler.submissions), len(ops.ORDER))
        self.assertIn("--partition=cpu-killable", self.scheduler.submissions[0])
        self.assertIn("--dependency=afterany:" + result["job_ids"]["guardian"], self.scheduler.submissions[1])
        for name, command in zip(ops.ORDER, self.scheduler.submissions):
            self.assertIn("--no-requeue", command)
            self.assertIn("--kill-on-invalid-dep=yes", command)
            self.assertIn("--export=NONE", command)
            self.assertEqual(any(arg.startswith("--gpus=") for arg in command), self.plan["stages"][name]["gpu"])
        fit = self.scheduler.submissions[list(ops.ORDER).index("gpu_fits")]
        self.assertIn("--array=0-23%8", fit)
        self.assertIn("--cpus-per-task=6", fit)
        self.assertIn("--mem=32000M", fit)
        self.assertFalse(result["laptop_close_ready"])

    def test_repeated_committed_submission_never_duplicates_jobs(self):
        original = self.submit()
        again = self.submit()
        self.assertEqual(original, again)
        self.assertEqual(len(self.scheduler.submissions), len(ops.ORDER))
        self.assertEqual(json.loads((Path(self.config["root"]) / "execution.json").read_text()),
                         ops.execution_record(self.plan, self.authorization))

    def test_ambiguous_accepted_job_reconciles_without_duplicate(self):
        self.scheduler.failure_at = 3
        with self.assertRaises(ops.AmbiguousSubmission):
            self.submit()
        path = ops._intent_path(self.plan, "prepare")
        self.assertEqual(json.loads(path.read_text())["status"], "ambiguous")
        self.scheduler.matches = ["990003"]
        result = ops.reconcile_intents(self.plan, self.scheduler, ops.digest(self.authorization))
        self.assertEqual(result["job_ids"]["prepare"], "990003")
        self.assertFalse(result["unresolved"])
        self.assertEqual(len(self.scheduler.submissions), 3)
        with self.assertRaisesRegex(ops.PlanError, "never expand"):
            self.submit()
        self.assertEqual(len(self.scheduler.submissions), 3)

    def test_zero_or_multiple_acceptance_matches_never_retry(self):
        self.scheduler.failure_at = 1
        with self.assertRaises(ops.AmbiguousSubmission):
            self.submit()
        for matches in ([], ["999001", "999002"]):
            self.scheduler.matches = matches
            with self.subTest(matches=matches), self.assertRaises(ops.AmbiguousSubmission):
                self.submit()
        self.assertEqual(len(self.scheduler.submissions), 1)

    def test_invalid_sbatch_receipt_remains_ambiguous(self):
        with patch.object(self.scheduler, "submit", return_value="accepted maybe"):
            with self.assertRaises(ops.AmbiguousSubmission):
                self.submit()
        self.assertEqual(json.loads(ops._intent_path(self.plan, "guardian").read_text())["status"], "ambiguous")

    def test_guardian_wrong_allocation_and_array_are_rejected_before_side_effects(self):
        jobs = self.submit()["job_ids"]
        for env in ({"SLURM_JOB_ID": "999999"}, {"SLURM_JOB_ID": jobs["guardian"], "SLURM_ARRAY_TASK_ID": "0"},
                    {"SLURM_JOB_ID": jobs["guardian"], "SLURM_ARRAY_JOB_ID": jobs["guardian"]}):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True), self.assertRaises(ops.PlanError):
                ops._bound_guard_job(self.plan, ops.digest(self.authorization), self.scheduler)
        self.assertEqual(self.scheduler.reconciliations, [])
        self.assertEqual(self.scheduler.cancellations, [])
        self.assertFalse((ops._control(self.plan) / "manifests/terminal.json").exists())

    def test_initial_guardian_ack_race_is_bounded_and_can_resolve(self):
        jobs = self.submit()["job_ids"]
        path = ops._intent_path(self.plan, "guardian")
        original = json.loads(path.read_text())
        intent = {**original, "status": "submitting"}
        intent.pop("job_id")
        ops.atomic_json(path, intent)

        def ack(_seconds):
            ops.atomic_json(path, original)

        with self.env(jobs["guardian"]):
            self.assertEqual(ops._bound_guard_job(self.plan, ops.digest(self.authorization), wait_for_ack=True,
                                                  sleep=ack), jobs["guardian"])
        self.assertEqual(len(self.scheduler.submissions), len(ops.ORDER))

    def test_ack_wait_has_a_finite_deadline(self):
        jobs = self.submit()["job_ids"]
        path = ops._intent_path(self.plan, "guardian")
        intent = json.loads(path.read_text())
        intent.update(status="submitting")
        intent.pop("job_id")
        ops.atomic_json(path, intent)
        clock = [self.now.timestamp()]

        def advance(seconds):
            clock[0] += seconds

        with self.env(jobs["guardian"]), self.assertRaisesRegex(ops.PlanError, "acknowledgement"):
            ops._bound_guard_job(self.plan, ops.digest(self.authorization), wait_for_ack=True,
                                 clock=lambda: clock[0], sleep=advance)
        self.assertLessEqual(clock[0] - self.now.timestamp(), 300)

    def test_lost_guardian_ack_is_reconciled_without_new_sbatch(self):
        jobs = self.submit()["job_ids"]
        path = ops._intent_path(self.plan, "guardian")
        intent = json.loads(path.read_text())
        intent.update(status="ambiguous")
        intent.pop("job_id")
        ops.atomic_json(path, intent)
        self.scheduler.matches = [jobs["guardian"]]
        with self.env(jobs["guardian"]):
            self.assertEqual(ops._bound_guard_job(self.plan, ops.digest(self.authorization), self.scheduler,
                                                  wait_for_ack=True), jobs["guardian"])
        self.assertTrue(json.loads(path.read_text())["reconciled"])
        self.assertEqual(len(self.scheduler.submissions), len(ops.ORDER))

    def test_stage_binding_wrong_ID_cpu_GPU_and_requeue_rejected(self):
        jobs = self.submit()["job_ids"]
        with self.env("999999"), self.assertRaisesRegex(ops.PlanError, "recorded"):
            ops._stage_identity(self.plan, "linear_fit", jobs)
        with self.env(jobs["linear_fit"], SLURM_JOB_GPUS="0"), self.assertRaisesRegex(ops.PlanError, "CPU-only"):
            ops._stage_identity(self.plan, "linear_fit", jobs)
        with self.env(jobs["linear_fit"], SLURM_RESTART_COUNT="1"), self.assertRaisesRegex(ops.PlanError, "retries"):
            ops._stage_identity(self.plan, "linear_fit", jobs)
        with self.env("123", SLURM_ARRAY_JOB_ID=jobs["gpu_fits"], SLURM_ARRAY_TASK_ID="48", SLURM_GPUS="1"):
            with self.assertRaisesRegex(ops.PlanError, "array index"):
                ops._stage_identity(self.plan, "gpu_fits", jobs)

    def test_every_array_element_and_exit_code_is_required(self):
        jobs = self.submit()["job_ids"]
        states = self.successful_states(jobs)
        self.assertEqual(ops.stage_states(self.plan, jobs, states)["gpu_fits"], "COMPLETED")
        states.pop(jobs["gpu_fits"] + "_23")
        self.assertEqual(ops.stage_states(self.plan, jobs, states)["gpu_fits"], "UNKNOWN")
        states[jobs["gpu_fits"] + "_23"] = {"state": "COMPLETED", "exit_code": "1:0"}
        self.assertEqual(ops.stage_states(self.plan, jobs, states)["gpu_fits"], "FAILED")

    def test_guardian_six_phase_receipts_are_immutable_and_bind_task_history(self):
        jobs = self.submit()["job_ids"]
        self.populate_task_receipts(jobs)
        statuses = ops.stage_states(self.plan, jobs, self.successful_states(jobs))
        ops._freeze_phases(self.plan, self.authorization, jobs, statuses)
        phases = ops._control(self.plan) / "manifests/phases"
        self.assertEqual({p.stem for p in phases.glob("*.json")}, set(ops.PHASES))
        before = (phases / "bases.json").read_bytes()
        path = ops._receipt_path(self.plan, "cpu_fits", jobs["cpu_fits"], 0)
        value = json.loads(path.read_text())
        value["completed_utc"] = (self.now + dt.timedelta(seconds=1)).isoformat()
        ops.atomic_json(path, value)
        with self.assertRaisesRegex(ops.PlanError, "frozen artifact"):
            ops._freeze_phases(self.plan, self.authorization, jobs, statuses)
        self.assertEqual((phases / "bases.json").read_bytes(), before)

    def test_guardian_needs_all_25_outputs_not_a_bare_success_marker(self):
        statuses = {name: "COMPLETED" for name in ops.ORDER if name not in ops.GUARDS}
        evidence = self.evidence()
        self.assertEqual(ops.guardian_decision(self.plan, statuses, evidence, now=self.now,
                                              launch={"status": "submitted"})["status"], "complete")
        for bad in ({"complete": True}, {**evidence, "method_names": list(ops.METHODS[:-1])},
                    {**evidence, "linear_fits": 5}, {**evidence, "diagnostic_histories": 27},
                    {**evidence, "bootstrap_draws": 9999}, {**evidence, "files": {}}):
            with self.subTest(bad=bad.get("method_names", "other")):
                self.assertEqual(ops.guardian_decision(self.plan, statuses, bad, now=self.now,
                                                      launch={"status": "submitted"})["status"], "incomplete")

    def test_deadlines_and_failed_required_stages_fail_closed(self):
        states = {name: "PENDING" for name in ops.ORDER if name not in ops.GUARDS}
        result = ops.guardian_decision(self.plan, states, now=self.now + dt.timedelta(minutes=91),
                                      launch={"status": "submitted"})
        self.assertEqual(result["reason"], "base_deadline")
        result = ops.guardian_decision(self.plan, {**states, "gpu_fits": "OUT_OF_MEMORY"}, now=self.now,
                                      launch={"status": "submitted"})
        self.assertEqual(result["reason"], "required_stage_failed")
        self.assertEqual(ops.guardian_decision(self.plan, states, now=self.now,
                                              launch={"status": "incomplete_submission"})["reason"],
                         "incomplete_submission")

    def test_incomplete_terminal_is_write_once_and_cannot_be_promoted(self):
        jobs = self.submit()["job_ids"]
        path = ops._control(self.plan) / "manifests/terminal.json"
        with self.env(jobs["guardian"]):
            self.assertEqual(ops._finish_guardian(self.plan, {"status": "incomplete", "reason": "first_failure"},
                                                  jobs, self.scheduler, ops.digest(self.authorization)), 2)
            before, modified = path.read_bytes(), path.stat().st_mtime_ns
            self.assertEqual(ops._finish_guardian(self.plan, {"status": "complete"}, jobs, self.scheduler,
                                                  ops.digest(self.authorization)), 2)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(path.stat().st_mtime_ns, modified)
        self.assertEqual(len(self.scheduler.cancellations), 1)
        self.assertEqual(set(self.scheduler.cancellations[0]),
                         {job for name, job in jobs.items() if name not in ops.GUARDS})

    def test_running_guardian_cannot_publish_success_before_accounting(self):
        jobs = self.submit()["job_ids"]
        with self.env(jobs["guardian"]), self.assertRaisesRegex(ops.PlanError, "afterany"):
            ops._finish_guardian(self.plan, {"status": "complete", "evidence": self.evidence()},
                                 jobs, self.scheduler, ops.digest(self.authorization))

    def test_afterany_guard_rejects_failed_guardian_history_despite_complete_candidate(self):
        jobs = self.submit()["job_ids"]
        evidence = self.evidence()
        candidate = {"status": "complete", "workflow_sha256": ops.digest(self.plan),
                     "guardian_job_id": jobs["guardian"], "evidence": evidence}
        states = self.successful_states(jobs)
        self.assertEqual(ops.failure_guard_decision(self.plan, jobs, states, candidate, evidence)["status"], "complete")
        for state, code in (("FAILED", "1:0"), ("OUT_OF_MEMORY", "0:9"), ("CANCELLED", "0:15"),
                            ("COMPLETED", "1:0"), ("RUNNING", "0:0")):
            changed = {**states, jobs["guardian"]: {"state": state, "exit_code": code}}
            result = ops.failure_guard_decision(self.plan, jobs, changed, candidate, evidence)
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["reason"], "guardian_accounting_not_successful")

    def test_afterany_terminal_success_is_idempotent_and_history_bound(self):
        jobs = self.submit()["job_ids"]
        states, evidence = self.successful_states(jobs), self.evidence()
        decision = {"status": "complete", "reason": "all_history_verified", "evidence": evidence}
        path = ops._control(self.plan) / "manifests/terminal.json"
        with self.env(jobs["failure_guard"]):
            self.assertEqual(ops._finish_guardian(self.plan, decision, jobs, self.scheduler,
                                                  ops.digest(self.authorization), states, publisher="failure_guard"), 0)
            before = path.read_bytes()
            self.assertEqual(ops._finish_guardian(self.plan, {"status": "incomplete"}, jobs, self.scheduler,
                                                  ops.digest(self.authorization), publisher="failure_guard"), 0)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.scheduler.cancellations, [])

    def test_foreign_terminal_cannot_be_overwritten(self):
        jobs = self.submit()["job_ids"]
        path = ops._control(self.plan) / "manifests/terminal.json"
        with self.env(jobs["guardian"]):
            ops._finish_guardian(self.plan, {"status": "incomplete"}, jobs, self.scheduler, ops.digest(self.authorization))
            value = json.loads(path.read_text())
            value["authorization_sha256"] = "0" * 64
            ops.atomic_json(path, value)
            before = path.read_bytes()
            with self.assertRaisesRegex(ops.PlanError, "terminal receipt"):
                ops._finish_guardian(self.plan, {"status": "incomplete"}, jobs, self.scheduler,
                                     ops.digest(self.authorization))
        self.assertEqual(path.read_bytes(), before)

    def assert_concurrent_terminal(self, second_decision):
        jobs = self.submit()["job_ids"]
        path = ops._control(self.plan) / "manifests/terminal.json"
        entered, release, contended = threading.Event(), threading.Event(), threading.Event()
        first, second = MockScheduler(), MockScheduler()
        original_cancel, original_flock, original_atomic = first.cancel, ops.fcntl.flock, ops.atomic_json
        writes = []

        def blocking_cancel(ids):
            original_cancel(ids)
            entered.set()
            if not release.wait(10):
                raise RuntimeError("Mock terminal synchronization timed out")

        def flock(fd, operation):
            try:
                return original_flock(fd, operation)
            except BlockingIOError:
                contended.set()
                raise

        def atomic(file, value):
            original_atomic(file, value)
            if Path(file) == path:
                writes.append(path.read_bytes())

        first.cancel = blocking_cancel
        with (self.env(jobs["guardian"]), patch.object(ops.fcntl, "flock", side_effect=flock),
              patch.object(ops, "atomic_json", side_effect=atomic), ThreadPoolExecutor(max_workers=2) as pool):
            a = pool.submit(ops._finish_guardian, self.plan, {"status": "incomplete", "reason": "immutable_first"},
                            jobs, first, ops.digest(self.authorization))
            try:
                self.assertTrue(entered.wait(10))
                b = pool.submit(ops._finish_guardian, self.plan, second_decision, jobs, second,
                                ops.digest(self.authorization))
                self.assertTrue(contended.wait(10))
            finally:
                release.set()
            self.assertEqual(a.result(timeout=10), 2)
            self.assertEqual(b.result(timeout=10), 2)
        self.assertEqual(len(writes), 1)
        self.assertEqual(path.read_bytes(), writes[0])
        self.assertEqual(len(first.cancellations), 1)
        self.assertEqual(second.cancellations, [])

    def test_concurrent_identical_terminal_publication_is_idempotent(self):
        self.assert_concurrent_terminal({"status": "incomplete", "reason": "immutable_first"})

    def test_concurrent_conflicting_terminal_cannot_upgrade_failure(self):
        self.assert_concurrent_terminal({"status": "complete", "reason": "forbidden_upgrade"})

    def test_terminal_lock_wait_is_bounded(self):
        self.submit()
        clock = [0.0]

        def advance(seconds):
            clock[0] += seconds

        with patch.object(ops.legacy.fcntl, "flock", side_effect=BlockingIOError):
            with self.assertRaisesRegex(ops.PlanError, "lock wait expired"):
                with ops._terminal_lock(ops._control(self.plan) / "manifests/terminal.json",
                                        clock=lambda: clock[0], sleep=advance):
                    self.fail("Contended terminal lock cannot permit publication")
        self.assertLessEqual(clock[0], 35)

    def test_evaluation_receipt_binds_inventory_and_all_registered_methods(self):
        root = Path(self.config["root"]) / "evaluation"
        root.mkdir()
        report = {"scope_id": ops.SCOPE, "campaign_sha256": self.config["campaign_binding"]["campaign_sha256"],
                  "method_names": list(ops.METHODS), "primary_contrasts": [list(row) for row in ops.CONTRASTS],
                  "seeds": list(ops.SEEDS), "bootstrap_draws": 10000}
        ops.atomic_json(root / "report.json", report)
        for name in self.config["evaluation_files"]:
            if name not in ("report.json", "complete.json"):
                (root / name).write_text("synthetic output identity; no numerical data\n")
        complete = {"complete": True, "files": {name: ops.file_sha256(root / name)
                                               for name in self.config["evaluation_files"] if name != "complete.json"}}
        ops.atomic_json(root / "complete.json", complete)
        self.assertEqual(ops.validate_evaluation_outputs(self.plan), complete)
        report["method_names"].pop()
        ops.atomic_json(root / "report.json", report)
        complete["files"]["report.json"] = ops.file_sha256(root / "report.json")
        ops.atomic_json(root / "complete.json", complete)
        with self.assertRaisesRegex(ops.PlanError, "25-method"):
            ops.validate_evaluation_outputs(self.plan)
        complete["files"].pop("scores.npz")
        ops.atomic_json(root / "complete.json", complete)
        with self.assertRaisesRegex(ops.PlanError, "entire declared"):
            ops.validate_evaluation_outputs(self.plan)

    def test_laptop_handoff_needs_full_DAG_live_guardian_and_afterany_guard(self):
        jobs = self.submit()["job_ids"]
        self.scheduler.snapshot = {job: {"state": "PENDING", "exit_code": "0:0"} for job in jobs.values()}
        self.scheduler.snapshot[jobs["guardian"]]["state"] = "RUNNING"
        heartbeat = ops._control(self.plan) / "manifests/guardian.json"
        ops.atomic_json(heartbeat, {"status": "watching", "workflow_sha256": ops.digest(self.plan),
                                   "authorization_sha256": ops.digest(self.authorization),
                                   "job_id": jobs["guardian"], "checked_utc": ops.utc_now().isoformat()})
        self.assertTrue(ops.handoff_state(self.plan, self.authorization, self.scheduler)["laptop_close_ready"])
        self.scheduler.snapshot[jobs["failure_guard"]]["state"] = "FAILED"
        self.assertFalse(ops.handoff_state(self.plan, self.authorization, self.scheduler)["laptop_close_ready"])

    def test_preflight_handoff_is_never_full_launch_permission(self):
        self.plan = self.preflight()
        self.authorization = self.approve(self.plan)
        jobs = self.submit()["job_ids"]
        self.scheduler.snapshot = {job: {"state": "PENDING", "exit_code": "0:0"} for job in jobs.values()}
        self.scheduler.snapshot[jobs["guardian"]]["state"] = "RUNNING"
        ops.atomic_json(ops._control(self.plan) / "manifests/guardian.json", {
            "status": "watching", "workflow_sha256": ops.digest(self.plan),
            "authorization_sha256": ops.digest(self.authorization), "job_id": jobs["guardian"],
            "checked_utc": ops.utc_now().isoformat(),
        })
        result = ops.handoff_state(self.plan, self.authorization, self.scheduler)
        self.assertTrue(result["server_durable"])
        self.assertFalse(result["laptop_close_ready"])
        self.assertFalse((Path(self.config["root"]) / "execution.json").exists())


if __name__ == "__main__":
    unittest.main()
