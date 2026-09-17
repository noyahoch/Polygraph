"""Stdlib-only tests for the resume-only DAG. Scheduler and scientific commands are mocked."""
from __future__ import annotations

import copy
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from . import resume_slurm as rs
from . import slurm as ops
from . import test_slurm as base_tests

GuardianPaused = base_tests.GuardianPaused
MockScheduler = base_tests.MockScheduler


class ResumeTests(unittest.TestCase):
    approve = base_tests.FinalSlurmTests.approve
    env = base_tests.FinalSlurmTests.env
    successful_states = base_tests.FinalSlurmTests.successful_states
    populate_task_receipts = base_tests.FinalSlurmTests.populate_task_receipts

    def setUp(self):
        base_tests.FinalSlurmTests.setUp(self)
        # Parent benchmark: everything up to the evaluation gate completed, dev_gpu failed.
        self.parent_scheduler = self.scheduler
        jobs = ops.submit_plan(self.plan, self.authorization, self.parent_scheduler, now=self.now)["job_ids"]
        self.parent_jobs = jobs
        self.populate_task_receipts(jobs)
        for index in range(len(self.plan["stages"]["dev_gpu"]["tasks"])):
            path = ops._receipt_path(self.plan, "dev_gpu", jobs["dev_gpu"], index)
            value = json.loads(path.read_text())
            value.update(status="failed", error_type="TimeoutError")
            ops.atomic_json(path, value)
        ops._receipt_path(self.plan, "analysis", jobs["analysis"], 0).unlink()
        statuses = {name: "COMPLETED" for name in ops.ORDER if name not in ops.GUARDS}
        statuses.update(dev_gpu="FAILED", analysis="CANCELLED")
        ops._freeze_phases(self.plan, self.authorization, jobs, statuses)
        self.parent_control = ops._control(self.plan)
        ops.atomic_json(self.parent_control / "manifests/terminal.json", {
            "complete": False, "status": "incomplete", "reason": "required_stage_failed",
            "run_id": self.plan["run_id"], "workflow_sha256": ops.digest(self.plan),
            "authorization_sha256": ops.digest(self.authorization), "job_ids": jobs})
        root = Path(self.config["root"])
        ops.atomic_json(root / "evaluation_gate.json", {"synthetic": "gate"})
        # Resume release: parent bytes plus the two resume files only.
        control = self.directory / "resume_control"
        release = control / "releases/synthetic-resume"
        parent_release = Path(self.config["code_root"])
        inventory = dict(self.config["source_sha256"])
        for name in inventory:
            (release / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(parent_release / name, release / name)
        for name in sorted(rs.RESUME_FILES):
            (release / name).parent.mkdir(parents=True, exist_ok=True)
            (release / name).write_text("# synthetic resume source; never executed\n")
            inventory[name] = ops.file_sha256(release / name)
        ops.atomic_json(release / "source_manifest.json", {"release_id": release.name, "source_sha256": inventory})
        self.inventory = inventory
        claims = self.plan["resource_claims"]
        self.rconfig = {
            **{name: copy.deepcopy(self.config[name]) for name in rs.SHARED},
            "mode": "resume", "approval_id": "synthetic-resume", "control_root": str(control),
            "code_root": str(release), "source_sha256": inventory,
            "caps": {"guardian": 130, "failure_guard": 5, "cpu_validation": 5, "dev_gpu": 60, "analysis": 1},
            "deadlines": {name: self.config["deadlines"][name] for name in ("predictions", "evaluation")},
            "max_concurrent_gpus": 6, "external_reserved_gpus": 0,
            "gpu_budget_minutes": 10**6, "cpu_budget_minutes": 10**6,
            "external_gpu_minutes": claims["workflow_gpu_minutes"],
            "external_cpu_minutes": claims["workflow_cpu_minutes"],
            "test_selectors": [ops.MODULE + "test_slurm", ops.MODULE + "test_resume_slurm"],
            "resume": rs.resume_binding(self.parent_control, "synthetic dev_gpu timeout at its cap"),
        }
        self.rplan = rs.build_plan(self.rconfig, now=self.now)
        self.rauth = self.approve(self.rplan)
        self.rauth.update(full_ready_at=(self.now - dt.timedelta(minutes=2)).isoformat(),
                          measured_resource_approval=True)
        self.scheduler = MockScheduler()

    def evidence(self, plan):
        result = base_tests.FinalSlurmTests.evidence(self, plan)
        result["files"] = {name: "a" * 64 for name in rs.required_evidence_paths(plan)}
        return result

    def test_plan_reruns_only_parent_dev_tasks_and_analysis(self):
        self.assertEqual(self.rplan["order"], list(rs.RESUME_ORDER))
        self.assertEqual(self.rplan["stages"]["dev_gpu"]["tasks"], self.plan["stages"]["dev_gpu"]["tasks"])
        self.assertEqual(self.rplan["stages"]["analysis"]["tasks"], self.plan["stages"]["analysis"]["tasks"])
        self.assertEqual([name for name, row in self.rplan["stages"].items() if row["gpu"]], ["dev_gpu"])
        self.assertEqual(len(self.rplan["stages"]["dev_gpu"]["tasks"]), 6)
        claims = self.rplan["resource_claims"]
        self.assertEqual(claims["workflow_gpu_minutes"], 360)
        self.assertEqual(claims["reserved_gpu_minutes"],
                         360 + self.config["preflight"]["reserved_gpu_minutes"]
                         + self.plan["resource_claims"]["workflow_gpu_minutes"])
        command = rs.sbatch_command(self.rplan, "dev_gpu", {"cpu_validation": "5"}, "f" * 64)
        self.assertIn("pilots.final_comparison_20260916.resume_slurm", command[-1])
        self.assertIn("--array=0-5%6", command)
        # Test artifact paths may themselves contain "resume" (e.g. a resume control job_work dir),
        # so check the runner module, not the whole command string.
        parent_runner = ops.sbatch_command(self.plan, "guardian", {}, "f" * 64)[-1]
        self.assertIn("-m pilots.final_comparison_20260916.slurm run-stage", parent_runner)
        self.assertNotIn("resume_slurm", parent_runner)
        self.assertIn("-m pilots.final_comparison_20260916.resume_slurm run-stage", command[-1])

    def test_budget_counts_the_parent_reservation(self):
        config = dict(self.rconfig, gpu_budget_minutes=self.rplan["resource_claims"]["reserved_gpu_minutes"] - 1)
        with self.assertRaisesRegex(ops.PlanError, "NEW budgets"):
            rs.build_plan(config, now=self.now)
        wrong = dict(self.rconfig, external_gpu_minutes=0)
        with self.assertRaisesRegex(ops.PlanError, "full reservation"):
            rs.verify_launch_inputs(rs.build_plan(wrong, now=self.now))

    def test_launch_refuses_changed_parent_bindings(self):
        rs.verify_launch_inputs(self.rplan)
        root = Path(self.config["root"])
        cases = {
            "execution record": lambda: ops.atomic_json(root / "execution.json", {"tampered": True}),
            "evaluation gate": lambda: ops.atomic_json(root / "evaluation_gate.json", {"other": 1}),
            "receipts": lambda: ops._receipt_path(self.plan, "heads", self.parent_jobs["heads"], 0).write_text("{}"),
            "phase freezes": lambda: (self.parent_control / "manifests/phases/gate.json").unlink(),
        }
        for label, mutate in cases.items():
            with self.subTest(label=label):
                snapshot = {path: path.read_bytes() for path in self.directory.rglob("*.json")}
                mutate()
                with self.assertRaises((ops.PlanError, OSError, json.JSONDecodeError, KeyError)):
                    rs.verify_launch_inputs(self.rplan)
                for path in list(self.directory.rglob("*.json")):
                    if path not in snapshot:
                        path.unlink()
                for path, data in snapshot.items():
                    path.write_bytes(data)
                rs.verify_launch_inputs(self.rplan)

    def test_existing_results_block_submission_but_not_completion_audit(self):
        # Resume2 guardian 904070 failed its completion audit: verify_launch_inputs (also run by
        # completion_evidence) refused the evaluation outputs the resumed analysis had just written.
        root = Path(self.config["root"])
        ops.atomic_json(root / "evaluation/report.json", {"synthetic": True})
        frozen = json.loads(ops.json_bytes(self.rplan))
        rs.verify_launch_inputs(frozen)
        with self.assertRaisesRegex(ops.PlanError, "cannot overwrite"):
            rs.submit_plan(frozen, self.rauth, self.scheduler, now=self.now)
        self.assertEqual(self.scheduler.submissions, [])
        self.assertFalse(rs._record_path(self.rplan).exists())
        self.assertFalse((rs._control(self.rplan) / "manifests").exists())

    def test_resume_cannot_change_cutoffs_science_source_or_tasks(self):
        later = dict(self.rconfig, deadlines={
            "predictions": self.rconfig["deadlines"]["predictions"],
            "evaluation": (self.now + dt.timedelta(minutes=115)).isoformat()})
        with self.assertRaisesRegex(ops.PlanError, "extend"):
            rs.verify_launch_inputs(rs.build_plan(later, now=self.now))
        name = "pilots/final_comparison_20260916/predict.py"
        changed = dict(self.rconfig, source_sha256={**self.inventory, name: "0" * 64})
        with self.assertRaisesRegex(ops.PlanError, "only add its own"):
            rs.verify_launch_inputs(rs.build_plan(changed, now=self.now))
        with self.assertRaisesRegex(ops.PlanError, "exact"):
            rs.build_plan(dict(self.rconfig, evaluation_files=["complete.json"]), now=self.now)
        with self.assertRaisesRegex(ops.PlanError, "own control"):
            rs.build_plan(dict(self.rconfig, control_root=str(self.parent_control),
                               code_root=str(self.parent_control / "releases/x")), now=self.now)

    def test_complete_parent_or_foreign_terminal_cannot_be_resumed(self):
        path = self.parent_control / "manifests/terminal.json"
        value = json.loads(path.read_text())
        ops.atomic_json(path, {**value, "complete": True})
        with self.assertRaisesRegex(ops.PlanError, "terminal"):
            rs.resume_binding(self.parent_control, "synthetic dev_gpu timeout at its cap")

    def test_authorization_requires_measured_approval(self):
        for changes in ({"measured_resource_approval": False}, {"full_ready_at": None}, {"approved": False},
                        {"expires_at": self.rconfig["deadlines"]["predictions"]}):
            with self.subTest(changes=changes), self.assertRaises((ops.PlanError, TypeError)):
                rs.validate_authorization(self.rplan, {**self.rauth, **changes}, now=self.now)
        rs.validate_authorization(self.rplan, self.rauth, now=self.now)

    def test_full_resume_lifecycle_through_the_frozen_sorted_workflow(self):
        root = Path(self.config["root"])
        execution = (root / "execution.json").read_bytes()
        launch = rs.submit_plan(self.rplan, self.rauth, self.scheduler, now=self.now)
        jobs = launch["job_ids"]
        self.assertEqual(len(self.scheduler.submissions), len(rs.RESUME_ORDER))
        self.assertEqual((root / "execution.json").read_bytes(), execution)
        record = json.loads(rs._record_path(self.rplan).read_text())
        self.assertEqual(record["parent"]["parent_workflow_sha256"], ops.digest(self.plan))
        self.assertEqual(record["rerun_stages"], ["dev_gpu", "analysis"])
        control = rs._control(self.rplan)
        workflow, approval = control / "workflow.json", control / "authorization.json"
        frozen = json.loads(workflow.read_text())
        self.assertEqual(frozen, self.rplan)
        # Sorted-key JSON reverses the canonical predictions -> evaluation phase order.
        self.assertEqual(list(frozen["phases"]), ["evaluation", "predictions"])
        self.assertEqual([name for name, _ in rs._phase_items(frozen)], ["predictions", "evaluation"])
        argv = ["run-stage", "--workflow", str(workflow), "--sha256", rs.digest(self.rplan),
                "--authorization", str(approval), "--authorization-sha256", rs.digest(self.rauth)]
        snapshot = self.scheduler.snapshot
        snapshot[jobs["guardian"]] = {"state": "RUNNING", "exit_code": "0:0"}
        snapshot[jobs["failure_guard"]] = {"state": "PENDING", "exit_code": "0:0"}
        executed = []
        evidence = self.evidence(frozen)
        evidence["resume_parent"] = {"receipts_sha256": self.rconfig["resume"]["parent_receipts_sha256"],
                                     "phases": self.rconfig["resume"]["parent_phases"]}
        original_verify = rs.verify_release

        def run_command(command, cwd, environment, stop_at):
            executed.append(command)
            return 0

        def pause(_seconds):
            raise GuardianPaused()

        def main(stage, job, **extra):
            with self.env(job, **extra):
                return rs.main([*argv, "--stage", stage], scheduler=self.scheduler)

        def guardian_cycle():
            with self.assertRaises(GuardianPaused):
                main("guardian", jobs["guardian"])
            self.assertEqual(json.loads((control / "manifests/guardian.json").read_text())["status"], "watching")

        def run_task(stage, index):
            row, job = frozen["stages"][stage], jobs[stage]
            extra = {"SLURM_GPUS": "1"} if row["gpu"] else {}
            if len(row["tasks"]) > 1:
                extra.update(SLURM_ARRAY_JOB_ID=job, SLURM_ARRAY_TASK_ID=str(index))
                return main(stage, str(int(job) + 1000 + index), **extra), f"{job}_{index}"
            return main(stage, job, **extra), job

        with (patch.object(rs, "verify_release", lambda plan, executing=False: original_verify(plan)),
              patch.object(rs, "_stage_environment", return_value=({}, {})),
              patch.object(rs, "_run_command", side_effect=run_command),
              patch.object(rs, "_allocated_audit", return_value=evidence),
              patch.object(rs.time, "sleep", side_effect=pause)):
            guardian_cycle()
            for stage in ("cpu_validation", "dev_gpu"):
                for index in range(len(frozen["stages"][stage]["tasks"])):
                    code, key = run_task(stage, index)
                    self.assertEqual(code, 0)
                    snapshot[key] = {"state": "COMPLETED", "exit_code": "0:0"}
            with self.assertRaises(GuardianPaused):
                run_task("analysis", 0)  # waits for the guardian's predictions freeze
            guardian_cycle()
            self.assertTrue((control / "manifests/phases/predictions.json").is_file())
            code, key = run_task("analysis", 0)
            self.assertEqual(code, 0)
            snapshot[key] = {"state": "COMPLETED", "exit_code": "0:0"}
            self.assertEqual(executed, [task["commands"][0] for name in ("cpu_validation", "dev_gpu", "analysis")
                                        for task in self.rplan["stages"][name]["tasks"]])
            self.assertTrue(all("predict" in " ".join(command) or "unittest" in command or "evaluate" in " ".join(command)
                                for command in executed))
            self.assertEqual(main("guardian", jobs["guardian"]), 0)
            snapshot[jobs["guardian"]] = {"state": "COMPLETED", "exit_code": "0:0"}
            snapshot[jobs["failure_guard"]] = {"state": "RUNNING", "exit_code": "0:0"}
            self.assertEqual(main("failure_guard", jobs["failure_guard"]), 0)
        terminal = json.loads((control / "manifests/terminal.json").read_text())
        self.assertTrue(terminal["complete"])
        self.assertEqual(terminal["publisher"], "failure_guard")
        self.assertEqual(self.scheduler.cancellations, [])
        parent_terminal = json.loads((self.parent_control / "manifests/terminal.json").read_text())
        self.assertFalse(parent_terminal["complete"])
        self.assertEqual(ops.file_sha256(self.parent_control / "manifests/terminal.json"),
                         self.rconfig["resume"]["parent_terminal_sha256"])

    def test_evidence_requires_bound_parent_receipts(self):
        evidence = self.evidence(self.rplan)
        self.assertFalse(rs.evidence_is_complete(self.rplan, evidence))
        evidence["resume_parent"] = {"receipts_sha256": "0" * 64, "phases": self.rconfig["resume"]["parent_phases"]}
        self.assertFalse(rs.evidence_is_complete(self.rplan, evidence))
        evidence["resume_parent"]["receipts_sha256"] = self.rconfig["resume"]["parent_receipts_sha256"]
        self.assertTrue(rs.evidence_is_complete(self.rplan, evidence))
        self.assertIn("resumes/synthetic-resume.json", rs.required_evidence_paths(self.rplan))


if __name__ == "__main__":
    unittest.main()
