"""Synthetic replication contract tests; execute only inside Slurm."""
from __future__ import annotations

import copy
import datetime as dt
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from . import combine, evaluate, predict, protocol, replication, train


class ReplicationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        protocol.require_slurm()
        cls.group = protocol.cache_cohort()
        cls.roles = protocol.make_roles(cls.group)
        cls.record_map = {row["record_id"]: row for row in cls.group["records"]}

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="polygraph-replication-test-")
        self.addCleanup(self.temporary.cleanup)
        self.parent = Path(self.temporary.name)
        self.source, self.root = self.parent / "historical", self.parent / "replication"
        self.cache = self.source / "feature_cache"
        self.cache.mkdir(parents=True)
        self.write(self.cache / "cohort.json", self.group)
        self.write(self.cache / "protocol.json", protocol.cache_protocol())
        self.write(self.cache / "manifest.json", {
            "complete": True, "diagnostic_only": False, "records": 28800,
            "cohort_sha256": protocol.digest(self.group),
            "protocol_sha256": protocol.digest(protocol.cache_protocol()),
        })
        self.write(self.source / "role_map.json", self.roles)
        self.write(self.source / "execution.json", protocol.make_execution(self.cache, self.source / "role_map.json"))
        for name in ("base_freeze.json", "heads_freeze.json", "predictions/meta.json", "predictions/dev_eval.json"):
            self.write(self.source / name, {"synthetic": True})
        self.source_inputs = {
            "roles_sha256": protocol.file_sha256(self.source / "role_map.json"),
            "execution_sha256": protocol.file_sha256(self.source / "execution.json"),
            "base_freeze_sha256": protocol.file_sha256(self.source / "base_freeze.json"),
            "heads_freeze_sha256": protocol.file_sha256(self.source / "heads_freeze.json"),
            "cache_manifest_sha256": protocol.file_sha256(self.cache / "manifest.json"),
            "dev_prediction_sidecar_sha256": protocol.file_sha256(self.source / "predictions/dev_eval.json"),
        }
        self.write(self.source / "evaluation/report.json", {
            "scope_id": protocol.SCOPE, "complete": True, "seed": 7, "base_epochs": 20,
            "records": 7200, "test_evaluated": False, "inputs": self.source_inputs,
        })
        for name in ("REPORT.md", "scores.npz", "bootstrap.json", "bootstrap_source_counts.npz"):
            path = self.source / "evaluation" / name
            path.write_bytes(b"synthetic provenance fixture; not scientific data\n")
        self.write(self.source / "evaluation/complete.json", {
            "complete": True, "inputs": self.source_inputs,
            "files": {name: protocol.file_sha256(self.source / "evaluation" / name) for name in
                      ("report.json", "REPORT.md", "scores.npz", "bootstrap.json", "bootstrap_source_counts.npz")},
        })
        self.write(self.source / "evaluation/late_diagnostic.json", {
            "scope_id": protocol.SCOPE, "complete": True, "status": "complete_late_diagnostic",
            "reused_on_time_meta_export": True, "original_protocol_status": "incomplete: late development predictions",
            "warning": "Do not report as the original fixed-protocol primary result.",
        })
        now = dt.datetime.now(dt.timezone.utc)
        self.deadlines = replication.normalized_cutoffs(
            *[(now + dt.timedelta(days=offset)).isoformat() for offset in (1, 2, 3)])
        self.manifest = replication.prepare(self.source, self.root, "synthetic-fixed-replication", self.deadlines)

    @staticmethod
    def write(path, value):
        combine.atomic_json(path, value)

    def seed_root(self, seed):
        return self.root / f"seed{seed}"

    def execution(self, seed):
        return protocol.read(self.seed_root(seed) / "execution.json")

    def populate_bases(self, seed):
        root = self.seed_root(seed)
        execution = self.execution(seed)
        implementation = protocol.implementation_identity()
        for arm in protocol.ARMS:
            directory = root / "runs" / arm / f"seed{seed}"
            config = {
                "scope_id": replication.SCOPE, "arm": arm, "seed": seed,
                "training": protocol.TRAINING, "fresh_initialization": True,
                "execution_sha256": protocol.file_sha256(root / "execution.json"),
                "roles_sha256": protocol.file_sha256(root / "role_map.json"),
                "cache_manifest_sha256": execution["cache_manifest_sha256"],
                "implementation": implementation, "implementation_sha256": protocol.digest(implementation),
            }
            self.write(directory / "config.json", config)
            self.write(directory / "history.json", [{"epoch": i, "checkpoint_auroc": 0.6} for i in range(1, 21)])
            self.write(directory / "checkpoint.json", {"best_epoch": 1})
            for name in ("best.safetensors", "latest.pt", "checkpoint.npz"):
                (directory / name).write_bytes(f"synthetic {arm} seed{seed} {name}".encode())
            self.write(directory / "complete.json", {
                "scope_id": replication.SCOPE, "complete": True, "arm": arm, "seed": seed,
                "completed_epochs": 20, "selection_role": "checkpoint", "best_epoch": 1,
                "completed_unix": self.manifest["created_unix"] + 0.000001,
                "artifacts": {name: protocol.file_sha256(directory / name) for name in train.ARTIFACTS},
            })
        return predict.create_base_freeze(root / "base_freeze.json", root / "runs",
                                          self.cache, root / "execution.json", root / "role_map.json")

    def populate_heads(self, seed):
        root = self.seed_root(seed)
        self.populate_bases(seed)
        execution, _, bindings = combine.context(root)
        for name, columns in (("stack", [0, 1, 2, 3]), ("last_only", [3])):
            self.write(root / "heads" / (name + ".json"), {
                "name": name, "scope_id": replication.SCOPE, "seed": seed,
                "columns": columns, "training_role": "meta", "rows": 3600, "source_photos": 400,
                "converged": True, "recipe": combine.RECIPE, "serialization_audit": {"passed": True},
                "scaler": {"mean": [0.0] * len(columns), "scale": [1.0] * len(columns)},
                "model": {"classes": [0, 1], "coef": [1.0] * len(columns), "intercept": 0.0},
            })
        self.write(root / "heads_freeze.json", {
            "complete": True, "scope_id": replication.SCOPE, "seed": seed,
            "arms": list(protocol.ARMS), "recipe": combine.RECIPE, **bindings,
            "implementation": combine.code_identity(execution),
            "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "files": {name: protocol.file_sha256(root / name) for name in ("heads/stack.json", "heads/last_only.json")},
        })

    def prediction_export(self, seed, role):
        root = self.seed_root(seed)
        execution, _, bindings = combine.context(root)
        rows = [self.record_map[r] for r in self.roles["roles"][role]["record_ids"]]
        count = len(rows)
        labels = np.arange(count, dtype=np.int64) % 2
        logits = np.column_stack((labels, -labels, np.arange(count) % 11, -labels)).astype(np.float32)
        data = {name: np.asarray([row[name] for row in rows], dtype=np.int64) for name in
                ("record_id", "image_id", "source_id", "severity", "split_id")}
        data.update(y=labels, label=np.zeros(count, dtype=np.int64), pred=labels, logits=logits)
        out = root / "predictions" / (role + ".npz")
        out.parent.mkdir(exist_ok=True)
        evaluate.atomic_npz(out, data)
        sidecar = {
            "schema_version": 1, "scope_id": replication.SCOPE, "role": role,
            "arms": list(protocol.ARMS), "seed": seed, "rows": count,
            "npz_sha256": protocol.file_sha256(out), "completed_unix": time.time(), **bindings,
            "replication_manifest_sha256": execution["replication_manifest_sha256"],
        }
        if role == "dev_eval":
            sidecar["heads_freeze_sha256"] = protocol.file_sha256(root / "heads_freeze.json")
            sidecar["joint_heads_freeze_sha256"] = protocol.file_sha256(self.root / "joint_heads_freeze.json")
        self.write(out.with_suffix(".json"), sidecar)
        return out

    def test_seed_independent_exact_roles_and_cache(self):
        original_bytes = (self.source / "role_map.json").read_bytes()
        for seed in replication.SEEDS:
            directory = self.seed_root(seed)
            execution, roles = protocol.validate_inputs(self.cache, directory / "execution.json", directory / "role_map.json")
            self.assertEqual(execution["seed"], seed)
            self.assertEqual(execution["training"], protocol.TRAINING)
            self.assertEqual(execution["head_random_state"], 7)
            self.assertEqual(execution["matrix"], [[arm, seed] for arm in protocol.ARMS])
            self.assertEqual((directory / "role_map.json").read_bytes(), original_bytes)
            self.assertEqual(roles, self.roles)
            self.assertEqual((directory / "feature_cache").resolve(), self.cache)

    def test_legacy_execution_and_deadlines_are_unchanged(self):
        before = protocol.make_execution(self.cache, self.source / "role_map.json")
        actual, _ = protocol.validate_inputs(self.cache, self.source / "execution.json", self.source / "role_map.json")
        self.assertEqual(actual, before)
        self.assertEqual(actual["seed"], 7)
        self.assertEqual(actual["deadlines"], protocol.DEADLINES)
        with patch.object(predict, "check_prediction_deadline") as check:
            predict.check_execution_deadline(actual)
            check.assert_called_once_with()
        with patch.object(predict, "check_prediction_deadline", side_effect=AssertionError("legacy bypass")):
            predict.check_execution_deadline(self.execution(17))

    def test_prepare_idempotent_and_never_relabels_source(self):
        original = {name: protocol.file_sha256(self.source / name) for name in replication.SOURCE_FILES}
        frozen_sha = protocol.file_sha256(self.root / "replication.json")
        repeated = replication.prepare(self.source, self.root, "synthetic-fixed-replication", self.deadlines)
        self.assertEqual(repeated, self.manifest)
        self.assertEqual(protocol.file_sha256(self.root / "replication.json"), frozen_sha)
        self.assertEqual(original, {name: protocol.file_sha256(self.source / name) for name in original})
        self.assertEqual(protocol.read(self.source / "evaluation/late_diagnostic.json")["status"], "complete_late_diagnostic")

    def test_reject_changed_cutoffs_or_run_id(self):
        changed = dict(self.deadlines)
        changed["evaluation_complete_before"] = changed["predictions_complete_before"]
        with self.assertRaisesRegex(RuntimeError, "identity or cutoffs"):
            replication.prepare(self.source, self.root, "synthetic-fixed-replication", changed)
        with self.assertRaisesRegex(RuntimeError, "identity or cutoffs"):
            replication.prepare(self.source, self.root, "different", self.deadlines)

    def test_reject_unapproved_seeds_budget_and_analysis(self):
        for key, replacement in (
                ("seeds", [7, 17, 27]), ("matrix", [["block11", 17]]),
                ("training", {**protocol.TRAINING, "epochs": 21}), ("head_random_state", 17),
                ("analysis", {**replication.ANALYSIS, "primary_seeds": [7, 17, 27]}),
                ("original_test_access", True)):
            changed = copy.deepcopy(self.manifest)
            changed[key] = replacement
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                replication.validate_contract(changed)
        for seed in (7, 1, 18, True):
            with self.subTest(seed=seed), self.assertRaises(RuntimeError):
                replication.seed_execution(self.root, seed, self.manifest)

    def test_reject_past_or_timezone_free_cutoffs(self):
        changed = copy.deepcopy(self.manifest)
        changed["created_unix"] = replication.timestamp(changed["deadlines"]["base_complete_before"])
        with self.assertRaisesRegex(RuntimeError, "future"):
            replication.validate_contract(changed)
        with self.assertRaisesRegex(ValueError, "timezone"):
            replication.timestamp("2026-01-01T12:00:00")
        with patch.object(replication.time, "time", return_value=replication.timestamp(self.deadlines["base_complete_before"])):
            with self.assertRaises(TimeoutError):
                replication.check_deadline(self.manifest, "base_complete_before")

    def test_reject_changed_pilot_provenance(self):
        self.write(self.source / "evaluation/late_diagnostic.json", {"status": "complete"})
        with self.assertRaisesRegex(RuntimeError, "pilot provenance"):
            replication.validate_manifest(self.root)

    def test_reject_cross_seed_execution_and_roles(self):
        self.write(self.seed_root(17) / "execution.json", self.execution(27))
        with self.assertRaisesRegex(RuntimeError, "namespace"):
            replication.validate_seed_execution(self.seed_root(17) / "execution.json", self.seed_root(17) / "role_map.json")

    def test_reject_changed_role_membership(self):
        roles = copy.deepcopy(self.roles)
        roles["roles"]["meta"]["record_ids"].reverse()
        self.write(self.seed_root(17) / "role_map.json", roles)
        with self.assertRaisesRegex(RuntimeError, "role map changed"):
            protocol.validate_inputs(self.cache, self.seed_root(17) / "execution.json", self.seed_root(17) / "role_map.json")

    def test_complete_fixed20_and_earliest_tie(self):
        self.populate_bases(17)
        directory = self.seed_root(17) / "runs/block2/seed17"
        complete, _ = train.verify_complete(directory, self.cache, self.seed_root(17) / "execution.json", self.seed_root(17) / "role_map.json")
        self.assertEqual(complete["best_epoch"], 1)
        complete["best_epoch"] = 2
        self.write(directory / "complete.json", complete)
        with self.assertRaisesRegex(RuntimeError, "earliest greatest"):
            train.verify_complete(directory, self.cache, self.seed_root(17) / "execution.json", self.seed_root(17) / "role_map.json")

    def test_incomplete_epoch_matrix_is_not_a_freeze(self):
        self.populate_bases(27)
        directory = self.seed_root(27) / "runs/block11/seed27"
        complete = protocol.read(directory / "complete.json")
        complete["completed_epochs"] = 19
        self.write(directory / "complete.json", complete)
        with self.assertRaisesRegex(RuntimeError, "complete fixed20"):
            predict.create_base_freeze(self.seed_root(27) / "base_freeze.json", self.seed_root(27) / "runs",
                                       self.cache, self.seed_root(27) / "execution.json", self.seed_root(27) / "role_map.json")

    def test_joint_freeze_requires_both_seeds_and_is_idempotent(self):
        self.populate_heads(17)
        with self.assertRaises(FileNotFoundError):
            replication.freeze_heads(self.root)
        self.populate_heads(27)
        frozen = replication.freeze_heads(self.root)
        self.assertEqual(frozen["seeds"], [17, 27])
        self.assertEqual(replication.freeze_heads(self.root), frozen)
        self.assertEqual(replication.validate_joint_heads_freeze(self.root), frozen)

    def test_joint_freeze_rejects_prior_development_output(self):
        for seed in replication.SEEDS:
            self.populate_heads(seed)
        path = self.seed_root(17) / "predictions/dev_eval.npz"
        path.parent.mkdir()
        path.write_bytes(b"forbidden early output")
        with self.assertRaisesRegex(RuntimeError, "before any new"):
            replication.freeze_heads(self.root)

    def test_prediction_gate_precedes_development_dataset(self):
        self.populate_heads(17)
        root = self.seed_root(17)
        args = SimpleNamespace(cache=self.cache, execution=root / "execution.json",
                               roles=root / "role_map.json", run_root=root / "runs",
                               base_freeze=root / "base_freeze.json", heads_freeze=root / "heads_freeze.json",
                               role="dev_eval", out=root / "predictions/dev_eval.npz")
        with patch.object(predict, "initialize"), patch.object(predict, "RoleDataset") as dataset:
            with self.assertRaises(FileNotFoundError):
                predict.predict(args)
            dataset.assert_not_called()

    def test_head_identity_cannot_cross_seeds(self):
        self.populate_heads(17)
        head_path = self.seed_root(17) / "heads/stack.json"
        head = protocol.read(head_path)
        head["seed"] = 27
        self.write(head_path, head)
        frozen_path = self.seed_root(17) / "heads_freeze.json"
        frozen = protocol.read(frozen_path)
        frozen["files"]["heads/stack.json"] = protocol.file_sha256(head_path)
        self.write(frozen_path, frozen)
        with self.assertRaisesRegex(RuntimeError, "different training seed"):
            combine.validate_heads(self.seed_root(17))

    def test_meta_only_head_fit_keeps_negative_last_coefficient(self):
        self.populate_bases(17)
        path = self.prediction_export(17, "meta")
        root = self.seed_root(17)
        frozen = combine.fit_heads(root, path, root / "heads")
        self.assertEqual(frozen["seed"], 17)
        heads, _, _ = combine.validate_heads(root)
        self.assertLess(heads["last_only"]["model"]["coef"][0], 0)
        self.assertTrue(all(head["serialization_audit"]["passed"] for head in heads.values()))
        self.assertEqual(combine.fit_heads(root, path, root / "heads"), frozen)

    def test_development_export_uses_joint_identity_and_exact_records(self):
        for seed in replication.SEEDS:
            self.populate_heads(seed)
        replication.freeze_heads(self.root)
        path = self.prediction_export(17, "dev_eval")
        data, _, _ = combine.load_predictions(self.seed_root(17), path, "dev_eval")
        self.assertEqual(data["logits"].shape, (7200, 4))
        sidecar = protocol.read(path.with_suffix(".json"))
        sidecar["joint_heads_freeze_sha256"] = "changed"
        self.write(path.with_suffix(".json"), sidecar)
        with self.assertRaisesRegex(RuntimeError, "prior joint seed freeze"):
            combine.load_predictions(self.seed_root(17), path, "dev_eval")

    def test_original_test_role_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "meta/dev_eval"):
            combine.load_predictions(self.seed_root(17), self.seed_root(17) / "predictions/test.npz", "test")

    def test_per_seed_evaluation_is_bound_and_idempotent(self):
        for seed in replication.SEEDS:
            self.populate_heads(seed)
        replication.freeze_heads(self.root)
        path = self.prediction_export(17, "dev_eval")
        root = self.seed_root(17)
        result = evaluate.evaluate(root, path, root / "heads", root / "evaluation")
        self.assertEqual(result["seed"], 17)
        self.assertEqual(result["scope_id"], replication.SCOPE)
        self.assertIn("seed17", result["primary"]["scope"])
        self.assertFalse(result["test_evaluated"])
        complete = protocol.read(root / "evaluation/complete.json")
        self.assertEqual(complete["seed"], 17)
        self.assertEqual(result, evaluate.evaluate(root, path, root / "heads", root / "evaluation"))
        del complete["files"]["bootstrap.json"]
        self.write(root / "evaluation/complete.json", complete)
        with self.assertRaisesRegex(RuntimeError, "inventory is incomplete"):
            evaluate.evaluate(root, path, root / "heads", root / "evaluation")

    def test_existing_synthetic_numerical_checks(self):
        from .smoke import cpu_checks
        result = cpu_checks()
        self.assertTrue(result["weighted_auc_ties_and_replication"])
        self.assertTrue(result["head_json_scores_and_probabilities"])
        self.assertTrue(result["source_photo_grouping"])

    def test_slurm_required_before_preparation(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Slurm"):
                replication.prepare(self.source, self.parent / "forbidden", "no-mac", self.deadlines)
        self.assertFalse((self.parent / "forbidden").exists())


if __name__ == "__main__":
    protocol.require_slurm()
    unittest.main()
