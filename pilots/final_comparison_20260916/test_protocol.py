"""Synthetic contract/gate tests; the runner must be an allocated Slurm job."""
import copy
import datetime as dt
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import freeze, protocol


def contract(root):
    return {
        "schema_version": 1, "scope_id": protocol.SCOPE, "run_id": "synthetic-final",
        "root": str(root), "hidden_root": str(root / "hidden"),
        "seeds": list(protocol.SEEDS), "arms": list(protocol.ARMS), "families": list(protocol.FAMILIES),
        "training": copy.deepcopy(protocol.TRAINING), "model_specs": copy.deepcopy(protocol.MODEL_SPECS),
        "neural_matrix": protocol.neural_matrix(), "missing_matrix": protocol.missing_matrix(),
        "head_recipe": copy.deepcopy(protocol.HEAD_RECIPE),
        "linear_recipe": copy.deepcopy(protocol.LINEAR_RECIPE),
        "analysis": copy.deepcopy(protocol.ANALYSIS), "diagnostics": copy.deepcopy(protocol.DIAGNOSTICS),
        "method_names": list(protocol.METHODS), "original_test_access": False,
        "neural_fits": 39, "imported_neural_fits": 12, "new_neural_fits": 27,
        "linear_fits": 1, "meta_heads": 18, "reported_methods": 25,
        "reuse": {protocol.model_key("G", arm, seed): {"path": f"/synthetic/G/{arm}/seed{seed}"}
                  for seed in protocol.IMPORTED_SEEDS for arm in protocol.ARMS},
        "reuse_groups": {str(seed): {"status": "complete_late_diagnostic" if seed == 7 else "complete"}
                         for seed in protocol.IMPORTED_SEEDS},
    }


class FinalContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        protocol.require_slurm()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="final-contract-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.campaign = contract(self.root)

    def test_exact_matrix_and_reuse(self):
        protocol._validate_contract(self.campaign)
        all_rows, missing = protocol.neural_matrix(), protocol.missing_matrix()
        self.assertEqual(protocol.SEEDS, (7, 17, 27))
        self.assertEqual(protocol.IMPORTED_SEEDS, protocol.SEEDS)
        self.assertEqual(len(all_rows), 39)
        self.assertEqual(len(missing), 27)
        self.assertEqual(sum(row["family"] == "G" for row in missing), 0)
        self.assertEqual(sum(row["family"] == "H" for row in missing), 12)
        self.assertEqual(sum(row["family"] == "S" for row in missing), 12)
        self.assertEqual(sum(row["family"] == "O" for row in missing), 3)
        self.assertEqual(protocol.COUNTS, {
            "neural_fits": 39, "imported_neural_fits": 12, "new_neural_fits": 27, "linear_fits": 1,
            "meta_heads": 18, "imported_meta_heads": 6, "new_meta_heads": 12, "reported_methods": 25})

    def test_six_primary_contrasts_and_secondary_singleton(self):
        self.assertEqual(len(protocol.METHODS), 25)
        self.assertEqual(len(set(protocol.METHODS)), 25)
        self.assertEqual(len(protocol.PRIMARY_CONTRASTS), 6)
        self.assertFalse(any("L" in contrast for contrast in protocol.PRIMARY_CONTRASTS))
        self.assertEqual(protocol.LINEAR_RECIPE["fits"], 1)
        self.assertFalse(protocol.LINEAR_RECIPE["primary"])
        self.assertEqual(protocol.ANALYSIS["family_size"], 6)

    def test_changed_scope_seed_recipe_or_matrix_is_rejected(self):
        for key, value in (
            ("seeds", [1, 2, 7, 17, 27]), ("training", {**protocol.TRAINING, "epochs": 21}),
            ("original_test_access", True), ("linear_fits", 3), ("meta_heads", 30),
            ("missing_matrix", protocol.missing_matrix()[:-1]),
            ("diagnostics", {**protocol.DIAGNOSTICS, "action": "extend training"}),
        ):
            changed = copy.deepcopy(self.campaign)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                protocol._validate_contract(changed)

    def test_import_cannot_be_dropped(self):
        self.campaign["reuse"].pop("G/block2/seed7")
        with self.assertRaisesRegex(RuntimeError, "historical G fits"):
            protocol._validate_contract(self.campaign)

    def test_invalid_identity_and_test_role(self):
        for args in (("G", "block2", True), ("H", "block3", 7),
                     ("O", "block11", 7), ("L", "logits", 7), ("G", "block2", 9)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                protocol.model_key(*args)
        with self.assertRaisesRegex(ValueError, "test is closed"):
            protocol.role_spec(self.root, "test")

    def test_import_resolution_preserves_original_path(self):
        with patch.object(protocol, "read_campaign", return_value=self.campaign):
            self.assertEqual(protocol.resolve_run(self.root, "G", "block2", 7),
                             Path("/synthetic/G/block2/seed7"))
            self.assertEqual(protocol.resolve_run(self.root, "H", "block2", 7),
                             self.root / "runs/H/block2/seed7")

    def test_frozen_json_does_not_replace_history(self):
        path = self.root / "frozen.json"
        protocol.write_frozen(path, {"status": "incomplete"})
        before = path.read_bytes()
        protocol.write_frozen(path, {"status": "incomplete"})
        self.assertEqual(path.read_bytes(), before)
        with self.assertRaisesRegex(RuntimeError, "replace"):
            protocol.write_frozen(path, {"status": "complete"})
        self.assertEqual(path.read_bytes(), before)

    def test_read_requires_real_allocation(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, "Slurm"):
            protocol.read_campaign(self.root)

    def execution(self):
        now = dt.datetime.now(dt.timezone.utc)
        return {"approved": True, "scope_id": protocol.SCOPE, "run_id": self.campaign["run_id"],
                "campaign_sha256": "a" * 64, "original_test_access": False,
                "authorized_at": (now - dt.timedelta(minutes=1)).isoformat(),
                "deadlines": {name: (now + dt.timedelta(hours=i)).isoformat()
                              for i, name in enumerate(("base", "predictions", "evaluation"), 1)}}

    def test_scientific_execution_missing_or_unapproved_is_blocked(self):
        with patch.object(protocol, "read_campaign", return_value=self.campaign), \
             patch.object(protocol, "campaign_sha", return_value="a" * 64):
            with self.assertRaisesRegex(RuntimeError, "separately approved"):
                protocol.read_execution(self.root)
            value = self.execution()
            value["approved"] = False
            protocol.atomic_json(self.root / "execution.json", value)
            with self.assertRaisesRegex(RuntimeError, "approval"):
                protocol.read_execution(self.root)

    def test_cutoff_is_strict_and_timezone_qualified(self):
        value = self.execution()
        with patch.object(protocol, "read_execution", return_value=value):
            deadline = protocol._timestamp(value["deadlines"]["base"])
            with patch.object(protocol.time, "time", return_value=deadline - 1):
                protocol.check_cutoff(self.root, "base")
            with patch.object(protocol.time, "time", return_value=deadline):
                with self.assertRaises(TimeoutError):
                    protocol.check_cutoff(self.root, "base")
        with self.assertRaisesRegex(ValueError, "timezone"):
            protocol._timestamp("2026-09-16T14:00:00")

    def test_zero_budget_and_extra_primary_are_not_accepted_as_defaults(self):
        changed = copy.deepcopy(self.campaign)
        changed["analysis"]["primary_contrasts"].append(["G_stack", "L"])
        with self.assertRaises(RuntimeError):
            protocol._validate_contract(changed)
        changed = copy.deepcopy(self.campaign)
        changed["training"]["epochs"] = 0
        with self.assertRaises(RuntimeError):
            protocol._validate_contract(changed)


class GlobalGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        protocol.require_slurm()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="final-gate-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.now = dt.datetime.now(dt.timezone.utc)
        self.execution = {
            "authorized_at": (self.now - dt.timedelta(minutes=1)).isoformat(),
            "deadlines": {name: (self.now + dt.timedelta(hours=i)).isoformat()
                          for i, name in enumerate(("base", "predictions", "evaluation"), 1)},
        }
        self.header = {"schema_version": 1, "scope_id": protocol.SCOPE, "complete": True,
                       "campaign_sha256": "a" * 64, "source_identity": {"source.py": "b" * 64}}
        for target, value in (("_header", self.header), ("read_execution", self.execution)):
            patcher = patch.object(freeze, target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_missing_evaluation_gate_is_not_a_default_success(self):
        with self.assertRaises(FileNotFoundError):
            freeze.validate_evaluation_gate(self.root)

    def test_incomplete_and_foreign_headers_are_rejected(self):
        value = {**self.header, "frozen_utc": self.now.isoformat()}
        freeze._validate_header(self.root, value, "base")
        for key, item in (("complete", False), ("campaign_sha256", "c" * 64)):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                freeze._validate_header(self.root, {**value, key: item}, "base")

    def test_late_freeze_cannot_be_relabelled(self):
        value = {**self.header, "frozen_utc": self.execution["deadlines"]["predictions"]}
        with self.assertRaisesRegex(RuntimeError, "cutoff"):
            freeze._validate_header(self.root, value, "predictions")

    def test_global_evaluation_requires_all_fits_heads_and_linear(self):
        value = {**self.header, "frozen_utc": self.now.isoformat(),
                 "neural_fits": 39, "meta_heads": 18, "linear_fits": 1,
                 "original_test_access": False, "files": {"diagnostics/complete.json": "d" * 64}}
        protocol.atomic_json(self.root / "evaluation_gate.json", value)
        with patch.object(freeze, "_evaluation_bindings", return_value=value["files"]):
            self.assertEqual(freeze.validate_evaluation_gate(self.root), value)
            for key, number in (("neural_fits", 65), ("meta_heads", 30), ("linear_fits", 3)):
                protocol.atomic_json(self.root / "evaluation_gate.json", {**value, key: number})
                with self.subTest(key=key), self.assertRaises(RuntimeError):
                    freeze.validate_evaluation_gate(self.root)

    def test_changed_diagnostic_binding_is_rejected(self):
        value = {**self.header, "frozen_utc": self.now.isoformat(),
                 "neural_fits": 39, "meta_heads": 18, "linear_fits": 1,
                 "original_test_access": False, "files": {"diagnostics/complete.json": "d" * 64}}
        protocol.atomic_json(self.root / "evaluation_gate.json", value)
        with patch.object(freeze, "_evaluation_bindings", return_value={"diagnostics/complete.json": "e" * 64}):
            with self.assertRaisesRegex(RuntimeError, "changed"):
                freeze.validate_evaluation_gate(self.root)

    def test_gate_memo_runs_once_and_revalidates_on_gate_file_change(self):
        self.addCleanup(freeze._EVALUATION_GATE_CACHE.clear)
        value = {**self.header, "frozen_utc": self.now.isoformat(),
                 "neural_fits": 39, "meta_heads": 18, "linear_fits": 1,
                 "original_test_access": False, "files": {"diagnostics/complete.json": "d" * 64}}
        protocol.atomic_json(self.root / "evaluation_gate.json", value)
        with patch.object(freeze, "_evaluation_bindings", return_value=value["files"]) as bindings:
            for _ in range(3):
                self.assertEqual(freeze.validate_evaluation_gate(self.root), value)
            self.assertEqual(bindings.call_count, 1)
            protocol.atomic_json(self.root / "evaluation_gate.json", {**value, "files": {}})
            with self.assertRaises(RuntimeError):
                freeze.validate_evaluation_gate(self.root)
            self.assertEqual(bindings.call_count, 2)
            with self.assertRaises(RuntimeError):  # failures are never cached
                freeze.validate_evaluation_gate(self.root)
            self.assertEqual(bindings.call_count, 3)


if __name__ == "__main__":
    protocol.require_slurm()
    unittest.main()
