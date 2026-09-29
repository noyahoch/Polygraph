"""Focused protocol/provenance regression tests; run only on Slurm."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from . import data, rewire
from .protocol import atomic_json, file_sha256, require_slurm


def diagnostic(split, fraction, edges=30, accepted=0):
    return {"record_id": 1, "split": split, "edges": edges, "changed_fraction": fraction,
            "accepted": accepted, "attempts": 600, "swap_target_reached": accepted >= 2 * edges,
            "cls_in_neighbors_replaced": 0, "cls_out_neighbors_replaced": 0, "cls_neighbors_changed": False}


class RewireGateTests(unittest.TestCase):
    def test_mean_gate_retains_weak_graph_and_is_not_swap_target_gate(self):
        rows = [diagnostic("train", .65), diagnostic("val", 1.0)]
        summary = rewire.development_summary(rows, .8)
        self.assertTrue(summary["passed"])
        self.assertAlmostEqual(summary["mean_changed_fraction"], .825)
        self.assertEqual(summary["by_split"]["combined"]["below_swap_target_graphs"], 2)
        self.assertEqual(summary["by_split"]["combined"]["weak_fraction_eligible_graphs"], .5)
        self.assertEqual(len(rows), 2)

    def test_small_and_empty_graphs_retained_outside_gate(self):
        summary = rewire.development_summary([diagnostic("train", .9), diagnostic("val", 0, edges=0),
                                               diagnostic("train", 0, edges=19)], .8)
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["development_graphs"], 3)
        self.assertEqual(summary["excluded_from_gate_but_retained"], 2)

    def test_test_diagnostics_cannot_influence_gate(self):
        # Deliberately omit diagnostic values for held-out rows: they must not be read.
        rows = [diagnostic("train", .79), {"record_id": 30000, "split": "test", "integrity_passed": True}]
        first = rewire.development_summary(rows, .8)
        rows[1].update(changed_fraction=1e12, accepted=10**9, edges=10**9)
        second = rewire.development_summary(rows, .8)
        self.assertEqual(first, second)
        self.assertFalse(second["passed"])

    def test_no_eligible_development_graph_is_not_diagnostic(self):
        summary = rewire.development_summary([diagnostic("train", 1, edges=19)], .8)
        self.assertFalse(summary["passed"])
        self.assertIsNone(summary["mean_changed_fraction"])

    def test_degree_and_input_integrity_with_compiled_swaps(self):
        source = np.repeat(np.arange(12), 3)
        target = np.concatenate([(np.arange(i + 1, i + 4) % 12) for i in range(12)])
        original = np.stack([source, target])
        before = original.copy()
        changed, accepted, attempts = rewire.directed_swaps(source, target, 12, 123)
        rewire.verify_rewire(original, changed, 12)
        self.assertTrue(np.array_equal(original, before))
        self.assertTrue(np.array_equal(np.bincount(source, minlength=12), np.bincount(original[0], minlength=12)))
        self.assertLessEqual(accepted, 2 * len(source))
        self.assertLessEqual(attempts, 20 * len(source))
        with self.assertRaisesRegex(RuntimeError, "Invalid graph endpoint"):
            rewire.verify_rewire(original, np.full_like(target, 12), 12)
        empty = np.empty((2, 0), dtype=np.int64)
        rewire.verify_rewire(empty, np.empty(0, dtype=np.int64), 12)


class CacheIdentityTests(unittest.TestCase):
    def test_hash_reused_until_file_identity_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.pt"
            path.write_bytes(b"original")
            expected = file_sha256(path)
            with patch.object(data, "file_sha256", wraps=file_sha256) as hasher:
                data.verify_cached_file(path, expected)
                data.verify_cached_file(path, expected)
                self.assertEqual(hasher.call_count, 1)
                replacement = path.with_suffix(".tmp")
                replacement.write_bytes(b"replaced")
                os.replace(replacement, path)
                with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                    data.verify_cached_file(path, expected)
                self.assertEqual(hasher.call_count, 2)

    def test_changed_source_provenance_fails_before_model_loading(self):
        from . import extract
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            atomic_json(cache / "data_provenance.json", {"gaussian_noise.npy": {"sha256": "old"}})
            images = SimpleNamespace(provenance=lambda: {"gaussian_noise.npy": {"sha256": "changed"}})
            args = SimpleNamespace(cache=cache, protocol=None, max_records=27, shard_size=9,
                                   batch_size=9, resume=True, data_root=cache / "source", device="cuda")
            with patch.object(extract, "OfficialImages", return_value=images), patch.object(extract, "Capture") as model:
                with self.assertRaisesRegex(RuntimeError, "different frozen input"):
                    extract.capture(args)
                model.assert_not_called()

    def test_changed_processor_config_fails_before_shard_capture(self):
        from . import extract
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            atomic_json(cache / "processor.json", {"configuration": {"size": "old"}})
            images = SimpleNamespace(provenance=lambda: {"source.npy": {"sha256": "stable"}})
            capture = SimpleNamespace(processor=SimpleNamespace(to_dict=lambda: {"size": "changed"}))
            args = SimpleNamespace(cache=cache, protocol=None, max_records=27, shard_size=9,
                                   batch_size=9, resume=True, data_root=cache / "source", device="cuda")
            with patch.object(extract, "OfficialImages", return_value=images), patch.object(extract, "Capture", return_value=capture):
                with self.assertRaisesRegex(RuntimeError, "different frozen input"):
                    extract.capture(args)


if __name__ == "__main__":
    require_slurm()
    unittest.main()
