"""Fast results-only checks; no training, original-test access, or bootstrap jobs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from . import collect, data


def synthetic_scores():
    images = np.repeat(np.arange(800), 9)
    views = np.tile(np.asarray(data.CONDITIONS), (800, 1))
    labels = images % 2
    score = labels.astype(float) + (np.arange(7200) % 11) / 100
    result = {
        "image_id": images, "record_id": np.arange(7200),
        "source_id": views[:, 0], "severity": views[:, 1],
        "split_id": np.ones(7200, dtype=int), "y": labels,
        "label": np.zeros(7200, dtype=int), "pred": labels,
    }
    result.update({name: score.copy() for name in data.METHODS})
    return result


class NumericalResultsTests(unittest.TestCase):
    def test_average_ranks_handle_ties(self):
        np.testing.assert_allclose(data.rank_percentiles([2, 1, 2, 1]), [.75, .25, .75, .25])

    def test_rank_is_invariant_to_positive_affine_change(self):
        values = np.array([.5, -.2, 9, 9, 0])
        np.testing.assert_allclose(data.rank_percentiles(values), data.rank_percentiles(values * 10 + 3))

    def test_rank_rejects_nonfinite_or_empty(self):
        for values in ([], [1, np.nan], [[1, 2]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                data.rank_percentiles(values)

    def test_spearman_tracks_rank_not_scale(self):
        result = data.rank_correlation([np.array([1, 2, 2, 4]), np.array([10, 20, 20, 40]),
                                        np.array([-1, -2, -2, -4])])
        np.testing.assert_allclose(result, [[1, 1, -1], [1, 1, -1], [-1, -1, 1]])

    def test_constant_correlation_explicitly_undefined(self):
        result = data.rank_correlation([np.ones(4), np.arange(4)])
        self.assertIsNone(result[0][1])
        self.assertIsNone(result[0][0])
        self.assertAlmostEqual(result[1][1], 1)

    def test_roc_known_auc_and_tie_points(self):
        points = np.asarray(data.roc_points([0, 0, 1, 1], [0, 2, 1, 3]))
        self.assertAlmostEqual(np.trapezoid(points[:, 1], points[:, 0]), .75)
        self.assertEqual(points[0].tolist(), [0, 0])
        self.assertEqual(points[-1].tolist(), [1, 1])
        self.assertEqual(data.roc_points([0, 1, 0, 1], [4, 4, 4, 4]), [[0, 0], [1, 1]])

    def test_roc_display_is_bounded_and_monotone(self):
        points = np.asarray(data.roc_points(np.arange(1000) % 2, np.arange(1000), max_points=31))
        self.assertLessEqual(len(points), 31)
        self.assertTrue((np.diff(points, axis=0) >= 0).all())
        self.assertEqual(points[[0, -1]].tolist(), [[0, 0], [1, 1]])

    def test_roc_rejects_single_class_and_invalid_budget(self):
        with self.assertRaises(ValueError):
            data.roc_points([0, 0], [1, 2])
        with self.assertRaises(ValueError):
            data.roc_points([0, 1], [1, 2], max_points=1)

    def test_histograms_normalized_per_outcome(self):
        result = data.rank_histograms(synthetic_scores())
        self.assertEqual(len(result["edges"]), 21)
        for method in result["methods"].values():
            self.assertAlmostEqual(sum(method["correct"]), 1)
            self.assertAlmostEqual(sum(method["error"]), 1)

    def test_conditions_keep_all_nine_views(self):
        rows = data.condition_rows(synthetic_scores())
        self.assertEqual(len(rows), 9)
        self.assertEqual([row["records"] for row in rows], [800] * 9)
        self.assertTrue(all(row["delta"] == 0 and row["defined"] for row in rows))

    def test_negative_slice_is_retained_without_selection(self):
        scores = synthetic_scores()
        mask = (scores["source_id"] == 2) & (scores["severity"] == 3)
        scores["learned_stack"][mask] *= -1
        rows = data.condition_rows(scores)
        row = next(row for row in rows if row["source_id"] == 2 and row["severity"] == 3)
        self.assertEqual(row["delta"], -1)
        self.assertEqual(len(rows), 9)

    def test_single_outcome_slice_is_null_not_success(self):
        scores = synthetic_scores()
        scores["y"][scores["source_id"] == 0] = 0
        rows = data.condition_rows(scores)
        self.assertFalse(rows[0]["defined"])
        self.assertIsNone(rows[0]["delta"])
        self.assertIn("undefined", rows[0]["undefined_reason"])

    def test_source_errors_are_grouped_by_photo_not_family(self):
        result = data.source_error_counts(synthetic_scores())
        self.assertEqual(result["photos"], 800)
        self.assertEqual(result["counts"], [400, 0, 0, 0, 0, 0, 0, 0, 0, 400])

    def test_metric_parity_rejects_mismatch(self):
        data.assert_close(.75, .75, "equal")
        with self.assertRaisesRegex(RuntimeError, "metric mismatch"):
            data.assert_close(.75, .76, "changed")


class ResultArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_saved_scores_require_exact_schema(self):
        scores = synthetic_scores()
        path = self.root / "scores.npz"
        np.savez_compressed(path, **scores)
        loaded = data.load_scores(path)
        self.assertEqual(set(loaded), set(scores))
        scores["unknown"] = np.zeros(7200)
        np.savez_compressed(path, **scores)
        with self.assertRaisesRegex(RuntimeError, "schema"):
            data.load_scores(path)

    def test_reject_missing_source_view(self):
        scores = synthetic_scores()
        scores["image_id"][0] = 900
        path = self.root / "scores.npz"
        np.savez_compressed(path, **scores)
        with self.assertRaisesRegex(RuntimeError, "nine views"):
            data.load_scores(path)

    def test_reject_invalid_targets_and_nonfinite(self):
        for field, value in (("y", 2), ("learned_stack", np.nan)):
            scores = synthetic_scores()
            scores[field][0] = value
            path = self.root / "scores.npz"
            np.savez_compressed(path, **scores)
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                data.load_scores(path)

    def test_download_inventory_excludes_heavy_artifacts_and_old_test_predictions(self):
        groups = collect.inventory()
        names = [name for _, files in groups.values() for name in files]
        self.assertTrue(all(name.endswith((".json", ".csv", ".npz")) for name in names))
        self.assertFalse(any(part in name for name in names for part in ("latest.pt", ".safetensors", "feature_cache")))
        self.assertFalse(any("predictions" in name for name in groups["september10"][1]))
        self.assertEqual(sum(name.endswith("scores.npz") for name in names), 3)

    def test_hash_binding_rejects_modified_copy(self):
        path = self.root / "report.json"
        path.write_text('{"complete": true}')
        expected = collect.sha(path)
        collect.verify_hash(path, expected)
        path.write_text('{"complete": false}')
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            collect.verify_hash(path, expected)

    def test_per_file_limit_prevents_download(self):
        response = SimpleNamespace(stdout=str(collect.MAX_FILE_BYTES + 1) + "\n")
        with patch.object(collect.subprocess, "run", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "per-file ceiling"):
                collect.remote_sizes(Path("/not-executed-wrapper"), "/remote", ["report.json"])

    def test_total_limit_is_checked_before_any_archive_download(self):
        with patch.object(collect, "remote_sizes", side_effect=lambda wrapper, root, names:
                          [collect.MAX_FILE_BYTES] * len(names)), patch.object(collect, "fetch_group") as fetch:
            with self.assertRaisesRegex(RuntimeError, "before download"):
                collect.collect(Path("/not-executed-wrapper"), self.root / "raw")
            fetch.assert_not_called()

    def test_save_json_and_csv_are_portable(self):
        value = {"schema_version": 1, "csv_rows": [
            {"study": "current", "scope": "descriptive", "seed": 17, "method": "stack_minus_last",
             "metric": "delta", "value": -.01, "status": "posthoc_descriptive_no_ci"}]}
        data.save(copy.deepcopy(value), self.root)
        self.assertEqual(json.loads((self.root / "results.json").read_text()), value)
        self.assertIn("-0.01", (self.root / "results.csv").read_text())


if __name__ == "__main__":
    unittest.main()
