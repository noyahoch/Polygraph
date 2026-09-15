"""Synthetic aggregation tests; run only inside an allocated CPU Slurm job.

Fixtures use POLYGRAPH_TEST_ARTIFACT_ROOT (or the working directory), never a
system temp directory. No historical outcomes, feature cache, trained models or
cluster submission commands are accessed.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import time
import unittest
from unittest.mock import patch
import uuid

import numpy as np
from sklearn.metrics import roc_auc_score

from . import aggregate_replications as aggregate
from . import evaluate


def setUpModule():
    aggregate.require_slurm()


def synthetic_data(sources=4, views=9, late=True):
    n = sources * views
    row = np.arange(n, dtype=np.int64)
    y = (row % 3 == 0).astype(np.int64)
    label = row % 7
    metadata = {
        "record_id": row + 50000, "image_id": np.repeat(np.arange(sources) + 10000, views),
        "source_id": np.tile(np.arange(views) % 3, sources),
        "severity": np.tile(np.arange(views) // 3, sources),
        "split_id": np.ones(n, dtype=np.int64), "y": y,
        "label": label, "pred": label + y,
    }
    result = {}
    for seed in (17, 27, 7) if late else (17, 27):
        raw = np.column_stack([((row * factor + seed) % 11 - 5).astype(np.float64)
                               for factor in (1, 3, 5, 7)])
        result[seed] = {name: values.copy() for name, values in metadata.items()}
        result[seed].update(
            learned_stack=((row * 3 + seed) % 13).astype(np.float64) - 10 * y,
            learned_last_only=((row * 7 + seed) % 17).astype(np.float64) + 20 * y,
            fixed_probability_mean=aggregate.sigmoid(raw).mean(axis=1),
        )
        result[seed].update({"raw_" + arm: raw[:, index].copy()
                             for index, arm in enumerate(aggregate.ARMS)})
    return result


def roles_for(data):
    return {"original_test_access": False, "roles": {"dev_eval": {
        "record_ids": data[17]["record_id"].tolist(),
        "photo_ids": np.unique(data[17]["image_id"]).tolist(), "original_split": "val",
    }}}


class PairedStatisticsTests(unittest.TestCase):
    def test_weighted_auc_matches_sklearn_for_ties_and_zero_weights(self):
        labels = np.array([0, 1, 0, 1, 0, 1])
        scores = np.array([0., 0., 1., 2., 2., 2.])
        auc = aggregate.WeightedAUC(labels, scores)
        for weights in (np.ones(6), np.array([3, 0, 2, 1, 0, 4]), np.array([0, 2, 0, 0, 3, 1])):
            with self.subTest(weights=weights.tolist()):
                self.assertAlmostEqual(auc(weights), roc_auc_score(labels, scores, sample_weight=weights), places=14)
        self.assertIsNone(auc(np.array([0, 1, 0, 1, 0, 1])))
        self.assertIsNone(auc(np.zeros(6)))

    def test_shared_source_multiplicities_match_sklearn_every_method_and_seed(self):
        data = synthetic_data()
        order = np.r_[np.arange(0, 36, 2), np.arange(1, 36, 2)]
        data = {seed: {name: values[order] for name, values in arrays.items()}
                for seed, arrays in data.items()}
        aggregate.validate_alignment(data, roles_for(data), source_photographs=4)
        result = aggregate.paired_seed_statistics(data, draws=20)
        rng = np.random.default_rng(20260914)
        expected_counts = np.asarray([np.bincount(rng.integers(0, 4, size=4), minlength=4)
                                      for _ in range(20)])
        np.testing.assert_array_equal(result["source_counts"]["multiplicity"], expected_counts)
        groups, membership = np.unique(data[17]["image_id"], return_inverse=True)
        np.testing.assert_array_equal(result["source_counts"]["image_id"], groups)
        self.assertEqual(len(groups), 4)
        self.assertEqual(len(np.unique(data[17]["source_id"])), 3)
        self.assertTrue(np.all(expected_counts.sum(axis=1) == 4))
        for draw, multiplicity in enumerate(expected_counts):
            weights = multiplicity[membership]
            for seed_index, seed in enumerate((17, 27, 7)):
                for method_index, method in enumerate(aggregate.METHODS):
                    wanted = roc_auc_score(data[seed]["y"], data[seed][method], sample_weight=weights)
                    self.assertAlmostEqual(result["draws"]["auroc"][draw, seed_index, method_index],
                                           wanted, places=14)
            differences = result["draws"]["auroc"][draw, :, 0] - result["draws"]["auroc"][draw, :, 1]
            np.testing.assert_array_equal(result["draws"]["delta_auroc"][draw], differences)
            self.assertEqual(result["draws"]["primary_mean_delta_auroc"][draw], differences[:2].mean())
            self.assertEqual(result["draws"]["all3_mean_delta_auroc"][draw], differences.mean())
        np.testing.assert_allclose(result["primary"]["interval_95"],
                                   np.percentile(result["draws"]["primary_mean_delta_auroc"],
                                                 [2.5, 97.5], method="linear"))

    def test_mean_delta_and_sample_sd_are_not_pooled_or_prediction_averaged(self):
        data = synthetic_data(sources=4, views=1, late=False)
        y = np.array([0, 0, 1, 1], dtype=np.int64)
        for arrays in data.values():
            arrays.update(y=y.copy(), label=np.zeros(4, dtype=np.int64), pred=y.copy())
        data[17]["learned_stack"] = np.array([0., 1., 2., 3.])
        data[17]["learned_last_only"] = np.array([0., 1., 2., 3.])
        data[27]["learned_stack"] = np.array([106., 104., 102., 100.])
        data[27]["learned_last_only"] = np.array([100., 101., 102., 103.])
        result = aggregate.paired_seed_statistics(data, draws=32)
        stats = result["primary"]["statistics"]
        self.assertEqual(result["primary"]["estimate"], -0.5)
        self.assertEqual(stats["learned_stack"]["mean"], 0.5)
        self.assertEqual(stats["learned_last_only"]["mean"], 1.0)
        self.assertEqual(stats["learned_last_only"]["sample_sd"], 0.0)
        self.assertAlmostEqual(stats["delta_auroc"]["sample_sd"], np.std([0., -1.], ddof=1))
        self.assertAlmostEqual(stats["learned_stack"]["sample_sd"], np.std([1., 0.], ddof=1))
        self.assertNotEqual(stats["delta_auroc"]["sample_sd"], np.std([0., -1.], ddof=0))
        self.assertEqual(stats["delta_auroc"]["ddof"], 1)
        pooled = (roc_auc_score(np.tile(y, 2), np.r_[data[17]["learned_stack"], data[27]["learned_stack"]])
                  - roc_auc_score(np.tile(y, 2), np.r_[data[17]["learned_last_only"], data[27]["learned_last_only"]]))
        averaged = (roc_auc_score(y, (data[17]["learned_stack"] + data[27]["learned_stack"]) / 2)
                    - roc_auc_score(y, (data[17]["learned_last_only"] + data[27]["learned_last_only"]) / 2))
        self.assertNotEqual(result["primary"]["estimate"], pooled)
        self.assertNotEqual(result["primary"]["estimate"], averaged)
        data[7] = copy.deepcopy(data[17])
        data[7]["learned_last_only"] = data[7]["learned_stack"][::-1].copy()
        with_late = aggregate.paired_seed_statistics(data, draws=32)
        self.assertEqual(result["primary"], with_late["primary"])
        np.testing.assert_array_equal(result["draws"]["primary_mean_delta_auroc"],
                                      with_late["draws"]["primary_mean_delta_auroc"])
        self.assertEqual(with_late["all3_descriptive"]["estimate"], 0.0)
        self.assertEqual(with_late["all3_descriptive"]["seeds"], [7, 17, 27])
        self.assertEqual(with_late["per_seed"]["7"]["status"], "complete_late_diagnostic")
        self.assertEqual(result["all3_descriptive"]["status"], "unavailable")

    def test_one_class_draws_are_not_redrawn_or_dropped(self):
        data = synthetic_data(sources=2)
        y = np.repeat(np.array([0, 1], dtype=np.int64), 9)
        for arrays in data.values():
            arrays.update(y=y.copy(), label=np.zeros(18, dtype=np.int64), pred=y.copy())
        result = aggregate.paired_seed_statistics(data, draws=64)
        rng = np.random.default_rng(20260914)
        counts = np.asarray([np.bincount(rng.integers(0, 2, size=2), minlength=2) for _ in range(64)])
        np.testing.assert_array_equal(result["source_counts"]["multiplicity"], counts)
        undefined = np.flatnonzero((counts == 0).any(axis=1)).tolist()
        self.assertTrue(undefined)
        self.assertLess(len(undefined), 64)
        self.assertEqual(result["draws"]["auroc"].shape, (64, 3, 7))
        for row in result["per_seed"].values():
            self.assertEqual(row["primary"]["undefined_draw_indices"], undefined)
            self.assertIsNone(row["primary"]["interval_95"])
            self.assertTrue(all(metric["interval_95"] is None for metric in row["metrics"].values()))
        for name in ("primary", "all3_descriptive"):
            self.assertIsNone(result[name]["interval_95"])
            self.assertEqual(result[name]["undefined_draw_indices"]["delta_auroc"], undefined)
        self.assertTrue(np.isnan(result["draws"]["auroc"][undefined]).all())
        json.dumps({name: result[name] for name in ("primary", "all3_descriptive", "per_seed")},
                   allow_nan=False)
        for arrays in data.values():
            arrays["y"] = np.zeros(18, dtype=np.int64)
        with self.assertRaisesRegex(RuntimeError, "Point AUROC is undefined"):
            aggregate.paired_seed_statistics(data, draws=2)

    def test_linear_percentiles_and_no_conditional_interval(self):
        np.testing.assert_allclose(aggregate._interval(np.array([0., 1., 2., 4.])), [0.075, 3.85])
        self.assertIsNone(aggregate._interval(np.array([0., 1., np.nan])))


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_data()
        self.roles = roles_for(self.data)

    def validate(self, data=None, roles=None):
        aggregate.validate_alignment(self.data if data is None else data,
                                     self.roles if roles is None else roles, source_photographs=4)

    def test_valid_alignment_and_fixed_primary_seed_set(self):
        self.validate()
        for keys in ((17,), (27,), (7, 17), (7, 27), (17, 27, 37)):
            with self.subTest(seeds=keys):
                changed = {seed: copy.deepcopy(self.data.get(seed, self.data[17])) for seed in keys}
                with self.assertRaisesRegex(RuntimeError, "Exactly seeds17/27"):
                    self.validate(changed)
                with self.assertRaises(RuntimeError):
                    aggregate.paired_seed_statistics(changed, draws=2)

    def test_every_metadata_column_must_match_in_exact_order(self):
        for name in aggregate.METADATA:
            with self.subTest(column=name):
                changed = copy.deepcopy(self.data)
                changed[27][name][0] += 1
                with self.assertRaisesRegex(RuntimeError, "ordered metadata"):
                    self.validate(changed)
        changed = copy.deepcopy(self.data)
        changed[27] = {name: values[::-1].copy() for name, values in changed[27].items()}
        with self.assertRaisesRegex(RuntimeError, "ordered metadata"):
            self.validate(changed)

    def test_role_order_duplicates_and_missing_views_are_rejected(self):
        roles = copy.deepcopy(self.roles)
        roles["roles"]["dev_eval"]["record_ids"].reverse()
        with self.assertRaisesRegex(RuntimeError, "frozen record order"):
            self.validate(roles=roles)
        changed = copy.deepcopy(self.data)
        for arrays in changed.values():
            arrays["record_id"][1] = arrays["record_id"][0]
        with self.assertRaises(RuntimeError):
            self.validate(changed, roles_for(changed))
        changed = copy.deepcopy(self.data)
        for arrays in changed.values():
            arrays["image_id"][0] = arrays["image_id"][-1]
        with self.assertRaisesRegex(RuntimeError, "all nine views"):
            self.validate(changed)
        changed = {seed: {name: values[:-1] for name, values in arrays.items()}
                   for seed, arrays in self.data.items()}
        with self.assertRaisesRegex(RuntimeError, "record count"):
            self.validate(changed)

    def test_binary_targets_and_all_predeclared_finite_scores_are_required(self):
        changed = copy.deepcopy(self.data)
        for arrays in changed.values():
            arrays["y"][0] = 0
        with self.assertRaisesRegex(RuntimeError, "frozen-classifier error"):
            self.validate(changed)
        for mutation in ("missing", "extra", "nonfinite", "fixed_mean", "dtype"):
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(self.data)
                if mutation == "missing":
                    del changed[27]["raw_block11"]
                elif mutation == "extra":
                    changed[27]["unplanned_method"] = np.ones(36)
                elif mutation == "nonfinite":
                    changed[27]["learned_stack"][0] = np.nan
                elif mutation == "fixed_mean":
                    changed[27]["fixed_probability_mean"][0] += 0.1
                else:
                    changed[27]["record_id"] = changed[27]["record_id"].astype(np.float64)
                with self.assertRaises(RuntimeError):
                    self.validate(changed)


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = synthetic_data(sources=800)
        cls.full_statistics = aggregate.paired_seed_statistics(cls.data)

    def setUp(self):
        artifact_root = Path(os.environ.get("POLYGRAPH_TEST_ARTIFACT_ROOT", "."))
        self.directory = artifact_root / (".aggregate_replications_fixture_" + uuid.uuid4().hex)
        self.directory.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.directory)
        self.directory = self.directory.resolve()
        self.source, self.root = self.directory / "historical", self.directory / "replication"
        self.cache, self.out = self.source / "feature_cache", self.root / "evaluation"
        self.cache.mkdir(parents=True)
        self.root.mkdir()
        self.roles = roles_for(self.data)
        self.created = time.time() - 120
        self.write(self.cache / "manifest.json", {"complete": True, "synthetic": True})
        self.write(self.root / "role_map.json", self.roles)
        self.seed_inputs(7)
        self.evaluation(7)
        self.write(self.source / "evaluation/late_diagnostic.json", {
            "scope_id": aggregate.PILOT_SCOPE, "complete": True, "status": "complete_late_diagnostic",
            "reused_on_time_meta_export": True,
            "original_protocol_status": "incomplete: original fixed cutoff was missed",
            "warning": "Never promote the late diagnostic to the original protocol's primary result.",
        })
        self.manifest = {
            "scope_id": aggregate.SCOPE, "root": str(self.root), "source_root": str(self.source),
            "cache_path": str(self.cache), "created_unix": self.created, "seeds": [17, 27],
            "deadlines": {"evaluation_complete_before":
                          dt.datetime.fromtimestamp(time.time() + 3600, dt.timezone.utc).isoformat()},
            "roles_sha256": aggregate.sha256(self.root / "role_map.json"),
            "cache_manifest_sha256": aggregate.sha256(self.cache / "manifest.json"),
            "source_files": {name: aggregate.sha256(self.source / name) for name in aggregate.SOURCE_FILES},
        }
        self.write(self.root / "replication.json", self.manifest)
        self.joint = {"complete": True, "scope_id": aggregate.SCOPE,
                      "frozen_unix": self.created + 10, "seeds": [17, 27]}
        self.write(self.root / "joint_heads_freeze.json", self.joint)
        for seed in (17, 27):
            self.seed_inputs(seed)
            self.evaluation(seed)
        # Only the external eight-fit training/freeze validators are mocked.
        # All aggregate input/output checks, score checks and file hashes are real.
        for name, filename in (("validate_manifest", "replication.json"),
                               ("validate_joint_heads_freeze", "joint_heads_freeze.json")):
            mock = patch.object(aggregate, name,
                                side_effect=lambda root, name=filename: aggregate._read(Path(root) / name))
            mock.start()
            self.addCleanup(mock.stop)

    @staticmethod
    def write(path, value):
        aggregate.atomic_json(path, value)

    def seed_root(self, seed):
        return self.source if seed == 7 else self.root / f"seed{seed}"

    def seed_inputs(self, seed):
        root = self.seed_root(seed)
        self.write(root / "role_map.json", self.roles)
        if seed != 7:
            (root / "feature_cache").symlink_to(self.cache, target_is_directory=True)
        scope = aggregate.PILOT_SCOPE if seed == 7 else aggregate.SCOPE
        self.write(root / "execution.json", {"scope_id": scope, "seed": seed})
        for name in ("base_freeze.json", "heads_freeze.json", "predictions/meta.json",
                     "predictions/dev_eval.json"):
            self.write(root / name, {"scope_id": scope, "seed": seed, "synthetic": True})
        arrays = {name: self.data[seed][name] for name in aggregate.METADATA}
        arrays["logits"] = np.column_stack([self.data[seed]["raw_" + arm] for arm in aggregate.ARMS])
        aggregate.atomic_npz(root / "predictions/dev_eval.npz", arrays)

    def evaluation(self, seed):
        root = self.seed_root(seed)
        out = root / "evaluation"
        out.mkdir()
        inputs = {key: aggregate.sha256(root / name) for key, name in (
            ("execution_sha256", "execution.json"), ("roles_sha256", "role_map.json"),
            ("base_freeze_sha256", "base_freeze.json"), ("heads_freeze_sha256", "heads_freeze.json"),
            ("dev_prediction_npz_sha256", "predictions/dev_eval.npz"),
            ("dev_prediction_sidecar_sha256", "predictions/dev_eval.json"),
        )}
        inputs["cache_manifest_sha256"] = aggregate.sha256(self.cache / "manifest.json")
        inputs["implementation"] = ({"combine.py": "a" * 64, "evaluate.py": "b" * 64} if seed == 7 else
                                    aggregate.code_identity(aggregate.read(root / "execution.json")))
        if seed != 7:
            inputs.update(replication_manifest_sha256=aggregate.sha256(self.root / "replication.json"),
                          joint_heads_freeze_sha256=aggregate.sha256(self.root / "joint_heads_freeze.json"))
        primary = {
            **self.full_statistics["per_seed"][str(seed)]["primary"],
            "draws": aggregate.BOOTSTRAP_DRAWS, "bootstrap_seed": aggregate.BOOTSTRAP_SEED,
            "source_photographs": 800, "views_per_source": 9, "percentile_method": "linear",
            "group": "image_id (source photograph), not source_id (corruption family)",
            "undefined_draw_policy": "no redraw; withhold interval if any draw is undefined",
            "scope": f"Synthetic source-image uncertainty conditional on seed{seed}",
        }
        index = (17, 27, 7).index(seed)
        values = self.full_statistics["draws"]["delta_auroc"][:, index]
        self.write(out / "bootstrap.json", {**primary, "bootstrap_values":
                   [float(value) if np.isfinite(value) else None for value in values]})
        aggregate.atomic_npz(out / "scores.npz", self.data[seed])
        aggregate.atomic_npz(out / "bootstrap_source_counts.npz", self.full_statistics["source_counts"])
        (out / "REPORT.md").write_text("Synthetic report; no historical outcomes.\n", encoding="utf-8")
        scope = aggregate.PILOT_SCOPE if seed == 7 else aggregate.SCOPE
        report = {
            "scope_id": scope, "seed": seed, "complete": True, "status": "complete",
            "records": 7200, "source_photographs": 800, "base_epochs": 20,
            "base_models": list(aggregate.ARMS), "test_evaluated": False,
            "inputs": inputs, "primary": primary,
            "metrics": {name: {"auroc": metric["auroc"]}
                        for name, metric in self.full_statistics["per_seed"][str(seed)]["metrics"].items()},
            "errors": int(self.data[seed]["y"].sum()),
            "correct": int((self.data[seed]["y"] == 0).sum()),
        }
        self.write(out / "report.json", report)
        complete = {"scope_id": scope, "complete": True, "inputs": inputs,
                    "files": {name: aggregate.sha256(out / name) for name in aggregate.EVALUATION_FILES}}
        if seed != 7:
            complete.update(seed=seed, completed_unix=self.created + 30)
        self.write(out / "complete.json", complete)

    def rebind_evaluation(self, seed=17):
        out = self.seed_root(seed) / "evaluation"
        report, complete = aggregate.read(out / "report.json"), aggregate.read(out / "complete.json")
        complete["inputs"] = report["inputs"]
        complete["files"] = {name: aggregate.sha256(out / name) for name in aggregate.EVALUATION_FILES}
        self.write(out / "complete.json", complete)

    def bundles(self):
        return aggregate._input_bundles(self.root, self.manifest, self.joint, None)[0]

    def run_cached(self):
        statistics = copy.deepcopy(self.full_statistics)
        with patch.object(aggregate, "paired_seed_statistics", return_value=statistics):
            return aggregate.aggregate(self.root, self.out)

    def test_existing_single_seed_bootstrap_matches_all_2000_shared_draws(self):
        data = self.data[17]
        original, image_ids, counts = evaluate.paired_bootstrap(
            data["y"], data["learned_stack"], data["learned_last_only"], data["image_id"], seed=17)
        np.testing.assert_array_equal(image_ids, self.full_statistics["source_counts"]["image_id"])
        np.testing.assert_array_equal(counts, self.full_statistics["source_counts"]["multiplicity"])
        np.testing.assert_allclose(np.asarray(original["bootstrap_values"], dtype=np.float64),
                                   self.full_statistics["draws"]["delta_auroc"][:, 0],
                                   rtol=1e-14, atol=1e-14, equal_nan=True)
        current = self.full_statistics["per_seed"]["17"]["primary"]
        for name in ("estimate", "stack_auroc", "last_only_auroc", "interval_95", "undefined_draw_indices"):
            self.assertEqual(original[name], current[name])

    def test_end_to_end_late_is_readonly_and_completed_outputs_are_idempotent(self):
        source_before = {name: aggregate.sha256(self.source / name) for name in aggregate.SOURCE_FILES}
        result = aggregate.aggregate(self.root, self.out)
        self.assertEqual(result["primary"], self.full_statistics["primary"])
        self.assertLess(result["primary"]["estimate"], 0)
        self.assertTrue(result["all3_descriptive_complete"])
        self.assertEqual(result["per_seed"]["7"]["status"], "complete_late_diagnostic")
        self.assertTrue(result["per_seed"]["7"]["original_protocol_status"].startswith("incomplete"))
        self.assertIn("original protocol", result["per_seed"]["7"]["warning"])
        self.assertEqual(result["records_per_seed"], 7200)
        self.assertFalse(result["test_evaluated"])
        with np.load(self.out / "scores.npz", allow_pickle=False) as scores:
            self.assertEqual(scores["scores"].shape, (3, 7200, 7))
            self.assertEqual(len(np.unique(scores["record_id"])), 7200)
            np.testing.assert_array_equal(scores["seeds"], [17, 27, 7])
            for seed_index, seed in enumerate((17, 27, 7)):
                for method_index, name in enumerate(aggregate.METHODS):
                    np.testing.assert_array_equal(scores["scores"][seed_index, :, method_index], self.data[seed][name])
            for name in aggregate.METADATA:
                np.testing.assert_array_equal(scores[name], self.data[17][name])
        with np.load(self.out / "bootstrap_draws.npz", allow_pickle=False) as draws:
            self.assertEqual(draws["auroc"].shape, (2000, 3, 7))
            np.testing.assert_array_equal(draws["primary_mean_delta_auroc"], draws["delta_auroc"][:, :2].mean(axis=1))
            np.testing.assert_array_equal(draws["all3_mean_delta_auroc"], draws["delta_auroc"].mean(axis=1))
        with np.load(self.out / "bootstrap_source_counts.npz", allow_pickle=False) as sources:
            self.assertEqual(sources["multiplicity"].shape, (2000, 800))
            np.testing.assert_array_equal(sources["multiplicity"], self.full_statistics["source_counts"]["multiplicity"])
        for name in ("report.json", "bootstrap.json", "complete.json"):
            aggregate._read(self.out / name)
        complete = aggregate.read(self.out / "complete.json")
        self.assertEqual(set(complete["files"]), aggregate.OUTPUT_FILES)
        before = {name: aggregate.sha256(self.out / name) for name in (*aggregate.OUTPUT_FILES, "complete.json")}
        after_cutoff = aggregate._evaluation_cutoff(self.manifest) + 1
        with patch.object(aggregate.time, "time", return_value=after_cutoff), \
                patch.object(aggregate, "paired_seed_statistics", side_effect=AssertionError("must not recompute")), \
                patch.object(aggregate, "check_deadline", side_effect=AssertionError("completed validation only")):
            self.assertEqual(aggregate.aggregate(self.root, self.out, self.source), result)
        self.assertEqual(before, {name: aggregate.sha256(self.out / name) for name in before})
        self.assertEqual(source_before, {name: aggregate.sha256(self.source / name) for name in source_before})

    def test_default_includes_prebound_late7_and_matches_explicit_source_identity(self):
        result = self.run_cached()
        self.assertTrue(result["complete"])
        self.assertTrue(result["all3_descriptive_complete"])
        self.assertEqual(result["included_seeds"], [17, 27, 7])
        self.assertEqual(result["all3_descriptive"]["status"], "complete")
        self.assertEqual(result["per_seed"]["7"]["status"], "complete_late_diagnostic")
        self.assertEqual(result["inputs"]["late_seed7_root"], str(self.source))
        self.assertEqual(result["late_seed7_marker"]["status"], "complete_late_diagnostic")
        with np.load(self.out / "bootstrap_draws.npz", allow_pickle=False) as draws:
            self.assertIn("all3_mean_delta_auroc", draws.files)
        with patch.object(aggregate, "paired_seed_statistics", side_effect=AssertionError("must not recompute")):
            self.assertEqual(aggregate.aggregate(self.root, self.out, self.source), result)
        with self.assertRaisesRegex(RuntimeError, "prebound"):
            aggregate.aggregate(self.root, self.out, self.seed_root(17))

    def test_missing_any_seed_prevents_subset_success(self):
        for seed in (17, 27, 7):
            with self.subTest(seed=seed):
                path = self.seed_root(seed) / "evaluation/complete.json"
                unavailable = path.with_name("complete.unavailable.json")
                path.rename(unavailable)
                try:
                    with patch.object(aggregate, "paired_seed_statistics", side_effect=AssertionError("must not compute")):
                        with self.assertRaisesRegex(RuntimeError, "Missing or redirected artifact"):
                            aggregate.aggregate(self.root, self.out)
                    self.assertFalse((self.out / "complete.json").exists())
                finally:
                    unavailable.rename(path)

    def test_incomplete_all3_calculation_cannot_write_success(self):
        incomplete = copy.deepcopy(self.full_statistics)
        incomplete["all3_descriptive"]["status"] = "unavailable"
        with patch.object(aggregate, "paired_seed_statistics", return_value=incomplete):
            with self.assertRaisesRegex(RuntimeError, "No subset success"):
                aggregate.aggregate(self.root, self.out)
        self.assertFalse((self.out / "complete.json").exists())

    def test_every_required_completed_output_is_verified(self):
        self.run_cached()
        for name in aggregate.OUTPUT_FILES:
            with self.subTest(name=name):
                path = self.out / name
                original = path.read_bytes()
                path.write_bytes(original + b"\nchanged\n")
                with self.assertRaisesRegex(RuntimeError, "Completed artifact changed"):
                    aggregate.aggregate(self.root, self.out, self.source)
                path.write_bytes(original)
        complete = aggregate.read(self.out / "complete.json")
        del complete["files"]["bootstrap_draws.npz"]
        self.write(self.out / "complete.json", complete)
        with self.assertRaisesRegex(RuntimeError, "inventory"):
            aggregate.aggregate(self.root, self.out, self.source)

    def test_wrong_seed_and_incomplete_input_inventory_are_rejected(self):
        out = self.seed_root(17) / "evaluation"
        report = aggregate.read(out / "report.json")
        report["seed"] = 27
        self.write(out / "report.json", report)
        self.rebind_evaluation()
        with self.assertRaisesRegex(RuntimeError, "wrong seed"):
            self.bundles()
        report["seed"] = 17
        self.write(out / "report.json", report)
        self.rebind_evaluation()
        complete = aggregate.read(out / "complete.json")
        complete["seed"] = 27
        self.write(out / "complete.json", complete)
        with self.assertRaisesRegex(RuntimeError, "completion missed"):
            self.bundles()
        complete["seed"] = 17
        del complete["files"]["scores.npz"]
        self.write(out / "complete.json", complete)
        with self.assertRaisesRegex(RuntimeError, "inventory"):
            self.bundles()

    def test_input_file_hashes_and_complete_input_bindings_are_checked(self):
        out = self.seed_root(17) / "evaluation"
        path = out / "scores.npz"
        original = path.read_bytes()
        path.write_bytes(original + b"\nchanged\n")
        with self.assertRaisesRegex(RuntimeError, "Completed artifact changed"):
            self.bundles()
        path.write_bytes(original)
        report = aggregate.read(out / "report.json")
        del report["inputs"]["dev_prediction_npz_sha256"]
        self.write(out / "report.json", report)
        self.rebind_evaluation()
        with self.assertRaisesRegex(RuntimeError, "input binding"):
            self.bundles()

    def test_completed_seed_must_follow_joint_freeze_and_precede_cutoff(self):
        path = self.seed_root(17) / "evaluation/complete.json"
        original = aggregate.read(path)
        cutoff = aggregate._evaluation_cutoff(self.manifest)
        for completed in (self.joint["frozen_unix"], cutoff, cutoff + 1):
            with self.subTest(completed=completed):
                self.write(path, {**original, "completed_unix": completed})
                with self.assertRaisesRegex(RuntimeError, "frozen cutoff"):
                    self.bundles()

    def test_rebound_mismatched_metadata_is_rejected_before_statistics(self):
        out = self.seed_root(27) / "evaluation"
        changed = copy.deepcopy(self.data[27])
        changed["source_id"][0] += 1
        aggregate.atomic_npz(out / "scores.npz", changed)
        self.rebind_evaluation(27)
        with patch.object(aggregate, "paired_seed_statistics", side_effect=AssertionError("must not compute")):
            with self.assertRaisesRegex(RuntimeError, "ordered metadata: source_id"):
                aggregate.aggregate(self.root, self.out, self.source)
        self.assertFalse((self.out / "complete.json").exists())

    def test_changed_frozen_head_and_nan_json_are_rejected(self):
        path = self.seed_root(17) / "heads_freeze.json"
        original = path.read_bytes()
        self.write(path, {"changed": True})
        with self.assertRaisesRegex(RuntimeError, "input changed"):
            self.bundles()
        path.write_bytes(original)
        report_path = self.seed_root(17) / "evaluation/report.json"
        report = aggregate.read(report_path)
        report["metrics"]["learned_stack"]["auroc"] = float("nan")
        report_path.write_text(json.dumps(report, allow_nan=True), encoding="utf-8")
        self.rebind_evaluation()
        with self.assertRaisesRegex(RuntimeError, "NaN or infinity"):
            self.bundles()

    def test_late_source_requires_exact_prebound_root_and_complete_binding(self):
        with self.assertRaisesRegex(RuntimeError, "prebound"):
            aggregate._late_source(self.manifest, self.seed_root(17))
        changed = copy.deepcopy(self.manifest)
        del changed["source_files"]["evaluation/bootstrap_source_counts.npz"]
        with self.assertRaisesRegex(RuntimeError, "inventory"):
            aggregate._late_source(changed, self.source)
        changed = copy.deepcopy(self.manifest)
        changed["source_files"]["evaluation/scores.npz"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "artifact changed"):
            aggregate._late_source(changed, self.source)
        self.assertEqual(self.bundles()[7]["identity"]["inputs"]["implementation"]["evaluate.py"], "b" * 64)

    def test_late_marker_never_erases_original_incomplete_warning(self):
        path = self.source / "evaluation/late_diagnostic.json"
        original = aggregate.read(path)
        for name, value in (("status", "complete"), ("original_protocol_status", "complete"),
                            ("warning", ""), ("reused_on_time_meta_export", False), ("complete", False)):
            with self.subTest(field=name):
                self.write(path, {**original, name: value})
                manifest = copy.deepcopy(self.manifest)
                manifest["source_files"]["evaluation/late_diagnostic.json"] = aggregate.sha256(path)
                with self.assertRaisesRegex(RuntimeError, "late marker"):
                    aggregate._late_source(manifest, self.source)

    def test_saved_statistics_and_shared_draws_must_agree(self):
        bundles = self.bundles()
        bundles[17]["report"]["metrics"]["learned_stack"]["auroc"] += 0.1
        with self.assertRaisesRegex(RuntimeError, "disagrees"):
            aggregate._verify_seed_statistics(bundles, self.full_statistics, self.data)
        out = self.seed_root(17) / "evaluation"
        bootstrap = aggregate.read(out / "bootstrap.json")
        bootstrap["bootstrap_values"][0] += 0.125
        self.write(out / "bootstrap.json", bootstrap)
        self.rebind_evaluation()
        with self.assertRaisesRegex(RuntimeError, "bootstrap values disagree"):
            aggregate._verify_seed_statistics(self.bundles(), self.full_statistics, self.data)

    def test_saved_source_counts_must_match_rng_not_just_row_totals(self):
        out = self.seed_root(17) / "evaluation"
        counts = copy.deepcopy(self.full_statistics["source_counts"])
        source = int(np.flatnonzero(counts["multiplicity"][0] > 0)[0])
        target = (source + 1) % 800
        counts["multiplicity"][0, source] -= 1
        counts["multiplicity"][0, target] += 1
        aggregate.atomic_npz(out / "bootstrap_source_counts.npz", counts)
        self.rebind_evaluation()
        with self.assertRaisesRegex(RuntimeError, "exact shared image_id draws"):
            aggregate._verify_seed_statistics(self.bundles(), self.full_statistics, self.data)

    def test_noncanonical_or_redirected_output_and_partial_overwrite_are_refused(self):
        for out in (self.root / "elsewhere", self.source / "evaluation"):
            with self.subTest(out=str(out)):
                with self.assertRaisesRegex(RuntimeError, "canonical"):
                    aggregate.aggregate(self.root, out, self.source)
        self.out.symlink_to(self.source / "evaluation", target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "canonical"):
            aggregate.aggregate(self.root, self.out, self.source)
        self.out.unlink()
        self.out.mkdir()
        original = '{"preexisting": "incomplete"}\n'
        (self.out / "report.json").write_text(original, encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "silent overwrite"):
            aggregate.aggregate(self.root, self.out, self.source)
        self.assertEqual((self.out / "report.json").read_text(), original)

    def test_hard_deadline_blocks_computation_and_completion(self):
        cutoff = aggregate._evaluation_cutoff(self.manifest)
        with patch.object(aggregate.time, "time", return_value=cutoff), \
                patch.object(aggregate, "paired_seed_statistics", side_effect=AssertionError("must not compute")):
            with self.assertRaises(TimeoutError):
                aggregate.aggregate(self.root, self.out, self.source)
        self.assertFalse((self.out / "complete.json").exists())
        now = {"value": cutoff - 10}
        original_npz = aggregate.atomic_npz

        def expire_after_scores(path, arrays):
            original_npz(path, arrays)
            if path == self.out / "scores.npz":
                now["value"] = cutoff

        with patch.object(aggregate.time, "time", side_effect=lambda: now["value"]), \
                patch.object(aggregate, "atomic_npz", side_effect=expire_after_scores):
            with self.assertRaises(TimeoutError):
                self.run_cached()
        self.assertFalse((self.out / "complete.json").exists())
        with self.assertRaisesRegex(RuntimeError, "silent overwrite"):
            self.run_cached()

    def test_inputs_are_revalidated_after_computation(self):
        original_npz = aggregate.atomic_npz

        def change_input_after_draws(path, arrays):
            original_npz(path, arrays)
            if path == self.out / "bootstrap_draws.npz":
                self.write(self.seed_root(27) / "heads_freeze.json", {"changed_during_aggregation": True})

        with patch.object(aggregate, "atomic_npz", side_effect=change_input_after_draws):
            with self.assertRaisesRegex(RuntimeError, "input changed"):
                self.run_cached()
        self.assertFalse((self.out / "complete.json").exists())


if __name__ == "__main__":
    aggregate.require_slurm()
    unittest.main()
