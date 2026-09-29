"""Synthetic analysis tests (diagnostics, heads, linear probe, statistics); Slurm-only runner."""
import csv
import inspect
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from pilots.layer_ensemble_20260914.evaluate import WeightedAUC, sigmoid
from . import diagnostics, heads, linear, protocol
from . import evaluate as ev

PACKAGE = __name__.rsplit(".", 1)[0]
SEEDS = tuple(protocol.SEEDS)
IMPORTED = tuple(protocol.IMPORTED_SEEDS)
NEURAL_FITS = len(protocol.neural_matrix())
IMPORTED_FITS = len(protocol.ARMS) * len(IMPORTED)


# --------------------------------------------------------------------------- helpers
def make_history(aucs, losses, running=True):
    rows, best = [], 0
    for index, (auc, loss) in enumerate(zip(aucs, losses)):
        if auc > aucs[best]:
            best = index
        row = {"epoch": index + 1, "checkpoint_auroc": auc, "training_loss": loss,
               "best_epoch": best + 1}
        if running:
            row["best_checkpoint_auroc"] = aucs[best]
        rows.append(row)
    return rows, best + 1


EPOCHS = range(1, 21)
PATTERNS = {
    # strictly rising AUROC (selected 20) and falling loss: the only warning pattern
    "rising": ([0.5 + 0.01 * e for e in EPOCHS], [1.0 - 0.03 * e for e in EPOCHS]),
    # peak at epoch 10; the running best (0.70) differs from actual epochs 15/20
    "peaked": ([0.5 + 0.02 * e if e <= 10 else 0.70 - 0.005 * (e - 10) for e in EPOCHS],
               [1.0 - 0.03 * e for e in EPOCHS]),
    # selected 20 and AUROC rising, but loss rising: no warning
    "loss_up": ([0.5 + 0.01 * e for e in EPOCHS], [0.2 + 0.01 * e for e in EPOCHS]),
    # zero loss from epoch 10: percentage undefined
    "zero": ([0.5 + 0.02 * e if e <= 10 else 0.70 - 0.005 * (e - 10) for e in EPOCHS],
             [1.0 - 0.1 * e if e < 10 else 0.0 for e in EPOCHS]),
}


def pattern_for(index):
    return "zero" if index == 4 else ("rising", "peaked", "loss_up")[index % 3]


def ols_slope(losses):
    return float(np.polyfit(np.arange(16, 21), np.asarray(losses[15:20]), 1)[0])


def manual_head(columns, mean, scale, coef, intercept, classes=(0, 1)):
    return {"columns": list(columns),
            "scaler": {"mean": list(mean), "scale": list(scale),
                       "var": [value * value for value in scale]},
            "model": {"coef": list(coef), "intercept": intercept, "classes": list(classes)}}


def brute_auc(labels, scores, weights=None):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    weights = np.ones(len(labels)) if weights is None else np.asarray(weights, dtype=np.float64)
    pos, neg = labels == 1, labels == 0
    wp, wn = weights[pos].sum(), weights[neg].sum()
    if not wp or not wn:
        return None
    total = 0.0
    for i in np.flatnonzero(pos):
        for j in np.flatnonzero(neg):
            credit = 1.0 if scores[i] > scores[j] else 0.5 if scores[i] == scores[j] else 0.0
            total += weights[i] * weights[j] * credit
    return total / (wp * wn)


def logistic_data(rows, columns, weights, bias, seed):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(rows, columns))
    probability = 1.0 / (1.0 + np.exp(-(values @ np.asarray(weights) + bias)))
    labels = (rng.random(rows) < probability).astype(np.int64)
    return values, labels


def sklearn_reference(values, labels, recipe):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    options = {key: recipe[key] for key in
               ("penalty", "C", "solver", "max_iter", "tol", "class_weight", "random_state")}
    if isinstance(options["class_weight"], dict):
        options["class_weight"] = {int(k): v for k, v in options["class_weight"].items()}
    scaler = StandardScaler().fit(values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LogisticRegression(**options).fit(scaler.transform(values), labels)
    return scaler, model


# --------------------------------------------------------------------------- diagnostics
class DiagnoseHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        protocol.require_slurm()

    def test_actual_checkpoint_auroc_not_running_best(self):
        aucs, losses = PATTERNS["peaked"]
        history, best = make_history(aucs, losses)
        self.assertEqual(best, 10)
        result = diagnostics.diagnose_history(history, best)
        self.assertEqual(result["checkpoint_auroc_15"], aucs[14])
        self.assertEqual(result["checkpoint_auroc_20"], aucs[19])
        self.assertNotEqual(result["checkpoint_auroc_15"], history[14]["best_checkpoint_auroc"])
        self.assertAlmostEqual(result["late_auc_change"], aucs[19] - aucs[14], places=12)
        self.assertLess(result["late_auc_change"], 0)  # signed, not absolute
        self.assertEqual(result["selected_epoch"], 10)
        self.assertFalse(result["selected_epoch_20"])
        self.assertFalse(result["budget_sensitivity_warning"])

    def test_loss_decrease_absolute_relative_and_tail_slope(self):
        aucs = PATTERNS["peaked"][0]
        losses = [1.0 - 0.01 * e for e in range(1, 16)] + [0.40, 0.38, 0.37, 0.33, 0.30]
        history, best = make_history(aucs, losses)
        result = diagnostics.diagnose_history(history, best)
        self.assertEqual(result["training_loss_15"], losses[14])
        self.assertEqual(result["training_loss_20"], 0.30)
        self.assertAlmostEqual(result["loss_decrease_absolute"], losses[14] - 0.30, places=12)
        self.assertAlmostEqual(result["loss_decrease_percent"],
                               100.0 * (losses[14] - 0.30) / losses[14], places=10)
        self.assertIsNone(result["loss_decrease_percent_undefined_reason"])
        self.assertAlmostEqual(result["loss_slope_16_20"], ols_slope(losses), places=12)
        self.assertAlmostEqual(result["loss_slope_16_20"], -0.025, places=12)
        rising = diagnostics.diagnose_history(*make_history(*PATTERNS["loss_up"]))
        self.assertLess(rising["loss_decrease_absolute"], 0)
        self.assertLess(rising["loss_decrease_percent"], 0)
        self.assertGreater(rising["loss_slope_16_20"], 0)

    def test_zero_loss15_percentage_undefined_with_reason(self):
        history, best = make_history(*PATTERNS["zero"])
        result = diagnostics.diagnose_history(history, best)
        self.assertEqual(result["training_loss_15"], 0.0)
        self.assertIsNone(result["loss_decrease_percent"])
        self.assertIsInstance(result["loss_decrease_percent_undefined_reason"], str)
        self.assertIn("zero", result["loss_decrease_percent_undefined_reason"])
        self.assertEqual(result["loss_decrease_absolute"], 0.0)
        self.assertEqual(result["loss_slope_16_20"], 0.0)

    def test_warning_and_no_warning_cases(self):
        warned = diagnostics.diagnose_history(*make_history(*PATTERNS["rising"]))
        self.assertTrue(warned["selected_epoch_20"])
        self.assertGreater(warned["late_auc_change"], 0)
        self.assertGreater(warned["loss_decrease_absolute"], 0)
        self.assertIs(warned["budget_sensitivity_warning"], True)
        boundary_only = diagnostics.diagnose_history(*make_history(*PATTERNS["loss_up"]))
        self.assertTrue(boundary_only["selected_epoch_20"])
        self.assertIs(boundary_only["budget_sensitivity_warning"], False)
        for result in (boundary_only, diagnostics.diagnose_history(*make_history(*PATTERNS["peaked"]))):
            self.assertFalse(result["budget_sensitivity_warning"])
            self.assertNotIn("converged", json.dumps(result).lower())
            self.assertFalse(any("converge" in key for key in result))
        self.assertFalse(diagnostics.POLICY["warning_is_failure"])
        self.assertEqual(diagnostics.POLICY["absence_of_warning"], "not proof of convergence")
        self.assertFalse(any("converge" in field for field in diagnostics.CSV_FIELDS))

    def test_earliest_tie_is_selected(self):
        aucs = [0.6] * 5 + [0.9] + [0.5] * 13 + [0.9]
        history, best = make_history(aucs, PATTERNS["rising"][1])
        self.assertEqual(best, 6)
        self.assertEqual(diagnostics.diagnose_history(history, 6)["selected_epoch"], 6)
        with self.assertRaises(ValueError):
            diagnostics.diagnose_history(history, 20)

    def test_missing_or_inconsistent_history_fails_closed(self):
        history, best = make_history(*PATTERNS["rising"])
        broken = []
        broken.append((history[:-1], best))
        broken.append((history[:14] + history[15:], best))
        broken.append((history + [dict(history[-1], epoch=21)], best))
        swapped = [dict(row) for row in history]
        swapped[3], swapped[4] = swapped[4], swapped[3]
        broken.append((swapped, best))
        for field, value in (("checkpoint_auroc", None), ("training_loss", float("nan")),
                             ("checkpoint_auroc", True), ("training_loss", -0.1),
                             ("checkpoint_auroc", 1.5), ("best_epoch", 3),
                             ("best_checkpoint_auroc", 0.1), ("epoch", 15.0)):
            changed = [dict(row) for row in history]
            changed[14][field] = value
            broken.append((changed, best))
        missing = [dict(row) for row in history]
        del missing[16]["training_loss"]
        broken.append((missing, best))
        broken.extend(((history, 19), (history, 20.0), (history, True), (history, 0),
                       ("not a list", best)))
        for index, (candidate, selected) in enumerate(broken):
            with self.subTest(case=index), self.assertRaises(ValueError):
                diagnostics.diagnose_history(candidate, selected)
        without_running = make_history(*PATTERNS["rising"], running=False)
        self.assertTrue(diagnostics.diagnose_history(*without_running)["budget_sensitivity_warning"])


def matrix_rows():
    rows = []
    for index, fit in enumerate(protocol.neural_matrix()):
        pattern = pattern_for(index)
        history, best = make_history(*PATTERNS[pattern])
        rows.append({**fit, "pattern": pattern, "history": history, "best": best,
                     "imported": fit["family"] == "G" and fit["seed"] in IMPORTED})
    return rows


class DiagnosticSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        protocol.require_slurm()

    def rows(self):
        return [{"family": item["family"], "arm": item["arm"], "seed": item["seed"],
                 "imported": item["imported"], "pattern": item["pattern"],
                 **diagnostics.diagnose_history(item["history"], item["best"])}
                for item in matrix_rows()]

    def expected_group(self, items):
        return {"denominator": len(items),
                "selected_epoch_20_count": sum(i["pattern"] in ("rising", "loss_up") for i in items),
                "late_auc_increase_count": sum(i["pattern"] in ("rising", "loss_up") for i in items),
                "late_loss_decrease_count": sum(i["pattern"] in ("rising", "peaked") for i in items),
                "budget_sensitivity_warning_count": sum(i["pattern"] == "rising" for i in items),
                "undefined_percent": sum(i["pattern"] == "zero" for i in items)}

    def check_group(self, actual, items):
        expected = self.expected_group(items)
        for key in ("denominator", "selected_epoch_20_count", "late_auc_increase_count",
                    "late_loss_decrease_count", "budget_sensitivity_warning_count"):
            self.assertEqual(actual[key], expected[key], key)
        self.assertEqual(actual["seeds"], sorted({i["seed"] for i in items}))
        percent = actual["loss_decrease_percent"]
        self.assertEqual(percent["undefined_count"], expected["undefined_percent"])
        self.assertEqual(percent["defined_count"], len(items) - expected["undefined_percent"])
        auc = [i["late_auc_change"] for i in items]
        self.assertAlmostEqual(actual["late_auc_change"]["mean"], statistics.mean(auc), places=12)
        self.assertEqual(actual["late_auc_change"]["minimum"], min(auc))
        self.assertEqual(actual["late_auc_change"]["maximum"], max(auc))

    def test_counts_and_denominators_by_family_layer_and_family(self):
        rows = self.rows()
        self.assertEqual(len(rows), NEURAL_FITS)
        self.assertEqual(sum(row["imported"] for row in rows), IMPORTED_FITS)
        summary = diagnostics.summarize(rows)
        self.check_group(summary["all"], rows)
        self.assertEqual({k: v["denominator"] for k, v in summary["by_family"].items()},
                         {"G": 4 * len(SEEDS), "H": 4 * len(SEEDS), "S": 4 * len(SEEDS), "O": len(SEEDS)})
        self.assertEqual(len(summary["by_family_layer"]), 13)
        self.assertEqual(summary["by_family_layer"]["O/logits"]["denominator"], len(SEEDS))
        for family in ("G", "H", "S", "O"):
            self.check_group(summary["by_family"][family],
                             [r for r in rows if r["family"] == family])
        for name, group in summary["by_family_layer"].items():
            family, arm = name.split("/")
            items = [r for r in rows if (r["family"], r["arm"]) == (family, arm)]
            self.assertEqual(len(items), len(SEEDS))
            self.check_group(group, items)
        imported_g = [r for r in rows if r["imported"]]
        self.assertEqual({r["seed"] for r in imported_g}, set(IMPORTED))
        self.assertEqual(summary["by_family"]["G"]["seeds"], sorted(SEEDS))
        self.assertNotIn("converged", json.dumps(summary).lower())

    def test_undefined_percent_group_and_rejections(self):
        rows = self.rows()
        zero = [r for r in rows if r["pattern"] == "zero"]
        only = diagnostics.summarize(zero)["all"]["loss_decrease_percent"]
        self.assertEqual((only["mean"], only["defined_count"], only["undefined_count"]), (None, 0, 1))
        with self.assertRaises(ValueError):
            diagnostics.summarize([])
        with self.assertRaises(ValueError):
            diagnostics.summarize(rows + rows[:1])
        for change in ({"family": "L"}, {"family": "O", "arm": "block2"},
                       {"family": "G", "arm": "logits"}, {"seed": 3}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                diagnostics.summarize([{**rows[0], **change}])


class DiagnosticCollectionTests(unittest.TestCase):
    """_collect/build over every registered synthetic run directory, with the campaign context mocked."""

    @classmethod
    def setUpClass(cls):
        protocol.require_slurm()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="final-diagnostics-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.items = matrix_rows()
        for item in self.items:
            directory = self.directory(item["family"], item["arm"], item["seed"])
            directory.mkdir(parents=True)
            protocol.atomic_json(directory / "history.json", item["history"])
            protocol.atomic_json(directory / "complete.json", {
                "complete": True, "completed_epochs": 20, "selection_role": "checkpoint",
                "best_epoch": item["best"]})
            scope = ("layer_ensemble_20260914_core_seed7" if item["imported"] and item["seed"] == 7
                     else "layer_ensemble_replication_seed17_27" if item["imported"]
                     else protocol.SCOPE)
            protocol.atomic_json(directory / "config.json", {"scope_id": scope})
            (directory / "best.safetensors").write_bytes(b"synthetic")
        self.campaign = {
            "reuse": {protocol.model_key("G", arm, seed): {} for arm in protocol.ARMS
                      for seed in IMPORTED},
            "reuse_groups": {str(seed): {"status": "complete_late_diagnostic" if seed == 7 else "complete"}
                             for seed in IMPORTED}}
        fake_train = types.ModuleType(PACKAGE + ".train")

        def verify_complete(root, family, arm, seed):
            directory = self.directory(family, arm, seed)
            return (protocol.read(directory / "complete.json"),
                    protocol.read(directory / "config.json"))

        fake_train.verify_complete = verify_complete
        for patcher in (
            patch.dict(sys.modules, {PACKAGE + ".train": fake_train}),
            patch.object(diagnostics, "_context", side_effect=lambda root, with_base=False: (
                self.root, self.campaign, {}, {"campaign_sha256": "a" * 64})),
            patch.object(protocol, "resolve_run", side_effect=lambda root, f, a, s: self.directory(f, a, s)),
            patch.object(protocol, "check_cutoff"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def directory(self, family, arm, seed):
        return self.root / "runs" / family / arm / f"seed{seed}"

    def test_collect_includes_imported_g_with_original_status(self):
        _, inputs, rows = diagnostics._collect(self.root)
        self.assertEqual(len(rows), NEURAL_FITS)
        self.assertEqual(len(inputs["runs"]), NEURAL_FITS)
        imported = [row for row in rows if row["imported"]]
        self.assertEqual(len(imported), IMPORTED_FITS)
        self.assertTrue(all(row["family"] == "G" for row in imported))
        self.assertEqual({row["original_status"] for row in imported if row["seed"] == 7},
                         {"complete_late_diagnostic"})
        self.assertEqual({row["original_status"] for row in imported if row["seed"] != 7}, {"complete"})
        self.assertEqual(sum(row["budget_sensitivity_warning"] for row in rows),
                         sum(item["pattern"] == "rising" for item in self.items))

    def test_missing_epoch_or_shorter_budget_or_dropped_import_fails_closed(self):
        item = self.items[7]
        directory = self.directory(item["family"], item["arm"], item["seed"])
        original = protocol.read(directory / "history.json")
        (directory / "history.json").unlink()
        protocol.atomic_json(directory / "history.json", original[:19])
        with self.assertRaises(ValueError):
            diagnostics._collect(self.root)
        (directory / "history.json").unlink()
        protocol.atomic_json(directory / "history.json", original)
        complete = protocol.read(directory / "complete.json")
        (directory / "complete.json").unlink()
        protocol.atomic_json(directory / "complete.json", {**complete, "completed_epochs": 19})
        with self.assertRaisesRegex(RuntimeError, "shorter-budget"):
            diagnostics._collect(self.root)
        (directory / "complete.json").unlink()
        protocol.atomic_json(directory / "complete.json", complete)
        self.campaign["reuse"].pop(protocol.model_key("G", "block2", 7))
        with self.assertRaisesRegex(RuntimeError, "may not be dropped"):
            diagnostics._collect(self.root)
        self.campaign["reuse"][protocol.model_key("G", "block2", 7)] = {}
        self.campaign["reuse_groups"]["7"]["status"] = "complete"
        with self.assertRaisesRegex(RuntimeError, "lateness"):
            diagnostics._collect(self.root)

    def test_build_freezes_json_and_csv_and_revalidates(self):
        first = diagnostics.build(self.root)
        self.assertEqual((first["neural_fit_count"], first["imported_fit_count"],
                          first["fresh_fit_count"]),
                         (NEURAL_FITS, IMPORTED_FITS, NEURAL_FITS - IMPORTED_FITS))
        artifact = protocol.read(self.root / "diagnostics/training_diagnostics.json")
        self.assertEqual(artifact["summary"]["by_family"]["G"]["denominator"], 4 * len(SEEDS))
        with (self.root / "diagnostics/training_diagnostics.csv").open(newline="") as stream:
            table = list(csv.DictReader(stream))
        self.assertEqual(len(table), NEURAL_FITS)
        self.assertEqual(list(table[0]), list(diagnostics.CSV_FIELDS))
        zero = [row for row in table if row["loss_decrease_percent_undefined_reason"]]
        self.assertEqual(len(zero), 1)
        self.assertEqual(zero[0]["loss_decrease_percent"], "")
        self.assertEqual(diagnostics.build(self.root), first)


# --------------------------------------------------------------------------- heads
class HeadScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        protocol.require_slurm()
        cls.values, cls.labels = logistic_data(240, 4, [1.5, 0.0, -1.2, 0.8], -0.3, seed=11)

    def test_recipe_and_columns_are_fixed(self):
        self.assertEqual(heads.HEAD_COLUMNS, {"stack": [0, 1, 2, 3], "last_only": [3]})
        self.assertEqual(heads.RECIPE, protocol.HEAD_RECIPE)
        self.assertEqual(heads.RECIPE["standardize_on"], "meta")

    def test_manual_head_is_signed_standardized_linear_score(self):
        values = np.array([[0.0, 1.0, 2.0, 3.0], [1.0, -1.0, 0.5, -2.0]])
        head = manual_head([0, 1, 2, 3], [0.5, 0.0, 1.0, 1.0], [2.0, 1.0, 0.5, 4.0],
                           [1.0, -2.0, 0.5, -1.0], 0.25)
        expected = (((values - [0.5, 0.0, 1.0, 1.0]) / [2.0, 1.0, 0.5, 4.0])
                    @ [1.0, -2.0, 0.5, -1.0] + 0.25)
        np.testing.assert_allclose(heads.head_scores(head, values), expected, rtol=0, atol=1e-15)
        last = manual_head([3], [1.0], [4.0], [-1.0], 0.25)
        np.testing.assert_allclose(heads.head_scores(last, values), -(values[:, 3] - 1.0) / 4.0 + 0.25)
        bad = []
        bad.append(manual_head([3], [1.0, 0.0], [4.0, 1.0], [-1.0, 0.0], 0.0))
        bad.append(manual_head([4], [1.0], [4.0], [-1.0], 0.0))
        bad.append(manual_head([np.int64(3)], [1.0], [4.0], [-1.0], 0.0))
        bad.append(manual_head([True], [1.0], [4.0], [-1.0], 0.0))
        bad.append(manual_head([3, 3], [1.0, 1.0], [4.0, 4.0], [1.0, 1.0], 0.0))
        bad.append(manual_head([], [], [], [], 0.0))
        bad.append(manual_head([3], [1.0], [0.0], [-1.0], 0.0))
        bad.append(manual_head([3], [1.0], [4.0], [-1.0], float("nan")))
        bad.append(manual_head([3], [1.0], [4.0], [-1.0], 0.0, classes=(1, 0)))
        for index, head in enumerate(bad):
            with self.subTest(case=index), self.assertRaises(ValueError):
                heads.head_scores(head, values)
        with self.assertRaises(ValueError):
            heads.head_scores(last, np.array([[0.0, 1.0, np.inf, 3.0]]))

    def test_fitted_heads_match_sklearn_and_keep_orientation(self):
        header = {"name": "stack"}
        stack = heads._fit_serialized(self.values, self.labels, [0, 1, 2, 3], heads.RECIPE, header)
        last = heads._fit_serialized(self.values, self.labels, [3], heads.RECIPE, {"name": "last_only"})
        for model, columns in ((stack, [0, 1, 2, 3]), (last, [3])):
            self.assertTrue(model["converged"])
            self.assertTrue(model["serialization_audit"]["passed"])
            self.assertEqual(model["model"]["classes"], [0, 1])
            self.assertEqual(model["scaler"]["n_samples_seen"], 240)
            heads._validate_solver(model, columns, 240, heads.RECIPE)
            np.testing.assert_allclose(model["scaler"]["mean"], self.values[:, columns].mean(axis=0),
                                       rtol=0, atol=1e-12)
            scaler, reference = sklearn_reference(self.values[:, columns], self.labels, heads.RECIPE)
            scores = heads.head_scores(model, self.values)
            standardized = scaler.transform(self.values[:, columns])
            np.testing.assert_allclose(scores, reference.decision_function(standardized),
                                       rtol=1e-10, atol=1e-10)
            np.testing.assert_array_equal(scores > 0, reference.predict(standardized) == 1)
            np.testing.assert_allclose(sigmoid(scores), reference.predict_proba(standardized)[:, 1],
                                       rtol=1e-9, atol=1e-12)
            self.assertGreater(WeightedAUC(self.labels, scores)(), 0.6)  # higher = error
        self.assertEqual(len(stack["model"]["coef"]), 4)
        self.assertEqual(len(last["model"]["coef"]), 1)
        self.assertLess(stack["model"]["coef"][2], 0)  # negative coefficient kept, not flipped
        perturbed = self.values.copy()
        perturbed[:, :3] += 5.0
        np.testing.assert_array_equal(heads.head_scores(last, perturbed), heads.head_scores(last, self.values))
        self.assertFalse(np.allclose(heads.head_scores(stack, perturbed), heads.head_scores(stack, self.values)))

    def test_unconverged_solver_is_not_eligible(self):
        recipe = {**heads.RECIPE, "max_iter": 1}
        model = heads._fit_serialized(self.values, self.labels, [0, 1, 2, 3], recipe, {})
        self.assertFalse(model["converged"])
        with self.assertRaises(RuntimeError):
            heads._validate_solver(model, [0, 1, 2, 3], 240, recipe)
        good = heads._fit_serialized(self.values, self.labels, [0, 1, 2, 3], heads.RECIPE, {})
        for change in ({"warnings": [{"category": "ConvergenceWarning", "message": "x"}]},
                       {"model": {**good["model"], "n_iter": [1000]}},
                       {"converged": False},
                       {"serialization_audit": {**good["serialization_audit"], "passed": False}}):
            with self.subTest(change=list(change)), self.assertRaises(RuntimeError):
                heads._validate_solver({**good, **change}, [0, 1, 2, 3], 240, heads.RECIPE)
        with self.assertRaises(RuntimeError):
            heads._validate_solver(good, [0, 1, 2, 3], 239, heads.RECIPE)
        with self.assertRaises(ValueError):
            heads._fit_serialized(self.values, np.zeros(240, dtype=np.int64), [0, 1, 2, 3], heads.RECIPE, {})


# --------------------------------------------------------------------------- linear probe
def base_arrays(seed=5):
    weights = np.zeros(100)
    weights[[0, 17, 42, 99]] = [1.2, -0.9, 0.7, 0.5]
    logits, labels = logistic_data(14400, 100, weights, -1.3, seed=seed)
    logits = logits * np.linspace(0.5, 3.0, 100) + np.linspace(-2.0, 2.0, 100)
    index = np.arange(14400)
    arrays = {"record_id": index, "image_id": index // 9, "source_id": index % 5,
              "severity": np.zeros(14400, dtype=np.int64), "split_id": np.zeros(14400, dtype=np.int64),
              "y": labels, "label": np.zeros(14400, dtype=np.int64), "pred": labels.copy(),
              "logits": logits}
    return arrays


class LinearProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        protocol.require_slurm()
        cls.arrays = base_arrays()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="final-linear-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.loader_roles = []
        fake_data = types.ModuleType(PACKAGE + ".data")

        def load_logits(root, role):
            self.loader_roles.append(role)
            return {name: value.copy() for name, value in self.arrays.items()}

        fake_data.load_logits = load_logits
        for patcher in (
            patch.dict(sys.modules, {PACKAGE + ".data": fake_data}),
            patch.object(linear, "_context", side_effect=lambda root, with_base=False: (
                self.root, {}, {"roles": {}}, {"campaign_sha256": "a" * 64})),
            patch.object(linear, "_validate_role_arrays"),
            patch.object(protocol, "check_cutoff"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_exactly_100_ordered_features_and_base_train_recipe(self):
        self.assertEqual(linear.FEATURES, [f"classifier_logit_{i}" for i in range(100)])
        self.assertEqual(linear.COLUMNS, list(range(100)))
        root, arrays, inputs, recipe = linear._fit_context(self.root)
        self.assertEqual(self.loader_roles, ["base_train"])
        positives = int(self.arrays["y"].sum())
        negatives = 14400 - positives
        self.assertGreater(negatives, positives)
        self.assertEqual(recipe["standardize_on"], "base_train")
        self.assertEqual(recipe["class_weight"], {"0": 1.0, "1": negatives / positives})
        self.assertEqual((recipe["penalty"], recipe["C"], recipe["solver"], recipe["max_iter"],
                          recipe["tol"], recipe["random_state"]), ("l2", 1.0, "lbfgs", 1000, 1e-6, 7))
        self.assertEqual(inputs["training_role"], "base_train")
        self.assertEqual(inputs["training_counts"], {"records": 14400, "source_photographs": 1600,
                                                     "positive": positives, "negative": negatives})
        self.assertEqual(inputs["feature_order"], linear.FEATURES)
        self.assertFalse(protocol.LINEAR_RECIPE["primary"])
        self.assertEqual(protocol.LINEAR_RECIPE["fits"], 1)

    def test_context_rejects_one_class_or_wrong_row_count(self):
        original = self.arrays
        try:
            type(self).arrays = {**original, "y": np.zeros(14400, dtype=np.int64)}
            with self.assertRaises(ValueError):
                linear._fit_context(self.root)
            type(self).arrays = {name: value[:-9] for name, value in original.items()}
            with self.assertRaises(ValueError):
                linear._fit_context(self.root)
        finally:
            type(self).arrays = original

    def test_single_fit_parity_orientation_and_no_refit(self):
        with patch.object(linear, "_fit_serialized", wraps=heads._fit_serialized) as fitter:
            complete = linear.fit(self.root)
            self.assertEqual(fitter.call_count, 1)
            self.assertEqual(linear.fit(self.root), complete)
            self.assertEqual(fitter.call_count, 1)
        self.assertEqual((complete["fit_count"], complete["family"], complete["training_role"]),
                         (1, "L", "base_train"))
        self.assertIs(complete["dev_eval_used_for_fit"], False)
        self.assertIsNone(complete["seed_sd"])
        model = protocol.read(self.root / "linear/model.json")
        self.assertEqual(model["columns"], list(range(100)))
        self.assertEqual(len(model["model"]["coef"]), 100)
        self.assertTrue(model["converged"])
        self.assertTrue(model["serialization_audit"]["passed"])
        # base_train-only standardization
        np.testing.assert_allclose(model["scaler"]["mean"], self.arrays["logits"].mean(axis=0),
                                   rtol=0, atol=1e-9)
        np.testing.assert_allclose(model["scaler"]["scale"], self.arrays["logits"].std(axis=0),
                                   rtol=1e-10, atol=0)
        self.assertEqual(model["scaler"]["n_samples_seen"], 14400)
        scores = heads.head_scores(model, self.arrays["logits"])
        scaler, reference = sklearn_reference(self.arrays["logits"], self.arrays["y"], model["recipe"])
        standardized = scaler.transform(self.arrays["logits"])
        np.testing.assert_allclose(scores, reference.decision_function(standardized), rtol=1e-9, atol=1e-9)
        np.testing.assert_array_equal(scores > 0, reference.predict(standardized) == 1)
        self.assertGreater(WeightedAUC(self.arrays["y"], scores)(), 0.6)  # higher = more likely error
        self.assertGreater(model["model"]["coef"][0], 0)
        self.assertLess(model["model"]["coef"][17], 0)
        visible = sorted(p.name for p in (self.root / "linear").iterdir() if not p.name.startswith("."))
        self.assertEqual(visible, ["complete.json", "model.json"])

    def test_convergence_failure_blocks_and_is_not_refit(self):
        def unconverged(*args, **kwargs):
            return {**heads._fit_serialized(*args, **kwargs), "converged": False}

        with patch.object(linear, "_fit_serialized", side_effect=unconverged) as fitter:
            with self.assertRaises(RuntimeError):
                linear.fit(self.root)
            self.assertFalse((self.root / "linear/complete.json").exists())
            self.assertTrue((self.root / "linear/.fit_started.json").exists())
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                linear.fit(self.root)
            self.assertEqual(fitter.call_count, 1)

    def test_analytic_scores_are_error_oriented_and_stable(self):
        from scipy.special import log_softmax, softmax

        rng = np.random.default_rng(9)
        logits = rng.normal(scale=3.0, size=(6, 100))
        logits[0] = 0.0
        logits[1] = 0.0
        logits[1, 7] = 50.0
        logits[2] *= 400.0
        scores = linear.analytic_scores(logits)
        np.testing.assert_allclose(scores["MSP"], 1 - softmax(logits, axis=1).max(axis=1), rtol=0, atol=1e-12)
        np.testing.assert_allclose(scores["entropy"],
                                   -(softmax(logits, axis=1) * log_softmax(logits, axis=1)).sum(axis=1),
                                   rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(scores["MSP"][0], 0.99, places=12)
        self.assertAlmostEqual(scores["entropy"][0], math.log(100), places=12)
        self.assertGreater(scores["MSP"][0], scores["MSP"][1])  # flat row = more error-like
        self.assertGreater(scores["entropy"][0], scores["entropy"][1])
        self.assertTrue(all(np.isfinite(value).all() for value in scores.values()))
        for bad in (logits[:, :99], logits[0], np.where(np.eye(6, 100) > 0, np.nan, logits)):
            with self.assertRaises(ValueError):
                linear.analytic_scores(bad)

    def test_gate_binding_hashes_the_evaluation_gate_file(self):
        protocol.atomic_json(self.root / "heads_freeze.json", {"kind": "heads"})
        protocol.atomic_json(self.root / "evaluation_gate.json", {"kind": "gate"})
        with patch.object(protocol, "require_evaluation_gate",
                          return_value={"complete": True, "scope_id": linear.SCOPE}):
            bound = linear._gate_inputs(self.root)
        self.assertEqual(bound["heads_freeze_sha256"], protocol.sha256(self.root / "heads_freeze.json"))
        self.assertEqual(bound["evaluation_gate_sha256"],
                         protocol.sha256(self.root / "evaluation_gate.json"))


# --------------------------------------------------------------------------- statistics
def bootstrap_inputs(photos=12, positive_photo=None, seed=3):
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(photos) * 7 + 100, 9)
    ids = ids[rng.permutation(len(ids))]
    if positive_photo is None:
        labels = rng.integers(0, 2, size=len(ids))
        labels[:2] = [0, 1]
    else:
        labels = (ids == positive_photo).astype(np.int64)
    scores = {key: np.round(rng.normal(size=len(ids)) + 0.8 * labels, 1) for key in ev.SCORE_KEYS}
    return labels, ids, scores


class StatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        protocol.require_slurm()

    def test_tie_aware_auroc_exact_small_cases(self):
        self.assertEqual(WeightedAUC([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])(), 0.75)
        self.assertEqual(WeightedAUC([0, 1, 0, 1], [1.0, 1.0, 2.0, 2.0])(), 0.5)
        self.assertAlmostEqual(WeightedAUC([0, 1, 1, 0, 1], [0.2, 0.2, 0.5, 0.5, 0.9])(), 4 / 6, places=15)
        self.assertEqual(WeightedAUC([0, 1, 0, 1], [3.0, 3.0, 3.0, 3.0])(), 0.5)
        self.assertIsNone(WeightedAUC([1, 1, 1], [0.1, 0.2, 0.3])())
        self.assertIsNone(WeightedAUC([0, 1, 1], [0.1, 0.2, 0.3])([1, 0, 0]))
        weighted = WeightedAUC([0, 1, 1, 0], [0.3, 0.3, 0.9, 0.5])([2, 1, 3, 0])
        self.assertAlmostEqual(weighted, (2 * 1 * 0.5 + 2 * 3 * 1.0) / (4 * 2), places=15)
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(1)
        for trial in range(20):
            labels = rng.integers(0, 2, size=30)
            labels[:2] = [0, 1]
            scores = rng.integers(0, 5, size=30).astype(float)
            weights = rng.integers(0, 3, size=30)
            weights[:2] = 1
            with self.subTest(trial=trial):
                self.assertAlmostEqual(WeightedAUC(labels, scores)(), roc_auc_score(labels, scores), places=12)
                self.assertAlmostEqual(WeightedAUC(labels, scores)(weights),
                                       brute_auc(labels, scores, weights), places=12)
        metrics = ev.score_metrics(np.zeros(5, dtype=int), np.arange(5.0))[0]
        self.assertIsNone(metrics["auroc"])
        self.assertEqual(metrics["auroc_undefined_reason"], "one outcome class")

    def test_family_scores_mean_of_sigmoids_and_columns(self):
        logits = np.array([[-4.0, 0.0, 1.0, 3.0], [2.0, -1.0, 0.5, -3.0]])
        models = {"stack": manual_head([0, 1, 2, 3], [0.0] * 4, [1.0] * 4, [1.0, -1.0, 0.5, 2.0], 0.1),
                  "last_only": manual_head([3], [1.0], [2.0], [-1.5], 0.0)}
        result = ev.family_scores("G", logits, models)
        self.assertEqual(set(result), {"G_block2", "G_block5", "G_block8", "G_block11",
                                       "G_mean", "G_stack", "G_last_only"})
        for column, arm in enumerate(ev.ARMS):
            np.testing.assert_array_equal(result[f"G_{arm}"], logits[:, column])
        expected_mean = (1 / (1 + np.exp(-logits))).mean(axis=1)
        np.testing.assert_allclose(result["G_mean"], expected_mean, rtol=0, atol=1e-15)
        self.assertAlmostEqual(result["G_mean"][0], 0.5504047289, places=8)
        self.assertNotAlmostEqual(result["G_mean"][0], float(sigmoid(logits.mean(axis=1))[0]), places=3)
        np.testing.assert_allclose(result["G_stack"], logits @ [1.0, -1.0, 0.5, 2.0] + 0.1)
        np.testing.assert_allclose(result["G_last_only"], -1.5 * (logits[:, 3] - 1.0) / 2.0)
        np.testing.assert_array_equal(ev.family_scores("O", logits[:, :1])["O"], logits[:, 0])
        for args in (("O", logits[:, :1], models), ("G", logits, None),
                     ("G", logits, {"stack": models["stack"]}), ("H", logits[:, :3], models),
                     ("L", logits, models), ("O", logits, None)):
            with self.subTest(family=args[0]), self.assertRaises(ValueError):
                ev.family_scores(*args)

    def test_intervals_linear_percentiles_and_bonferroni_six(self):
        values = np.array([30.0, 0.0, 100.0, 10.0, 90.0, 20.0, 80.0, 40.0, 70.0, 50.0, 60.0])
        ordinary = ev.intervals(values, primary=False)
        np.testing.assert_allclose(ordinary["interval_95"], [2.5, 97.5], rtol=0, atol=1e-12)
        self.assertNotIn("interval_bonferroni", ordinary)
        self.assertEqual(ordinary["undefined_draw_count"], 0)
        adjusted = ev.intervals(values, primary=True)
        np.testing.assert_allclose(adjusted["interval_95"], [2.5, 97.5], rtol=0, atol=1e-12)
        # linear interpolation at position (n - 1) * q = 10 * q on values spaced by 10
        np.testing.assert_allclose(adjusted["interval_bonferroni"], [100 * 0.05 / 12, 100 - 100 * 0.05 / 12],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(adjusted["bonferroni_quantiles"], [0.05 / 12, 1 - 0.05 / 12],
                                   rtol=0, atol=1e-15)
        self.assertEqual(adjusted["primary_family_size"], 6)
        self.assertAlmostEqual(adjusted["adjusted_individual_nominal_coverage"], 1 - 0.05 / 6, places=15)
        self.assertEqual(adjusted["percentile_method"], "linear")
        withheld = ev.intervals(np.r_[values, np.nan], primary=True)
        self.assertIsNone(withheld["interval_95"])
        self.assertIsNone(withheld["interval_bonferroni"])
        self.assertEqual((withheld["undefined_draw_indices"], withheld["undefined_draw_count"]), ([11], 1))
        for bad in (np.r_[values, np.inf], values.reshape(1, -1), np.array([])):
            with self.assertRaises(ValueError):
                ev.intervals(bad, primary=True)

    def test_exactly_six_primary_contrasts_without_l(self):
        self.assertEqual(len(ev.PRIMARY), 6)
        self.assertEqual(tuple(ev.PRIMARY), tuple(protocol.PRIMARY_CONTRASTS))
        self.assertEqual(len(set(ev.PRIMARY_NAMES)), 6)
        self.assertFalse(any(method in pair for pair in ev.PRIMARY for method in ev.STATIC_METHODS))
        self.assertEqual(set(ev.METHODS), set(protocol.METHODS))
        self.assertEqual(len(ev.METHODS), 25)
        self.assertEqual(len(ev.SCORE_KEYS), 22 * len(SEEDS) + 3)
        self.assertEqual(tuple(ev.SEEDS), SEEDS)
        self.assertEqual(tuple(heads.SEEDS), SEEDS)
        self.assertEqual(ev.STATISTICS["training_seeds"], list(SEEDS))
        self.assertEqual(ev.STATISTICS["primary_family_size"], 6)
        self.assertEqual((ev.BOOTSTRAP_DRAWS, ev.BOOTSTRAP_SEED), (10000, 20260914))
        parameters = inspect.signature(ev.paired_bootstrap).parameters
        self.assertEqual(parameters["rng_seed"].default, 20260914)
        self.assertEqual(parameters["draws"].default, 10000)

    def test_mean_within_seed_difference_and_sample_sd(self):
        rng = np.random.default_rng(8)
        left = rng.uniform(0.6, 0.9, size=len(SEEDS)).tolist()
        right = rng.uniform(0.6, 0.9, size=len(SEEDS)).tolist()
        aucs = {f"G_stack/seed{s}": a for s, a in zip(ev.SEEDS, left)}
        aucs.update({f"S_stack/seed{s}": a for s, a in zip(ev.SEEDS, right)})
        aucs.update({f"O/seed{s}": a for s, a in zip(ev.SEEDS, right)}, L=0.77)
        result = ev._mean_seed_difference(aucs, "G_stack", "S_stack")
        differences = [a - b for a, b in zip(left, right)]
        self.assertAlmostEqual(result["estimate"], statistics.mean(differences), places=15)
        self.assertAlmostEqual(result["paired_difference_sample_sd"], statistics.stdev(differences), places=15)
        self.assertEqual(list(result["per_seed"]), [str(seed) for seed in SEEDS])
        static = ev._mean_seed_difference(aucs, "L", "O")
        np.testing.assert_allclose(list(static["per_seed"].values()), [0.77 - b for b in right])
        aucs[f"S_stack/seed{SEEDS[0]}"] = None
        undefined = ev._mean_seed_difference(aucs, "G_stack", "S_stack")
        self.assertIsNone(undefined["estimate"])
        self.assertIsNone(undefined["paired_difference_sample_sd"])

    def test_method_summary_seed_sd_and_static_not_applicable(self):
        rng = np.random.default_rng(4)
        per_vector = {key: {name: float(rng.random()) for name in ev.METRIC_NAMES}
                      for key in ev.SCORE_KEYS}
        summary = ev.summarize_metrics(per_vector)
        self.assertEqual(set(summary), set(ev.METHODS))
        values = [per_vector[f"H_mean/seed{s}"]["auroc"] for s in ev.SEEDS]
        self.assertAlmostEqual(summary["H_mean"]["auroc"]["mean"], statistics.mean(values), places=14)
        self.assertAlmostEqual(summary["H_mean"]["auroc"]["sample_sd"], statistics.stdev(values), places=14)
        self.assertEqual(summary["H_mean"]["training_seed_observations"], len(SEEDS))
        self.assertEqual(summary["H_mean"]["score_vector_count"], len(SEEDS))
        for method in ev.STATIC_METHODS:
            entry = summary[method]
            self.assertEqual(entry["score_vector_count"], 1)
            self.assertEqual(entry["training_seed_observations"], 0)
            self.assertIs(entry["training_seed_sd_applicable"], False)
            self.assertEqual(entry["auroc"]["mean"], per_vector[method]["auroc"])
            self.assertIsNone(entry["auroc"]["sample_sd"])
            self.assertTrue(entry["auroc"]["sample_sd_reason"].startswith("not applicable"))
        self.assertEqual((summary["L"]["static_fit_count"], summary["MSP"]["static_fit_count"]), (1, 0))
        per_vector[f"O/seed{SEEDS[-1]}"]["auroc"] = None
        broken = ev.summarize_metrics(per_vector)["O"]["auroc"]
        self.assertIsNone(broken["mean"])
        self.assertIsNone(broken["sample_sd"])
        per_vector.pop("L")
        with self.assertRaises(ValueError):
            ev.summarize_metrics(per_vector)


class PairedBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        protocol.require_slurm()

    def run_bootstrap(self, labels, ids, scores, draws=40, **kwargs):
        return ev.paired_bootstrap(labels, scores, ids, draws=draws, expected_photos=12, **kwargs)

    def test_image_groups_shared_draws_and_determinism(self):
        labels, ids, scores = bootstrap_inputs()
        calls = []
        report, groups, counts, arrays = self.run_bootstrap(labels, ids, scores, check_cutoff=lambda: calls.append(1))
        self.assertEqual(calls, [1])
        again = self.run_bootstrap(labels, ids, scores)
        np.testing.assert_array_equal(counts, again[2])
        for name in arrays:
            np.testing.assert_array_equal(arrays[name], again[3][name])
        self.assertEqual(json.dumps(report, sort_keys=True), json.dumps(again[0], sort_keys=True))
        expected_groups, membership = np.unique(ids, return_inverse=True)
        np.testing.assert_array_equal(groups, expected_groups)
        rng = np.random.default_rng(20260914)
        for draw in range(40):
            multiplicity = np.bincount(rng.integers(0, 12, size=12), minlength=12)
            np.testing.assert_array_equal(counts[draw], multiplicity)
        self.assertEqual(counts.shape, (40, 12))
        self.assertTrue(np.all(counts.sum(axis=1) == 12))
        keys = list(arrays["score_keys"])
        self.assertIn("L", keys)
        self.assertFalse(any(key.startswith(("MSP", "entropy", "H_block", "G_last")) for key in keys))
        for draw in (0, 1, 17, 39):
            weights = counts[draw][membership]
            for group in range(12):
                self.assertEqual(len(set(weights[membership == group].tolist())), 1)
            expanded = np.repeat(np.arange(len(labels)), weights)
            for column, key in enumerate(keys):  # the same draw is shared by every method/seed
                expected = WeightedAUC(labels, scores[key])(weights)
                self.assertEqual(arrays["weighted_auroc"][draw, column], expected)
                self.assertAlmostEqual(expected, WeightedAUC(labels[expanded], scores[key][expanded])(),
                                       places=12)
        column = {key: index for index, key in enumerate(keys)}
        for index, (left, right) in enumerate(ev.PRIMARY):
            per_seed = np.stack([arrays["weighted_auroc"][:, column[f"{left}/seed{s}"]]
                                 - arrays["weighted_auroc"][:, column[f"{right}/seed{s}"]]
                                 for s in ev.SEEDS], axis=1)
            np.testing.assert_allclose(arrays["primary_seed_differences"][:, index], per_seed, rtol=0, atol=0)
            np.testing.assert_allclose(arrays["primary_mean_differences"][:, index], per_seed.mean(axis=1),
                                       rtol=0, atol=1e-15)
            point = [WeightedAUC(labels, scores[f"{left}/seed{s}"])()
                     - WeightedAUC(labels, scores[f"{right}/seed{s}"])() for s in ev.SEEDS]
            entry = report["primary"][ev.PRIMARY_NAMES[index]]
            self.assertAlmostEqual(entry["estimate"], statistics.mean(point), places=14)
            self.assertAlmostEqual(entry["paired_difference_sample_sd"], statistics.stdev(point), places=14)
            np.testing.assert_allclose(entry["interval_95"], np.quantile(
                per_seed.mean(axis=1), [0.025, 0.975], method="linear"), rtol=0, atol=1e-15)
            np.testing.assert_allclose(entry["interval_bonferroni"], np.quantile(
                per_seed.mean(axis=1), [0.05 / 12, 1 - 0.05 / 12], method="linear"), rtol=0, atol=1e-15)
            self.assertEqual(set(entry["per_seed_descriptive_intervals"]), {str(seed) for seed in SEEDS})
        secondary = arrays["secondary_L_minus_O_seed_differences"]
        np.testing.assert_allclose(secondary, np.stack(
            [arrays["weighted_auroc"][:, column["L"]] - arrays["weighted_auroc"][:, column[f"O/seed{s}"]]
             for s in ev.SEEDS], axis=1), rtol=0, atol=0)
        self.assertEqual(set(report["primary"]), set(ev.PRIMARY_NAMES))
        self.assertEqual(set(report["secondary"]), {"L_minus_O"})
        self.assertIs(report["secondary"]["L_minus_O"]["primary"], False)
        self.assertNotIn("interval_bonferroni", report["secondary"]["L_minus_O"])
        self.assertEqual(report["group"], "image_id")
        different = self.run_bootstrap(labels, ids, scores, rng_seed=7)[2]
        self.assertFalse(np.array_equal(different, counts))

    def test_one_class_draws_are_recorded_not_redrawn(self):
        labels, ids, scores = bootstrap_inputs(positive_photo=100 + 7 * 5)
        report, groups, counts, arrays = self.run_bootstrap(labels, ids, scores, draws=60)
        positive_group = int(np.flatnonzero(groups == 135)[0])
        missing = np.flatnonzero(counts[:, positive_group] == 0).tolist()
        self.assertGreater(len(missing), 0)
        self.assertEqual(counts.shape, (60, 12))
        rng = np.random.default_rng(20260914)
        for draw in range(60):  # the RNG stream is untouched: nothing was redrawn
            np.testing.assert_array_equal(counts[draw], np.bincount(rng.integers(0, 12, size=12), minlength=12))
        undefined_rows = np.flatnonzero(np.isnan(arrays["weighted_auroc"]).all(axis=1)).tolist()
        self.assertEqual(undefined_rows, missing)
        self.assertFalse(np.isnan(np.delete(arrays["weighted_auroc"], missing, axis=0)).any())
        for name in ev.PRIMARY_NAMES:
            entry = report["primary"][name]
            self.assertIsNotNone(entry["estimate"])
            self.assertIsNone(entry["interval_95"])
            self.assertIsNone(entry["interval_bonferroni"])
            self.assertEqual(entry["undefined_draw_indices"], missing)
            for seed_entry in entry["per_seed_descriptive_intervals"].values():
                self.assertIsNone(seed_entry["interval_95"])
        self.assertIsNone(report["secondary"]["L_minus_O"]["interval_95"])
        self.assertEqual(report["secondary"]["L_minus_O"]["undefined_draw_count"], len(missing))
        self.assertIsNone(ev._contrast_rows(report["primary"])[0]["bonferroni_lower"])

    def test_malformed_bootstrap_inputs_are_rejected(self):
        labels, ids, scores = bootstrap_inputs()
        partial = dict(scores)
        partial.pop("MSP")
        first, last = SEEDS[0], SEEDS[-1]
        extra = {**scores, f"L/seed{first}": scores["L"]}
        short = {**scores, f"O/seed{first}": scores[f"O/seed{first}"][:-1]}
        infinite = {**scores, f"G_mean/seed{last}": np.r_[np.inf, scores[f"G_mean/seed{last}"][1:]]}
        for index, candidate in enumerate((partial, extra, short, infinite)):
            with self.subTest(case=index), self.assertRaises(ValueError):
                self.run_bootstrap(labels, ids, candidate, draws=2)
        for kwargs in ({"draws": 0}, {"draws": 2.0}, {"draws": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ev.paired_bootstrap(labels, scores, ids, expected_photos=12, **kwargs)
        with self.assertRaises(ValueError):
            ev.paired_bootstrap(labels, scores, ids, draws=2, expected_photos=11)
        uneven = ids.copy()
        uneven[np.flatnonzero(ids == 100)[0]] = 107
        with self.assertRaises(ValueError):
            self.run_bootstrap(labels, uneven, scores, draws=2)
        with self.assertRaises(ValueError):
            self.run_bootstrap(np.zeros_like(labels), ids, scores, draws=2)


if __name__ == "__main__":
    protocol.require_slurm()
    unittest.main()
