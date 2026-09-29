"""Independent hand-computed scientific checks; execute exclusively on Slurm.

Run: python -m unittest pilots.logit_dynamics_20260919.test_science -v
Small L/K values below are mathematical test fixtures, not experiment arms.
"""
from __future__ import annotations

import copy
import hashlib
import math
import unittest

import numpy as np

from .features import build_features, feature_names, fit_normalizer, normalize
from .protocol import make_roles, require_slurm


def setUpModule():
    require_slurm()


def trajectory(states, final, low=-100.0):
    """Construct independently specified last-block class scores."""
    heads = np.full((1, 12, 100), low, dtype=np.float32)
    # Irrelevant early blocks intentionally have a different winning class.
    heads[:, :, 99] = 80.0
    for position, values in enumerate(states, start=12 - len(states)):
        heads[0, position] = low
        for class_id, value in values.items():
            heads[0, position, class_id] = value
    terminal = np.full((1, 100), low, dtype=np.float32)
    for class_id, value in final.items():
        terminal[0, class_id] = value
    return heads, terminal


class PaperFeatureCases(unittest.TestCase):
    def test_numeric_exclusion_and_all_seven_dynamics(self):
        # Final prediction is class 2 at every numeric depth. Its logit is
        # excluded from competitors, but class 2 belongs to every dynamic set.
        # Raw top-1 identities: 0,1,2. Raw top-two sets: {0,2},{1,2},{2,0}.
        heads, final = trajectory(
            [{0: 3.0, 2: 2.0, 1: 1.0}, {1: 3.0, 2: 2.0, 0: 1.0}],
            {2: 3.0, 0: 2.0, 1: 1.0},
        )
        actual = build_features(heads, final, layers=2, k=2)[0]
        minority_mass = 1.0 / (math.e + 1.0)
        expected = [
            2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 3.0, 2.0, 1.0,
            1.0, minority_mass / (2.0 - minority_mass),
            3.0, 1.0 / 3.0, math.log(3.0), 3.0, 1.0,
        ]
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
        self.assertEqual(feature_names(2, 2)[0], "depth11_final_predicted_class_logit")
        self.assertEqual(feature_names(2, 2)[6], "depthclassifier_final_predicted_class_logit")

    def test_tied_logits_choose_lower_class_id(self):
        # Equal top scores resolve to classes 0 then 1. At the terminal depth,
        # the tie for second place resolves to class 0 rather than class 1.
        # Each adjacent shared mass is exactly 1/2, so Jaccard is 1/3.
        heads, final = trajectory(
            [{0: 2.0, 1: 2.0, 2: 1.0}, {1: 2.0, 2: 2.0, 0: 1.0}],
            {2: 3.0, 0: 2.0, 1: 2.0},
        )
        actual = build_features(heads, final, layers=2, k=2)[0, -7:]
        expected = [1.0, 1.0 / 3.0, 3.0, 1.0 / 3.0, math.log(3.0), 3.0, 1.0]
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_topk_union_counts_members_even_after_softmax_underflow(self):
        # The second class remains in every top-two set even when exp(-1000)
        # underflows. Counting nonzero softmax weights would incorrectly yield 1.
        heads, final = trajectory([{0: 0.0, 1: -1000.0}] * 12,
                                  {0: 0.0, 1: -1000.0}, low=-10000.0)
        actual = build_features(heads, final, layers=12, k=2)[0, -7:]
        np.testing.assert_allclose(actual, [0, 1, 2, 1, 0, 1, 0], atol=1e-7, rtol=0)

    def test_commitment_is_start_of_final_unbroken_suffix(self):
        cases = [([3, 3, 3, 3], 0.0),
                 ([3, 1, 3, 3], 0.5),
                 ([3, 3, 3, 1], 1.0)]
        for identities, expected in cases:
            with self.subTest(identities=identities):
                heads, final = trajectory([{class_id: 2.0} for class_id in identities], {3: 2.0})
                actual = build_features(heads, final, layers=4, k=1)[0, -1]
                self.assertAlmostEqual(float(actual), expected, places=7)

    def test_full_contract_dimensions_and_all_tied_class_order(self):
        heads = np.zeros((1, 12, 100), dtype=np.float32)
        final = np.zeros((1, 100), dtype=np.float32)
        actual = build_features(heads, final)
        self.assertEqual(actual.shape, (1, 85))
        self.assertEqual(len(feature_names()), 85)
        np.testing.assert_allclose(actual[0, -7:], [0, 1, 5, 1, 0, 1, 0], atol=1e-7, rtol=0)


class NormalizationCases(unittest.TestCase):
    def test_train_population_statistics_constant_feature_and_frozen_validation(self):
        training = np.asarray([[1.0, 2.0, 9.0], [3.0, 2.0, 5.0]], dtype=np.float32)
        scaler = fit_normalizer(training)
        original = copy.deepcopy(scaler)
        np.testing.assert_array_equal(scaler["mean"], [2, 2, 7])
        np.testing.assert_array_equal(scaler["scale"], [1, 1, 2])
        np.testing.assert_array_equal(normalize([[100, 2, -1]], scaler), [[98, 0, -4]])
        self.assertEqual(scaler, original)
        self.assertEqual(scaler["records"], 2)
        self.assertEqual(scaler["ddof"], 0)
        wrong_role = {**scaler, "fit_role": "probe_val"}
        with self.assertRaises(ValueError):
            normalize(training, wrong_role)


class SourceAllocationCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_slurm()
        cls.original = {"roles": {
            "base_train": {"photo_ids": list(range(1600))},
            "checkpoint": {"photo_ids": list(range(1600, 2000))},
            "meta": {"photo_ids": list(range(2000, 2400))},
            "dev_eval": {"photo_ids": list(range(2400, 3200))},
        }}
        cls.rows = [
            {"image_id": image_id, "record_id": image_id * 9 + condition,
             # This intentionally repeats across every photograph: grouping
             # by source_id would produce nine groups, not 3,200 photographs.
             "source_id": condition, "severity": 0, "split_id": 0,
             "label": image_id % 100, "pred": (image_id % 100 + image_id % 2) % 100,
             "y": image_id % 2}
            for image_id in range(3200) for condition in range(9)
        ]

    def test_fixed_hash_membership_disjoint_sources_and_nine_views(self):
        result = make_roles(self.rows, self.original)["roles"]
        # Hard-code the documented salt independently of the production constant.
        ordered = sorted(range(1600), key=lambda image_id: (
            hashlib.sha256(f"logit_dynamics_20260919:head_split:v1:{image_id}".encode("utf-8")).hexdigest(),
            image_id,
        ))
        expected_heads = set(ordered[:1200])
        self.assertEqual(set(result["head_train"]["photo_ids"]), expected_heads)
        self.assertEqual(set(result["probe_train"]["photo_ids"]),
                         (set(range(1600)) - expected_heads) | set(range(2000, 2400)))
        self.assertEqual(set(result["probe_val"]["photo_ids"]), set(range(1600, 2000)))
        self.assertEqual(set(result["dev_eval"]["photo_ids"]), set(range(2400, 3200)))
        observed = set()
        for role, count in [("head_train", 1200), ("probe_train", 800),
                            ("probe_val", 400), ("dev_eval", 800)]:
            selected = result[role]
            photographs = set(selected["photo_ids"])
            self.assertFalse(observed & photographs)
            observed |= photographs
            self.assertEqual(len(photographs), count)
            self.assertEqual(selected["records"], 9 * count)
            self.assertEqual(set(selected["record_ids"]),
                             {9 * image_id + view for image_id in photographs for view in range(9)})
        self.assertEqual(observed, set(range(3200)))

    def test_original_list_order_does_not_change_membership(self):
        reversed_roles = copy.deepcopy(self.original)
        for spec in reversed_roles["roles"].values():
            spec["photo_ids"].reverse()
        first = make_roles(self.rows, self.original)
        reordered = make_roles(self.rows, reversed_roles)
        self.assertEqual(first, reordered)

    def test_overlap_or_lost_view_fails(self):
        overlapping = copy.deepcopy(self.original)
        overlapping["roles"]["checkpoint"]["photo_ids"][0] = 0
        with self.assertRaises(RuntimeError):
            make_roles(self.rows, overlapping)
        with self.assertRaises(RuntimeError):
            make_roles(self.rows[:-1], self.original)

    def test_missing_auxiliary_class_or_error_label_fails(self):
        missing_class = [{**row, "label": 0} for row in self.rows]
        with self.assertRaises(RuntimeError):
            make_roles(missing_class, self.original)
        no_errors = [{**row, "y": 0} for row in self.rows]
        with self.assertRaises(RuntimeError):
            make_roles(no_errors, self.original)


def pairwise_auc(labels, scores, weights):
    """Independent weighted positive-negative pair definition, including ties."""
    positive, negative = labels == 1, labels == 0
    denominator = weights[positive].sum() * weights[negative].sum()
    if denominator == 0:
        return float("nan")
    greater = scores[positive, None] > scores[None, negative]
    equal = scores[positive, None] == scores[None, negative]
    pair_weight = weights[positive, None] * weights[None, negative]
    return float(((greater + 0.5 * equal) * pair_weight).sum() / denominator)


class PairedBootstrapCases(unittest.TestCase):
    def test_estimand_averages_seed_aurocs_not_seed_predictions(self):
        from .evaluate import paired_bootstrap
        labels = np.tile([0, 0, 0, 0, 0, 1, 1, 1, 1], 3)
        image_ids = np.repeat([50, 7, 91], 9)
        g = np.column_stack([labels, -labels, labels]).astype(float)
        ld = np.column_stack([-labels, 10 * labels, -labels]).astype(float)
        result, draws = paired_bootstrap(labels, image_ids, g, ld, draws=13, rng_seed=42)
        # Per-seed differences are +1,-1,+1. Averaging scores would instead
        # yield perfect rankings for both methods, giving the incorrect zero.
        self.assertAlmostEqual(result["estimate"], 1.0 / 3.0)
        self.assertEqual(list(result["per_seed"].values()), [1.0, -1.0, 1.0])
        np.testing.assert_allclose(draws["mean_differences"], 1.0 / 3.0)
        np.testing.assert_allclose(result["interval_95"], [1.0 / 3.0] * 2)

    def test_each_draw_uses_shared_photograph_weights_for_all_seeds_and_methods(self):
        from .evaluate import paired_bootstrap
        image_ids = np.repeat([50, 7, 91], 9)
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 1,
                             0, 0, 0, 1, 0, 1, 0, 1, 0,
                             1, 0, 1, 0, 0, 0, 0, 0, 1])
        rank = np.arange(27)
        g = np.column_stack([rank % 5, -(rank % 7), rank % 3]).astype(float)
        ld = np.column_stack([rank % 4, rank % 6, -(rank % 5)]).astype(float)
        result, draws = paired_bootstrap(labels, image_ids, g, ld, draws=13, rng_seed=42)
        self.assertEqual(result["undefined_draw_indices"], [])
        np.testing.assert_array_equal(draws["image_id"], [7, 50, 91])
        np.testing.assert_array_equal(draws["multiplicity"].sum(axis=1), np.full(13, 3))
        self.assertTrue(np.all(draws["multiplicity"] >= 0))
        self.assertTrue(np.any(draws["multiplicity"] != 1))
        for draw, counts in enumerate(draws["multiplicity"]):
            by_photo = dict(zip([7, 50, 91], counts))
            weights = np.asarray([by_photo[int(image_id)] for image_id in image_ids])
            expected = [pairwise_auc(labels, g[:, seed], weights) - pairwise_auc(labels, ld[:, seed], weights)
                        for seed in range(3)]
            np.testing.assert_allclose(draws["per_seed_differences"][draw], expected, atol=1e-14, rtol=0)
            self.assertAlmostEqual(draws["mean_differences"][draw], sum(expected) / 3.0, places=14)
        # source_id names corruption conditions and repeats across photographs.
        with self.assertRaises(ValueError):
            paired_bootstrap(labels, np.tile(np.arange(9), 3), g, ld, draws=2)

    def test_undefined_draws_are_retained_and_withhold_interval(self):
        from .evaluate import paired_bootstrap
        labels = np.repeat([0, 1], 9)
        image_ids = np.repeat([100, 200], 9)
        g = np.column_stack([labels] * 3).astype(float)
        ld = -g
        result, draws = paired_bootstrap(labels, image_ids, g, ld, draws=30, rng_seed=42)
        self.assertEqual(draws["multiplicity"].shape, (30, 2))
        self.assertTrue(result["undefined_draw_indices"])
        self.assertIsNone(result["interval_95"])
        expected_undefined = [i for i, counts in enumerate(draws["multiplicity"]) if 0 in counts]
        self.assertEqual(result["undefined_draw_indices"], expected_undefined)


if __name__ == "__main__":
    require_slurm()
    unittest.main()
