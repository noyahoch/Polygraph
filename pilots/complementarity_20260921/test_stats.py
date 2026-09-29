"""Small scientific correctness fixtures. Execute only inside Slurm."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .common import atomic_json, require_slurm


def run_tests():
    require_slurm()
    import numpy as np
    from .fusion import FitFailure, fit_combiner, predict_combiner
    from .statistics import (CONTRASTS, DRAWS, PRIMARY_QUANTILES, SECONDARY_QUANTILES,
                             intervals, literal_photo_rows, metrics, ranking_diagnostics,
                             read_draw, within_photo)

    checked = []
    def expect_failure(function, expected):
        try:
            function()
        except expected:
            return
        raise AssertionError("Expected failure was not raised")

    photos = np.repeat(np.arange(400, dtype=np.int64), 9)
    view = np.tile(np.arange(9), 400)
    level = photos % 5
    x = np.column_stack((level.astype(np.float64), 0.25 + 2 * level))
    y = (view < 2 + level).astype(np.int64)
    counts = np.ones(400, dtype=np.int64)
    counts[:100], counts[100:200] = 0, 2
    weight = counts[photos]
    repeated = literal_photo_rows({"image_id": photos}, np.arange(400), counts)
    assert len(repeated) == 3600 and np.count_nonzero(weight == 0) == 900
    assert np.all(photos[repeated].reshape(400, 9) == photos[repeated].reshape(400, 9)[:, :1])
    weighted, literal = fit_combiner(x, y, weight), fit_combiner(x[repeated], y[repeated])
    for field in ("mean", "scale"):
        np.testing.assert_allclose(weighted[field], literal[field], atol=1e-12, rtol=1e-12)
    left, right = predict_combiner(weighted, x), predict_combiner(literal, x)
    np.testing.assert_allclose(left, right, atol=1e-6, rtol=1e-6)
    assert len(np.unique(left)) == 5 and len(np.unique(right)) == 5
    assert np.all(np.diff(np.unique(left)) > 1e-4), "Fixture score groups must stay separated"
    a, b = metrics(y, left, weight), metrics(y[repeated], right[repeated])
    for name in ("auroc", "average_precision"):
        np.testing.assert_allclose(a[name], b[name], atol=1e-12, rtol=0)
    checked.append("zero-weight literal nine-row photo duplication: scaler, decision score, tied AUROC/AP")

    # A separate tie fixture checks metric multiplicity independently of the optimizer.
    tied_y = np.asarray([0, 1, 1, 0, 1, 0], dtype=np.int64)
    tied_score = np.asarray([0., 0., 1., 1., 2., 2.])
    tied_weight = np.asarray([2, 3, 0, 1, 4, 2], dtype=np.int64)
    repeated_ties = np.repeat(np.arange(6), tied_weight)
    a, b = metrics(tied_y, tied_score, tied_weight), metrics(tied_y[repeated_ties], tied_score[repeated_ties])
    for name in ("auroc", "average_precision"):
        np.testing.assert_allclose(a[name], b[name], atol=1e-12, rtol=0)
    assert metrics(np.zeros(3, dtype=int), np.arange(3))["auroc"] is None
    assert metrics(np.zeros(3, dtype=int), np.arange(3))["average_precision"] is None
    assert metrics(np.ones(3, dtype=int), np.arange(3))["average_precision"] == 1.0
    checked.append("independent AUROC/AP tie, zero-weight and undefined-class fixtures")

    expect_failure(lambda: fit_combiner(x, np.zeros(3600, dtype=int), weight), FitFailure)
    expect_failure(lambda: fit_combiner(x, y, weight / 3600), FitFailure)
    malformed = x.copy()
    malformed[0, 0] = np.nan
    expect_failure(lambda: fit_combiner(malformed, y, weight), FitFailure)
    reloaded = json.loads(json.dumps(weighted, allow_nan=False))
    np.testing.assert_array_equal(left, predict_combiner(reloaded, x))
    checked.append("missing class, normalized weights and nonfinite input rejected; model save/reload exact")

    photo_y = np.asarray([1]*4+[0]*5 + [1]+[0]*8 + [0]*9 + [1]*9)
    photo_score = np.asarray([2]*4+[0]*5 + [0, 0]+[1]*7 + [0]*9 + [0]*9, dtype=float)
    within = within_photo(photo_y, photo_score, np.repeat(np.arange(4), 9))
    assert within["eligible_photographs"] == 2 and within["total_photographs"] == 4
    np.testing.assert_allclose(within["value"], (1 + 0.5/8)/2, atol=1e-15, rtol=0)
    order = ranking_diagnostics(np.asarray([0., 1., 1., 2.]), np.asarray([0., 1., 1.00001, 2.]))
    assert order["average_rank_records_differ"] == 2 and order["weighted_fit_tied_records"] == 2
    checked.append("equal-photo all-error/correct pair ranking, half ties and explicit ranking diagnostics")

    assert PRIMARY_QUANTILES == (0.0083333333, 0.9916666667)
    assert SECONDARY_QUANTILES == (0.025, 0.975) and DRAWS == 2000
    points = {"contrasts": {name: {"estimate": -0.314, "primary": primary}
                            for name, (_cohort, _a, _b, primary) in CONTRASTS.items()}}
    completed = [{"draw_id": draw, "contrasts": {name: {"valid": True, "mean": draw/2000,
                   "per_seed": [draw/2000]*3} for name in CONTRASTS}} for draw in range(DRAWS)]
    output = intervals(points, completed)
    for name, (_cohort, _a, _b, primary) in CONTRASTS.items():
        assert output[name]["estimate"] == -0.314
        quantiles = PRIMARY_QUANTILES if primary else SECONDARY_QUANTILES
        np.testing.assert_array_equal(output[name]["interval"],
                                      np.quantile(np.arange(DRAWS)/2000, quantiles, method="linear"))
    completed[17]["contrasts"]["DG-D"] = {"valid": False, "mean": None, "per_seed": [0, None, 0]}
    output = intervals(points, completed)
    assert output["DG-D"]["interval"] is None and output["DG-D"]["invalid_draw_ids"] == [17]
    assert output["C-B"]["interval"] is not None
    expect_failure(lambda: intervals(points, completed[:-1]), RuntimeError)
    checked.append("original point estimate, exact decimal quantiles, fixed 2000 IDs and no dropped failures")

    with tempfile.TemporaryDirectory(prefix="complementarity-stat-tests-") as temporary:
        identity = {"fixture": "fixed-ID-resume"}
        directory = Path(temporary)
        expected = []
        # The first 50 remain in place; continuation reloads them and adds the rest.
        for stop in (50, 60):
            for draw in range(stop):
                path = directory / f"{draw:04d}.json"
                if not path.exists():
                    atomic_json(path, {"draw_id": draw, "identity": identity, "processed": True, "value": draw*7})
                saved = read_draw(path, draw, identity)
                assert saved["value"] == draw*7
            if stop == 50:
                expected = [(directory / f"{draw:04d}.json").read_bytes() for draw in range(50)]
        assert expected == [(directory / f"{draw:04d}.json").read_bytes() for draw in range(50)]
        expect_failure(lambda: read_draw(directory / "0000.json", 1, identity), RuntimeError)
        expect_failure(lambda: read_draw(directory / "0000.json", 0, {"fixture": "changed"}), RuntimeError)
    checked.append("fixed-ID continuation preserves the retained first 50 and rejects changed identity")
    return {"complete": True, "passed": True, "tests": checked,
            "execution": "Slurm only", "scaler_atol_rtol": 1e-12,
            "decision_atol_rtol": 1e-6, "fixture_metric_atol": 1e-12}
