"""Statistical regression checks; numerical execution is Slurm-only."""
import os
import json
import pytest

if not os.environ.get("SLURM_JOB_ID"):
    pytest.skip("Numerical evaluation checks are Slurm-only", allow_module_level=True)

import numpy as np
from pilots.topology_20260910 import evaluate as evaluation


def metadata(groups=4):
    image_id = np.repeat(np.arange(groups), 9)
    y = (np.tile(np.arange(9), groups) % 2).astype(int)
    return {"record_id": np.arange(len(y)), "image_id": image_id,
            "source_id": np.tile([0, 1, 1, 2, 2, 3, 3, 4, 4], groups),
            "severity": np.tile([0, 3, 5, 3, 5, 3, 5, 3, 5], groups),
            "split_id": np.full(len(y), 2), "y": y, "label": np.zeros(len(y), dtype=int),
            "pred": y, "confidence": np.full(len(y), .95), "margin": np.zeros(len(y))}


def contestant_scores(data, arms=evaluation.CORE_ARMS):
    return {f"{arm}/seed{seed}": data["y"].astype(float) for arm in arms for seed in evaluation.SEEDS}


def test_weighted_auc_matches_expanded_rows_and_ties():
    labels, scores, counts = np.array([0, 1, 0, 1]), np.array([0., 0., 1., 2.]), np.array([2, 3, 0, 1])
    weighted = evaluation.WeightedAUC(labels, scores)(counts)
    expanded = evaluation.WeightedAUC(np.repeat(labels, counts), np.repeat(scores, counts))()
    assert weighted == pytest.approx(expanded)
    assert evaluation.WeightedAUC(labels, scores)() == pytest.approx(.625)
    assert evaluation.WeightedAUC(np.zeros(2), np.arange(2))() is None
    assert evaluation.WeightedAUC(np.array([]), np.array([]))() is None


def test_bootstrap_uses_photographs_and_identical_paired_draws():
    data = metadata(6)
    bootstrap, images, counts = evaluation.paired_bootstrap(data, contestant_scores(data), draws=12, seed=17)
    assert bootstrap["group"] == "image_id"
    assert len(images) == 6  # Not the five source_id corruption categories.
    assert counts.shape == (12, 6)
    assert np.all(counts.sum(axis=1) == 6)
    _, repeated_images, repeated_counts = evaluation.paired_bootstrap(data, contestant_scores(data), draws=12, seed=17)
    np.testing.assert_array_equal(images, repeated_images)
    np.testing.assert_array_equal(counts, repeated_counts)
    for comparison in bootstrap["comparisons"]["mixture"].values():
        assert comparison["estimate"] == 0
        assert comparison["interval_95"] == [0., 0.]
        assert comparison["bootstrap_values"] == [0.] * 12


def test_undefined_draws_are_not_replaced_and_withhold_interval():
    data = metadata(2)
    data["y"] = np.repeat([0, 1], 9)
    result, _, counts = evaluation.paired_bootstrap(data, contestant_scores(data), draws=40, seed=3)
    primary = result["comparisons"]["mixture"]["full_graph_minus_full_set"]
    assert counts.shape[0] == 40
    assert 0 < primary["undefined_draw_count"] < 40
    assert primary["interval_95"] is None
    assert len(primary["bootstrap_values"]) == 40
    assert evaluation.practical_conclusion(primary) == "insufficient_support_no_confirmatory_interval"


def test_registered_interaction_is_full_gap_minus_raw_gap():
    aucs = {f"{arm}/seed{seed}": .6 for arm in evaluation.FULL_ARMS for seed in evaluation.SEEDS}
    for seed in evaluation.SEEDS:
        aucs[f"full_graph/seed{seed}"] = .7
        aucs[f"raw_graph/seed{seed}"] = .9
    result = evaluation._contrast_values(aucs, evaluation.FULL_ARMS)
    assert result["full_gap_minus_raw_gap"]["estimate"] == pytest.approx(-.2)
    assert result["full_gap_minus_raw_gap"]["seed_differences"] == pytest.approx([-.2] * 5)


def test_seed_contrast_is_mean_of_auc_differences():
    data = metadata(2)
    scores = contestant_scores(data)
    for seed in evaluation.SEEDS[1:]:
        scores[f"full_graph/seed{seed}"] = 1 - data["y"]
    scores[f"full_graph/seed1"] = data["y"] * 100.
    result, _, _ = evaluation.paired_bootstrap(data, scores, draws=8)
    primary = result["comparisons"]["mixture"]["full_graph_minus_full_set"]
    assert primary["seed_differences"] == [0., -1., -1., -1., -1.]
    assert primary["estimate"] == pytest.approx(-.8)
    averaged_score = np.mean([scores[f"full_graph/seed{seed}"] for seed in evaluation.SEEDS], axis=0)
    assert evaluation.WeightedAUC(data["y"], averaged_score)() == 1.


def test_confidence_support_counts_distinct_photographs_per_outcome():
    insufficient = metadata(199)
    sufficient = metadata(200)
    assert not evaluation.confidence_support(insufficient)["inferential_support"]
    assert evaluation.confidence_support(sufficient)["inferential_support"]
    sufficient["confidence"][sufficient["y"] == 1] = .89
    support = evaluation.confidence_support(sufficient)
    assert support["source_photographs_with_error"] == 0
    assert not support["inferential_support"]


def test_threshold_metrics_missing_classes_and_zero_acceptance():
    data = metadata(2)
    result = evaluation.metrics(data, np.ones(len(data["y"])), threshold=.5)
    assert result["coverage"] == 0.
    assert result["accepted_risk"] is None
    assert result["false_alarm_rate"] == result["error_recall"] == 1.
    only_correct = evaluation.metrics(data, np.zeros(len(data["y"])), threshold=0., mask=data["y"] == 0)
    assert only_correct["auroc"] is None and only_correct["error_recall"] is None
    assert only_correct["coverage"] == 1.


def test_matrix_requires_every_seed_and_explicit_reduction():
    full = {"matrix": "full", "runs": {key: {} for key in evaluation._required_runs(evaluation.FULL_ARMS)}}
    assert evaluation._check_matrix(full) == evaluation.FULL_ARMS
    del full["runs"]["raw_graph/seed27"]
    with pytest.raises(RuntimeError, match="every declared arm"):
        evaluation._check_matrix(full)
    reduced = {"matrix": "primary-only", "runs": {key: {} for key in evaluation._required_runs(evaluation.CORE_ARMS)}}
    with pytest.raises(RuntimeError, match="approval"):
        evaluation._check_matrix(reduced)
    with pytest.raises(RuntimeError, match="approval"):
        evaluation._approval(None)


def test_rewired_training_identity_binds_null_manifest(tmp_path):
    for name in ("protocol.json", "cohort.json", "manifest.json"):
        (tmp_path / name).write_text(json.dumps({"name": name}))
    (tmp_path / "rewire").mkdir()
    null = tmp_path / "rewire/manifest.json"
    null.write_text(json.dumps({"complete": True, "version": 1}))
    admission = tmp_path / "rewire/admission.json"
    admission.write_text(json.dumps({"admitted": True, "version": 1}))
    common = evaluation._cache_identity(tmp_path)
    first = evaluation._cache_identity(tmp_path, "full_rewired")
    assert "rewire_manifest_sha256" not in common
    assert "rewire_manifest_sha256" in first
    null.write_text(json.dumps({"complete": True, "version": 2}))
    assert evaluation._cache_identity(tmp_path) == common
    assert evaluation._cache_identity(tmp_path, "full_rewired") != first
    second = evaluation._cache_identity(tmp_path, "full_rewired")
    admission.write_text(json.dumps({"admitted": True, "version": 2}))
    assert evaluation._cache_identity(tmp_path, "full_rewired") != second


def scoring_fixture(tmp_path, monkeypatch):
    frozen = {"run_root": str(tmp_path / "runs"), "implementation_sha256": "e" * 64,
              "runs": {"full_graph/seed1": {"config_sha256": "c" * 64, "best_sha256": "b" * 64}}}
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(frozen))
    prediction = metadata(2)
    prediction.update(score=prediction["y"].astype(np.float32), logit=prediction["y"].astype(np.float32))
    monkeypatch.setattr(evaluation, "_check_frozen", lambda cache, freeze: frozen)
    monkeypatch.setattr(evaluation, "_verify_cohort", lambda cache, data, split: None)
    monkeypatch.setattr(evaluation, "predict_split", lambda *args, **kwargs: {key: value.copy() for key, value in prediction.items()})
    out = tmp_path / "results"
    path = out / "predictions/full_graph__seed1.npz"
    receipt = path.with_suffix(".receipt.json")
    return freeze_path, out, path, receipt, prediction


def test_orphan_prediction_recovery_keeps_original_npz_bytes(tmp_path, monkeypatch):
    freeze, out, path, receipt, _ = scoring_fixture(tmp_path, monkeypatch)
    first = evaluation.score_one(tmp_path, freeze, out, "full_graph", 1, "cpu")
    original_bytes, original_mtime = path.read_bytes(), path.stat().st_mtime_ns
    receipt.unlink()  # Simulate interruption after the atomic NPZ and before receipt.
    recovered = evaluation.score_one(tmp_path, freeze, out, "full_graph", 1, "cpu")
    assert recovered["recovered_uncommitted_artifact"] is True
    assert recovered["prediction_sha256"] == first["prediction_sha256"]
    assert path.read_bytes() == original_bytes
    assert path.stat().st_mtime_ns == original_mtime
    assert receipt.exists()


def test_orphan_recovery_rejects_even_small_score_change_without_overwrite(tmp_path, monkeypatch):
    freeze, out, path, receipt, prediction = scoring_fixture(tmp_path, monkeypatch)
    evaluation.score_one(tmp_path, freeze, out, "full_graph", 1, "cpu")
    original_bytes = path.read_bytes()
    original_sha = evaluation.file_sha256(path)
    receipt.unlink()
    prediction["score"][0] += np.float32(1e-7)  # Inside restore tolerance, forbidden for artifact replacement.
    with pytest.raises(RuntimeError, match="exact frozen recomputation"):
        evaluation.score_one(tmp_path, freeze, out, "full_graph", 1, "cpu")
    assert path.read_bytes() == original_bytes and evaluation.file_sha256(path) == original_sha
    assert not receipt.exists()


@pytest.mark.parametrize("legacy", [False, True])
def test_orphan_recovery_rejects_wrong_or_missing_provenance(tmp_path, monkeypatch, legacy):
    freeze, out, path, receipt, _ = scoring_fixture(tmp_path, monkeypatch)
    evaluation.score_one(tmp_path, freeze, out, "full_graph", 1, "cpu")
    receipt.unlink()
    artifact = evaluation._arrays(path)
    if legacy:
        artifact.pop("__scoring_provenance")
    else:
        provenance = json.loads(artifact["__scoring_provenance"].item())
        provenance["best_sha256"] = "0" * 64
        artifact["__scoring_provenance"] = np.asarray(json.dumps(provenance, sort_keys=True, separators=(",", ":")))
    evaluation._atomic_npz(path, artifact)
    original_bytes, original_sha = path.read_bytes(), evaluation.file_sha256(path)
    with pytest.raises(RuntimeError, match="matching frozen scoring provenance"):
        evaluation.score_one(tmp_path, freeze, out, "full_graph", 1, "cpu")
    assert path.read_bytes() == original_bytes and evaluation.file_sha256(path) == original_sha
    assert not receipt.exists()


@pytest.mark.parametrize("interval,expected", [([.006, .02], "supports_benefit_exceeding_0.005"),
    ([-.002, .004], "rules_out_benefit_as_large_as_0.005_for_this_comparison"),
    ([.003, .007], "practical_benefit_unresolved"), ([.005, .006], "practical_benefit_unresolved")])
def test_practical_margin_is_registered_not_zero(interval, expected):
    assert evaluation.practical_conclusion({"interval_95": interval}) == expected
