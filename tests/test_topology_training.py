"""Focused trainer checks. Execute only under a Slurm allocation."""
import copy
import json
import os

import pytest

if not os.environ.get("SLURM_JOB_ID"):
    pytest.skip("Numerical trainer checks are Slurm-only", allow_module_level=True)

import numpy as np
import torch

from pilots.topology_20260910 import train


def test_threshold_conservative_ties_and_auroc():
    labels = np.r_[np.zeros(20), np.ones(3)]
    scores = np.r_[np.arange(18), 18., 18., 18., 19., 20.]
    summary = train.validation_threshold(labels, scores)
    assert summary["correct_rank"] == 19
    assert summary["threshold"] == 18.0
    assert summary["false_alarm_count"] == 0
    assert summary["error_alarm_count"] == 2
    assert train.auroc(np.array([0, 1, 0, 1]), np.array([0., 0., 1., 2.])) == pytest.approx(0.625)
    with pytest.raises(ValueError, match="both classes"):
        train.auroc(np.ones(3), np.arange(3))


def test_restoration_tolerance_and_near_tie_report():
    reference = np.array([0., 0.000001, 1., 2.])
    near_tie = np.array([0.000001, 0., 1., 2.])
    report = train.compare_scores(reference, near_tie, threshold=0.0000005)
    assert report["near_tie_order_changes"] == 1
    assert report["near_threshold_decision_changes"] == 2
    with pytest.raises(RuntimeError, match="exceed"):
        train.compare_scores(reference, reference + 0.001)


class ToyCache:
    """Local synthetic logits, never a production cache or benchmark result."""
    def __init__(self, cache, split, arm, freeze=None):
        self.split = split
        self.values = torch.randn(18 if split == "train" else 12, 100,
                                  generator=torch.Generator().manual_seed(123 if split == "train" else 456))
        self.errors = (np.arange(len(self.values)) % 3 == 0).astype(np.float32)

    def __len__(self):
        return len(self.errors)

    def labels(self):
        return self.errors

    def logits(self):
        return self.values

    def metadata(self):
        return {"record_id": np.arange(len(self)), "image_id": np.arange(len(self)),
                "source_id": np.zeros(len(self), dtype=int), "severity": np.zeros(len(self), dtype=int),
                "split_id": np.full(len(self), 0 if self.split == "train" else 1),
                "y": self.errors, "pred": self.errors.astype(int), "label": np.zeros(len(self), dtype=int),
                "confidence": np.full(len(self), .5), "margin": np.zeros(len(self))}

    def shard_blocks(self):
        return [list(range(0, len(self) // 2)), list(range(len(self) // 2, len(self)))]


def test_epoch_boundary_resume_and_artifact_binding(tmp_path, monkeypatch):
    machine = copy.deepcopy(train.protocol())
    machine["training"]["epochs"] = 3
    machine["training"]["logit_batch_size"] = 6
    monkeypatch.setattr(train, "protocol", lambda: machine)
    monkeypatch.setattr(train, "CachedDataset", ToyCache)
    cache = tmp_path / "cache"
    cache.mkdir()
    for name, value in (("protocol.json", machine), ("cohort.json", {"diagnostic": True}),
                        ("manifest.json", {"diagnostic": True, "complete": True})):
        (cache / name).write_text(json.dumps(value))
    direct, resumed = tmp_path / "direct", tmp_path / "resumed"
    direct.mkdir()
    resumed.mkdir()
    monkeypatch.setattr(train, "_STOP_REQUESTED", False)
    assert train._fit(cache, direct, "logit", 1, torch.device("cpu"))
    monkeypatch.setattr(train, "_STOP_REQUESTED", True)
    assert not train._fit(cache, resumed, "logit", 1, torch.device("cpu"))
    assert not (resumed / "complete.json").exists()
    first = torch.load(resumed / "latest.pt", map_location="cpu", weights_only=False)
    assert first["completed_epochs"] == first["sampler"]["epoch"] == 1
    monkeypatch.setattr(train, "_STOP_REQUESTED", False)
    assert train._fit(cache, resumed, "logit", 1, torch.device("cpu"))
    state_a = torch.load(direct / "latest.pt", map_location="cpu", weights_only=False)
    state_b = torch.load(resumed / "latest.pt", map_location="cpu", weights_only=False)
    assert state_a["sampler"] == state_b["sampler"]
    for name in state_a["model"]:
        torch.testing.assert_close(state_a["model"][name], state_b["model"][name], atol=1e-5, rtol=1e-5)
    assert torch.equal(state_a["rng"]["torch"], state_b["rng"]["torch"])
    scores_a, scores_b = np.load(direct / "validation.npz"), np.load(resumed / "validation.npz")
    train.compare_scores(scores_a["score"], scores_b["score"])
    config = json.loads((resumed / "config.json").read_text())
    expected_mean = ToyCache(cache, "train", "logit").values.mean(0)
    torch.testing.assert_close(torch.tensor(config["preprocessing"]["mean"]), expected_mean)
    assert config["preprocessing"]["fit_split"] == "train"
    assert config["pos_weight"] == 2.
    model, restored_config = train.load_run(resumed, "cpu")
    assert not model.training and restored_config == config
    prediction = train.predict_split(resumed, cache, "val", "cpu")
    train.compare_scores(scores_b["score"], prediction["score"])
    with pytest.raises(RuntimeError, match="frozen manifest"):
        train.predict_split(resumed, cache, "test", "cpu")
    metadata = json.loads((resumed / "validation.json").read_text())
    metadata["threshold"] += 1
    (resumed / "validation.json").write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="artifact changed"):
        train.load_run(resumed, "cpu")


def test_error_label_semantics():
    dataset = ToyCache(None, "train", "logit")
    train._check_labels(dataset)
    dataset.metadata = lambda: {"pred": np.zeros(len(dataset)), "label": np.zeros(len(dataset)), "y": dataset.errors}
    with pytest.raises(ValueError, match="predicted class"):
        train._check_labels(dataset)
