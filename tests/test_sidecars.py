from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polygraph.data.sidecars import (AlignedSidecar, atomic_torch_save, build_manifest,
                                     compact_class_evidence, derive_attention_edge_features, store_key_sha256,
                                     validate_manifest, value_message_statistics)


def fixture(root: Path):
    store, side = root / "store", root / "side"
    store.mkdir(); side.mkdir()
    (store / "store_keys.json").write_text(json.dumps([["clean_test", 0, i] for i in range(3)]))
    (store / "manifest.json").write_text(json.dumps(
        {"records": 3, "shard_records": [2, 1], "model_id": "model"}))
    manifest = build_manifest(store, "model", {"x": ["N", 4]}, "float16", ["test"], 12)
    (side / "manifest.json").write_text(json.dumps(manifest))
    atomic_torch_save({"records": 2, "x": torch.ones(2, 4)}, side / "x_00000.pt")
    atomic_torch_save({"records": 1, "x": torch.zeros(1, 4)}, side / "x_00001.pt")
    return store, side


def test_alignment_and_offsets():
    with tempfile.TemporaryDirectory() as tmp:
        store, side = fixture(Path(tmp))
        reader = AlignedSidecar(side, store, "x", model_id="model", layer=12)
        payload, offset = reader.locate(2)
        assert offset == 0 and payload["x"].shape == (1, 4)


def test_hash_and_shard_count_mutations_are_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        store, side = fixture(Path(tmp))
        data = json.loads((side / "manifest.json").read_text())
        data["store_key_sha256"] = "wrong"
        (side / "manifest.json").write_text(json.dumps(data))
        try:
            validate_manifest(side, store, "model", 12)
            raise AssertionError("hash mutation accepted")
        except ValueError as exc:
            assert "hash" in str(exc)
        data["store_key_sha256"] = store_key_sha256(store)
        data["shard_records"] = [3]
        (side / "manifest.json").write_text(json.dumps(data))
        try:
            validate_manifest(side, store, "model", 12)
            raise AssertionError("shard-count mutation accepted")
        except ValueError as exc:
            assert "shard counts" in str(exc)


def test_wrong_layer_and_atomic_resume():
    with tempfile.TemporaryDirectory() as tmp:
        store, side = fixture(Path(tmp))
        try:
            validate_manifest(side, store, "model", 11)
            raise AssertionError("wrong layer accepted")
        except ValueError as exc:
            assert "layer" in str(exc)
        target = side / "atomic.pt"
        atomic_torch_save({"records": 1}, target)
        assert target.exists() and not target.with_suffix(".pt.tmp").exists()


def test_message_norm_algebra():
    torch.manual_seed(4)
    value = torch.randn(3, 7, 12)
    weight = torch.randn(12, 12)
    direction = torch.randn(3, 12)
    raw, projected, support = value_message_statistics(value, weight, direction, heads=3)
    explicit = []
    for h in range(3):
        block = weight[:, h * 4:(h + 1) * 4]
        explicit.append((value.reshape(3, 7, 3, 4)[:, :, h] @ block.T).norm(dim=-1))
    assert torch.allclose(projected, torch.stack(explicit, -1), atol=2e-5, rtol=2e-5)
    assert raw.shape == support.shape == (3, 7, 3)


def test_edge_features_use_source_not_target():
    edge_index = torch.tensor([[2], [1]])  # 2 -> 1
    attention = torch.tensor([[0.5, 0.25]])
    projected = torch.tensor([[1., 1.], [10., 10.], [2., 4.]])
    support = torch.tensor([[0., 0.], [9., 9.], [-2., 3.]])
    features = derive_attention_edge_features(attention, edge_index, projected, support,
                                              "evidence_flow")
    assert torch.allclose(features[0, 2:4], torch.log1p(torch.tensor([1., 1.])))
    assert torch.allclose(features[0, 4:6], torch.asinh(torch.tensor([-1., .75])))


def test_compact_evidence_uses_predicted_and_runner_not_true_class():
    import inspect
    from torch import nn

    torch.manual_seed(8)
    hidden = torch.randn(2, 5, 6)
    predicted, runner = torch.tensor([1, 3]), torch.tensor([2, 0])
    layernorm, classifier = nn.LayerNorm(6), nn.Linear(6, 4)
    features, normalized = compact_class_evidence(hidden, predicted, runner, layernorm, classifier)
    assert features.shape == (2, 5, 4) and normalized.shape == hidden.shape
    assert "true" not in inspect.signature(compact_class_evidence).parameters
    changed, _ = compact_class_evidence(hidden, runner, predicted, layernorm, classifier)
    assert not torch.allclose(features, changed)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test(); print(f"[PASS] {test.__name__}")
    print(f"All {len(tests)} tests passed.")
