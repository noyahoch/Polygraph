from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polygraph.data.sidecars import (AlignedSidecar, atomic_torch_save, build_manifest,
                                     store_key_sha256, validate_manifest)


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


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test(); print(f"[PASS] {test.__name__}")
    print(f"All {len(tests)} tests passed.")
