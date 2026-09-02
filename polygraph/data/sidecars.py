"""Aligned, resumable sidecars over the immutable graph-store record order."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch


def store_key_sha256(store_dir: Path) -> str:
    return hashlib.sha256((Path(store_dir) / "store_keys.json").read_bytes()).hexdigest()


def source_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def atomic_torch_save(payload: Any, path: Path) -> None:
    """Durably publish a tensor shard; a `.tmp` is never treated as complete."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_json_save(payload: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def build_manifest(store_dir: Path, model_id: str, feature_schema: Dict[str, Any],
                   dtype: str, command: Sequence[str], layer: Optional[int] = None) -> Dict[str, Any]:
    store = json.loads((Path(store_dir) / "manifest.json").read_text())
    result = {
        "store_key_sha256": store_key_sha256(store_dir),
        "records": int(store["records"]),
        "shard_records": list(map(int, store["shard_records"])),
        "model_id": model_id,
        "feature_schema": feature_schema,
        "dtype": dtype,
        "creation_command": list(command),
        "source_git_commit": source_commit(),
    }
    if layer is not None:
        result["layer"] = int(layer)
    return result


def validate_manifest(sidecar_dir: Path, store_dir: Path, model_id: Optional[str] = None,
                      layer: Optional[int] = None) -> Dict[str, Any]:
    sidecar = json.loads((Path(sidecar_dir) / "manifest.json").read_text())
    store = json.loads((Path(store_dir) / "manifest.json").read_text())
    checks = {
        "store key hash": (sidecar.get("store_key_sha256"), store_key_sha256(store_dir)),
        "record count": (sidecar.get("records"), store.get("records")),
        "shard counts": (sidecar.get("shard_records"), store.get("shard_records")),
    }
    expected_model = model_id if model_id is not None else store.get("model_id")
    if expected_model is not None:
        checks["model ID"] = (sidecar.get("model_id"), expected_model)
    if layer is not None:
        checks["layer"] = (sidecar.get("layer"), int(layer))
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"sidecar {name} mismatch: {actual!r} != {expected!r}")
    return sidecar


class AlignedSidecar:
    """Shard/offset reader with strict manifest and per-shard record checks."""

    def __init__(self, sidecar_dir: Path, store_dir: Path, prefix: str,
                 model_id: Optional[str] = None, layer: Optional[int] = None,
                 cache_shards: int = 2):
        self.sidecar_dir, self.prefix = Path(sidecar_dir), prefix
        self.manifest = validate_manifest(sidecar_dir, store_dir, model_id, layer)
        self.counts = list(map(int, self.manifest["shard_records"]))
        self.bounds = [0]
        for count in self.counts:
            self.bounds.append(self.bounds[-1] + count)
        self.cache_shards = max(1, cache_shards)
        self.cache: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()

    def shard(self, shard_index: int) -> Dict[str, Any]:
        if shard_index not in self.cache:
            path = self.sidecar_dir / f"{self.prefix}_{shard_index:05d}.pt"
            payload = torch.load(path, map_location="cpu")
            if int(payload.get("records", -1)) != self.counts[shard_index]:
                raise ValueError(f"sidecar shard {shard_index} record count mismatch")
            self.cache[shard_index] = payload
            while len(self.cache) > self.cache_shards:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(shard_index)
        return self.cache[shard_index]

    def locate(self, store_index: int) -> Tuple[Dict[str, Any], int]:
        if not 0 <= store_index < self.bounds[-1]:
            raise IndexError(store_index)
        shard_index = bisect_right(self.bounds, store_index) - 1
        return self.shard(shard_index), store_index - self.bounds[shard_index]
