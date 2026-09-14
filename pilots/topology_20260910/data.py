"""Final-layer-only cache and a shared PyG dataset for all matched arms."""
from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from polygraph.data.sidecars import derive_attention_edge_features
from .protocol import ARMS, SPLIT_NAMES, digest, file_sha256, protocol
from .rewiring_decision import admission_bindings, read_admission


_VERIFIED_FILES = {}


def verify_cached_file(path, expected_sha256):
    """Hash once per process and file identity; changes force reverification.

    Stat checks are cheap on repeated graph access. A matching path is insufficient:
    replacing an inode or changing size/mtime/ctime invalidates its cached proof.
    """
    path = Path(path).resolve()
    def fingerprint():
        info = path.stat()
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    before = fingerprint()
    key = str(path)
    if _VERIFIED_FILES.get(key) != (before, expected_sha256):
        if not expected_sha256 or file_sha256(path) != expected_sha256:
            raise RuntimeError(f"Cached tensor checksum mismatch: {path.name}")
        if before != fingerprint():
            raise RuntimeError(f"Cached tensor changed during verification: {path.name}")
        _VERIFIED_FILES[key] = (before, expected_sha256)
    return before


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


class GraphData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in {"record_id", "store_index", "image_id", "source_id", "split_id"}:
            return 0
        return super().__inc__(key, value, *args, **kwargs)


def check_freeze(cache, freeze):
    if freeze is None:
        raise RuntimeError("Test access requires an explicit frozen manifest.")
    cache = Path(cache)
    frozen = json.loads(Path(freeze).read_text())
    if frozen.get("frozen") is not True:
        raise RuntimeError("Manifest is not frozen.")
    manifest = json.loads((cache / "manifest.json").read_text())
    if not manifest.get("complete") or manifest.get("diagnostic_only"):
        raise RuntimeError("Test scoring requires a complete, non-diagnostic feature cache")
    expected = {"protocol_sha256": digest(json.loads((cache / "protocol.json").read_text())),
                "cohort_sha256": digest(json.loads((cache / "cohort.json").read_text())),
                "cache_manifest_sha256": file_sha256(cache / "manifest.json")}
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise RuntimeError(f"Frozen manifest mismatch: {key}")
    run_root = Path(frozen["run_root"])
    if not run_root.is_absolute():
        run_root = Path(freeze).resolve().parent / run_root
    runs = frozen.get("runs")
    if not isinstance(runs, dict) or not runs:
        raise RuntimeError("Freeze must bind all completed runs")
    artifacts = {"config_sha256": "config.json", "best_sha256": "best.safetensors",
                 "validation_npz_sha256": "validation.npz", "validation_json_sha256": "validation.json",
                 "complete_sha256": "complete.json"}
    for relative, bindings in runs.items():
        directory = (run_root / relative).resolve()
        if not directory.is_relative_to(run_root.resolve()):
            raise RuntimeError("Frozen run path escapes run_root")
        for key, filename in artifacts.items():
            if not bindings.get(key) or file_sha256(directory / filename) != bindings[key]:
                raise RuntimeError(f"Frozen run artifact mismatch: {relative}/{filename}")
    if any(relative.startswith("full_rewired/") for relative in runs):
        if frozen.get("rewire_manifest_sha256") != file_sha256(cache / "rewire" / "manifest.json"):
            raise RuntimeError("Frozen rewiring manifest mismatch")
        for key, value in admission_bindings(cache).items():
            if frozen.get(key) != value:
                raise RuntimeError("Frozen rewiring admission/decision mismatch")
    return frozen


class CachedDataset(torch.utils.data.Dataset):
    """Same immutable tensors/order for every arm; topology changes only for rewired."""

    def __init__(self, cache, split, arm, freeze=None, require_complete=True):
        self.cache = Path(cache)
        self.arm = arm
        self.spec = ARMS[arm]
        self.split = split
        if split not in SPLIT_NAMES:
            raise ValueError(split)
        if split == "test":
            check_freeze(self.cache, freeze)
        self.manifest = json.loads((self.cache / "manifest.json").read_text())
        if require_complete and not self.manifest.get("complete"):
            raise RuntimeError("Feature cache is not complete.")
        if self.manifest["protocol_sha256"] != digest(json.loads((self.cache / "protocol.json").read_text())):
            raise RuntimeError("Cache/protocol identity mismatch")
        if self.manifest["cohort_sha256"] != digest(json.loads((self.cache / "cohort.json").read_text())):
            raise RuntimeError("Cache/cohort identity mismatch")
        if file_sha256(self.cache / "index.json") != self.manifest["index_sha256"]:
            raise RuntimeError("Record-index checksum mismatch")
        self.index = json.loads((self.cache / "index.json").read_text())
        self.entries = [e for e in self.index if e["split"] == split]
        if not self.entries:
            raise ValueError(f"No {split} records in cache")
        self._shards = OrderedDict()
        self._rewired = OrderedDict()
        self._loaded_fingerprints = {}
        self._shard_hashes = {Path(entry["path"]).name: entry["sha256"] for entry in self.manifest["shards"]}
        self._rewire_hashes = {}
        if self.spec["rewired"]:
            rewire = json.loads((self.cache / "rewire" / "manifest.json").read_text())
            if rewire["cache_manifest_sha256"] != file_sha256(self.cache / "manifest.json"):
                raise RuntimeError("Rewiring cache is absent, incomplete, or misaligned")
            if rewire.get("schema_version") != 2 or rewire.get("configuration") != protocol()["rewire"]:
                raise RuntimeError("Rewiring cache protocol/gating schema mismatch")
            self.rewiring_admission = read_admission(self.cache)
            self._rewire_hashes = {Path(entry["path"]).name: entry["sha256"] for entry in rewire["shards"]}
            if set(self._rewire_hashes) != set(self._shard_hashes):
                raise RuntimeError("Original/rewired shard inventories disagree")

    def __len__(self):
        return len(self.entries)

    def shard_blocks(self):
        blocks = {}
        for position, entry in enumerate(self.entries):
            blocks.setdefault(entry["shard"], []).append(position)
        return list(blocks.values())

    def _load(self, filename, rewired=False):
        memory = self._rewired if rewired else self._shards
        hashes = self._rewire_hashes if rewired else self._shard_hashes
        if Path(filename).name != filename or filename not in hashes:
            raise RuntimeError("Unrecognized cache shard")
        path = self.cache / ("rewire" if rewired else "shards") / filename
        fingerprint = verify_cached_file(path, hashes[filename])
        key = (rewired, filename)
        if filename not in memory or self._loaded_fingerprints.get(key) != fingerprint:
            memory[filename] = torch.load(path, map_location="cpu", weights_only=True)
            if verify_cached_file(path, hashes[filename]) != fingerprint:
                raise RuntimeError("Cached tensor changed during load")
            self._loaded_fingerprints[key] = fingerprint
            while len(memory) > 1:
                memory.popitem(last=False)
        return memory[filename]

    def labels(self):
        return np.asarray([e["y"] for e in self.entries], dtype=np.float32)

    def metadata(self):
        names = ("record_id", "image_id", "source_id", "severity", "split_id", "y", "pred", "label", "confidence", "margin")
        return {name: np.asarray([e[name] for e in self.entries]) for name in names}

    def logits(self):
        rows = []
        for entry in self.entries:
            shard = self._load(entry["shard"])
            if int(shard["record_id"][entry["offset"]]) != entry["record_id"]:
                raise RuntimeError("Logit index/shard source identity mismatch")
            rows.append(shard["logits"][entry["offset"]])
        return torch.stack(rows).float()

    def __getitem__(self, position):
        entry = self.entries[position]
        shard = self._load(entry["shard"])
        offset = entry["offset"]
        if int(shard["record_id"][offset]) != entry["record_id"]:
            raise RuntimeError("Index/shard source identity mismatch")
        start, stop = map(int, shard["edge_offsets"][offset:offset + 2])
        edge_index = shard["edge_index"][:, start:stop].long().clone()
        raw_edges = shard["edge_attr"][start:stop].float()
        base = shard["base_x"][offset].float()
        x = torch.cat([base, shard["hidden"][offset].float()], 1) if self.spec["full"] else base
        if self.spec["rewired"]:
            rewired = self._load(entry["shard"], rewired=True)
            if int(rewired["record_id"][offset]) != entry["record_id"]:
                raise RuntimeError("Rewired/original source identity mismatch")
            if int(rewired["edge_offsets"][offset]) != start or int(rewired["edge_offsets"][offset + 1]) != stop:
                raise RuntimeError("Rewiring edge-offset mismatch")
            edge_index[1] = rewired["targets"][start:stop].long()
        attr = derive_attention_edge_features(raw_edges, edge_index,
                    shard["projected_value_norm"][offset].float(),
                    shard["decision_support_proxy"][offset].float(),
                    "evidence_flow" if self.spec["full"] else "attention")
        return GraphData(x=x, edge_index=edge_index, edge_attr=attr,
                         y=torch.tensor([entry["y"]], dtype=torch.float32),
                         record_id=torch.tensor([entry["record_id"]]), store_index=torch.tensor([entry["record_id"]]),
                         image_id=torch.tensor([entry["image_id"]]), source_id=torch.tensor([entry["source_id"]]),
                         severity=torch.tensor([entry["severity"]]), split_id=torch.tensor([entry["split_id"]]),
                         pred=torch.tensor([entry["pred"]]), label=torch.tensor([entry["label"]]),
                         confidence=torch.tensor([entry["confidence"]], dtype=torch.float32),
                         margin=torch.tensor([entry["margin"]], dtype=torch.float32),
                         output_logits=shard["logits"][offset].float().unsqueeze(0))
