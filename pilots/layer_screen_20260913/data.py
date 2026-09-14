"""Materialize a selected-layer union from sparse per-head attention on demand."""
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np
import torch
from polygraph.data.graphs import node_coordinates
from pilots.topology_20260910.data import GraphData, atomic_torch, verify_cached_file
from .protocol import ARMS, cohort, digest, file_sha256, protocol

def check_freeze(*args, **kwargs):
    raise RuntimeError("This development-only study cannot access test")

def materialize(payload, offset, layers):
    lo, hi = payload["sparse_offsets"][offset:offset+2].tolist()
    flat = payload["sparse_positions"][lo:hi].long()
    values = payload["sparse_values"][lo:hi].float()
    channel, pair = flat.div(197*197, rounding_mode="floor"), flat.remainder(197*197)
    channel_map = torch.full((144,), -1, dtype=torch.long)
    for local, layer in enumerate(layers):
        channel_map[layer*12:(layer+1)*12] = torch.arange(local*12, (local+1)*12)
    channels = channel_map[channel]
    keep = channels >= 0
    pair, channels, values = pair[keep], channels[keep], values[keep]
    unique, inverse = torch.unique(pair, sorted=True, return_inverse=True)
    edge_attr = torch.zeros((len(unique), 12*len(layers)), dtype=torch.float32)
    edge_attr[inverse, channels] = values
    edge_index = torch.stack((unique.remainder(197), unique.div(197, rounding_mode="floor")))
    coords = node_coordinates(197, 0, 1)
    diagonal = payload["diagonals"][offset, layers].permute(1, 0, 2).reshape(197, -1).float()
    x = torch.cat((coords, diagonal, payload["hidden"][offset].float()), dim=1)
    return x, edge_index, edge_attr

class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, cache, split, arm, freeze=None, require_complete=True, diagnostic=False):
        if split not in ("train", "val"):
            raise RuntimeError("Test is forbidden in layer_screen_20260913")
        self.cache, self.arm, self.layers = Path(cache), arm, ARMS[arm]["layers"]
        self.manifest = json.loads((self.cache / "manifest.json").read_text())
        if require_complete and not self.manifest.get("complete"):
            raise RuntimeError("Incomplete cache")
        if self.manifest.get("diagnostic_only") and not diagnostic:
            raise RuntimeError("Diagnostic cache cannot be used for scientific training")
        for name, expected in (("protocol", protocol()), ("cohort", cohort())):
            actual = json.loads((self.cache / f"{name}.json").read_text())
            if digest(actual) != digest(expected) or self.manifest[f"{name}_sha256"] != digest(actual):
                raise RuntimeError(f"Cache {name} identity mismatch")
        verify_cached_file(self.cache / "index.json", self.manifest["index_sha256"])
        all_rows = json.loads((self.cache / "index.json").read_text())
        expected_rows = {r["record_id"]: r for r in cohort()["records"]}
        ids = [r["record_id"] for r in all_rows]
        if len(set(ids)) != len(ids):
            raise RuntimeError("Duplicate cache records")
        for row in all_rows:
            original = expected_rows.get(row["record_id"])
            if original is None or any(row[k] != original[k] for k in original):
                raise RuntimeError("Cache contains foreign/held-out/misaligned record")
        if not diagnostic and set(ids) != set(expected_rows):
            raise RuntimeError("Development cache must contain exactly 28,800 records")
        self.entries = [row for row in all_rows if row["split"] == split]
        self.shards = {Path(s["path"]).name: s["sha256"] for s in self.manifest["shards"]}
        self.loaded = OrderedDict()

    def __len__(self): return len(self.entries)
    def _load(self, name):
        if name not in self.loaded:
            path = self.cache / "shards" / name
            verify_cached_file(path, self.shards[name])
            self.loaded[name] = torch.load(path, map_location="cpu", weights_only=True)
            verify_cached_file(path, self.shards[name])
            while len(self.loaded) > 1: self.loaded.popitem(last=False)
        return self.loaded[name]
    def shard_blocks(self):
        blocks = OrderedDict()
        for i, row in enumerate(self.entries): blocks.setdefault(row["shard"], []).append(i)
        return list(blocks.values())
    def labels(self): return np.asarray([r["y"] for r in self.entries], dtype=np.float32)
    def metadata(self):
        keys = ("record_id", "image_id", "source_id", "severity", "split_id", "y", "pred", "label", "confidence", "margin")
        return {k: np.asarray([r[k] for r in self.entries]) for k in keys}
    def __getitem__(self, index):
        row = self.entries[index]
        payload = self._load(row["shard"])
        offset = row["offset"]
        if int(payload["record_id"][offset]) != row["record_id"]:
            raise RuntimeError("Record/shard alignment changed")
        x, edge_index, edge_attr = materialize(payload, offset, self.layers)
        return GraphData(x=x, edge_index=edge_index, edge_attr=edge_attr,
                         y=torch.tensor([row["y"]], dtype=torch.float32),
                         record_id=torch.tensor([row["record_id"]]), image_id=torch.tensor([row["image_id"]]),
                         source_id=torch.tensor([row["source_id"]]), split_id=torch.tensor([row["split_id"]]),
                         output_logits=payload["logits"][offset].float().reshape(1,100))
