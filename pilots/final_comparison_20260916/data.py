"""Role-isolated, read-only views of the immutable cache and hidden sidecar.

H exposes coordinates and its own hidden layer only. S discards incidence before
batching: its artificial endpoints identify only the owning graph. O never
materializes attention graphs. Diagnostic views are fixed, small and non-fittable.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch

from polygraph.data.graphs import node_coordinates
from pilots.layer_screen_20260913.data import materialize as graph_materialize
from pilots.layer_screen_20260913.protocol import digest
from pilots.topology_20260910.data import GraphData, atomic_torch, verify_cached_file

from .models import GRAPH_INDEX, HIDDEN_INDEX, validate_arm
from .protocol import (
    ROLES, SCOPE, campaign_sha, read, read_campaign, require_evaluation_gate,
    require_slurm, role_spec, sha256,
)

METADATA = ("record_id", "image_id", "source_id", "severity", "split_id", "y", "label", "pred")
DIAGNOSTIC_ROLES = ("base_train", "checkpoint")
DIAGNOSTIC_ROWS_PER_ROLE = 48
HIDDEN_INDICES = (3, 6, 9)
SIDECAR_SCHEMA = {
    "schema_version": 1,
    "sidecar_kind": "hidden_states_3_6_9",
    "scope_id": SCOPE,
    "hidden_indices": list(HIDDEN_INDICES),
    "tokens": 197,
    "hidden_dim": 768,
    "storage_dtype": "float16",
    "h12_source": "immutable cache; parity reference only",
    "coordinates": "row,col,CLS,constant_zero; same as the single-layer G inputs",
}


def _role_guard(root, role, diagnostic=False):
    if role not in ROLES:
        raise ValueError("Only frozen development roles are supported; original test is closed")
    if diagnostic and role not in DIAGNOSTIC_ROLES:
        raise RuntimeError("Diagnostic loading is restricted to base_train and checkpoint")
    if role == "dev_eval":
        require_evaluation_gate(root)


def _inside(directory, name):
    directory = Path(directory).resolve()
    path = directory / name
    if not path.resolve().is_relative_to(directory):
        raise RuntimeError("Cache/sidecar path escapes its registered directory")
    return path


def cache_index(root):
    """Validate metadata without opening any tensor shard."""
    require_slurm()
    campaign = read_campaign(root)
    cache = Path(campaign["cache"])
    manifest = read(cache / "manifest.json")
    verify_cached_file(cache / "index.json", manifest["index_sha256"])
    group, feature_protocol = read(cache / "cohort.json"), read(cache / "protocol.json")
    if (digest(group) != campaign["cache_cohort_sha256"]
            or digest(feature_protocol) != campaign["cache_protocol_sha256"]):
        raise RuntimeError("Immutable cache cohort/feature protocol changed")
    rows = read(cache / "index.json")
    original = group["records"]
    if len(rows) != 28800 or len(original) != len(rows):
        raise RuntimeError("The immutable cache must contain exactly 28,800 development records")
    shards = {}
    for spec in manifest["shards"]:
        name = Path(spec["path"]).name
        if spec["path"] != "shards/" + name or name in shards:
            raise RuntimeError("Invalid or duplicate immutable shard path")
        shards[name] = spec
    seen, offsets = set(), {}
    for row, expected in zip(rows, original):
        record = row["record_id"]
        if type(record) is not int or record in seen:
            raise RuntimeError("Duplicate or invalid immutable record ID")
        seen.add(record)
        if any(row.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Cache index record membership/order changed")
        if row["split"] not in ("train", "val") or row["shard"] not in shards:
            raise RuntimeError("Foreign split or shard in development cache")
        if any(type(row[key]) is not int for key in METADATA):
            raise RuntimeError("Immutable record metadata must be integer-valued")
        if not (0 <= row["label"] < 100 and 0 <= row["pred"] < 100
                and row["y"] == int(row["pred"] != row["label"])):
            raise RuntimeError("The target must remain frozen classifier error")
        if type(row["offset"]) is not int or row["offset"] < 0:
            raise RuntimeError("Invalid immutable shard offset")
        offsets.setdefault(row["shard"], []).append(row["offset"])
    if set(offsets) != set(shards):
        raise RuntimeError("Immutable shard/index inventories disagree")
    for name, values in offsets.items():
        if values != list(range(shards[name]["records"])):
            raise RuntimeError("Immutable shard offsets are not a complete ordered partition")
    verify_cached_file(cache / "index.json", manifest["index_sha256"])
    return campaign, manifest, rows


def select_role(root, role, rows, *, diagnostic=False):
    _role_guard(root, role, diagnostic)
    spec = role_spec(root, role)
    wanted = set(spec["record_ids"])
    selected = [row for row in rows if row["record_id"] in wanted]
    counts = Counter(row["image_id"] for row in selected)
    if ([row["record_id"] for row in selected] != spec["record_ids"]
            or set(counts) != set(spec["photo_ids"])
            or any(count != 9 for count in counts.values())
            or any(row["split"] != spec["original_split"] for row in selected)):
        raise RuntimeError("Role order, photograph grouping or original split changed")
    return selected[:DIAGNOSTIC_ROWS_PER_ROLE] if diagnostic else selected


def diagnostic_rows(root, rows=None):
    require_slurm()
    if rows is None:
        _, _, rows = cache_index(root)
    selected = {row["record_id"] for role in DIAGNOSTIC_ROLES
                for row in select_role(root, role, rows, diagnostic=True)}
    return [row for row in rows if row["record_id"] in selected]


def metadata_arrays(rows):
    require_slurm()
    return {key: np.asarray([row[key] for row in rows], dtype=np.int64) for key in METADATA}


def role_metadata(root, role):
    require_slurm()
    _role_guard(root, role)
    _, _, rows = cache_index(root)
    return metadata_arrays(select_role(root, role, rows))


def validate_metadata(root, role, values, *, expected=None):
    require_slurm()
    _role_guard(root, role)
    expected = role_metadata(root, role) if expected is None else expected
    for key in METADATA:
        value = np.asarray(values[key])
        if value.dtype != np.int64 or value.shape != expected[key].shape:
            raise RuntimeError("Prediction metadata dtype/shape changed: " + key)
        if not np.array_equal(value, expected[key]):
            raise RuntimeError("Prediction records were reordered or relabeled: " + key)
    if not np.array_equal(values["y"], values["pred"] != values["label"]):
        raise RuntimeError("Prediction targets must remain frozen classifier errors")


def sidecar_identity(root, campaign, cache_manifest, rows, diagnostic):
    return {
        **SIDECAR_SCHEMA,
        "campaign_sha256": campaign_sha(root),
        "cache_manifest_sha256": campaign["cache_manifest_sha256"],
        "cache_index_sha256": cache_manifest["index_sha256"],
        "cache_cohort_sha256": campaign["cache_cohort_sha256"],
        "cache_protocol_sha256": campaign["cache_protocol_sha256"],
        "roles_sha256": campaign["roles_sha256"],
        "source_identity": campaign["source_identity"],
        "diagnostic_only": bool(diagnostic),
        "records": len(rows),
        "record_ids_sha256": digest([row["record_id"] for row in rows]),
        "diagnostic_rows_per_role": DIAGNOSTIC_ROWS_PER_ROLE if diagnostic else None,
        "diagnostic_roles": list(DIAGNOSTIC_ROLES) if diagnostic else [],
    }


def hidden_inventory(root, campaign, cache_manifest, cache_rows, diagnostic=False):
    hidden = Path(root) / "preflight" / "hidden" if diagnostic else Path(campaign["hidden_root"])
    if hidden.resolve() == Path(campaign["cache"]).resolve() or hidden.is_symlink():
        raise RuntimeError("Hidden sidecars must not redirect into the immutable cache")
    rows = diagnostic_rows(root, cache_rows) if diagnostic else cache_rows
    identity = sidecar_identity(root, campaign, cache_manifest, rows, diagnostic)
    manifest = read(hidden / "manifest.json")
    if any(manifest.get(key) != value for key, value in identity.items()):
        raise RuntimeError("Hidden sidecar source/role/index/schema identity changed")
    if manifest.get("complete") is not True or manifest.get("completed_records") != len(rows):
        raise RuntimeError("Only a complete, source-bound hidden sidecar may be loaded")
    verify_cached_file(hidden / "index.json", manifest["index_sha256"])
    index = read(hidden / "index.json")
    if [row["record_id"] for row in index] != [row["record_id"] for row in rows]:
        raise RuntimeError("Hidden sidecar record order changed")
    shards, offsets = {}, {}
    for spec in manifest["shards"]:
        name = Path(spec["path"]).name
        if spec["path"] != "shards/" + name or name in shards:
            raise RuntimeError("Invalid hidden shard inventory")
        shards[name] = spec
    role_of = {record: role for role in ROLES
               for record in role_spec(root, role)["record_ids"]}
    for entry, original in zip(index, rows):
        if (entry["cache_shard"] != original["shard"]
                or entry["cache_offset"] != original["offset"]
                or entry["role"] != role_of[entry["record_id"]]
                or entry["hidden_shard"] not in shards):
            raise RuntimeError("Hidden sidecar points to a different immutable record")
        offsets.setdefault(entry["hidden_shard"], []).append(entry["hidden_offset"])
    if set(offsets) != set(shards):
        raise RuntimeError("Hidden sidecar index/shard inventories disagree")
    for name, values in offsets.items():
        if values != list(range(shards[name]["records"])):
            raise RuntimeError("Hidden offsets do not partition the sidecar shard")
    return hidden, manifest, {row["record_id"]: row for row in index}, shards


class RoleDataset(torch.utils.data.Dataset):
    def __init__(self, root, role, family, arm, *, diagnostic=False):
        require_slurm()
        _role_guard(root, role, diagnostic)
        validate_arm(family, arm)
        self.root, self.role = Path(root).resolve(), role
        self.family, self.arm, self.diagnostic = family, arm, bool(diagnostic)
        self.campaign, self.manifest, rows = cache_index(self.root)
        self.cache = Path(self.campaign["cache"])
        self.entries = select_role(self.root, role, rows, diagnostic=diagnostic)
        self.shards = {Path(spec["path"]).name: spec for spec in self.manifest["shards"]}
        self.loaded, self.loaded_hidden, self.fingerprints = OrderedDict(), OrderedDict(), {}
        self.hidden = self.hidden_manifest = self.hidden_entries = self.hidden_shards = None
        if diagnostic or (family == "H" and HIDDEN_INDEX[arm] != 12):
            (self.hidden, self.hidden_manifest, self.hidden_entries,
             self.hidden_shards) = hidden_inventory(
                self.root, self.campaign, self.manifest, rows, diagnostic)
        self.feature_binding = {
            "cache_manifest_sha256": self.campaign["cache_manifest_sha256"],
            "cache_index_sha256": self.manifest["index_sha256"],
            "family": family, "arm": arm, "diagnostic_only": self.diagnostic,
        }
        if family == "H" and HIDDEN_INDEX[arm] != 12:
            self.feature_binding["hidden_manifest_sha256"] = sha256(self.hidden / "manifest.json")
            self.feature_binding["hidden_index_sha256"] = self.hidden_manifest["index_sha256"]
        self._logits = None

    def __len__(self):
        return len(self.entries)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(loaded=OrderedDict(), loaded_hidden=OrderedDict(), fingerprints={}, _logits=None)
        return state

    def labels(self):
        require_slurm()
        return np.asarray([row["y"] for row in self.entries], dtype=np.float32)

    def metadata(self):
        return metadata_arrays(self.entries)

    def shard_blocks(self):
        blocks = OrderedDict()
        for index, row in enumerate(self.entries):
            blocks.setdefault(row["shard"], []).append(index)
        return list(blocks.values())

    def _load(self, name):
        require_slurm()
        if name not in self.shards or Path(name).name != name:
            raise RuntimeError("Unknown immutable cache shard")
        path, spec = _inside(self.cache / "shards", name), self.shards[name]
        before = verify_cached_file(path, spec["sha256"])
        key = ("cache", name)
        if name not in self.loaded or self.fingerprints.get(key) != before:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if verify_cached_file(path, spec["sha256"]) != before:
                raise RuntimeError("Immutable shard changed during loading")
            needed = (("record_id", "logits") if self.family == "O"
                      else ("record_id", "hidden") if self.family == "H"
                      else ("record_id", "hidden", "diagonals", "sparse_offsets",
                            "sparse_positions", "sparse_values"))
            payload = {name: payload[name] for name in needed}
            if payload["record_id"].dtype != torch.int64 or payload["record_id"].shape != (spec["records"],):
                raise RuntimeError("Invalid immutable shard record IDs")
            self.loaded[name], self.fingerprints[key] = payload, before
            while len(self.loaded) > 1:
                self.loaded.popitem(last=False)
        return self.loaded[name]

    def _load_hidden(self, name):
        require_slurm()
        if self.hidden is None or name not in self.hidden_shards or Path(name).name != name:
            raise RuntimeError("Unknown or unauthorized hidden shard")
        spec = self.hidden_shards[name]
        path = _inside(self.hidden / "shards", name)
        before = verify_cached_file(path, spec["sha256"])
        receipt = path.with_suffix(".json")
        verify_cached_file(receipt, spec["receipt_sha256"])
        key = ("hidden", name)
        if name not in self.loaded_hidden or self.fingerprints.get(key) != before:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if verify_cached_file(path, spec["sha256"]) != before:
                raise RuntimeError("Hidden sidecar changed during loading")
            if set(payload) != {"record_id", "hidden_3", "hidden_6", "hidden_9"}:
                raise RuntimeError("Hidden sidecars must not contain attention, logits or outcome features")
            if (payload["record_id"].dtype != torch.int64
                    or payload["record_id"].shape != (spec["records"],)):
                raise RuntimeError("Invalid hidden record IDs")
            for layer in HIDDEN_INDICES:
                value = payload[f"hidden_{layer}"]
                if value.shape != (spec["records"], 197, 768) or value.dtype != torch.float16:
                    raise RuntimeError("Unexpected hidden layer shape/dtype")
                if not torch.isfinite(value).all():
                    raise FloatingPointError("Nonfinite stored hidden features")
            self.loaded_hidden[name], self.fingerprints[key] = payload, before
            while len(self.loaded_hidden) > 1:
                self.loaded_hidden.popitem(last=False)
        return self.loaded_hidden[name]

    def graph_at(self, index, *, payload=None, hidden_payload=None):
        require_slurm()
        if self.family == "O":
            raise RuntimeError("The output-only arm has no graph input")
        row = self.entries[index]
        target = {"y": torch.tensor([row["y"]], dtype=torch.float32),
                  **{key: torch.tensor([row[key]], dtype=torch.int64)
                     for key in ("record_id", "image_id", "source_id", "split_id")}}
        if self.family == "H" and HIDDEN_INDEX[self.arm] != 12:
            location = self.hidden_entries[row["record_id"]]
            if hidden_payload is None:
                hidden_payload = self._load_hidden(location["hidden_shard"])
            offset = location["hidden_offset"]
            if int(hidden_payload["record_id"][offset]) != row["record_id"]:
                raise RuntimeError("Hidden/cache record alignment changed")
            hidden = hidden_payload[f"hidden_{HIDDEN_INDEX[self.arm]}"][offset].float()
        else:
            payload = self._load(row["shard"]) if payload is None else payload
            offset = row["offset"]
            if int(payload["record_id"][offset]) != row["record_id"]:
                raise RuntimeError("Immutable record/shard alignment changed")
            if self.family in ("G", "S"):
                x, edge_index, edge_attr = graph_materialize(payload, offset, [GRAPH_INDEX[self.arm]])
                if x.shape != (197, 784) or edge_attr.shape != (edge_index.shape[1], 12):
                    raise RuntimeError("Unexpected matched graph/set feature dimensions")
                if self.family == "S":
                    edge_index = torch.zeros_like(edge_index)
                return GraphData(x=x, edge_index=edge_index, edge_attr=edge_attr, **target)
            hidden = payload["hidden"][offset].float()
        if hidden.shape != (197, 768) or not torch.isfinite(hidden).all():
            raise RuntimeError("All 197 own-layer hidden tokens are required")
        return GraphData(x=torch.cat((node_coordinates(197, 0, 1), hidden), dim=1), **target)

    def logits(self):
        require_slurm()
        if self.family != "O":
            raise RuntimeError("Classifier logits are not exposed by hidden/graph/set datasets")
        if self._logits is None:
            result = torch.empty((len(self), 100), dtype=torch.float32)
            for index, row in enumerate(self.entries):
                payload = self._load(row["shard"])
                offset = row["offset"]
                if int(payload["record_id"][offset]) != row["record_id"]:
                    raise RuntimeError("Logit/cache record alignment changed")
                value = payload["logits"][offset]
                if value.dtype != torch.float32 or value.shape != (100,) or not torch.isfinite(value).all():
                    raise RuntimeError("Expected 100 finite ordered FP32 classifier logits")
                if int(value.argmax()) != row["pred"]:
                    raise RuntimeError("Cached classifier prediction/logits disagree")
                result[index] = value
            self._logits = result
            self.loaded.clear()
        return self._logits

    def __getitem__(self, index):
        require_slurm()
        if self.family == "O":
            return self.logits()[index], torch.tensor(self.entries[index]["y"], dtype=torch.float32)
        return self.graph_at(index)


def load_logits(root, role):
    """All 100 frozen logits and unchanged integer metadata; no fitted scaler."""
    require_slurm()
    _role_guard(root, role)
    dataset = RoleDataset(root, role, "O", "logits")
    return {**dataset.metadata(), "logits": dataset.logits().numpy().copy()}
