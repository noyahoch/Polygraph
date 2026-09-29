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
from torch import Tensor


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
            payload = torch.load(path, map_location="cpu", weights_only=False)
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


def value_message_statistics(value: Tensor, output_weight: Tensor,
                             class_direction: Tensor, heads: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Raw norm, ||W_h V|| via Gram matrices, and projection onto a class direction."""
    batch, tokens, width = value.shape
    if width % heads:
        raise ValueError(f"width {width} is not divisible by {heads} heads")
    head_width = width // heads
    values = value.float().reshape(batch, tokens, heads, head_width)
    blocks = output_weight.float().reshape(output_weight.shape[0], heads, head_width).permute(1, 0, 2)
    gram = torch.einsum("hod,hoe->hde", blocks, blocks)
    projected_sq = torch.einsum("bthd,hde,bthe->bth", values, gram, values)
    direction = class_direction.float()
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    projected_direction = torch.einsum("hod,bo->bhd", blocks, direction)
    support = torch.einsum("bthd,bhd->bth", values, projected_direction)
    return values.norm(dim=-1), projected_sq.clamp_min(0).sqrt(), support


def reconstruct_attention_dense(attention: Tensor, value: Tensor, output_weight: Tensor,
                                output_bias: Optional[Tensor], heads: int) -> Tensor:
    """Reconstruct the attention output dense result before its residual connection."""
    batch, tokens, width = value.shape
    values = value.float().reshape(batch, tokens, heads, width // heads)
    mixed = torch.einsum("bhij,bjhd->bihd", attention.float(), values).reshape(batch, tokens, width)
    return torch.nn.functional.linear(mixed, output_weight.float(),
                                      None if output_bias is None else output_bias.float())


def derive_attention_edge_features(edge_attr: Tensor, edge_index: Tensor,
                                   projected_value_norm: Tensor,
                                   decision_support_proxy: Tensor, mode: str) -> Tensor:
    """Derive message features using row 0 (key/source j) of edge j -> i."""
    if mode == "attention":
        return edge_attr
    source = edge_index[0].long()
    message = torch.log1p((edge_attr * projected_value_norm[source]).clamp_min(0))
    decision = torch.asinh(edge_attr * decision_support_proxy[source])
    if mode == "attention_message":
        return torch.cat([edge_attr, message], dim=-1)
    if mode == "attention_decision":
        return torch.cat([edge_attr, decision], dim=-1)
    if mode == "evidence_flow":
        return torch.cat([edge_attr, message, decision], dim=-1)
    raise ValueError(f"unknown edge feature mode: {mode}")


def compact_class_evidence(hidden: Tensor, predicted_class: Tensor, runner_up_class: Tensor,
                           layernorm, classifier) -> Tuple[Tensor, Tensor]:
    """Four token features conditioned only on predicted and runner-up classes.

    Returns (features, normalized_hidden).  There is intentionally no true-class argument.
    """
    normalized = layernorm(hidden.float())
    weight, bias = classifier.weight, classifier.bias
    predicted_weight = weight[predicted_class]
    runner_weight = weight[runner_up_class]
    predicted = torch.einsum("btd,bd->bt", normalized, predicted_weight)
    runner = torch.einsum("btd,bd->bt", normalized, runner_weight)
    if bias is not None:
        predicted = predicted + bias[predicted_class, None]
        runner = runner + bias[runner_up_class, None]
    features = torch.stack([torch.asinh(predicted), torch.asinh(runner),
                            torch.asinh(predicted - runner),
                            torch.log1p(normalized.norm(dim=-1))], dim=-1)
    return features, normalized
