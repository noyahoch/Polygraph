"""Slurm integrity tests; never report held-out performance before the freeze."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load as safe_load, save as safe_save
from torch_geometric.data import Batch, Data

from polygraph.data.graphs import ThresholdGraphBuilder
from .data import CachedDataset, verify_cached_file
from .models import EXPECTED_PARAMETERS, build_model, parameter_count
from .protocol import ARMS, atomic_json, cohort, digest, file_sha256, protocol, require_slurm
from .rewire import directed_swaps, verify_rewire
from .rewiring_decision import admission_bindings, read_admission


def synthetic_graph(full=True, empty=False):
    nodes = 5
    x = torch.randn(nodes, 784 if full else 16)
    x[:, 2] = 0
    x[0, 2] = 1
    edge = torch.tensor([[0, 0, 1, 1, 2, 3, 4], [1, 2, 2, 3, 3, 4, 0]])
    if empty:
        edge = edge[:, :0]
    return Data(x=x, edge_index=edge, edge_attr=torch.randn(edge.shape[1], 36 if full else 12),
                y=torch.tensor([1.0]))


def model_checks(device):
    torch.manual_seed(1234)
    result = {}
    for arm, spec in ARMS.items():
        model = build_model(arm).to(device)
        count = sum(p.numel() for p in model.parameters())
        if count != EXPECTED_PARAMETERS[arm]:
            raise RuntimeError(f"Parameter mismatch for {arm}: {count} vs {EXPECTED_PARAMETERS[arm]}")
        if arm == "logit":
            batch = torch.randn(2, 100, device=device)
            logits = model(batch).view(-1)
        else:
            batch = Batch.from_data_list([synthetic_graph(spec["full"]), synthetic_graph(spec["full"], empty=True)]).to(device)
            logits = model(batch)[0]
        if logits.shape != (2,) or not torch.isfinite(logits).all():
            raise RuntimeError(f"Invalid outputs for {arm}, including empty-edge graph")
        torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.], device=device)).backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients) or not any(torch.any(g != 0) for g in gradients):
            raise RuntimeError(f"Invalid gradients for {arm}")
        model.eval()
        with torch.no_grad():
            before = model(batch) if arm == "logit" else model(batch)[0]
            restored = build_model(arm).to(device).eval()
            restored.load_state_dict(safe_load(safe_save({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()})))
            after = restored(batch) if arm == "logit" else restored(batch)[0]
            torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-5)
            if spec["architecture"] in {"m5_node_edge_set", "m5_endpoint_set"}:
                graph = synthetic_graph(spec["full"])
                first = Batch.from_data_list([graph]).to(device)
                changed = copy.deepcopy(graph)
                order = torch.randperm(changed.edge_index.shape[1])
                changed.edge_index = changed.edge_index[:, order]
                changed.edge_attr = changed.edge_attr[order]
                second = Batch.from_data_list([changed]).to(device)
                torch.testing.assert_close(model(first)[0], model(second)[0], atol=1e-5, rtol=1e-5)
                if spec["architecture"] == "m5_node_edge_set":
                    changed.edge_index[1] = changed.edge_index[1].flip(0)
                    third = Batch.from_data_list([changed]).to(device)
                    torch.testing.assert_close(model(first)[0], model(third)[0], atol=1e-5, rtol=1e-5)
            if arm != "logit":
                graph = synthetic_graph(spec["full"])
                order = torch.tensor([2, 4, 0, 3, 1])
                inverse = torch.empty_like(order)
                inverse[order] = torch.arange(len(order))
                relabeled = Data(x=graph.x[order], edge_index=inverse[graph.edge_index], edge_attr=graph.edge_attr)
                torch.testing.assert_close(model(Batch.from_data_list([graph]).to(device))[0],
                                           model(Batch.from_data_list([relabeled]).to(device))[0], atol=1e-5, rtol=1e-5)
        result[arm] = {"parameters": count, "finite_forward_backward": True,
                       "empty_graph": arm != "logit", "restore_allclose": True}
    attention = torch.zeros(1, 12, 5, 5)
    attention[:, :, 2, 1] = 0.5
    graph = ThresholdGraphBuilder(0.02).build(attention)[0]
    if not torch.equal(graph.edge_index.long(), torch.tensor([[1], [2]])):
        raise RuntimeError("Attention graph direction is not key/source -> query/target")
    # Deterministic compiled swaps and exact degree invariants, independent of labels.
    source = np.repeat(np.arange(12), 3)
    target = np.concatenate([(np.arange(i + 1, i + 4) % 12) for i in range(12)])
    changed, accepted, attempts = directed_swaps(source, target, 12, 123)
    again, _, _ = directed_swaps(source, target, 12, 123)
    if not np.array_equal(changed, again):
        raise RuntimeError("Rewiring is not deterministic")
    verify_rewire(np.stack([source, target]), changed, 12)
    return result


def cache_checks(cache, require_rewire=False):
    cache = Path(cache)
    manifest = json.loads((cache / "manifest.json").read_text())
    if not manifest.get("complete") or manifest["protocol_sha256"] != digest(protocol()):
        raise RuntimeError("Cache not complete or protocol mismatch")
    if manifest["cohort_sha256"] != digest(cohort()):
        raise RuntimeError("Cohort mismatch")
    if file_sha256(cache / "index.json") != manifest["index_sha256"]:
        raise RuntimeError("Record-index checksum mismatch")
    index = json.loads((cache / "index.json").read_text())
    positions = {(entry["shard"], entry["offset"]): entry for entry in index}
    if len(positions) != len(index):
        raise RuntimeError("Duplicate shard/offset identity")
    canonical = {r["record_id"]: r for r in cohort()["records"]}
    seen = set()
    groups = {split: set() for split in ("train", "val", "test")}
    for entry in index:
        identity = entry["record_id"]
        if identity in seen:
            raise RuntimeError("Duplicate record ID")
        seen.add(identity)
        expected = canonical[identity]
        for key in ("image_id", "source", "source_id", "severity", "split", "split_id", "key"):
            if entry[key] != expected[key]:
                raise RuntimeError(f"Canonical identity mismatch for record {identity}: {key}")
        if entry["y"] != int(entry["pred"] != entry["label"]):
            raise RuntimeError("Error-label definition mismatch")
        groups[entry["split"]].add(entry["image_id"])
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if groups[a] & groups[b]:
            raise RuntimeError("Source-photo leakage")
    if not manifest.get("diagnostic_only") and seen != set(canonical):
        raise RuntimeError("Missing full-cohort records")
    record_count = 0
    for row in manifest["shards"]:
        path = cache / row["path"]
        verify_cached_file(path, row["sha256"])
        payload = torch.load(path, map_location="cpu", weights_only=True)
        for key, value in payload.items():
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise RuntimeError(f"Nonfinite tensor {key} in {path}")
        size = len(payload["record_id"])
        for offset, record_id in enumerate(payload["record_id"]):
            entry = positions.get((path.name, offset))
            if entry is None or entry["record_id"] != int(record_id):
                raise RuntimeError("Shard/offset does not identify the indexed source record")
            if entry["pred"] != int(payload["pred"][offset]) or entry["pred"] != int(payload["logits"][offset].argmax()):
                raise RuntimeError("Index prediction differs from frozen classifier tensor")
        shapes = {"base_x": (size, 197, 16), "hidden": (size, 197, 768), "logits": (size, 100),
                  "projected_value_norm": (size, 197, 12), "decision_support_proxy": (size, 197, 12)}
        for key, shape in shapes.items():
            if tuple(payload[key].shape) != shape:
                raise RuntimeError(f"Invalid shape {key}")
        offsets = payload["edge_offsets"]
        if len(offsets) != size + 1 or int(offsets[0]) != 0 or torch.any(offsets[1:] < offsets[:-1]):
            raise RuntimeError("Invalid edge offsets")
        if int(offsets[-1]) != payload["edge_index"].shape[1] or payload["edge_attr"].shape != (int(offsets[-1]), 12):
            raise RuntimeError("Invalid edge alignment")
        for start, stop in zip(offsets[:-1], offsets[1:]):
            edges = payload["edge_index"][:, int(start):int(stop)].long()
            if torch.any(edges < 0) or torch.any(edges >= 197) or torch.any(edges[0] == edges[1]):
                raise RuntimeError("Invalid original graph endpoints")
            if len(torch.unique(edges[0] * 197 + edges[1])) != edges.shape[1]:
                raise RuntimeError("Duplicate original graph edge")
        record_count += size
    if record_count != len(index):
        raise RuntimeError("Shard and index record counts differ")
    rewiring_checked = False
    rewiring_status = None
    if (cache / "rewire" / "manifest.json").exists() or require_rewire:
        original = CachedDataset(cache, "train", "full_graph")
        rewired = CachedDataset(cache, "train", "full_rewired")
        rewire_manifest = json.loads((cache / "rewire" / "manifest.json").read_text())
        rewiring_status = read_admission(cache, required=True)
        for row in rewire_manifest["shards"]:
            # All splits: checksums and opaque record/offset identity only, no
            # held-out changed-fraction summaries or diagnostic gate decisions.
            verify_cached_file(cache / row["path"], row["sha256"])
            null = torch.load(cache / row["path"], map_location="cpu", weights_only=True)
            name = Path(row["path"]).name
            for offset, record_id in enumerate(null["record_id"]):
                if positions.get((name, offset), {}).get("record_id") != int(record_id):
                    raise RuntimeError("Rewired shard/offset source mismatch")
            offsets = null["edge_offsets"]
            if len(offsets) != len(null["record_id"]) + 1 or int(offsets[0]) != 0 or torch.any(offsets[1:] < offsets[:-1]):
                raise RuntimeError("Rewired edge offsets are invalid")
            if int(offsets[-1]) != len(null["targets"]) or torch.any(null["targets"] < 0) or torch.any(null["targets"] >= 197):
                raise RuntimeError("Rewired target shape/index mismatch")
        for position in range(min(16, len(original))):
            a, b = original[position], rewired[position]
            if not torch.equal(a.x, b.x) or not torch.equal(a.edge_attr, b.edge_attr) or not torch.equal(a.edge_index[0], b.edge_index[0]):
                raise RuntimeError("Rewiring changed node/edge values or source-feature association")
            verify_rewire(a.edge_index.numpy(), b.edge_index[1].numpy(), a.num_nodes)
        rewiring_checked = True
    return {"records": record_count, "groups": {k: len(v) for k, v in groups.items()},
            "diagnostic_only": manifest.get("diagnostic_only", False), "rewiring_checked": rewiring_checked,
            "rewiring_status": rewiring_status,
            "rewiring_bindings": admission_bindings(cache) if rewiring_checked else None,
            "cache_manifest_sha256": file_sha256(cache / "manifest.json")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--require-rewire", action="store_true")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    require_slurm()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA validation but no allocated GPU is available")
    result = {"passed": True, "model_checks": model_checks(torch.device(args.device))}
    if args.cache:
        result["cache_checks"] = cache_checks(args.cache, args.require_rewire)
    output = args.out or (args.cache / "validation.json" if args.cache else None)
    if output:
        atomic_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
