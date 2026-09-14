"""One frozen ViT pass per presentation; retain final-layer features only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from polygraph.data.graphs import ThresholdGraphBuilder, node_coordinates
from polygraph.data.pipeline import FrozenClassifier
from polygraph.data.sidecars import value_message_statistics, derive_attention_edge_features
from .data import atomic_torch, verify_cached_file
from .protocol import (MODEL_ID, MODEL_REVISION, PROCESSOR_ID, PROCESSOR_REVISION, SOURCES,
                       atomic_json, cohort, digest, file_sha256, protocol, require_slurm, write_frozen)


class OfficialImages:
    def __init__(self, root):
        self.root = Path(root)
        clean = self.root / "cifar-100-python" / "test"
        md5 = hashlib.md5(clean.read_bytes()).hexdigest()
        if md5 != protocol()["data"]["clean_test_md5"]:
            raise RuntimeError(f"Canonical clean-test checksum mismatch: {md5}")
        # Deserialization follows verification against torchvision's official checksum.
        with clean.open("rb") as stream:
            payload = pickle.load(stream, encoding="bytes")
        self.clean = np.asarray(payload[b"data"], dtype=np.uint8).reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
        self.labels = np.asarray(payload[b"fine_labels"], dtype=np.int64)
        folder = self.root / "CIFAR-100-C"
        labels = np.load(folder / "labels.npy", mmap_mode="r", allow_pickle=False)
        if labels.shape != (50000,) or not np.array_equal(labels.reshape(5, 10000), np.broadcast_to(self.labels, (5, 10000))):
            raise RuntimeError("Official corruption labels do not repeat canonical clean labels in row order")
        self.corrupted = {name: np.load(folder / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in SOURCES[1:]}
        for name, values in self.corrupted.items():
            if values.shape != (50000, 32, 32, 3) or values.dtype != np.uint8:
                raise RuntimeError(f"Invalid official array: {name}, {values.shape}, {values.dtype}")

    def image(self, record):
        index = record["image_id"]
        values = self.clean[index] if record["source"] == "clean_test" else self.corrupted[record["source"]][(record["severity"] - 1) * 10000 + index]
        return Image.fromarray(values).convert("RGB")

    def provenance(self):
        files = [self.root / "cifar-100-python" / "test"]
        files += [self.root / "CIFAR-100-C" / f"{name}.npy" for name in (*SOURCES[1:], "labels")]
        return {str(path.relative_to(self.root)): {"bytes": path.stat().st_size, "sha256": file_sha256(path)} for path in files}


class Capture:
    def __init__(self, device="cuda"):
        from transformers import AutoImageProcessor, ViTForImageClassification
        if device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Frozen-feature extraction requires an allocated CUDA GPU; no fallback")
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(PROCESSOR_ID, revision=PROCESSOR_REVISION, use_fast=True)
        self.model = ViTForImageClassification.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                            use_safetensors=True, attn_implementation="eager").eval().to(self.device)
        self.model.requires_grad_(False)
        vit = self.model.vit
        blocks = vit.layers if hasattr(vit, "layers") else vit.encoder.layer
        attention = blocks[11].attention
        if hasattr(attention, "v_proj"):
            value_layer, self.output_weight = attention.v_proj, attention.o_proj.weight
        else:
            value_layer, self.output_weight = attention.attention.value, attention.output.dense.weight
        self.values = []
        self.hook = value_layer.register_forward_hook(lambda _module, _inputs, output: self.values.append(output.detach()))
        self.builder = ThresholdGraphBuilder(0.02)

    @torch.no_grad()
    def batch(self, images):
        self.values.clear()
        pixels = self.processor(images=list(images), return_tensors="pt")["pixel_values"].to(self.device)
        result = self.model(pixels, output_attentions=True, output_hidden_states=True)
        if len(self.values) != 1 or result.attentions is None or len(result.attentions) != 12:
            raise RuntimeError("Expected one final value capture and twelve eager attention tensors")
        logits = result.logits.float()
        top2 = logits.topk(2, dim=-1).indices
        direction = self.model.classifier.weight[top2[:, 0]] - self.model.classifier.weight[top2[:, 1]]
        _, projected, support = value_message_statistics(self.values[0], self.output_weight, direction, 12)
        attention = result.attentions[11]
        hidden = result.hidden_states[12]
        probabilities = logits.softmax(-1).topk(2, dim=-1).values
        diagonals = self.builder.diagonals(attention)
        coords = node_coordinates(197, 11, 12).to(self.device)
        base = torch.cat([coords.unsqueeze(0).expand(len(images), -1, -1), diagonals], dim=-1)
        payload = {"base_x": base.cpu().half(), "hidden": hidden.cpu().half(),
                   "projected_value_norm": projected.cpu().half(), "decision_support_proxy": support.cpu().half(),
                   "logits": logits.cpu(), "pred": top2[:, 0].cpu(),
                   "confidence": probabilities[:, 0].cpu(), "margin": (probabilities[:, 0] - probabilities[:, 1]).cpu()}
        graphs = self.builder.build(attention)
        for key, value in payload.items():
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise RuntimeError(f"Nonfinite captured tensor: {key}")
        return payload, graphs

    @torch.no_grad()
    def legacy_check(self, images):
        """Independent existing API calls on the same pinned model/processor, Slurm only."""
        captured, graphs = self.batch(images)
        legacy = FrozenClassifier.__new__(FrozenClassifier)
        legacy.model_id, legacy.device = MODEL_ID, self.device
        legacy.model, legacy.processor = self.model, self.processor
        self.values.clear()
        reference_logits = legacy.logits(images)
        if len(self.values) != 1:
            raise RuntimeError("Legacy value hook did not fire once")
        reference_top2 = reference_logits.to(self.device).topk(2, -1).indices
        direction = self.model.classifier.weight[reference_top2[:, 0]] - self.model.classifier.weight[reference_top2[:, 1]]
        _, projected, support = value_message_statistics(self.values[0], self.output_weight, direction, 12)
        torch.testing.assert_close(captured["logits"], reference_logits, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(captured["projected_value_norm"], projected.cpu().half(), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(captured["decision_support_proxy"], support.cpu().half(), atol=1e-5, rtol=1e-5)
        reference_hidden = []
        hidden_hook = self.model.register_forward_hook(
            lambda _m, _i, output: reference_hidden.append(output.hidden_states[12].detach().cpu().half()))
        try:
            pred, confidence, margin, attention, cls = legacy.analyse(images, attentions=True, want_cls=True)
        finally:
            hidden_hook.remove()
        if not np.array_equal(pred, captured["pred"].numpy()):
            raise RuntimeError("Legacy and one-pass classifier predictions differ")
        torch.testing.assert_close(captured["hidden"][:, 0], cls[:, 11], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(captured["hidden"], reference_hidden[0], atol=1e-5, rtol=1e-5)
        reference_base = torch.cat([node_coordinates(197, 11, 12).unsqueeze(0).expand(len(images), -1, -1),
                                  self.builder.diagonals(attention[11]).cpu()], dim=-1).half()
        torch.testing.assert_close(captured["base_x"], reference_base, atol=1e-5, rtol=1e-5)
        reference_graphs = self.builder.build(attention[11])
        for i, (graph, reference) in enumerate(zip(graphs, reference_graphs)):
            if not torch.equal(graph.edge_index, reference.edge_index):
                raise RuntimeError("Legacy and one-pass threshold-edge identities differ")
            torch.testing.assert_close(graph.edge_attr, reference.edge_attr, atol=1e-5, rtol=1e-5)
            first = derive_attention_edge_features(graph.edge_attr.float(), graph.edge_index.long(),
                       captured["projected_value_norm"][i].float(), captured["decision_support_proxy"][i].float(), "evidence_flow")
            second = derive_attention_edge_features(reference.edge_attr.float(), reference.edge_index.long(),
                       projected.cpu().half()[i].float(), support.cpu().half()[i].float(), "evidence_flow")
            torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-5)
        return {"passed": True, "records": len(images), "atol": 1e-5, "rtol": 1e-5,
                "checked": ["logits", "predictions", "all_hidden_tokens", "base_nodes", "attention_edges", "projected_value_norm", "class_support", "evidence_edges"]}

    def close(self):
        self.hook.remove()


def capture(args):
    require_slurm()
    cache = args.cache
    proposed = json.loads(args.protocol.read_text()) if args.protocol else protocol()
    if proposed != protocol():
        raise RuntimeError("Protocol differs from the frozen implementation")
    write_frozen(cache / "protocol.json", proposed)
    write_frozen(cache / "cohort.json", cohort())
    records = cohort()["records"]
    diagnostic = args.max_records is not None
    if diagnostic:
        if args.max_records < 27:
            raise ValueError("Diagnostic extraction needs at least 27 records across three splits")
        per_split = args.max_records // 3
        records = [r for split in ("train", "val", "test") for r in [e for e in records if e["split"] == split][:per_split]]
    cache.mkdir(parents=True, exist_ok=True)
    package_root = Path(__file__).resolve().parents[2]
    extraction_files = [Path(__file__).resolve(), Path(__file__).with_name("protocol.py"),
                        package_root / "polygraph/data/graphs.py", package_root / "polygraph/data/sidecars.py"]
    versions = {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "transformers", "torch-geometric", "numpy")}
    identity = {"schema_version": 1, "protocol_sha256": digest(proposed), "cohort_sha256": digest(cohort()),
                "record_ids_sha256": digest([r["record_id"] for r in records]), "diagnostic_only": diagnostic,
                "records": len(records), "shard_size": args.shard_size,
                "extraction_implementation_sha256": digest({str(path.relative_to(package_root)): file_sha256(path) for path in extraction_files}),
                "versions": versions}
    manifest_path = cache / "manifest.json"
    previous = None
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if any(previous.get(k) != v for k, v in identity.items()):
            raise RuntimeError("Existing cache has a different extraction identity")
        if previous.get("complete") and args.resume:
            # Completed cache reuse does not append source presentations. Verify
            # its immutable outputs without requiring original archives to remain.
            for row in previous["shards"]:
                verify_cached_file(cache / row["path"], row["sha256"])
            for name, key in (("index.json", "index_sha256"),
                              ("data_provenance.json", "data_provenance_sha256"),
                              ("processor.json", "processor_sha256")):
                verify_cached_file(cache / name, previous.get(key))
            if json.loads((cache / "processor.json").read_text()) != previous.get("processor"):
                raise RuntimeError("Completed cache processor identity mismatch")
            print(json.dumps({"event": "complete_cache_reused", "cached_artifacts_verified": True,
                              "source_files_rechecked": False}), flush=True)
            return
        if not args.resume:
            raise RuntimeError("Cache exists; use --resume to preserve immutable shards")
    images = OfficialImages(args.data_root)
    provenance_path = cache / "data_provenance.json"
    # A partial resume must never append presentations from changed input arrays.
    # Compare current checksums with immutable provenance before constructing any shard.
    write_frozen(provenance_path, images.provenance())
    capture_model = Capture(args.device)
    processor_identity = {"processor_id": PROCESSOR_ID, "revision": PROCESSOR_REVISION,
                          "use_fast": True, "configuration": capture_model.processor.to_dict(),
                          "classifier_id": MODEL_ID, "classifier_revision": MODEL_REVISION}
    write_frozen(cache / "processor.json", processor_identity)
    identity.update(processor=processor_identity, processor_sha256=file_sha256(cache / "processor.json"),
                    data_provenance_sha256=file_sha256(provenance_path),
                    feature_numeric={"storage": "float16 features; float32 classifier logits",
                                     "class_direction_norm_clamp_min": 1e-12,
                                     "attention_edge_selection_dtype": "float32"})
    if previous is not None and any(previous.get(key) != value for key, value in identity.items()):
        capture_model.close()
        raise RuntimeError("Partial cache resume source/processor identity differs")
    environment = {"python": platform.python_version(), "versions": versions,
                   "gpu": torch.cuda.get_device_name(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
                   "processor_sha256": identity["processor_sha256"]}
    atomic_json(cache / "attempts" / f"extraction-{os.environ['SLURM_JOB_ID']}-{time.time_ns()}.json", environment)
    rows, shards = [], []
    started = time.monotonic()
    try:
        preflight = cache / "extraction_preflight.json"
        if not preflight.exists():
            atomic_json(preflight, capture_model.legacy_check([images.image(r) for r in records[:min(18, args.batch_size)]]))
        for shard_number, start in enumerate(range(0, len(records), args.shard_size)):
            subset = records[start:start + args.shard_size]
            name = f"shard_{shard_number:05d}.pt"
            path = cache / "shards" / name
            row_path = cache / "shards" / name.replace(".pt", ".json")
            if path.exists() and row_path.exists() and args.resume:
                sidecar = json.loads(row_path.read_text())
                if sidecar["record_ids"] != [r["record_id"] for r in subset] or file_sha256(path) != sidecar["sha256"]:
                    raise RuntimeError(f"Completed shard identity/hash mismatch: {name}")
                rows.extend(sidecar["rows"])
                shards.append({"path": f"shards/{name}", "records": len(subset), "sha256": sidecar["sha256"]})
                continue
            buffer, indices, attrs, offsets, shard_rows = {}, [], [], [0], []
            for batch_start in range(0, len(subset), args.batch_size):
                batch_records = subset[batch_start:batch_start + args.batch_size]
                payload, graphs = capture_model.batch([images.image(r) for r in batch_records])
                for key, value in payload.items():
                    buffer.setdefault(key, []).append(value)
                for j, (record, graph) in enumerate(zip(batch_records, graphs)):
                    indices.append(graph.edge_index)
                    attrs.append(graph.edge_attr)
                    offsets.append(offsets[-1] + graph.num_edges)
                    label, pred = int(images.labels[record["image_id"]]), int(payload["pred"][j])
                    shard_rows.append({**record, "shard": name, "offset": len(shard_rows), "label": label,
                                       "pred": pred, "y": int(pred != label), "confidence": float(payload["confidence"][j]),
                                       "margin": float(payload["margin"][j]), "edges": graph.num_edges})
            output = {key: torch.cat(value) for key, value in buffer.items()}
            output.update(edge_index=torch.cat(indices, 1), edge_attr=torch.cat(attrs, 0),
                          edge_offsets=torch.tensor(offsets, dtype=torch.int64),
                          record_id=torch.tensor([r["record_id"] for r in subset], dtype=torch.int64))
            atomic_torch(path, output)
            sha = file_sha256(path)
            atomic_json(row_path, {"record_ids": [r["record_id"] for r in subset], "sha256": sha, "rows": shard_rows})
            rows.extend(shard_rows)
            shards.append({"path": f"shards/{name}", "records": len(subset), "sha256": sha})
            atomic_json(manifest_path, {**identity, "complete": False, "shards": shards, "completed_records": len(rows)})
            print(json.dumps({"event": "capture_shard", "records": len(rows), "total": len(records),
                              "elapsed_seconds": time.monotonic() - started, "shard": name}), flush=True)
        atomic_json(cache / "index.json", rows)
        atomic_json(manifest_path, {**identity, "complete": True, "shards": shards, "completed_records": len(rows),
                                  "index_sha256": file_sha256(cache / "index.json"),
                                  "data_provenance_sha256": file_sha256(provenance_path)})
    finally:
        capture_model.close()


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--data-root", type=Path, required=True)
    c = sub.add_parser("capture")
    c.add_argument("--data-root", type=Path, required=True)
    c.add_argument("--cache", type=Path, required=True)
    c.add_argument("--protocol", type=Path)
    c.add_argument("--device", choices=["cuda"], default="cuda")
    c.add_argument("--batch-size", type=int, default=32)
    c.add_argument("--shard-size", type=int, default=256)
    c.add_argument("--max-records", type=int)
    c.add_argument("--resume", action="store_true")
    args = p.parse_args()
    require_slurm()
    if args.command == "prepare":
        source = OfficialImages(args.data_root)
        print(json.dumps({"passed": True, "files": source.provenance()}))
    else:
        capture(args)


if __name__ == "__main__":
    main()
