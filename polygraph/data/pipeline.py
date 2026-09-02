"""The frozen classifier, plus the two stages that run it: scan and extract."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ..config import DEFAULT_MODEL_ID
from .graphs import ThresholdGraphBuilder
from ..records import RecordKey, ScanRecord, append_scan_records, scanned_keys
from .sources import get_pool, pool_for

# Scan and extraction run the same frozen model at different batch sizes, and MPS numerics
# can flip a genuinely borderline prediction. The scan is canonical (it defined the splits),
# so isolated flips are recorded; a rate above this means something real changed.
MAX_DRIFT_FRACTION = 0.001


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FrozenClassifier:
    """The ViT under study: loaded once, frozen, inference only."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: Optional[torch.device] = None):
        from transformers import AutoImageProcessor, ViTForImageClassification

        self.model_id, self.device = model_id, device or choose_device()
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
        except OSError:  # this checkpoint ships no preprocessor config; use the base ViT's
            self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", use_fast=True)
        # eager attention is required: fused kernels do not expose attention matrices
        self.model = ViTForImageClassification.from_pretrained(model_id, attn_implementation="eager")
        self.model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def analyse(self, images, attentions: bool = False, want_cls: bool = False):
        """-> (pred, confidence, margin) arrays, per-layer attentions, per-layer CLS states."""
        pixels = self.processor(images=list(images), return_tensors="pt")["pixel_values"].to(self.device)
        out = self.model(pixels, output_attentions=attentions, output_hidden_states=want_cls)
        top2 = torch.softmax(out.logits.float(), -1).topk(2, -1)
        pred = top2.indices[:, 0].cpu().numpy()
        confidence = top2.values[:, 0].cpu().numpy()
        margin = (top2.values[:, 0] - top2.values[:, 1]).cpu().numpy()
        cls_states = (torch.stack([h[:, 0] for h in out.hidden_states[1:]], 1).to("cpu", torch.float16)
                      if want_cls else None)
        return pred, confidence, margin, (list(out.attentions) if attentions else None), cls_states

    @torch.no_grad()
    def logits(self, images) -> torch.Tensor:
        """Full fp32 classifier logits, without attention or hidden-state materialization."""
        pixels = self.processor(images=list(images), return_tensors="pt")["pixel_values"].to(self.device)
        return self.model(pixels, output_attentions=False, output_hidden_states=False).logits.float().cpu()


def capture_logits(classifier: FrozenClassifier, data_root: Path, store_dir: Path,
                   out_dir: Path, batch_size: int = 128,
                   overwrite_corrupt_only: bool = False) -> int:
    """Capture full logits in graph-store order with per-record scan consistency checks."""
    import json
    import sys

    from .sidecars import atomic_json_save, atomic_torch_save, build_manifest
    from .storage import GraphShard

    keys = [RecordKey(*k) for k in json.loads((store_dir / "store_keys.json").read_text())]
    store_manifest = json.loads((store_dir / "manifest.json").read_text())
    counts = list(map(int, store_manifest["shard_records"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    position = disagreements = 0
    max_conf_delta = max_margin_delta = 0.0
    for shard_index, count in enumerate(counts):
        shard_keys = keys[position:position + count]
        position += count
        out_path = out_dir / f"logits_{shard_index:05d}.pt"
        if out_path.exists():
            try:
                payload = torch.load(out_path, map_location="cpu")
                assert int(payload["records"]) == count
                assert payload["model_id"] == classifier.model_id
                assert tuple(payload["logits"].shape) == (count, 100)
                continue
            except Exception:
                if not overwrite_corrupt_only:
                    raise RuntimeError(f"corrupt/incompatible completed sidecar: {out_path}; "
                                       "pass --overwrite-corrupt-only")
                out_path.unlink()
        graph_shard = GraphShard.load(store_dir / store_manifest["shards"][shard_index])
        buffers, classes = [], []
        for start in tqdm(range(0, count, batch_size), desc=f"logits shard {shard_index}"):
            chunk = shard_keys[start:start + batch_size]
            images = [pool_for(k, data_root).image(k.base_index) for k in chunk]
            logits = classifier.logits(images)
            probability = logits.softmax(-1)
            top2 = probability.topk(2, -1)
            expected_pred = graph_shard.meta["pred"][start:start + len(chunk)].long()
            disagreements += int((top2.indices[:, 0] != expected_pred).sum())
            conf_delta = (top2.values[:, 0] - graph_shard.meta["confidence"][
                start:start + len(chunk)]).abs().max().item()
            margin_delta = ((top2.values[:, 0] - top2.values[:, 1]) - graph_shard.meta["margin"][
                start:start + len(chunk)]).abs().max().item()
            max_conf_delta, max_margin_delta = max(max_conf_delta, conf_delta), max(max_margin_delta, margin_delta)
            buffers.append(logits.to(torch.float16))
            classes.append(top2.indices.to(torch.int16))
        if disagreements > MAX_DRIFT_FRACTION * max(position, 1) + 1:
            raise RuntimeError(f"systematic logit prediction drift: {disagreements}/{position}")
        if max_conf_delta > 2e-4 or max_margin_delta > 2e-4:
            raise RuntimeError(f"systematic probability drift: confidence {max_conf_delta:g}, "
                               f"margin {max_margin_delta:g}")
        atomic_torch_save({"logits": torch.cat(buffers), "top2_classes": torch.cat(classes),
                           "records": count, "model_id": classifier.model_id}, out_path)
    manifest = build_manifest(store_dir, classifier.model_id,
                              {"logits": ["N", 100], "top2_classes": ["N", 2]},
                              "float16/int16", sys.argv)
    manifest.update({"prediction_disagreements": disagreements,
                     "max_confidence_delta": max_conf_delta,
                     "max_margin_delta": max_margin_delta})
    atomic_json_save(manifest, out_dir / "manifest.json")
    return position


def capture_message_stats(classifier: FrozenClassifier, data_root: Path, store_dir: Path,
                          out_dir: Path, layer: int = 11, batch_size: int = 64) -> int:
    """Capture compact final-block value statistics; never materializes per-edge messages."""
    import json
    import sys

    from .sidecars import (atomic_json_save, atomic_torch_save, build_manifest,
                           value_message_statistics)
    from .storage import GraphShard

    keys = [RecordKey(*k) for k in json.loads((store_dir / "store_keys.json").read_text())]
    store_manifest = json.loads((store_dir / "manifest.json").read_text())
    counts = list(map(int, store_manifest["shard_records"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    block = classifier.model.vit.encoder.layer[layer]
    value_layer = block.attention.attention.value
    output_weight = block.attention.output.dense.weight.detach()
    classifier_weight = classifier.model.classifier.weight.detach()
    heads = int(block.attention.attention.num_attention_heads)
    captured: List[torch.Tensor] = []

    def hook(_module, _inputs, output):
        captured.append(output.detach())

    handle = value_layer.register_forward_hook(hook)
    position = disagreements = 0
    try:
        for shard_index, count in enumerate(counts):
            shard_keys = keys[position:position + count]
            position += count
            out_path = out_dir / f"message_stats_{shard_index:05d}.pt"
            if out_path.exists():
                payload = torch.load(out_path, map_location="cpu")
                if (int(payload.get("records", -1)) == count and payload.get("model_id") == classifier.model_id
                        and int(payload.get("layer", -1)) == layer):
                    continue
                raise RuntimeError(f"incompatible completed message-stat shard: {out_path}")
            graph_shard = GraphShard.load(store_dir / store_manifest["shards"][shard_index])
            raw_all, projected_all, support_all, predicted_all, runner_all = [], [], [], [], []
            for start in tqdm(range(0, count, batch_size), desc=f"message stats shard {shard_index}"):
                chunk = shard_keys[start:start + batch_size]
                images = [pool_for(k, data_root).image(k.base_index) for k in chunk]
                captured.clear()
                logits = classifier.logits(images).to(classifier.device)
                if len(captured) != 1:
                    raise RuntimeError(f"value hook fired {len(captured)} times")
                top2 = logits.topk(2, -1).indices
                expected = graph_shard.meta["pred"][start:start + len(chunk)].to(top2.device).long()
                disagreements += int((top2[:, 0] != expected).sum())
                direction = classifier_weight[top2[:, 0]] - classifier_weight[top2[:, 1]]
                raw, projected, support = value_message_statistics(
                    captured[0], output_weight, direction, heads)
                raw_all.append(raw.cpu().half()); projected_all.append(projected.cpu().half())
                support_all.append(support.cpu().half()); predicted_all.append(top2[:, 0].cpu().short())
                runner_all.append(top2[:, 1].cpu().short())
            if disagreements > MAX_DRIFT_FRACTION * max(position, 1) + 1:
                raise RuntimeError(f"systematic message-stat prediction drift: {disagreements}/{position}")
            atomic_torch_save({"value_norm": torch.cat(raw_all),
                               "projected_value_norm": torch.cat(projected_all),
                               "decision_support_proxy": torch.cat(support_all),
                               "predicted_class": torch.cat(predicted_all),
                               "runner_up_class": torch.cat(runner_all), "records": count,
                               "model_id": classifier.model_id, "layer": layer}, out_path)
    finally:
        handle.remove()
    manifest = build_manifest(store_dir, classifier.model_id,
                              {"value_norm": ["N", 197, heads],
                               "projected_value_norm": ["N", 197, heads],
                               "decision_support_proxy": ["N", 197, heads],
                               "predicted_class": ["N"], "runner_up_class": ["N"]},
                              "float16/int16", sys.argv, layer=layer)
    manifest["prediction_disagreements"] = disagreements
    atomic_json_save(manifest, out_dir / "manifest.json")
    return position


def scan(classifier: FrozenClassifier, data_root: Path, scan_path: Path,
         pairs: Sequence[Tuple[str, int]], batch_size: int = 64, limit_per_pool: int = 0) -> None:
    """Record the classifier's verdict over image pools. Resumable, edge-rule agnostic."""
    seen = scanned_keys(scan_path)
    for source, severity in pairs:
        pool = get_pool(source, severity, data_root)
        todo = [i for i in range(len(pool)) if pool.key(i).as_tuple() not in seen]
        todo = todo[:limit_per_pool] if limit_per_pool else todo
        if not todo:
            continue
        correct, confidence_sum, buffer = 0, 0.0, []
        for start in tqdm(range(0, len(todo), batch_size), desc=f"scan {source} s{severity}"):
            chunk = todo[start:start + batch_size]
            pred, conf, margin, _, _ = classifier.analyse([pool.image(i) for i in chunk])
            for pos, index in enumerate(chunk):
                record = ScanRecord(pool.key(index), pool.label(index), int(pred[pos]),
                                    float(conf[pos]), float(margin[pos]))
                buffer.append(record)
                correct += record.correct
                confidence_sum += record.confidence
            if len(buffer) >= 2000:
                append_scan_records(scan_path, buffer)
                buffer = []
        append_scan_records(scan_path, buffer)
        print(f"{source} s{severity}: {correct}/{len(todo)} correct "
              f"(errors {len(todo) - correct}, mean confidence {confidence_sum / len(todo):.4f})", flush=True)


def capture_hidden(classifier: FrozenClassifier, data_root: Path, store_dir: Path,
                   out_dir: Path, layer: int = 12, batch_size: int = 64) -> int:
    """Per-token hidden states of one ViT block for every record in the graph store,
    sharded in the SAME order and sizes as the store so readers align by (shard, offset).
    layer=12 means the final block's output (hidden_states[12]); resumable per shard."""
    import json
    import sys

    from .sidecars import atomic_json_save, atomic_torch_save, build_manifest

    keys = [RecordKey(*k) for k in json.loads((store_dir / "store_keys.json").read_text())]
    manifest = json.loads((store_dir / "manifest.json").read_text())
    counts = manifest["shard_records"]
    out_dir.mkdir(parents=True, exist_ok=True)

    position = 0
    for shard_index, count in enumerate(counts):
        shard_keys = keys[position:position + count]
        position += count
        out_path = out_dir / f"hidden_{shard_index:05d}.pt"
        if out_path.exists():
            continue
        buffers = []
        for start in tqdm(range(0, count, batch_size), desc=f"hidden shard {shard_index}"):
            chunk = shard_keys[start:start + batch_size]
            images = [pool_for(k, data_root).image(k.base_index) for k in chunk]
            pixels = classifier.processor(images=images, return_tensors="pt")["pixel_values"]
            with torch.no_grad():
                out = classifier.model(pixels.to(classifier.device), output_hidden_states=True)
            buffers.append(out.hidden_states[layer].to("cpu", torch.float16))
            if classifier.device.type == "mps":
                torch.mps.empty_cache()
        tensor = torch.cat(buffers)
        atomic_torch_save({"hidden": tensor, "layer": layer, "records": count,
                           "model_id": classifier.model_id}, out_path)
    manifest = build_manifest(store_dir, classifier.model_id, {"hidden": ["N", 197, 768]},
                              "float16", sys.argv, layer=layer)
    atomic_json_save(manifest, out_dir / "manifest.json")
    return position


def extract(classifier: FrozenClassifier, data_root: Path, builder: ThresholdGraphBuilder,
            keys: Sequence[RecordKey], scan_lookup: Dict[Tuple[str, int, int], ScanRecord],
            writer, batch_size: int = 32, want_cls: bool = True) -> Dict[str, int]:
    """Build and store attention graphs. Resumable; scan labels are canonical."""
    pending, already, shard_index = writer.plan(keys)
    if already:
        print(f"resuming after {already} records ({shard_index} complete shards)", flush=True)
    written, drifted = already, []

    for start in tqdm(range(0, len(pending), batch_size), desc="extract"):
        chunk = pending[start:start + batch_size]
        images = [pool_for(k, data_root).image(k.base_index) for k in chunk]
        pred, _, _, attentions, cls_states = classifier.analyse(images, attentions=True, want_cls=want_cls)
        per_layer = [builder.build(a) for a in attentions]
        diagonals = torch.stack([builder.diagonals(a).to("cpu", torch.float16) for a in attentions], 1)

        for pos, key in enumerate(chunk):
            record = scan_lookup.get(key.as_tuple())
            if record is None:
                raise KeyError(f"no scan record for {key.as_tuple()}; run scan first")
            if int(pred[pos]) != record.pred:
                drifted.append(key.as_tuple())
                if len(drifted) > MAX_DRIFT_FRACTION * max(len(keys), 1) + 1:
                    raise RuntimeError(
                        f"{len(drifted)} predictions disagree with the scan (e.g. {drifted[:3]}); "
                        "check that the checkpoint and preprocessing match the scan")
            writer.add(record, [per_layer[l][pos] for l in range(len(attentions))], diagonals[pos],
                       None if cls_states is None else cls_states[pos])
            written += 1
            if writer.pending >= writer.shard_size:
                writer.flush(shard_index)
                shard_index += 1
    writer.flush(shard_index)
    if drifted:
        print(f"note: {len(drifted)} borderline predictions differed from the scan; "
              "stored with the scan's canonical labels", flush=True)
    return {"written": written, "prediction_drift": len(drifted)}
