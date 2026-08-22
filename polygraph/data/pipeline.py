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
