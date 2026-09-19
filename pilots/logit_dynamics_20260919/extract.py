"""CLS-only extraction from the exact frozen images and pinned fast preprocessing."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path
import signal
import time

import torch

from pilots.final_comparison_20260916.data import cache_index
from pilots.final_comparison_20260916.extract import _recover_or_read, _write_shard
from pilots.final_comparison_20260916.train import initialize
from pilots.topology_20260910.extract import OfficialImages
from .protocol import (MODEL_ID, MODEL_REVISION, PROCESSOR_ID, PROCESSOR_REVISION, METADATA,
                       atomic_json, campaign, digest, frozen_json, lock, read, require_slurm, sha256, verify)

BATCH_SIZE = 4
STOP = False
PARITY = {"logits_atol": 1e-5, "logits_rtol": 1e-5, "cls12_fp16_atol": 1e-4,
          "cls12_fp16_rtol": 1e-3, "predictions": "exact", "labels": "exact"}


class Capture:
    def __init__(self):
        require_slurm()
        if not torch.cuda.is_available():
            raise RuntimeError("Frozen ViT extraction requires allocated CUDA")
        initialize(7)
        from transformers import AutoImageProcessor, ViTForImageClassification
        self.processor = AutoImageProcessor.from_pretrained(
            PROCESSOR_ID, revision=PROCESSOR_REVISION, use_fast=True, local_files_only=True)
        self.model = ViTForImageClassification.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, use_safetensors=True,
            attn_implementation="eager", local_files_only=True).eval().cuda().requires_grad_(False)
        if self.model.config._attn_implementation != "eager":
            raise RuntimeError("Pinned eager ViT attention is required")

    @torch.inference_mode()
    def batch(self, images):
        require_slurm()
        inputs = self.processor(images=images, return_tensors="pt").to("cuda")
        # Attention is computed by the frozen backbone but never exported or rebuilt into graphs.
        result = self.model(**inputs, output_attentions=False, output_hidden_states=True)
        if len(result.hidden_states) != 13 or result.logits.shape != (len(images), 100):
            raise RuntimeError("Expected 12 block outputs and 100 class logits")
        if any(x.shape != (len(images), 197, 768) or x.dtype != torch.float32
               for x in result.hidden_states[1:]):
            raise RuntimeError("Frozen hidden shape or precision changed")
        cls = torch.stack([value[:, 0] for value in result.hidden_states[1:]], dim=1).cpu().half()
        logits = result.logits.float().cpu()
        if not torch.isfinite(cls).all() or not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite frozen ViT capture")
        return {"cls": cls, "logits": logits}


def check_parity(values, original, rows, official_labels):
    require_slurm()
    offsets = torch.tensor([r["offset"] for r in rows], dtype=torch.int64)
    record_ids = torch.tensor([r["record_id"] for r in rows], dtype=torch.int64)
    if not torch.equal(original["record_id"][offsets], record_ids):
        raise RuntimeError("Original cache record IDs changed")
    logits = original["logits"][offsets]
    cls12 = original["hidden"][offsets, 0]
    if logits.dtype != torch.float32 or cls12.dtype != torch.float16:
        raise RuntimeError("Original cache precision changed")
    predicted = torch.tensor([r["pred"] for r in rows])
    if (not torch.equal(logits.argmax(-1), predicted)
            or not torch.equal(values["logits"].argmax(-1), predicted)
            or any(r["label"] != int(official_labels[r["image_id"]])
                   or r["y"] != int(r["pred"] != r["label"]) for r in rows)):
        raise RuntimeError("Frozen predictions, errors or official labels changed")
    torch.testing.assert_close(values["logits"], logits, atol=PARITY["logits_atol"], rtol=PARITY["logits_rtol"])
    torch.testing.assert_close(values["cls"][:, -1].float(), cls12.float(),
                               atol=PARITY["cls12_fp16_atol"], rtol=PARITY["cls12_fp16_rtol"])
    return {"records": len(rows), "logits_max_abs": float((values["logits"] - logits).abs().max()),
            "cls12_max_abs": float((values["cls"][:, -1].float() - cls12.float()).abs().max()),
            "predictions_exact": True, "labels_exact": True}


def extract(root, data_root, mode="full"):
    require_slurm()
    root = Path(root).resolve()
    current = campaign(root)
    old, manifest, all_rows = cache_index(current["baseline_root"])
    rows = all_rows
    if mode == "preflight":
        roles = read(root / "role_map.json")["roles"]
        chosen = set(roles["head_train"]["record_ids"][:48] + roles["probe_val"]["record_ids"][:48])
        rows = [r for r in all_rows if r["record_id"] in chosen]
    elif mode != "full":
        raise ValueError("Unknown extraction mode")
    directory = root / ("preflight_cls" if mode == "preflight" else "cls")
    identity = {"campaign_sha256": sha256(root / "campaign.json"), "records": len(rows),
                "record_ids_sha256": digest([r["record_id"] for r in rows]), "mode": mode,
                "shape_per_record": [12, 768], "dtype": "float16", "parity": PARITY,
                "cache_manifest_sha256": current["cache_manifest_sha256"]}
    started = time.monotonic()
    with lock(directory, "extract"):
        if (directory / "manifest.json").exists():
            previous = read(directory / "manifest.json")
            if any(previous.get(k) != v for k, v in identity.items()):
                raise RuntimeError("Capture resume identity changed")
            if previous.get("complete"):
                for spec in previous["shards"]:
                    verify(directory / spec["path"], spec["sha256"])
                verify(directory / "index.json", previous["index_sha256"])
                return previous
        images = OfficialImages(Path(data_root))
        cache = Path(old["cache"])
        provenance = images.provenance()
        verify(cache / "data_provenance.json", manifest["data_provenance_sha256"])
        if provenance != read(cache / "data_provenance.json"):
            raise RuntimeError("Raw image bytes differ from the original cache")
        frozen_json(directory / "data_provenance.json", provenance)
        capture = Capture()
        processor = json.loads(json.dumps(capture.processor.to_dict()))
        verify(cache / "processor.json", manifest["processor_sha256"])
        if processor != read(cache / "processor.json"):
            raise RuntimeError("Exact original fast-processor configuration is required")
        frozen_json(directory / "processor.json", processor)
        groups = OrderedDict()
        for row in rows:
            groups.setdefault(row["shard"], []).append(row)
        originals = {Path(s["path"]).name: s for s in manifest["shards"]}
        index, shards = [], []
        maximum = {"logits_max_abs": 0.0, "cls12_max_abs": 0.0}
        torch.cuda.reset_peak_memory_stats()
        for name, subset in groups.items():
            expected = {"capture_identity_sha256": digest(identity), "cache_shard": name,
                        "cache_sha256": originals[name]["sha256"],
                        "record_ids": [r["record_id"] for r in subset], "diagnostic_only": mode == "preflight"}
            path = directory / "shards" / name
            path.parent.mkdir(exist_ok=True)
            receipt = _recover_or_read(path, expected)
            if receipt is None:
                original_path = cache / "shards" / name
                before = original_path.stat()
                original = torch.load(original_path, map_location="cpu", weights_only=True)
                after = original_path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError("Immutable original shard changed during parity loading")
                buffers = {"cls": [], "logits": []}
                parity = {"records": 0, **{key: 0.0 for key in maximum}}
                for offset in range(0, len(subset), BATCH_SIZE):
                    selected = subset[offset:offset + BATCH_SIZE]
                    values = capture.batch([images.image(r) for r in selected])
                    checked = check_parity(values, original, selected, images.labels)
                    parity["records"] += len(selected)
                    for key in maximum:
                        parity[key] = max(parity[key], checked[key])
                    for key in buffers:
                        buffers[key].append(values[key])
                payload = {key: torch.cat(value) for key, value in buffers.items()}
                payload.update({key: torch.tensor([r[key] for r in subset], dtype=torch.int64) for key in METADATA})
                shard_rows = [{**r, "cls_shard": name, "cls_offset": i} for i, r in enumerate(subset)]
                receipt = _write_shard(path, payload, {**expected, "rows": shard_rows, "parity": parity,
                                                       "completed_unix": time.time()})
                del original, payload, buffers
            for key in maximum:
                maximum[key] = max(maximum[key], receipt["parity"][key])
            index.extend(receipt["rows"])
            shards.append({"path": "shards/" + name, "sha256": receipt["sha256"], "records": len(subset)})
            atomic_json(directory / "manifest.json", {**identity, "complete": False,
                                                       "shards": shards, "completed_records": len(index)})
            print(json.dumps({"event": "cls_shard_complete", "records": len(index), "total": len(rows)}), flush=True)
            if STOP:
                raise InterruptedError("Stopped after atomically committed shard; resume same command")
        frozen_json(directory / "index.json", index)
        result = {**identity, "complete": True, "shards": shards, "completed_records": len(index),
                  "index_sha256": sha256(directory / "index.json"), "parity_maximum": maximum,
                  "elapsed_seconds": time.monotonic() - started, "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                  "gpu": torch.cuda.get_device_name(), "job_id": os.environ["SLURM_JOB_ID"]}
        atomic_json(directory / "manifest.json", result)
        return result


def _stop(_signal, _frame):
    global STOP
    STOP = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "full"), default="full")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    print(json.dumps(extract(args.root, args.data_root, args.mode)), flush=True)


if __name__ == "__main__":
    main()
