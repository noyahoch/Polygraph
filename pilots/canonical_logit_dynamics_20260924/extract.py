"""CLS-only canonical-source capture; original scan parity is a hard fitting gate."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
from polygraph.data.sources import (get_pool, clean_path, corruption_path)
from .protocol import (MODEL_ID, MODEL_REVISION, PROCESSOR_ID, PROCESSOR_REVISION, METADATA, PARITY,
                       atomic_json, atomic_torch, campaign, digest, frozen_json, initialize,
                       lock, read, require_slurm, sha256, verify)

BATCH_SIZE = 32
SHARD_SIZE = 128
STOP = False


class Capture:
    def __init__(self):
        require_slurm()
        if not torch.cuda.is_available():
            raise RuntimeError("Extraction requires an allocated CUDA GPU")
        initialize(7)
        from transformers import AutoImageProcessor, ViTForImageClassification
        self.processor = AutoImageProcessor.from_pretrained(PROCESSOR_ID, revision=PROCESSOR_REVISION,
                                                           use_fast=True, local_files_only=True)
        self.model = ViTForImageClassification.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
            use_safetensors=True, attn_implementation="eager", local_files_only=True).eval().cuda().requires_grad_(False)
        if self.model.config._attn_implementation != "eager":
            raise RuntimeError("Pinned eager attention is required")
        from huggingface_hub import hf_hub_download
        artifacts = {}
        for repo, revision, filename in ((MODEL_ID, MODEL_REVISION, "model.safetensors"),
                                         (MODEL_ID, MODEL_REVISION, "config.json"),
                                         (PROCESSOR_ID, PROCESSOR_REVISION, "preprocessor_config.json")):
            path = Path(hf_hub_download(repo, filename, revision=revision, local_files_only=True))
            artifacts[f"{repo}/{filename}"] = {"revision": revision, "sha256": sha256(path),
                                              "bytes": path.stat().st_size}
        self.provenance = {"artifacts": artifacts, "id2label": self.model.config.id2label,
                           "label2id": self.model.config.label2id,
                           "processor_class": type(self.processor).__name__}

    @torch.inference_mode()
    def batch(self, images):
        pixels = self.processor(images=images, return_tensors="pt")["pixel_values"].to("cuda")
        result = self.model(pixels, output_attentions=False, output_hidden_states=True)
        if len(result.hidden_states) != 13 or result.logits.shape != (len(images), 100):
            raise RuntimeError("Frozen model dimensions differ")
        if any(t.shape != (len(images), 197, 768) or t.dtype != torch.float32 for t in result.hidden_states[1:]):
            raise RuntimeError("Post-block hidden dimensions/precision differ")
        cls = torch.stack([t[:, 0] for t in result.hidden_states[1:]], dim=1).cpu().half()
        logits = result.logits.float().cpu()
        if not torch.isfinite(cls).all() or not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite frozen feature capture")
        return {"cls": cls, "logits": logits}


def input_path(root, source, severity):
    return clean_path(root, source) if source == "clean_test" else corruption_path(root, source, severity)


def input_provenance(data_root, pairs):
    require_slurm()
    return {str(input_path(data_root, source, severity).relative_to(data_root)):
            {"bytes": input_path(data_root, source, severity).stat().st_size,
             "sha256": sha256(input_path(data_root, source, severity))}
            for source, severity in pairs}


def check_parity(values, rows, labels):
    require_slurm()
    logits = values["logits"]
    original_pred = torch.tensor([r["pred"] for r in rows], dtype=torch.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if (not torch.equal(logits.argmax(-1), original_pred)
            or not np.array_equal(labels, [r["label"] for r in rows])
            or any(r["y"] != int(r["pred"] != r["label"]) for r in rows)):
        raise RuntimeError("Raw-source label or fresh predicted class differs from canonical scan; no label replacement")
    top2 = torch.softmax(logits, -1).topk(2, -1).values
    confidence = top2[:, 0].numpy()
    margin = (top2[:, 0] - top2[:, 1]).numpy()
    np.testing.assert_allclose(confidence, [r["confidence"] for r in rows],
                               atol=PARITY["confidence_atol"], rtol=PARITY["confidence_rtol"])
    np.testing.assert_allclose(margin, [r["margin"] for r in rows],
                               atol=PARITY["margin_atol"], rtol=PARITY["margin_rtol"])
    return {"records": len(rows), "predictions_exact": True, "labels_exact": True,
            "confidence_max_abs": float(np.max(np.abs(confidence - [r["confidence"] for r in rows]))),
            "margin_max_abs": float(np.max(np.abs(margin - [r["margin"] for r in rows])))}


def extract(root, data_root, mode="full"):
    require_slurm()
    root, data_root = Path(root).resolve(), Path(data_root).resolve()
    current = campaign(root)
    total_began = time.monotonic()
    rows = [r for r in read(root / "records.json") if r["pool"] != "meta_val"]
    if mode == "preflight":
        wanted = set(read(root / "preflight_records.json")["record_ids"])
        rows = [r for r in rows if r["record_id"] in wanted]
    elif mode != "full":
        raise ValueError("Unknown extraction mode")
    if mode == "full":
        prior = read(root / "preflight" / "complete.json")
        if prior.get("complete") is not True or prior["campaign_sha256"] != sha256(root / "campaign.json"):
            raise RuntimeError("Preflight must pass before full capture")
    pairs = sorted({(r["source"], r["severity"]) for r in rows})
    source_manifest = read(data_root / "dataset_manifest.json")
    if source_manifest.get("complete") is not True:
        raise RuntimeError("Pinned canonical parquet source download is incomplete")
    for source, severity in pairs:
        path = input_path(data_root, source, severity)
        verify(path, source_manifest["files"][str(path.relative_to(data_root))]["sha256"])
    provenance = {"dataset_manifest_sha256": sha256(data_root / "dataset_manifest.json"),
                  "revisions": source_manifest["revisions"],
                  "files": input_provenance(data_root, pairs)}
    directory = root / ("preflight_cls" if mode == "preflight" else "cls")
    identity = {"campaign_sha256": sha256(root / "campaign.json"), "records": len(rows), "mode": mode,
                "record_ids_sha256": digest([r["record_id"] for r in rows]), "parity": PARITY,
                "shape_per_record": [12, 768], "dtype": "float16", "data_sha256": digest(provenance),
                "batch_size": BATCH_SIZE, "shard_size": SHARD_SIZE}
    with lock(directory, "extract"):
        frozen_json(directory / "data_provenance.json", provenance)
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            old = read(manifest_path)
            if any(old.get(k) != v for k, v in identity.items()):
                raise RuntimeError("Extraction resume identity changed")
            if old.get("complete"):
                verify(directory / "index.json", old["index_sha256"])
                for spec in old["shards"]:
                    verify(directory / spec["path"], spec["sha256"])
                return old
        capture_started = time.monotonic()
        capture = Capture()
        capture_load_seconds = time.monotonic() - capture_started
        processor = json.loads(json.dumps(capture.processor.to_dict()))
        frozen_json(directory / "processor.json", processor)
        model_provenance = json.loads(json.dumps(capture.provenance))
        frozen_json(directory / "model_provenance.json", model_provenance)
        if mode == "full":
            if processor != read(root / "preflight_cls" / "processor.json"):
                raise RuntimeError("Processor changed after preflight")
            if provenance != read(root / "preflight_cls" / "data_provenance.json"):
                raise RuntimeError("Raw source files changed after preflight")
            if model_provenance != read(root / "preflight_cls" / "model_provenance.json"):
                raise RuntimeError("Cached model/processor artifacts changed after preflight")
        began = time.monotonic()
        shards, index = [], []
        maxima = {"confidence_max_abs": 0.0, "margin_max_abs": 0.0}
        timers = {"decode_seconds": 0.0, "inference_seconds": 0.0, "parity_seconds": 0.0,
                  "newly_extracted_records": 0}
        torch.cuda.reset_peak_memory_stats()
        for start in range(0, len(rows), SHARD_SIZE):
            subset = rows[start:start + SHARD_SIZE]
            name = f"shards/shard_{start // SHARD_SIZE:05d}.pt"
            path, receipt_path = directory / name, (directory / name).with_suffix(".json")
            expected = {"capture_identity_sha256": digest(identity), "record_ids": [r["record_id"] for r in subset]}
            if receipt_path.exists():
                receipt = read(receipt_path)
                if receipt["identity"] != expected:
                    raise RuntimeError("Existing shard belongs to a different capture")
                verify(path, receipt["sha256"])
            else:
                # A payload without its receipt is an interrupted write; preserve it,
                # then recapture only this incomplete shard, never completed shards.
                if path.exists():
                    preserved = path.with_name(path.name + f".unsealed.{os.environ['SLURM_JOB_ID']}.{time.time_ns()}")
                    path.rename(preserved)
                buffers = {"cls": [], "logits": []}
                parity = {"records": 0, **{key: 0.0 for key in maxima}}
                for offset in range(0, len(subset), BATCH_SIZE):
                    batch = subset[offset:offset + BATCH_SIZE]
                    started = time.monotonic()
                    pools = [get_pool(r["source"], r["severity"], data_root) for r in batch]
                    images = [pool.image(r["image_id"]) for pool, r in zip(pools, batch)]
                    labels = [pool.label(r["image_id"]) for pool, r in zip(pools, batch)]
                    timers["decode_seconds"] += time.monotonic() - started
                    started = time.monotonic()
                    values = capture.batch(images)
                    timers["inference_seconds"] += time.monotonic() - started
                    started = time.monotonic()
                    try:
                        checked = check_parity(values, batch, labels)
                    except Exception as exc:
                        probabilities = torch.softmax(values["logits"], -1).topk(2, -1).values
                        atomic_json(directory / "failures" / f"parity_{os.environ['SLURM_JOB_ID']}_{time.time_ns()}.json",
                                    {"exception": repr(exc), "rows": batch, "source_labels": labels,
                                     "fresh_pred": values["logits"].argmax(-1).tolist(),
                                     "fresh_confidence": probabilities[:, 0].tolist(),
                                     "fresh_margin": (probabilities[:, 0] - probabilities[:, 1]).tolist(),
                                     "capture_identity": identity})
                        raise
                    timers["parity_seconds"] += time.monotonic() - started
                    timers["newly_extracted_records"] += len(batch)
                    parity["records"] += len(batch)
                    for key in maxima:
                        parity[key] = max(parity[key], checked[key])
                    for key in buffers:
                        buffers[key].append(values[key])
                payload = {key: torch.cat(values) for key, values in buffers.items()}
                payload.update({key: torch.tensor([r[key] for r in subset], dtype=torch.int64) for key in METADATA})
                atomic_torch(path, payload)
                receipt = {"identity": expected, "sha256": sha256(path), "parity": parity,
                           "rows": [{**r, "cls_shard": name, "cls_offset": i} for i, r in enumerate(subset)],
                           "job_id": os.environ["SLURM_JOB_ID"]}
                atomic_json(receipt_path, receipt)
            shards.append({"path": name, "sha256": receipt["sha256"], "records": len(subset)})
            index.extend(receipt["rows"])
            for key in maxima:
                maxima[key] = max(maxima[key], receipt["parity"][key])
            atomic_json(manifest_path, {**identity, "complete": False, "completed_records": len(index), "shards": shards})
            print(json.dumps({"event": "cls_shard_complete", "records": len(index), "total": len(rows)}), flush=True)
            if STOP:
                raise InterruptedError("Resume same capture after atomically saved shard")
        frozen_json(directory / "index.json", index)
        result = {**identity, "complete": True, "completed_records": len(index), "shards": shards,
                  "index_sha256": sha256(directory / "index.json"), "parity_maximum": maxima,
                  "processor_sha256": sha256(directory / "processor.json"),
                  "model_provenance_sha256": sha256(directory / "model_provenance.json"),
                  "timing": {**timers, "model_load_seconds": capture_load_seconds,
                             "total_invocation_seconds": time.monotonic() - total_began},
                  "elapsed_seconds": time.monotonic() - began, "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                  "gpu": torch.cuda.get_device_name(), "job_id": os.environ["SLURM_JOB_ID"]}
        atomic_json(manifest_path, result)
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
