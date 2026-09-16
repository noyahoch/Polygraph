"""Append-only H3/H6/H9 extraction; the attention cache and H12 stay read-only."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import datetime as dt
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import time

import torch

from pilots.layer_screen_20260913.protocol import (
    MODEL_ID, MODEL_REVISION, PROCESSOR_ID, PROCESSOR_REVISION, digest,
)
from pilots.topology_20260910.extract import OfficialImages
from pilots.topology_20260910.data import verify_cached_file

from .data import (
    HIDDEN_INDICES, atomic_torch, cache_index, diagnostic_rows, hidden_inventory,
    sidecar_identity,
)
from .protocol import (
    ROLES, SEEDS, atomic_json, campaign_sha, check_cutoff, read, require_slurm, role_spec,
    sha256, write_frozen,
)
from .train import initialize, record_failure, stage_guard

CAPTURE_BATCH_SIZE = 4
PARITY = {"logits_atol": 1e-5, "logits_rtol": 1e-5,
          "hidden_atol": 1e-5, "hidden_rtol": 1e-5,
          "hidden_comparison": "stored FP16 H12 promoted to FP32",
          "predictions": "exact", "labels": "exact; official raw-data lookup"}
STOP = False


class Capture:
    def __init__(self):
        require_slurm()
        if not torch.cuda.is_available():
            raise RuntimeError("Hidden extraction requires an allocated CUDA GPU")
        initialize(SEEDS[0])
        from transformers import AutoImageProcessor, ViTForImageClassification
        self.processor = AutoImageProcessor.from_pretrained(
            PROCESSOR_ID, revision=PROCESSOR_REVISION, use_fast=True, local_files_only=True)
        self.model = ViTForImageClassification.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, use_safetensors=True,
            attn_implementation="eager", local_files_only=True).eval().to("cuda")
        self.model.requires_grad_(False)
        if self.model.config._attn_implementation != "eager":
            raise RuntimeError("The frozen ViT must use the pinned eager attention implementation")

    @torch.inference_mode()
    def batch(self, images):
        require_slurm()
        inputs = self.processor(images=images, return_tensors="pt").to("cuda")
        result = self.model(**inputs, output_attentions=True, output_hidden_states=True)
        if (result.attentions is None or len(result.attentions) != 12
                or result.hidden_states is None or len(result.hidden_states) != 13):
            raise RuntimeError("Expected twelve eager-attention blocks and hidden_states[0..12]")
        values = {"logits": result.logits.float().cpu()}
        if values["logits"].shape != (len(images), 100):
            raise RuntimeError("Expected 100 frozen classifier outputs")
        for layer in (*HIDDEN_INDICES, 12):
            hidden = result.hidden_states[layer]
            if hidden.shape != (len(images), 197, 768) or hidden.dtype != torch.float32:
                raise RuntimeError("Every own-layer capture must retain all 197 FP32 tokens")
            values[f"hidden_{layer}"] = hidden.to("cpu", torch.float16)
        for name, value in values.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError("Nonfinite frozen ViT output: " + name)
        return values


def check_parity(captured, cache_payload, records, labels):
    """Fixed tolerances are applied to every exported record, not just a sample."""
    require_slurm()
    offsets = torch.tensor([row["offset"] for row in records], dtype=torch.int64)
    expected_ids = torch.tensor([row["record_id"] for row in records], dtype=torch.int64)
    if not torch.equal(cache_payload["record_id"][offsets], expected_ids):
        raise RuntimeError("Hidden extraction/cache record identity mismatch")
    logits, hidden = cache_payload["logits"][offsets], cache_payload["hidden"][offsets]
    if logits.dtype != torch.float32 or hidden.dtype != torch.float16:
        raise RuntimeError("Immutable cache storage precision changed")
    original_labels = torch.tensor([row["label"] for row in records], dtype=torch.int64)
    official_labels = torch.tensor([int(labels[row["image_id"]]) for row in records], dtype=torch.int64)
    expected_pred = torch.tensor([row["pred"] for row in records], dtype=torch.int64)
    if (not torch.equal(original_labels, official_labels)
            or not torch.equal(logits.argmax(1), expected_pred)
            or not torch.equal(captured["logits"].argmax(1), expected_pred)
            or any(row["y"] != int(row["pred"] != row["label"]) for row in records)):
        raise RuntimeError("Pinned ViT/raw-data labels/predictions do not match the immutable cache")
    torch.testing.assert_close(captured["logits"], logits, atol=PARITY["logits_atol"],
                               rtol=PARITY["logits_rtol"])
    torch.testing.assert_close(captured["hidden_12"].float(), hidden.float(),
                               atol=PARITY["hidden_atol"], rtol=PARITY["hidden_rtol"])
    return {
        "records": len(records),
        "logits_max_absolute_error": float((captured["logits"] - logits).abs().max()),
        "h12_max_absolute_error": float((captured["hidden_12"].float() - hidden.float()).abs().max()),
        "predictions_exact": True, "labels_exact": True,
    }


def _publish_link(source, target):
    if target.exists():
        if sha256(source) != sha256(target):
            raise RuntimeError("Refusing to replace an existing hidden artifact: " + str(target))
    else:
        os.link(source, target)


def _recover_or_read(path, expected):
    receipt = path.with_suffix(".json")
    pending = path.with_name(path.name + ".pending")
    pending_receipt = receipt.with_name(receipt.name + ".pending")
    if path.exists() and receipt.exists():
        done = read(receipt)
    elif pending.exists() and pending_receipt.exists():
        done = read(pending_receipt)
        if any(done.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Interrupted hidden-shard transaction has a different source/role identity")
        verify_cached_file(pending, done["sha256"])
        _publish_link(pending, path)
        _publish_link(pending_receipt, receipt)
        pending.unlink()
        pending_receipt.unlink()
    elif path.exists() or receipt.exists() or pending_receipt.exists():
        raise RuntimeError("Incomplete hidden-shard transaction lacks its bound payload/receipt")
    else:
        if pending.exists():
            # An uncommitted payload has no scientific receipt. Preserve it, then
            # recapture exactly the same fixed inputs on an authorized resume.
            orphan = pending.with_name(pending.name + f".uncommitted.{time.time_ns()}")
            pending.rename(orphan)
        return None
    if any(done.get(key) != value for key, value in expected.items()) or done.get("complete") is not True:
        raise RuntimeError("Completed hidden shard does not match its capture identity")
    verify_cached_file(path, done["sha256"])
    return done


def _write_shard(path, payload, receipt):
    pending = path.with_name(path.name + ".pending")
    pending_receipt = path.with_suffix(".json").with_name(path.with_suffix(".json").name + ".pending")
    if path.exists() or path.with_suffix(".json").exists() or pending.exists() or pending_receipt.exists():
        raise RuntimeError("Hidden shards are append-only; reconcile existing transactions first")
    atomic_torch(pending, payload)
    receipt = {**receipt, "sha256": sha256(pending), "complete": True}
    atomic_json(pending_receipt, receipt)
    _publish_link(pending, path)
    _publish_link(pending_receipt, path.with_suffix(".json"))
    pending.unlink()
    pending_receipt.unlink()
    return receipt


def _stop_handler(_signum, _frame):
    global STOP
    STOP = True


def extract(root, data_root, mode="full", *, deadline=None):
    require_slurm()
    if mode not in ("preflight", "full"):
        raise ValueError("Extraction mode must be preflight or full")
    if not torch.cuda.is_available():
        raise RuntimeError("Hidden extraction requires an allocated GPU; no CPU fallback")
    root = Path(root).resolve()
    campaign, cache_manifest, all_rows = cache_index(root)
    diagnostic = mode == "preflight"
    hidden = root / "preflight" / "hidden" if diagnostic else Path(campaign["hidden_root"])
    if hidden.resolve() != hidden or hidden.resolve().is_relative_to(Path(campaign["cache"])):
        raise RuntimeError("Sidecars must be in the new namespace, never the immutable cache")
    rows = diagnostic_rows(root, all_rows) if diagnostic else all_rows
    identity = sidecar_identity(root, campaign, cache_manifest, rows, diagnostic)
    if diagnostic:
        bounded = time.monotonic() + 1800
        deadline = bounded if deadline is None else min(deadline, bounded)

        def guard():
            if time.monotonic() >= deadline:
                raise TimeoutError("Bounded hidden preflight reached its internal deadline")
    else:
        guard = stage_guard(root, "base")
    guard()
    hidden.mkdir(parents=True, exist_ok=True)
    (hidden / "shards").mkdir(exist_ok=True)
    began = time.monotonic()
    with (hidden / ".extract.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = hidden / "manifest.json"
        if manifest_path.exists():
            previous = read(manifest_path)
            if any(previous.get(key) != value for key, value in identity.items()):
                raise RuntimeError("Cannot resume hidden extraction with different source/data/roles")
            if previous.get("complete") is True:
                hidden_inventory(root, campaign, cache_manifest, all_rows, diagnostic)
                for shard in previous["shards"]:
                    verify_cached_file(hidden / shard["path"], shard["sha256"])
                    verify_cached_file((hidden / shard["path"]).with_suffix(".json"),
                                       shard["receipt_sha256"])
                return previous
        else:
            atomic_json(manifest_path, {**identity, "complete": False, "completed_records": 0, "shards": []})
        attempt_path = hidden / "attempts.json"
        attempts = read(attempt_path) if attempt_path.exists() else []
        attempt = {"mode": mode, "started_unix": time.time(), "status": "running",
                   "job_id": os.environ["SLURM_JOB_ID"], "batch_size": CAPTURE_BATCH_SIZE}
        attempts.append(attempt)
        atomic_json(attempt_path, attempts)
        try:
            cache = Path(campaign["cache"])
            verify_cached_file(cache / "processor.json", cache_manifest["processor_sha256"])
            verify_cached_file(cache / "data_provenance.json", cache_manifest["data_provenance_sha256"])
            images = OfficialImages(Path(data_root))
            guard()
            provenance = images.provenance()
            if provenance != read(cache / "data_provenance.json"):
                raise RuntimeError("Official raw-data bytes differ from the original extraction")
            write_frozen(hidden / "data_provenance.json", provenance)
            capture = Capture()
            processor = json.loads(json.dumps(capture.processor.to_dict()))
            if processor != read(cache / "processor.json"):
                raise RuntimeError("Pinned fast processor configuration differs from the immutable cache")
            write_frozen(hidden / "processor.json", processor)
            capture_identity = {
                "identity_sha256": digest(identity), "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "processor_id": PROCESSOR_ID, "processor_revision": PROCESSOR_REVISION,
                "attention_implementation": "eager", "precision": "float32",
                "batch_size": CAPTURE_BATCH_SIZE, "parity": PARITY, "tf32": False,
                "execution_sha256": None if diagnostic else sha256(root / "execution.json"),
                "processor_sha256": sha256(hidden / "processor.json"),
                "data_provenance_sha256": sha256(hidden / "data_provenance.json"),
            }
            write_frozen(hidden / "capture_identity.json", capture_identity)
            capture_sha = sha256(hidden / "capture_identity.json")
            versions = {name: importlib.metadata.version(name)
                        for name in ("torch", "transformers", "numpy", "torchvision", "Pillow")}
            attempt.update(versions=versions, gpu=torch.cuda.get_device_name(),
                           cuda=torch.version.cuda, capture_identity_sha256=capture_sha)
            atomic_json(attempt_path, attempts)
            grouped = OrderedDict()
            for row in rows:
                grouped.setdefault(row["shard"], []).append(row)
            cache_shards = {Path(spec["path"]).name: spec for spec in cache_manifest["shards"]}
            role_of = {record: role for role in ROLES for record in role_spec(root, role)["record_ids"]}
            index, shards, captured_count = [], [], 0
            parity_max = {"logits_max_absolute_error": 0.0, "h12_max_absolute_error": 0.0}
            raw_remaining = len(rows) * 3 * 197 * 768 * 2
            if shutil.disk_usage(hidden).free < raw_remaining + 2 * 64 * 3 * 197 * 768 * 2:
                raise RuntimeError("Insufficient sidecar storage; no automatic resource upgrade")
            inference_started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            for name, subset in grouped.items():
                guard()
                expected = {
                    "capture_identity_sha256": capture_sha, "cache_shard": name,
                    "cache_sha256": cache_shards[name]["sha256"],
                    "record_ids": [row["record_id"] for row in subset], "diagnostic_only": diagnostic,
                }
                path = hidden / "shards" / name
                done = _recover_or_read(path, expected)
                if done is None:
                    original_path = cache / "shards" / name
                    before = verify_cached_file(original_path, cache_shards[name]["sha256"])
                    original = torch.load(original_path, map_location="cpu", weights_only=True)
                    if verify_cached_file(original_path, cache_shards[name]["sha256"]) != before:
                        raise RuntimeError("Immutable cache changed during hidden parity capture")
                    buffers = {layer: [] for layer in HIDDEN_INDICES}
                    shard_parity = {"records": 0, "logits_max_absolute_error": 0.0,
                                    "h12_max_absolute_error": 0.0, "predictions_exact": True, "labels_exact": True}
                    for start in range(0, len(subset), CAPTURE_BATCH_SIZE):
                        guard()
                        selected = subset[start:start + CAPTURE_BATCH_SIZE]
                        values = capture.batch([images.image(row) for row in selected])
                        guard()
                        parity = check_parity(values, original, selected, images.labels)
                        for key in parity_max:
                            shard_parity[key] = max(shard_parity[key], parity[key])
                        shard_parity["records"] += len(selected)
                        for layer in HIDDEN_INDICES:
                            buffers[layer].append(values[f"hidden_{layer}"])
                        captured_count += len(selected)
                    payload = {f"hidden_{layer}": torch.cat(buffers[layer]) for layer in HIDDEN_INDICES}
                    payload["record_id"] = torch.tensor(expected["record_ids"], dtype=torch.int64)
                    shard_rows = [{
                        "record_id": row["record_id"], "cache_shard": name, "cache_offset": row["offset"],
                        "hidden_shard": name, "hidden_offset": offset, "role": role_of[row["record_id"]],
                    } for offset, row in enumerate(subset)]
                    guard()
                    done = _write_shard(path, payload, {**expected, "rows": shard_rows,
                                                       "parity": shard_parity, "completed_unix": time.time()})
                    del original, payload, buffers
                if done["parity"]["records"] != len(subset):
                    raise RuntimeError("Every hidden record requires its own immutable-cache parity proof")
                for key in parity_max:
                    parity_max[key] = max(parity_max[key], done["parity"][key])
                index.extend(done["rows"])
                shards.append({
                    "path": "shards/" + name, "sha256": done["sha256"], "records": len(subset),
                    "cache_shard": name, "cache_sha256": cache_shards[name]["sha256"],
                    "receipt_sha256": sha256(path.with_suffix(".json")),
                })
                atomic_json(manifest_path, {**identity, "complete": False, "shards": shards,
                                           "completed_records": len(index)})
                print(json.dumps({"event": "hidden_shard_complete", "mode": mode,
                                  "records": len(index), "total": len(rows)}), flush=True)
                if STOP:
                    raise InterruptedError("Hidden extraction stopped after a committed shard; resume is explicit")
            if [entry["record_id"] for entry in index] != [row["record_id"] for row in rows]:
                raise RuntimeError("Hidden export did not preserve original shard/record order")
            guard()
            if not diagnostic:
                check_cutoff(root, "base")
            write_frozen(hidden / "index.json", index)
            elapsed = time.monotonic() - inference_started
            total_bytes = sum((hidden / spec["path"]).stat().st_size for spec in shards)
            summary = {
                "diagnostic_only": diagnostic, "records": len(rows), "total_bytes": total_bytes,
                "newly_captured_records": captured_count, "capture_and_io_seconds": elapsed,
                "startup_seconds": inference_started - began, "elapsed_seconds": time.monotonic() - began,
                "projected_28800_bytes": total_bytes * 28800 / len(rows),
                "projected_28800_capture_seconds": elapsed * 28800 / captured_count if captured_count else None,
                "projection_caveat": "Fixed small sample; no resource upgrade or outcome tuning; startup/contention excluded",
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                "free_bytes": shutil.disk_usage(hidden).free, "parity": {**PARITY, **parity_max},
            }
            atomic_json(hidden / "capture_summary.json", summary)
            final = {
                **identity, "complete": True, "shards": shards, "completed_records": len(index),
                "index_sha256": sha256(hidden / "index.json"), "capture_identity_sha256": capture_sha,
                "processor_sha256": sha256(hidden / "processor.json"),
                "data_provenance_sha256": sha256(hidden / "data_provenance.json"),
                "parity": {**PARITY, **parity_max}, "completed_unix": time.time(),
                "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "job_id": os.environ["SLURM_JOB_ID"],
            }
            guard()
            atomic_json(manifest_path, final)
            hidden_inventory(root, campaign, cache_manifest, all_rows, diagnostic)
            attempt.update(status="complete", ended_unix=time.time(), captured_records=captured_count)
            atomic_json(attempt_path, attempts)
            return final
        except BaseException as error:
            attempt.update(status="failed", complete=False, ended_unix=time.time(), error=repr(error))
            atomic_json(attempt_path, attempts)
            record_failure(root, "extract", {"mode": mode}, error)
            raise


def main():
    require_slurm()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "full"), required=True)
    args = parser.parse_args()
    handlers = {name: signal.signal(name, _stop_handler) for name in (signal.SIGTERM, signal.SIGINT)}
    try:
        extract(args.root, args.data_root, args.mode)
    except BaseException as error:
        record_failure(args.root, "extract", {"mode": args.mode}, error)
        raise
    finally:
        for name, handler in handlers.items():
            signal.signal(name, handler)


if __name__ == "__main__":
    main()
