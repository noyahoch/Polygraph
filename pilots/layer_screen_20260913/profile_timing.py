"""Slurm-only timing diagnosis; no scientific fits or persistent model weights.

Keep startup, first CUDA use, verified shard reloads, graph reconstruction,
transfer, training, evaluation and serialization costs separate. A Python-cache
eviction does NOT make the operating system or shared filesystem cache cold.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import statistics
import tempfile
import time

import numpy as np
import torch

from . import data as data_module
from .data import CachedDataset, atomic_torch
from .models import build_model
from .protocol import ARMS, atomic_json, digest, file_sha256, protocol, require_slurm
from .train import (BlockShuffleSampler, _atomic_npz, _atomic_safetensors,
                    _cpu_state, _finite, _forward, _loader, _rng_state, _seed)


def stats(values):
    values = list(values)
    return {"count": len(values), "median": statistics.median(values),
            "mean": statistics.mean(values), "min": min(values), "max": max(values)} if values else None


class TraceIO:
    """Observe the unmodified loader's real verification and torch.load calls."""
    def __init__(self):
        self.events = []
        self.phase = "unset"
        self.observed_verifications = set()

    @contextmanager
    def install(self):
        original_verify = data_module.verify_cached_file
        original_load = torch.load

        def verify(path, expected):
            key = str(Path(path).resolve())
            first = key not in self.observed_verifications
            started = time.monotonic()
            answer = original_verify(path, expected)
            seconds = time.monotonic() - started
            self.observed_verifications.add(key)
            self.events.append({"phase": self.phase, "kind": "verification", "path": key,
                                "seconds": seconds, "first_observed_in_process": first})
            return answer

        def load(path, *args, **kwargs):
            started = time.monotonic()
            answer = original_load(path, *args, **kwargs)
            seconds = time.monotonic() - started
            self.events.append({"phase": self.phase, "kind": "deserialization", "path": str(path),
                                "seconds": seconds})
            return answer

        data_module.verify_cached_file, torch.load = verify, load
        try:
            yield
        finally:
            data_module.verify_cached_file, torch.load = original_verify, original_load

    def elapsed_since(self, index):
        return sum(event["seconds"] for event in self.events[index:])


def sync():
    torch.cuda.synchronize()


def finite_gradients(model):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            _finite(parameter.grad, f"gradient {name}")


def timeout(signum, frame):
    raise TimeoutError("Technical profiling time limit reached; no production run admitted")


def profile(args):
    require_slurm()
    if not torch.cuda.is_available():
        raise RuntimeError("Allocated CUDA GPU required; no local/CPU fallback")
    start = time.monotonic()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.cache / "manifest.json").read_text())
    if not manifest.get("diagnostic_only") or not manifest.get("complete"):
        raise RuntimeError("This technical profile requires the completed diagnostic cache")
    if manifest.get("records") != 36:
        raise RuntimeError("Reviewed timing sample is the existing 36-record cache")
    shard_paths = [args.cache / row["path"] for row in manifest["shards"]]
    trace = TraceIO()
    result = {"schema_version": 1, "passed": False, "technical_only": True,
              "scientific_fits_created": 0, "test_evaluated": False,
              "started_unix": time.time(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "protocol_sha256": digest(protocol()), "profiler_sha256": file_sha256(__file__),
              "cache_manifest_sha256": file_sha256(args.cache / "manifest.json"),
              "sample_records": manifest["records"], "sample_shards": len(shard_paths),
              "sample_shard_bytes": [path.stat().st_size for path in shard_paths],
              "batch_size": args.batch_size, "warmup_rounds": args.warmup_rounds,
              "measured_rounds": args.repeats, "gpu": torch.cuda.get_device_name(),
              "versions": {n: importlib.metadata.version(n) for n in ("torch", "numpy", "torch-geometric", "safetensors")},
              "torch_num_threads": torch.get_num_threads(), "arms": {},
              "limitations": [
                  "Only 36 development records and one existing shard: no production multishard or contention measurement.",
                  "Clearing Dataset.loaded evicts the application cache only; OS/NFS page-cache status is unknown.",
                  "Production holds one shard; each later shard miss still needs deserialization every epoch. Do not assume the full cache fits RAM.",
                  "verify_cached_file hashes once per unchanged file per process, then checks its stat fingerprint; each fresh fit pays first verification again.",
                  "Evaluation-mode timing uses the diagnostic training rows, not validation outcomes; no model selection is performed.",
                  "Microbenchmark starts after Python imports. Scheduler/bootstrap/import durations must be added from the job log.",
                  "No admission decision is made by this profiler. Preserve the 14-fit/60-epoch upper bound and all prior diagnostic allocation budgets."]}
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.max_seconds)
    try:
        with trace.install(), tempfile.TemporaryDirectory(prefix="timing_disposable_", dir=args.out.parent) as temporary:
            temporary = Path(temporary)
            for arm in ARMS:
                arm_start = time.monotonic()
                trace.phase = f"{arm}/dataset_construction"
                began = time.monotonic()
                ds = CachedDataset(args.cache, "train", arm, diagnostic=True)
                construction = time.monotonic() - began
                if len(ds) < args.batch_size:
                    raise RuntimeError("Diagnostic sample smaller than frozen batch")
                measurements = {"dataset_construction_seconds": construction, "reloads": [],
                                "train_batches": [], "evaluation_batches": [], "serialization": {}}
                result["arms"][arm] = measurements

                # Read each actual available shard. Repetitions preserve the
                # production verification memo; only the one-shard LRU is cleared.
                for repetition in range(3):
                    for path in shard_paths:
                        ds.loaded.clear()
                        trace.phase = f"{arm}/reload/{repetition}/{path.name}"
                        before = len(trace.events)
                        began = time.monotonic()
                        ds._load(path.name)
                        duration = time.monotonic() - began
                        events = trace.events[before:]
                        measurements["reloads"].append({"repetition": repetition, "path": str(path),
                            "bytes": path.stat().st_size, "seconds": duration,
                            "verification_seconds": sum(e["seconds"] for e in events if e["kind"] == "verification"),
                            "deserialization_seconds": sum(e["seconds"] for e in events if e["kind"] == "deserialization"),
                            "application_cache": "evicted", "os_page_cache": "unknown"})
                _seed(7)
                began = time.monotonic()
                model = build_model(arm).to("cuda")
                optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0001)
                # Weight one changes no scientific model: this disposable timing
                # model cannot estimate full-cohort class prevalence from 36 rows.
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0, device="cuda"))
                sync()
                measurements["model_optimizer_setup_seconds"] = time.monotonic() - began
                sampler = BlockShuffleSampler(ds, 7)
                config = {"arm": arm, "batch_size": args.batch_size}
                torch.cuda.reset_peak_memory_stats()
                for repetition in range(args.warmup_rounds + args.repeats):
                    indices = list(sampler)[:args.batch_size]
                    # Alternate resident/reloaded shard behavior, retaining actual
                    # I/O in raw batch wall time and exposing its components.
                    evicted = repetition % 2 == 0
                    if evicted:
                        ds.loaded.clear()
                    trace.phase = f"{arm}/train_batch/{repetition}"
                    before = len(trace.events)
                    began = time.monotonic()
                    batch = next(iter(_loader(ds, config, indices=indices)))
                    loader_seconds = time.monotonic() - began
                    io_seconds = trace.elapsed_since(before)
                    sync(); began = time.monotonic()
                    batch = batch.to("cuda")
                    sync(); transfer_seconds = time.monotonic() - began
                    model.train()
                    sync(); began = time.monotonic()
                    optimizer.zero_grad(set_to_none=True)
                    scores, labels = _forward(model, batch, arm, torch.device("cuda"))
                    loss = criterion(scores, labels)
                    _finite(loss, "training loss")
                    loss.backward()
                    finite_gradients(model)
                    optimizer.step()
                    # Match the production loop's scalar loss synchronization.
                    float(loss.detach())
                    sync(); step_seconds = time.monotonic() - began
                    measurements["train_batches"].append({"round": repetition,
                        "warmup": repetition < args.warmup_rounds, "first_step_for_arm": repetition == 0,
                        "records": len(indices), "record_ids": [ds.entries[i]["record_id"] for i in indices],
                        "edges": int(batch.edge_index.shape[1]), "application_cache_evicted": evicted,
                        "loader_total_seconds": loader_seconds, "loader_io_seconds": io_seconds,
                        "materialization_collation_seconds": max(0.0, loader_seconds - io_seconds),
                        "transfer_seconds": transfer_seconds, "train_step_seconds": step_seconds,
                        "batch_wall_components_seconds": loader_seconds + transfer_seconds + step_seconds})
                    del batch, scores, labels, loss

                for repetition in range(3):
                    indices = list(range(args.batch_size))
                    ds.loaded.clear()
                    trace.phase = f"{arm}/evaluation_batch/{repetition}"
                    before = len(trace.events)
                    began = time.monotonic()
                    batch = next(iter(_loader(ds, config, indices=indices)))
                    loader_seconds = time.monotonic() - began
                    io_seconds = trace.elapsed_since(before)
                    sync(); began = time.monotonic()
                    batch = batch.to("cuda")
                    sync(); transfer_seconds = time.monotonic() - began
                    model.eval()
                    sync(); began = time.monotonic()
                    with torch.no_grad():
                        score, _ = _forward(model, batch, arm, torch.device("cuda"))
                        score.detach().cpu().numpy()
                    sync(); forward_seconds = time.monotonic() - began
                    measurements["evaluation_batches"].append({"round": repetition, "records": len(indices),
                        "loader_total_seconds": loader_seconds, "loader_io_seconds": io_seconds,
                        "materialization_collation_seconds": max(0.0, loader_seconds - io_seconds),
                        "transfer_seconds": transfer_seconds, "evaluation_forward_seconds": forward_seconds})
                    del batch, score

                trace.phase = f"{arm}/serialization"
                sync(); began = time.monotonic()
                snapshot = {"technical_only": True, "model": _cpu_state(model), "optimizer": optimizer.state_dict(),
                            "best_state": _cpu_state(model), "rng": _rng_state(), "sampler": sampler.state_dict(),
                            "history": measurements["train_batches"]}
                sync()
                measurements["serialization"]["snapshot_construction_seconds"] = time.monotonic() - began
                for kind in ("latest", "best", "validation_export"):
                    target = temporary / {"latest": "latest.pt", "best": "best.safetensors", "validation_export": "validation.npz"}[kind]
                    began = time.monotonic()
                    if kind == "latest": atomic_torch(target, snapshot)
                    elif kind == "best": _atomic_safetensors(target, snapshot["best_state"])
                    else:
                        # Conservative small-vector export timing, not predicted
                        # validation scores: explicitly synthetic shape-only data.
                        _atomic_npz(target, {"technical_shape_only": np.ones(7200, dtype=np.int64),
                                            "not_predictions": np.arange(7200, dtype=np.float32)})
                    write_seconds = time.monotonic() - began
                    began = time.monotonic(); file_sha256(target); hash_seconds = time.monotonic() - began
                    measurements["serialization"][kind] = {"bytes": target.stat().st_size,
                        "write_seconds": write_seconds, "hash_seconds": hash_seconds}
                    target.unlink()
                warmed = [row for row in measurements["train_batches"] if not row["warmup"]]
                measurements["warm_summary"] = {key: stats(row[key] for row in warmed) for key in
                    ("loader_total_seconds", "loader_io_seconds", "materialization_collation_seconds", "transfer_seconds", "train_step_seconds", "batch_wall_components_seconds")}
                measurements["reload_summary"] = {key: stats(row[key] for row in measurements["reloads"]) for key in
                    ("seconds", "verification_seconds", "deserialization_seconds")}
                measurements["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                measurements["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
                measurements["total_profile_seconds"] = time.monotonic() - arm_start
                result["io_events"] = trace.events
                result["elapsed_seconds"] = time.monotonic() - start
                atomic_json(args.out, result)
                print(json.dumps({"event": "timing_arm_complete", "arm": arm, "warm_summary": measurements["warm_summary"],
                                  "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
                del ds, model, optimizer, criterion, snapshot
                gc.collect(); torch.cuda.empty_cache()
        result["passed"] = True
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        signal.alarm(0)
        result["io_events"] = trace.events
        result["elapsed_seconds"] = time.monotonic() - start
        result["ended_unix"] = time.time()
        atomic_json(args.out, result)


def main():
    require_slurm()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-seconds", type=int, default=720)
    args = parser.parse_args()
    if args.batch_size != 24 or args.warmup_rounds != 2 or args.repeats != 5 or not 1 <= args.max_seconds <= 720:
        parser.error("Reviewed timing contract: batch24, two warmups, five measured rounds, <=720 seconds")
    profile(args)


if __name__ == "__main__":
    main()
