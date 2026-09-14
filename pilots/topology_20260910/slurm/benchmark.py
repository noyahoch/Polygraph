"""Disposable timing only, inside Slurm and on explicitly diagnostic data."""
import argparse
import json
import os
from pathlib import Path
import time

MODULE_STARTED = time.time()
import torch

from pilots.topology_20260910.protocol import atomic_json, protocol, require_slurm


def capture(a):
    from pilots.topology_20260910 import extract
    timings = []
    original = extract.Capture.batch

    def timed(self, images):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original(self, images)
        torch.cuda.synchronize()
        timings.append({"records": len(images), "seconds": time.perf_counter() - start})
        return result

    extract.Capture.batch = timed
    started = time.perf_counter()
    extract.capture(argparse.Namespace(cache=a.cache, protocol=None, max_records=270,
                                      shard_size=256, batch_size=18, data_root=a.data_root,
                                      device="cuda", resume=True))
    total = time.perf_counter() - started
    manifest = json.loads((a.cache / "manifest.json").read_text())
    count = manifest["completed_records"]
    size = sum((a.cache / x["path"]).stat().st_size for x in manifest["shards"])
    steady = timings[1:]
    rate = sum(t["seconds"] for t in steady) / sum(t["records"] for t in steady)
    full_records = sum(protocol()["photo_counts"].values()) * len(protocol()["views"])
    result = {"diagnostic_only": True, "records": count, "first_batch_warmup": timings[0],
              "steady_batches": steady, "steady_seconds_per_record": rate,
              "whole_capture_seconds": total, "whole_capture_seconds_per_record": total / count,
              "cache_shard_bytes": size, "projected_full_cache_bytes": size * full_records / count,
              "projected_full_inference_seconds": rate * full_records,
              "projection_caveat": "Small diagnostic sample; network cold start, I/O contention and graph density may differ"}
    atomic_json(a.cache / "timing.json", result)
    print(json.dumps(result), flush=True)


def training(a):
    from pilots.topology_20260910 import train
    from pilots.topology_20260910.data import CachedDataset
    from pilots.topology_20260910.models import build_model
    manifest = json.loads((a.cache / "manifest.json").read_text())
    if not manifest.get("diagnostic_only") and not a.development_only:
        raise RuntimeError("Production cache timing requires explicit development-only mode")
    if a.development_only and a.out is None:
        raise RuntimeError("Development-only timing must have an isolated output path")
    device = torch.device("cuda")
    train._seed(1)
    torch.cuda.reset_peak_memory_stats()
    data = CachedDataset(a.cache, "train", "full_graph")
    val = CachedDataset(a.cache, "val", "full_graph")
    shard_events = []
    def track(dataset):
        original_load = dataset._load
        def load(filename, rewired=False):
            memory = dataset._rewired if rewired else dataset._shards
            miss = filename not in memory
            start = time.perf_counter()
            result = original_load(filename, rewired=rewired)
            if miss:
                path = dataset.cache / ("rewire" if rewired else "shards") / filename
                shard_events.append({"split": dataset.split, "shard": filename,
                                     "logical_file_bytes": path.stat().st_size,
                                     "load_and_hash_seconds": time.perf_counter() - start})
            return result
        dataset._load = load
    track(data)
    track(val)
    # Fixed model/optimizer/seed; only disposable sampler block rotation differs
    # so simultaneous readers touch distinct shards. Baseline uses worker0 too.
    if a.development_only:
        blocks = data.shard_blocks()
        offset = a.worker_index * (len(blocks) // 2)
        rotated = blocks[offset:] + blocks[:offset]
        data.shard_blocks = lambda: rotated
    config = train._make_config(a.cache, data, "full_graph", 1, device)
    model = build_model("full_graph").to(device).train()
    optim = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"],
                             weight_decay=config["training"]["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config["pos_weight"], device=device))
    startup_seconds = time.time() - MODULE_STARTED
    barrier_seconds = 0.0
    if a.barrier_dir:
        if not a.development_only or a.worker_index not in (0, 1):
            raise RuntimeError("A paired barrier is only valid for explicit disposable production timing")
        a.barrier_dir.mkdir(parents=True, exist_ok=True)
        ready = a.barrier_dir / f"worker{a.worker_index}.json"
        if ready.exists():
            raise RuntimeError("Refusing a stale I/O barrier; use a fresh reviewed phase")
        started_wait = time.time()
        atomic_json(ready, {"job_id": os.environ["SLURM_JOB_ID"], "ready_epoch": started_wait})
        peers = [a.barrier_dir / f"worker{i}.json" for i in range(2)]
        while not all(path.is_file() for path in peers):
            if time.time() - started_wait > 300:
                raise RuntimeError("Two-worker I/O barrier timed out without both allocations ready")
            time.sleep(0.25)
        start_epoch = max(json.loads(path.read_text())["ready_epoch"] for path in peers) + 5
        if time.time() > start_epoch + 1:
            raise RuntimeError("Paired I/O benchmark missed its common start")
        time.sleep(max(0, start_epoch - time.time()))
        barrier_seconds = time.time() - started_wait
    rows = []
    sampler = train.BlockShuffleSampler(data, 1) if a.development_only else None
    iterator = iter(train._loader(data, config, sampler=sampler))
    measured_started = time.time()
    while len(rows) < (64 if a.development_only else 4):
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        collated = time.perf_counter()
        optim.zero_grad(set_to_none=True)
        scores, y = train._forward(model, batch, "full_graph", device)
        loss = loss_fn(scores, y)
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise RuntimeError("Nonfinite benchmark gradient")
        optim.step()
        torch.cuda.synchronize()
        rows.append({"records": len(y), "edges": int(batch.edge_index.shape[1]),
                     "data_seconds": collated - start, "total_seconds": time.perf_counter() - start})
    training_ended = time.time()
    distinct_train_shards = sorted({event["shard"] for event in shard_events if event["split"] == "train"})
    if a.development_only and (len(rows) < 32 or len(distinct_train_shards) < 3):
        raise RuntimeError("Production admission timing did not span32batches and3distinct shards")
    torch.cuda.synchronize()
    start = time.perf_counter()
    # Keep the I/O admission bounded; neither a saved model nor final-test scores
    # are produced, and the validation subset is never used for scientific tuning.
    indices = list(range(min(len(val), 96))) if a.development_only else None
    values = train._collect(model, val, config, device, indices=indices)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result = {"diagnostic_only": True, "cache_is_diagnostic": manifest.get("diagnostic_only", False),
              "development_only": a.development_only, "arm": "full_graph", "batch_size": config["batch_size"],
              "training_batches": rows, "warmup_batch_included_at_index": 0,
              "validation_records": len(values), "validation_seconds": elapsed,
              "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "test_opened": False,
              "model_saved": False,
              "worker_index": a.worker_index, "startup_seconds": startup_seconds,
              "barrier_seconds": barrier_seconds, "training_started_epoch": measured_started,
              "training_ended_epoch": training_ended, "shard_cache_misses": shard_events,
              "distinct_training_shards": distinct_train_shards,
              "logical_shard_read_bytes": sum(event["logical_file_bytes"] for event in shard_events),
              "io_caveat": "Logical cache misses include hash/read work; operating-system/NFS cache may be warm, not physical disk bytes",
              "claim": "Disposable train/validation throughput check, no predictive-performance result or model promotion"}
    atomic_json(a.out or a.cache / "training_timing.json", result)
    print(json.dumps(result), flush=True)


def main():
    require_slurm()
    if not torch.cuda.is_available():
        raise RuntimeError("Allocated CUDA required")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["capture", "train"])
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--development-only", action="store_true")
    p.add_argument("--worker-index", type=int, choices=[0, 1], default=0)
    p.add_argument("--barrier-dir", type=Path)
    a = p.parse_args()
    capture(a) if a.mode == "capture" else training(a)


if __name__ == "__main__":
    main()
