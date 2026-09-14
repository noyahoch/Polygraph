"""One bounded Slurm diagnostic for worker parity and real loader throughput.

Only the existing 36 diagnostic rows are read. Repeated throughput presentations
are engineering replay, never additional scientific data or fitted checkpoints.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import tempfile
import time
import torch
from pilots.topology_20260910 import data as original_data
from .data import CachedDataset, atomic_torch
from .loader_runtime import settings
from .models import build_model
from .protocol import ARMS, atomic_json, digest, file_sha256, protocol, require_slurm
from .train import (_cpu_state, _finite, _forward, _loader, _restore_rng,
                    _rng_state, _seed, BlockShuffleSampler)


def resident_bytes(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"): return int(line.split()[1]) * 1024
    except (OSError, ValueError): pass
    return None


class ObservedDataset(CachedDataset):
    """Same data; log only cache misses, separately for each worker PID."""
    def __init__(self, *args, trace_dir, phase, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_dir, self.phase = str(trace_dir), phase
    def _load(self, name):
        if name in self.loaded: return super()._load(name)
        path = self.cache / "shards" / name
        first = str(path.resolve()) not in original_data._VERIFIED_FILES
        started = time.monotonic()
        answer = super()._load(name)
        elapsed = time.monotonic() - started
        row = {"pid": os.getpid(), "phase": self.phase, "shard": name,
               "bytes": path.stat().st_size, "load_verify_deserialize_seconds": elapsed,
               "first_verification_in_process": first, "rss_bytes": resident_bytes(os.getpid()),
               "os_nfs_cache_state": "unknown"}
        with (Path(self.trace_dir) / f"worker_{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        return answer


class ReplaySampler(torch.utils.data.Sampler):
    def __init__(self, dataset, traversals=8):
        self.sampler, self.traversals = BlockShuffleSampler(dataset, 7), traversals
    def __len__(self): return len(self.sampler) * self.traversals
    def __iter__(self):
        for _ in range(self.traversals): yield from self.sampler


def cpu_tree(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k,v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def shutdown(loader):
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()
        loader._iterator = None


def tensor_hash(batch):
    h = hashlib.sha256()
    for name in ("x", "edge_index", "edge_attr", "y", "record_id", "image_id", "source_id", "split_id", "output_logits"):
        value = getattr(batch, name).detach().cpu().contiguous()
        h.update(name.encode()); h.update(str((value.dtype, tuple(value.shape))).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def step(model, optimizer, batch, arm):
    optimizer.zero_grad(set_to_none=True)
    scores, labels = _forward(model, batch, arm, torch.device("cuda"))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels,
               pos_weight=torch.tensor(1.0, device="cuda"))
    _finite(loss, "diagnostic loss")
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None: _finite(parameter.grad, f"gradient {name}")
    optimizer.step()
    value = float(loss.detach())
    torch.cuda.synchronize()
    return value


def start_iterator(loader, generator):
    # Verify pinned Torch's one-draw behavior and no parent dropout-RNG use.
    before = generator.get_state()
    expected = torch.Generator().set_state(before)
    torch.empty((), dtype=torch.int64).random_(generator=expected)
    global_before = _rng_state()
    iterator = iter(loader)
    assert torch.equal(generator.get_state(), expected.get_state()), "Loader seed accounting drift"
    after = _rng_state()
    assert torch.equal(global_before["torch"], after["torch"]), "Loader consumed parent CPU RNG"
    assert all(torch.equal(a,b) for a,b in zip(global_before["cuda"], after["cuda"])), "Loader consumed dropout CUDA RNG"
    return iterator


def parity_epoch(loader, generator, model, optimizer, arm):
    model.train()
    rows, ids = [], []
    for batch in start_iterator(loader, generator):
        identity = tensor_hash(batch)
        ids.extend(batch.record_id.tolist())
        loss = step(model, optimizer, batch, arm)
        gradients = torch.cat([p.grad.detach().cpu().flatten() for p in model.parameters() if p.grad is not None])
        rows.append({"input_sha256": identity, "loss": loss, "gradients": gradients})
    return rows, ids


def compare_steps(reference, actual):
    assert len(reference) == len(actual)
    maximum = 0.0
    for a,b in zip(reference, actual):
        assert a["input_sha256"] == b["input_sha256"], "Worker inputs/order changed"
        assert abs(a["loss"] - b["loss"]) <= 1e-5 + 1e-5 * abs(a["loss"]), "Worker loss changed"
        torch.testing.assert_close(a["gradients"], b["gradients"], atol=1e-5, rtol=1e-5)
        maximum = max(maximum, float((a["gradients"] - b["gradients"]).abs().max()))
    return maximum


def compare_state(reference, actual):
    if torch.is_tensor(reference):
        torch.testing.assert_close(reference.cpu(), actual.cpu(), atol=1e-5, rtol=1e-5)
    elif isinstance(reference, dict):
        assert reference.keys() == actual.keys()
        for key in reference: compare_state(reference[key], actual[key])
    elif isinstance(reference, (list, tuple)):
        assert len(reference) == len(actual)
        for a,b in zip(reference,actual): compare_state(a,b)
    else: assert reference == actual


def new_model(arm):
    model = build_model(arm).to("cuda")
    return model, torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0001)


def check_parity(ds, arm, workers, checkpoint_dir):
    _seed(7)
    model, optimizer = new_model(arm)
    sampler = BlockShuffleSampler(ds, 7)
    generator = torch.Generator().manual_seed(1000010)
    cfg = {"arm": arm, "batch_size": 24, "loader": settings(workers)}
    loader = _loader(ds, cfg, sampler=sampler, generator=generator)
    epochs = 2 if arm in ("block11", "union12") else 1
    results, snapshots = [], []
    try:
        for epoch in range(epochs):
            rows, ids = parity_epoch(loader, generator, model, optimizer, arm)
            assert sorted(ids) == sorted(r["record_id"] for r in ds.entries), "Dropped or duplicated diagnostic row"
            results.extend(rows)
            snapshots.append({"model": _cpu_state(model), "optimizer": cpu_tree(optimizer.state_dict()),
                "sampler": sampler.state_dict(), "loader_rng": generator.get_state(), "rng": _rng_state()})
        resume_checked = False
        if epochs == 2:
            shutdown(loader)
            path = checkpoint_dir / f"disposable_{arm}_{workers}.pt"
            atomic_torch(path, snapshots[0])
            saved = torch.load(path, map_location="cpu", weights_only=False)
            path.unlink()
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            sampler.load_state_dict(saved["sampler"])
            generator.set_state(saved["loader_rng"])
            loader = _loader(ds, cfg, sampler=sampler, generator=generator)
            _restore_rng(saved["rng"])
            resumed, ids = parity_epoch(loader, generator, model, optimizer, arm)
            compare_steps(results[len(results)//2:], resumed)
            compare_state(snapshots[1]["model"], model.state_dict())
            compare_state(snapshots[1]["optimizer"], optimizer.state_dict())
            assert sampler.state_dict() == snapshots[1]["sampler"]
            assert torch.equal(generator.get_state(), snapshots[1]["loader_rng"])
            restored_rng = _rng_state()
            assert torch.equal(restored_rng["torch"], snapshots[1]["rng"]["torch"])
            assert all(torch.equal(a,b) for a,b in zip(restored_rng["cuda"], snapshots[1]["rng"]["cuda"]))
            resume_checked = True
        return results, {"passed": True, "epochs": epochs, "resume_checked": resume_checked,
                         "loader_rng_accounting_passed": True}
    finally:
        shutdown(loader)
        del model, optimizer
        gc.collect(); torch.cuda.empty_cache()


def timing(ds, arm, workers):
    _seed(7)
    model, optimizer = new_model(arm)
    config = {"arm": arm, "batch_size": 24, "loader": settings(workers)}
    generator = torch.Generator().manual_seed(1000010)
    stream_started = time.monotonic()
    began = time.monotonic()
    loader = _loader(ds, config, sampler=ReplaySampler(ds), generator=generator)
    create_seconds = time.monotonic() - began
    began = time.monotonic(); iterator = start_iterator(loader, generator)
    initialize_seconds = time.monotonic() - began
    rows, validation, peak_rss, warm_started = [], [], 0, None
    model.train()
    torch.cuda.reset_peak_memory_stats()
    try:
        while True:
            began = time.monotonic()
            try: batch = next(iterator)
            except StopIteration: break
            wait = time.monotonic() - began
            count = len(batch.y)
            step(model, optimizer, batch, arm)
            elapsed = time.monotonic() - began
            pids = [w.pid for w in getattr(iterator, "_workers", [])]
            rss = {str(pid): resident_bytes(pid) for pid in [os.getpid(), *pids]}
            peak_rss = max(peak_rss, sum(v or 0 for v in rss.values()))
            rows.append({"batch": len(rows), "records": count, "loader_wait_seconds": wait,
                         "end_to_end_seconds": elapsed, "warmup": len(rows) < 2, "rss_bytes_by_pid": rss})
            del batch
            if len(rows) == 2: warm_started = time.monotonic()
        stream_ended = time.monotonic()
        assert len(rows) == 12 and sum(r["records"] for r in rows) == 288
        # Production validation constructs a fresh nonpersistent loader each call.
        # Keep persistent train workers alive here to measure the actual overlap
        # in process residency, while their completed queues do no further work.
        validation_started = time.monotonic()
        began = validation_started
        val_loader = _loader(ds, config)
        val_iterator = iter(val_loader)
        validation_setup = time.monotonic() - began
        model.eval()
        while True:
            began = time.monotonic()
            try: batch = next(val_iterator)
            except StopIteration: break
            wait = time.monotonic() - began
            with torch.no_grad():
                scores, _ = _forward(model, batch, arm, torch.device("cuda"))
                scores.detach().cpu().numpy()
            torch.cuda.synchronize()
            validation.append({"records": len(batch.y), "loader_wait_seconds": wait,
                               "end_to_end_seconds": time.monotonic() - began})
            pids = [w.pid for it in (iterator,val_iterator) for w in getattr(it,"_workers",[])]
            peak_rss = max(peak_rss, sum(resident_bytes(pid) or 0 for pid in [os.getpid(),*pids]))
            del batch, scores
        validation_lifecycle = time.monotonic() - validation_started
        warm = [r["end_to_end_seconds"] for r in rows if not r["warmup"]]
        return {"loader_creation_seconds": create_seconds, "iterator_initialization_seconds": initialize_seconds,
                "train_stream_wall_seconds": stream_ended - stream_started,
                "warm_24_consecutive_wall_seconds_per_batch": (stream_ended - warm_started) / len(warm),
                "train_batches": rows, "warm_24_record_batch_seconds": {"median":statistics.median(warm), "min":min(warm), "max":max(warm)},
                "validation_loader_initialization_seconds": validation_setup, "validation_batches": validation,
                "validation_lifecycle_seconds": validation_lifecycle,
                "validation_shutdown_and_instrumentation_seconds": validation_lifecycle - validation_setup - sum(r["end_to_end_seconds"] for r in validation),
                "sampled_process_rss_sum_peak_bytes": peak_rss, "gpu_max_allocated_bytes":torch.cuda.max_memory_allocated(),
                "gpu_max_reserved_bytes":torch.cuda.max_memory_reserved()}
    finally:
        shutdown(loader)
        del iterator, loader, model, optimizer
        gc.collect(); torch.cuda.empty_cache()


def main():
    require_slurm()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-seconds", type=int, default=720)
    args = p.parse_args()
    if not 1 <= args.max_seconds <= 720: p.error("Maximum internal duration is720seconds")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    trace_dir = args.out.parent / (args.out.stem + "_worker_io")
    trace_dir.mkdir(exist_ok=False)
    manifest = json.loads((args.cache / "manifest.json").read_text())
    if not manifest.get("complete") or not manifest.get("diagnostic_only") or manifest.get("records") != 36:
        raise RuntimeError("Requires the existing complete36-record diagnostic cache")
    start = time.monotonic()
    result = {"passed":False,"technical_only":True,"scientific_fits_created":0,"test_evaluated":False,
        "job_id":os.environ["SLURM_JOB_ID"],"source_sha256":file_sha256(__file__),
        "protocol_sha256":digest(protocol()),"torch":torch.__version__,"gpu":torch.cuda.get_device_name(),
        "sample_records":36,"sample_shards":len(manifest["shards"]),"trace_directory":str(trace_dir),"cases":{},
        "limitations":["Throughput replays the same36records eight times (288presentations), not new scientific observations.",
            "Only one shard: no measured production multi-shard NFS contention; each worker has its own LRU1 and first-read checksum cost.",
            "Persistent workers apply to training; production validation loader startup is measured and remains recurring.",
            "RSS sums may count shared pages more than once; observations are sampled, not a guaranteed peak.",
            "No runtime setting is admitted automatically. Missing cases or failed parity prohibit a complete-matrix speedup claim."]}
    def expired(signum, frame): raise TimeoutError("Final worker diagnostic internal time cap")
    signal.signal(signal.SIGALRM, expired); signal.alarm(args.max_seconds)
    try:
        with tempfile.TemporaryDirectory(prefix="worker_disposable_", dir=args.out.parent) as temporary:
            for arm in ("block11","union12","block2","block5","block8","union4"):
                reference = None
                for workers in (0,2,4):
                    key=f"{arm}/workers{workers}"
                    ds=ObservedDataset(args.cache,"train",arm,diagnostic=True,trace_dir=trace_dir,phase=key)
                    before=time.monotonic()
                    steps, parity=check_parity(ds,arm,workers,Path(temporary))
                    if reference is None: reference=steps
                    parity["gradient_max_abs_difference_vs_workers0"] = compare_steps(reference,steps)
                    result["cases"][key]={"loader":settings(workers),"parity":parity,"parity_seconds":time.monotonic()-before}
                    atomic_json(args.out,result)
                    measured=timing(ds,arm,workers)
                    result["cases"][key]["timing"]=measured
                    result["elapsed_seconds"]=time.monotonic()-start
                    atomic_json(args.out,result)
                    print(json.dumps({"event":"worker_case_complete","case":key,"elapsed_seconds":result["elapsed_seconds"],
                        "warm_batch":measured["warm_24_record_batch_seconds"],"parity":parity}),flush=True)
                    del ds,steps
                del reference
        result["passed"]=len(result["cases"])==18 and all("timing" in c for c in result["cases"].values())
    except BaseException as error:
        result["error"]=repr(error)
        raise
    finally:
        signal.alarm(0)
        result["elapsed_seconds"]=time.monotonic()-start
        result["ended_unix"]=time.time()
        atomic_json(args.out,result)


if __name__=="__main__": main()
