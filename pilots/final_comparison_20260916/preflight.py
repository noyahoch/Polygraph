"""Bounded allocated readiness check: one GPU task, internal deadline under 30 minutes.

Diagnostic-only inputs (48 base_train + 48 checkpoint rows and the preflight hidden
sidecar). It records parameter counts, cross-family alignment, FP32 numerics,
optimizer/RNG/sampler resume fidelity and measured loader/storage cost. It never fits
a scientific model, selects a checkpoint, scores meta/dev_eval or writes execution.json.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time
import traceback

import numpy as np
import torch

from pilots.layer_screen_20260913.profile_workers import deterministic_parity
from pilots.layer_screen_20260913.protocol import digest
from pilots.layer_screen_20260913.train import (
    BlockShuffleSampler, _cpu_state, _finite, _restore_rng, _rng_state, _seed,
)

from .data import DIAGNOSTIC_ROLES, DIAGNOSTIC_ROWS_PER_ROLE, METADATA, RoleDataset, atomic_torch
from .extract import extract
from .models import EXPECTED_PARAMETERS, HIDDEN_INDEX, build_model, parameter_count
from .protocol import (
    ARMS, FAMILIES, ROLE_PHOTOS, SCOPE, SEEDS, TRAINING, atomic_json, campaign_sha, read, read_campaign,
    require_slurm, source_identity,
)
from .train import check_device, forward, initialize, make_loader, runtime, versions

SEED = SEEDS[0]  # any registered training seed; readiness only, never a scientific fit
MAX_SECONDS = 1500
EXTRACT_SECONDS = 1020
RECORDS_PER_PHOTO = 9
ROLE_RECORDS = {role: photos * RECORDS_PER_PHOTO for role, photos in ROLE_PHOTOS.items()}
FULL_RECORDS = sum(ROLE_RECORDS.values())
AUDIT_TOLERANCE = 1e-5
FORBIDDEN = ("execution.json", "predictions", "runs", "evaluation_gate.json")
IDENTITIES = tuple((family, arm) for family in FAMILIES
                   for arm in (("logits",) if family == "O" else ARMS))
TARGET_KEYS = frozenset({"y", "record_id", "image_id", "source_id", "split_id"})
# Readiness-only O inputs: raw logits, no fitted scaler (fit_normalizer is scientific-only).
IDENTITY_PREPROCESSING = {"kind": "preflight_identity_no_fit", "mean": [0.0] * 100, "scale": [1.0] * 100}
DEADLINE = math.inf


def guard():
    if time.monotonic() >= DEADLINE:
        raise TimeoutError("Bounded preflight reached its internal deadline")


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def refuse_scientific_root(root):
    """A preflight may never run inside an authorized or partially scientific root."""
    root = Path(root).resolve()
    for name in FORBIDDEN:
        if (root / name).exists() or (root / name).is_symlink():
            raise RuntimeError("Preflight refuses a scientific/authorized root containing " + name)
    return root


def write_receipt(path, value):
    path = Path(path)
    if path.exists():
        previous = read(path)
        if previous.get("complete") is True:
            if previous != value:
                raise RuntimeError("Refusing to replace a complete preflight receipt: " + str(path))
            return
    atomic_json(path, value)


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def compare_tree(a, b, path="state", *, atol=0.0):
    """Exact when atol is 0, else the original absolute/relative audit tolerance.

    Integer tensors, keys, layouts and plain values are always exact. Returns max |a-b|.
    """
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError(path + ": tensor layout changed")
        a, b = a.detach().cpu(), b.detach().cpu()
        if not a.is_floating_point():
            if not torch.equal(a, b):
                raise AssertionError(path + ": exact integer state differs")
            return 0.0
        if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise AssertionError(path + ": nonfinite state")
        error = float((a.double() - b.double()).abs().max()) if a.numel() else 0.0
        same = torch.equal(a, b) if atol == 0 else torch.allclose(a, b, atol=atol, rtol=atol)
        if not same:
            raise AssertionError(f"{path}: state differs (max absolute error {error})")
        return error
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            raise AssertionError(path + ": keys changed")
        return max((compare_tree(a[key], b[key], f"{path}.{key}", atol=atol) for key in a), default=0.0)
    if isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            raise AssertionError(path + ": length changed")
        return max((compare_tree(x, y, f"{path}[{i}]", atol=atol) for i, (x, y) in enumerate(zip(a, b))),
                   default=0.0)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)) or not np.array_equal(a, b):
            raise AssertionError(path + ": array changed")
        return 0.0
    if type(a) is not type(b) or a != b:
        raise AssertionError(path + ": value changed")
    return 0.0


def _keys(item):
    return set(item.keys() if callable(item.keys) else item.keys)


def feature_digest(items, select):
    result = hashlib.sha256()
    for item in items:
        value = select(item).detach().cpu().contiguous()
        result.update(json.dumps([str(value.dtype), list(value.shape)]).encode("ascii"))
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _config(family):
    return {"family": family, "batch_size": TRAINING["batch_size"], "preprocessing": IDENTITY_PREPROCESSING}


def check_alignment(dataset, items, reference):
    """Same diagnostic record IDs, order, grouping metadata and error labels for every family."""
    metadata, labels = dataset.metadata(), dataset.labels()
    if not dataset.diagnostic or len(dataset) != DIAGNOSTIC_ROWS_PER_ROLE:
        raise RuntimeError("Preflight must use exactly the fixed diagnostic rows")
    if (not np.array_equal(labels, metadata["y"])
            or not np.array_equal(metadata["y"], metadata["pred"] != metadata["label"])):
        raise RuntimeError("Diagnostic targets must remain frozen classifier errors")
    if items is not None:
        if ([int(item.record_id) for item in items] != metadata["record_id"].tolist()
                or [float(item.y) for item in items] != labels.tolist()
                or [int(item.image_id) for item in items] != metadata["image_id"].tolist()):
            raise RuntimeError("Materialized graph/hidden/set records are misaligned with metadata")
    value = {name: metadata[name].tolist() for name in METADATA}
    if dataset.role not in reference:
        reference[dataset.role] = value
    elif reference[dataset.role] != value:
        raise RuntimeError("Diagnostic record/label alignment differs across families: "
                           + f"{dataset.family}/{dataset.arm}/{dataset.role}")
    return {"records": len(dataset), "record_ids_sha256": digest(value["record_id"]),
            "metadata_sha256": digest(value), "positives": int(labels.sum()),
            "photos": len(set(value["image_id"]))}


def cross_check(family, arm, role, items, features):
    """H/S see exactly their registered inputs; S equals G except for discarded incidence."""
    width = 772 if family == "H" else 784
    expected = TARGET_KEYS | ({"x"} if family == "H" else {"x", "edge_index", "edge_attr"})
    for item in items:
        if _keys(item) != expected or item.x.shape != (197, width) or item.x.dtype != torch.float32:
            raise RuntimeError(f"{family}/{arm} item exposes unexpected inputs: {sorted(_keys(item))}")
        if int(item.x[:, 2].sum()) != 1 or float(item.x[0, 2]) != 1.0:
            raise RuntimeError("Exactly one CLS token flag at token 0 is required")
        if family != "H":
            if item.edge_attr.shape != (item.edge_index.shape[1], 12) or item.edge_index.dtype != torch.int64:
                raise RuntimeError("Unexpected edge layout")
            if item.edge_index.numel() and (int(item.edge_index.min()) < 0 or int(item.edge_index.max()) >= 197):
                raise RuntimeError("Edge endpoints escape their own image")
            if family == "S" and item.edge_index.numel() and int(item.edge_index.abs().max()) != 0:
                raise RuntimeError("S must discard incidence before batching")
    own = {"coords": feature_digest(items, lambda item: item.x[:, :4])}
    if family == "H":
        own["hidden"] = feature_digest(items, lambda item: item.x[:, 4:])
    else:
        own.update(
            diagonal=feature_digest(items, lambda item: item.x[:, 4:16]),
            hidden12=feature_digest(items, lambda item: item.x[:, 16:]),
            x=feature_digest(items, lambda item: item.x),
            edge_attr=feature_digest(items, lambda item: item.edge_attr),
            edge_counts=feature_digest(items, lambda item: torch.tensor([item.edge_index.shape[1]])),
        )
    if family == "G" and arm != ARMS[0]:
        first = features[("G", ARMS[0], role)]
        if own["coords"] != first["coords"] or own["hidden12"] != first["hidden12"]:
            raise RuntimeError("G coordinates/H12 tokens must not depend on the attention block")
    if family == "S":
        graph = features[("G", arm, role)]
        if any(own[name] != graph[name] for name in ("x", "edge_attr", "edge_counts")):
            raise RuntimeError("S must see exactly G's node/edge multisets for the same block")
    if family == "H":
        graph = features[("G", arm, role)]
        if own["coords"] != graph["coords"]:
            raise RuntimeError("H coordinates differ from the matched G coordinates")
        if (HIDDEN_INDEX[arm] == 12) != (own["hidden"] == graph["hidden12"]):
            raise RuntimeError("H12 must equal cached G tokens; H3/H6/H9 must be distinct sidecar tokens")
    features[(family, arm, role)] = own
    return own


@torch.no_grad()
def score_pass(model, loader, family, device):
    model.eval()
    _sync(device)
    began, scores, labels = time.monotonic(), [], []
    for batch in loader:
        guard()
        value, target = forward(model, batch, family, device)
        if value.dtype != torch.float32:
            raise RuntimeError("Native error logits must remain FP32")
        scores.append(value.detach().cpu())
        labels.append(target.detach().cpu())
    _sync(device)
    return torch.cat(scores), torch.cat(labels), time.monotonic() - began


def train_epoch(model, optimizer, loader, family, device):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0, device=device))
    result = {"losses": [], "grads": {}, "record_ids": []}
    for batch in loader:
        guard()
        result["record_ids"].append((batch[1] if family == "O" else batch.record_id).detach().cpu().clone())
        optimizer.zero_grad(set_to_none=True)
        scores, labels = forward(model, batch, family, device)
        loss = criterion(scores, labels)
        _finite(loss, "preflight loss")
        loss.backward()
        step = len(result["losses"])
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                _finite(parameter.grad, "preflight gradient " + name)
                result["grads"][f"{step}/{name}"] = parameter.grad.detach().cpu().clone()
        optimizer.step()
        result["losses"].append(loss.detach().cpu().clone())
    result.update(model=_cpu_state(model), optimizer=cpu_tree(optimizer.state_dict()))
    return result


def _optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=TRAINING["lr"], weight_decay=TRAINING["weight_decay"])


def resume_check(root, family, arm, dataset, source, device):
    """Commit CPU state after one diagnostic epoch; restored continuation must match."""
    config = _config(family)

    def epoch(model, optimizer, sampler, generator):
        return train_epoch(model, optimizer, make_loader(source, config, sampler=sampler, generator=generator),
                           family, device)

    def run():
        _seed(SEED)
        model = build_model(family, arm).to(device)
        optimizer = _optimizer(model)
        sampler = BlockShuffleSampler(dataset, SEED)
        generator = torch.Generator().manual_seed(SEED + 1000003)
        first = epoch(model, optimizer, sampler, generator)
        state = {"model": _cpu_state(model), "optimizer": cpu_tree(optimizer.state_dict()),
                 "rng": _rng_state(), "sampler": sampler.state_dict(), "loader_rng": generator.get_state()}
        with tempfile.TemporaryDirectory(prefix="resume_", dir=root / "preflight") as temporary:
            path = Path(temporary) / "latest.pt"
            atomic_torch(path, state)
            expected = epoch(model, optimizer, sampler, generator)
            restored = torch.load(path, map_location="cpu", weights_only=False)
        clone = build_model(family, arm).to(device)
        clone.load_state_dict(restored["model"], strict=True)
        clone_optimizer = _optimizer(clone)
        clone_optimizer.load_state_dict(restored["optimizer"])
        compare_tree(restored["model"], _cpu_state(clone), "restored.model")
        compare_tree(restored["optimizer"], cpu_tree(clone_optimizer.state_dict()), "restored.optimizer")
        clone_sampler = BlockShuffleSampler(dataset, SEED)
        clone_sampler.load_state_dict(restored["sampler"])
        clone_generator = torch.Generator()
        clone_generator.set_state(restored["loader_rng"])
        _restore_rng(restored["rng"])
        actual = epoch(clone, clone_optimizer, clone_sampler, clone_generator)
        if (clone_sampler.state_dict() != sampler.state_dict()
                or not torch.equal(clone_generator.get_state(), generator.get_state())):
            raise RuntimeError("Restored sampler/loader generator cursor diverged")
        for value in (first, expected):
            ids = torch.cat(value["record_ids"])
            if len(ids) != len(dataset) or (family != "O" and sorted(ids.tolist())
                                            != sorted(dataset.metadata()["record_id"].tolist())):
                raise RuntimeError("Each diagnostic epoch must expose every row exactly once")
        if [x.tolist() for x in first["record_ids"]] == [x.tolist() for x in expected["record_ids"]] \
                and len(first["record_ids"]) > 1 and family != "O":
            raise RuntimeError("Sampler epoch cursor did not advance")
        return expected, actual, len(first["losses"]) + len(expected["losses"])

    fallback = None
    try:
        with deterministic_parity():
            expected, actual, steps = run()
        mode = "deterministic_audit"
    except RuntimeError as error:
        if "determinist" not in str(error).lower():
            raise
        fallback = repr(error)
        expected, actual, steps = run()
        mode = "ordinary_fp32"
    try:
        error, exact = compare_tree(expected, actual, "continuation"), True
    except AssertionError:
        error, exact = compare_tree(expected, actual, "continuation", atol=AUDIT_TOLERANCE), False
    return {"mode": mode, "deterministic_fallback_reason": fallback, "optimizer_steps": steps,
            "restored_state_exact": True, "continuation_bitwise_exact": exact,
            "continuation_max_absolute_error": error, "tolerance": AUDIT_TOLERANCE,
            "checked": ["model", "optimizer", "python/numpy/torch/cuda RNG", "sampler epoch",
                        "loader generator", "losses", "gradients", "record order"],
            "passed": True}


def check_identity(root, family, arm, device, reference, features):
    guard()
    key = f"{family}/{arm}"
    result = {"device": device.type}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    began = time.monotonic()
    datasets = {role: RoleDataset(root, role, family, arm, diagnostic=True) for role in DIAGNOSTIC_ROLES}
    construction = time.monotonic() - began
    materialized, sources, alignment = 0.0, {}, {}
    for role, dataset in datasets.items():
        guard()
        started = time.monotonic()
        if family == "O":
            logits = dataset.logits()
            _finite(logits, "diagnostic logits")
            items, sources[role] = None, dataset
        else:
            items = [dataset[index] for index in range(len(dataset))]
            sources[role] = items
        materialized += time.monotonic() - started
        alignment[role] = check_alignment(dataset, items, reference)
        if items is not None:
            alignment[role]["feature_digests"] = cross_check(family, arm, role, items, features)
    _seed(SEED)
    model = build_model(family, arm).to(device)
    count = parameter_count(model)
    if count != EXPECTED_PARAMETERS[family] or any(p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError("Preflight model differs from the approved FP32 architecture: " + key)
    numerics, cold, warm = {}, 0.0, 0.0
    for role, source in sources.items():
        loader = make_loader(source, _config(family))
        scores, labels, seconds = score_pass(model, loader, family, device)
        repeat, _, repeat_seconds = score_pass(model, make_loader(source, _config(family)), family, device)
        if (scores.shape != (DIAGNOSTIC_ROWS_PER_ROLE,) or not torch.isfinite(scores).all()
                or not np.array_equal(labels.numpy(), datasets[role].labels())):
            raise RuntimeError("Expected one finite FP32 error logit per aligned diagnostic record")
        if not torch.allclose(scores, repeat, atol=AUDIT_TOLERANCE, rtol=AUDIT_TOLERANCE):
            raise RuntimeError("Repeated eval forward exceeded the fixed 1e-5 audit tolerance")
        cold, warm = cold + seconds, warm + repeat_seconds
        numerics[role] = {"shape": list(scores.shape), "dtype": "float32", "finite": True,
                          "min": float(scores.min()), "max": float(scores.max()), "mean": float(scores.mean()),
                          "repeat_max_absolute_error": float((scores - repeat).abs().max())}
    base = sources["base_train"]
    trainer = _optimizer(model)
    _sync(device)
    started = time.monotonic()
    train_epoch(model, trainer, make_loader(base, _config(family)), family, device)
    _sync(device)
    train_seconds = time.monotonic() - started
    del model, trainer
    resume = resume_check(root, family, arm, datasets["base_train"], base, device)
    rows = sum(len(dataset) for dataset in datasets.values())
    base_rows = len(datasets["base_train"])
    load = materialized / rows
    train_per_record = load + train_seconds / base_rows
    eval_per_record = load + warm / rows
    throughput = {
        "dataset_construction_seconds": construction, "materialize_seconds": materialized,
        "records": rows, "cold_eval_seconds": cold, "warm_eval_seconds": warm,
        "warm_train_seconds": train_seconds, "train_records": base_rows,
        "cold_loader_forward_records_per_second": rows / max(materialized + cold, 1e-9),
        "warm_eval_records_per_second": rows / max(warm, 1e-9),
        "loader_plus_train_records_per_second": 1 / max(train_per_record, 1e-12),
        "loader_plus_eval_records_per_second": 1 / max(eval_per_record, 1e-12),
        "projected_fixed20_fit_seconds": TRAINING["epochs"] * (
            ROLE_RECORDS["base_train"] * train_per_record + ROLE_RECORDS["checkpoint"] * eval_per_record)
        + 2 * ROLE_RECORDS["checkpoint"] * eval_per_record,
        "projection_caveat": ("96 fixed diagnostic rows, num_workers=0; shard-load cost amortization, "
                              "node contention and restoration audits beyond one pass are not modeled"),
    }
    if device.type == "cuda":
        throughput["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    result.update(parameters=count, alignment=alignment, numerics=numerics,
                  resume=resume, throughput=throughput)
    del datasets, sources, base
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def storage(root, manifest):
    hidden = root / "preflight" / "hidden"
    shard_bytes = sum((hidden / spec["path"]).stat().st_size for spec in manifest["shards"])
    records = manifest["completed_records"]
    projected = shard_bytes / records * FULL_RECORDS
    free = shutil.disk_usage(root).free
    summary = hidden / "capture_summary.json"
    return {"preflight_sidecar_records": records, "preflight_sidecar_bytes": shard_bytes,
            "bytes_per_record": shard_bytes / records, "full_cohort_records": FULL_RECORDS,
            "projected_full_sidecar_bytes": projected, "free_bytes_at_root": free,
            "free_bytes_after_projection": free - projected,
            "sufficient_with_20_percent_margin": free >= 1.2 * projected,
            "capture_summary": read(summary) if summary.exists() else None}


def run_preflight(root, data_root, result, out):
    root = refuse_scientific_root(root)
    campaign = read_campaign(root)
    result.update(run_id=campaign["run_id"], campaign_sha256=campaign_sha(root),
                  source_identity=source_identity())
    if result["source_identity"] != campaign["source_identity"]:
        raise RuntimeError("Executed source differs from the frozen campaign")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Preflight requires exactly one allocated CUDA GPU")
    gpu, cpu = check_device("G", "cuda"), check_device("O", "cpu")
    write_receipt(out, result)
    guard()
    manifest = extract(root, data_root, "preflight", deadline=min(DEADLINE, time.monotonic() + EXTRACT_SECONDS))
    if (manifest.get("complete") is not True or manifest.get("diagnostic_only") is not True
            or manifest.get("completed_records") != len(DIAGNOSTIC_ROLES) * DIAGNOSTIC_ROWS_PER_ROLE
            or manifest.get("campaign_sha256") != result["campaign_sha256"]):
        raise RuntimeError("Diagnostic hidden sidecar is incomplete or foreign")
    result["hidden_extraction"] = {
        "records": manifest["completed_records"], "parity": manifest["parity"],
        "h12_parity_enforced_per_record": True,
        "index_sha256": manifest["index_sha256"], "capture_identity_sha256": manifest["capture_identity_sha256"],
        "extraction_job_id": manifest["job_id"],
        "reused_existing_sidecar": manifest["job_id"] != os.environ["SLURM_JOB_ID"],
    }
    result["storage"] = storage(root, manifest)
    write_receipt(out, result)
    initialize(SEED)
    before = runtime()
    result["runtime"] = {
        "numerical": before, "versions": {**versions(), "transformers": importlib.metadata.version("transformers")},
        "torch_cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(gpu),
        "gpu_count": torch.cuda.device_count(), "gpu_total_bytes": torch.cuda.get_device_properties(gpu).total_memory,
        "cpu_threads": torch.get_num_threads(), "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "hostname": os.uname().nodename, "dtype": "float32",
    }
    reference, features = {}, {}
    parameters, alignment, numerics, resume, throughput = {}, {}, {}, {}, {}
    for family, arm in IDENTITIES:
        key = f"{family}/{arm}"
        value = check_identity(root, family, arm, cpu if family == "O" else gpu, reference, features)
        parameters[key] = value["parameters"]
        alignment[key], numerics[key] = value["alignment"], value["numerics"]
        resume[key], throughput[key] = value["resume"], value["throughput"]
        result.update(parameter_counts=parameters, alignment=alignment, numerics=numerics,
                      resume=resume, throughput=throughput)
        write_receipt(out, result)
        print(json.dumps({"event": "preflight_identity_complete", "identity": key,
                          "parameters": value["parameters"], "resume_mode": value["resume"]["mode"],
                          "seconds_remaining": DEADLINE - time.monotonic()}), flush=True)
    if runtime() != before:
        raise RuntimeError("Preflight audits changed the ordinary numerical runtime")
    if set(parameters) != {f"{f}/{a}" for f, a in IDENTITIES} or any(
            parameters[f"{f}/{a}"] != EXPECTED_PARAMETERS[f] for f, a in IDENTITIES):
        raise RuntimeError("Parameter-count inventory is incomplete")
    families = {}
    for family in FAMILIES:
        rows = [throughput[f"{f}/{a}"] for f, a in IDENTITIES if f == family]
        families[family] = {
            "min_loader_plus_train_records_per_second": min(r["loader_plus_train_records_per_second"] for r in rows),
            "min_loader_plus_eval_records_per_second": min(r["loader_plus_eval_records_per_second"] for r in rows),
            "max_projected_fixed20_fit_seconds": max(r["projected_fixed20_fit_seconds"] for r in rows),
        }
    result["throughput_by_family"] = families
    result["alignment_summary"] = {role: {"records": len(value["record_id"]),
                                          "record_ids_sha256": digest(value["record_id"]),
                                          "identities": len(IDENTITIES)} for role, value in reference.items()}
    refuse_scientific_root(root)
    if campaign_sha(root) != result["campaign_sha256"]:
        raise RuntimeError("Campaign changed during preflight")
    read_campaign(root)
    return result


def main(argv=None):
    global DEADLINE
    require_slurm()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "data-root", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    root, out = args.root.resolve(), args.out.parent.resolve() / args.out.name
    if out != root / "preflight.json":
        parser.error("--out must be ROOT/preflight.json")
    # Refuse before writing anything: an authorized scientific root is never touched.
    refuse_scientific_root(root)
    began = time.monotonic()
    DEADLINE = began + MAX_SECONDS

    def timeout(_signum, _frame):
        raise TimeoutError("Preflight allocation internal deadline reached")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(MAX_SECONDS + 30)
    result = {"schema_version": 1, "scope_id": SCOPE, "mode": "preflight", "complete": False,
              "job_id": os.environ["SLURM_JOB_ID"], "run_id": None, "campaign_sha256": None,
              "started_utc": utc(), "diagnostic_only": True, "scientific_fits": 0,
              "dev_eval_scoring": False, "meta_scoring": False, "checkpoint_selection": False,
              "original_test_access": False, "execution_record_written": False,
              "max_seconds": MAX_SECONDS, "seed": SEED}
    try:
        write_receipt(out, result)
        run_preflight(root, args.data_root, result, out)
        result.update(complete=True, finished_utc=utc(), elapsed_seconds=time.monotonic() - began)
        write_receipt(out, result)
        print(json.dumps({"event": "preflight_complete", "job_id": result["job_id"],
                          "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    except BaseException as error:
        result.update(complete=False, error=repr(error), traceback=traceback.format_exc(),
                      finished_utc=utc(), elapsed_seconds=time.monotonic() - began)
        try:
            write_receipt(out, result)
        except Exception as receipt_error:
            print(json.dumps({"event": "preflight_receipt_not_written", "error": repr(receipt_error)}), flush=True)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        DEADLINE = math.inf


if __name__ == "__main__":
    main()
