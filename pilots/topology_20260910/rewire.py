"""Compiled, cached directed degree-preserving swaps; keep source-attached features."""
from __future__ import annotations

import argparse
import concurrent.futures
from functools import lru_cache
import json
import time
from pathlib import Path

import numpy as np
import torch
from numba import njit, __version__ as NUMBA_VERSION

from .data import atomic_torch
from .protocol import atomic_json, cohort, digest, file_sha256, protocol, require_slurm


@njit(cache=True)
def directed_swaps(source, target, nodes, seed, accepted_per_edge=2, attempts_per_edge=20):
    """No rejection-induced change in degree, multiplicity, or source-edge features."""
    np.random.seed(seed)
    result = target.copy()
    count = source.size
    adjacency = np.zeros((nodes, nodes), dtype=np.uint8)
    for k in range(count):
        adjacency[source[k], result[k]] = 1
    accepted = 0
    attempts = 0
    goal = accepted_per_edge * count
    while count >= 2 and accepted < goal and attempts < attempts_per_edge * count:
        attempts += 1
        first, second = np.random.randint(count), np.random.randint(count)
        a, b = source[first], result[first]
        c, d = source[second], result[second]
        if first == second or a == c or b == d or a == d or c == b:
            continue
        if adjacency[a, d] or adjacency[c, b]:
            continue
        adjacency[a, b] = 0
        adjacency[c, d] = 0
        adjacency[a, d] = 1
        adjacency[c, b] = 1
        result[first], result[second] = d, b
        accepted += 1
    return result, accepted, attempts


def verify_rewire(edge_index, targets, nodes):
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or targets.shape != (edge_index.shape[1],):
        raise RuntimeError("Original/rewired graph shape mismatch")
    if not np.issubdtype(edge_index.dtype, np.integer) or not np.issubdtype(targets.dtype, np.integer):
        raise RuntimeError("Graph endpoints must be integer indices")
    source, original = edge_index
    if np.any(edge_index < 0) or np.any(edge_index >= nodes) or np.any(targets < 0) or np.any(targets >= nodes):
        raise RuntimeError("Invalid graph endpoint")
    if np.any(source == original) or len(np.unique(source * nodes + original)) != len(source):
        raise RuntimeError("Original graph contains self-loops or duplicate directed edges")
    if not np.array_equal(np.bincount(original, minlength=nodes), np.bincount(targets, minlength=nodes)):
        raise RuntimeError("In-degree changed during rewiring")
    if np.any(source == targets):
        raise RuntimeError("Rewiring introduced self-loops")
    if len(np.unique(source * nodes + targets)) != len(source):
        raise RuntimeError("Rewiring introduced duplicate directed edges")


@lru_cache(maxsize=1)
def record_splits():
    # Immutable source assignment only; this mapping contains no labels or scores.
    return {record["record_id"]: record["split"] for record in cohort()["records"]}


def development_summary(graphs, minimum_changed_fraction):
    """The registered gate is one unweighted mean over eligible train/val graphs.

    Test graphs contribute only opaque identities/integrity records elsewhere.
    Neither their changed fractions nor accepted-swap counts enter this function.
    """
    development = [graph for graph in graphs if graph["split"] in {"train", "val"}]
    eligible = [graph for graph in development if graph["edges"] >= 20]
    fractions = np.asarray([graph["changed_fraction"] for graph in eligible], dtype=np.float64)
    mean = float(fractions.mean()) if len(fractions) else None
    summary = {"gate_split": ["train", "val"], "minimum_edges": 20,
               "minimum_mean_changed_fraction": minimum_changed_fraction,
               "eligible_graphs": len(eligible), "development_graphs": len(development),
               "excluded_from_gate_but_retained": len(development) - len(eligible),
               "mean_changed_fraction": mean, "passed": bool(mean is not None and mean >= minimum_changed_fraction),
               "quantile_method": "linear", "by_split": {}}
    for split in ("train", "val", "combined"):
        selected = development if split == "combined" else [g for g in development if g["split"] == split]
        eligible_selected = [g for g in selected if g["edges"] >= 20]
        values = np.asarray([g["changed_fraction"] for g in selected], dtype=np.float64)
        gate_values = np.asarray([g["changed_fraction"] for g in eligible_selected], dtype=np.float64)
        quantiles = (dict(zip(("0", "0.1", "0.25", "0.5", "0.75", "0.9", "1"),
                     map(float, np.quantile(values, [0, .1, .25, .5, .75, .9, 1], method="linear"))))
                     if len(values) else None)
        summary["by_split"][split] = {"graphs": len(selected), "eligible_graphs": len(eligible_selected),
             "changed_fraction_quantiles_all_graphs": quantiles,
             "weak_fraction_all_graphs": float(np.mean(values < minimum_changed_fraction)) if len(values) else None,
             "weak_fraction_eligible_graphs": float(np.mean(gate_values < minimum_changed_fraction)) if len(gate_values) else None,
             "accepted_swaps": sum(g["accepted"] for g in selected), "attempted_swaps": sum(g["attempts"] for g in selected),
             "below_swap_target_graphs": sum(not g["swap_target_reached"] for g in selected),
             "cls_neighbors_changed_graph_fraction": float(np.mean([g["cls_neighbors_changed"] for g in selected])) if selected else None,
             "cls_in_neighbors_replaced": sum(g["cls_in_neighbors_replaced"] for g in selected),
             "cls_out_neighbors_replaced": sum(g["cls_out_neighbors_replaced"] for g in selected)}
    return summary


def rewire_one(task):
    cache, entry, configuration, cache_sha = task
    cache = Path(cache)
    filename = Path(entry["path"]).name
    output = cache / "rewire" / filename
    sidecar = output.with_suffix(".json")
    identity = {"source_sha256": entry["sha256"], "configuration_sha256": digest(configuration),
                "cache_manifest_sha256": cache_sha, "schema_version": 2,
                "rewire_implementation_sha256": file_sha256(Path(__file__)),
                "versions": {"numpy": np.__version__, "numba": NUMBA_VERSION}}
    if file_sha256(cache / entry["path"]) != entry["sha256"]:
        raise RuntimeError(f"Input shard checksum changed: {filename}")
    if output.exists() and sidecar.exists():
        saved = json.loads(sidecar.read_text())
        if any(saved.get(k) != v for k, v in identity.items()) or file_sha256(output) != saved["sha256"]:
            raise RuntimeError(f"Existing rewiring cache mismatch: {filename}")
        return saved
    shard = torch.load(cache / entry["path"], map_location="cpu", weights_only=True)
    edges = shard["edge_index"].numpy().astype(np.int64)
    targets = edges[1].copy()
    offsets = shard["edge_offsets"].numpy()
    reports = []
    for row, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        record_id = int(shard["record_id"][row])
        split = record_splits()[record_id]
        source, original = edges[:, start:stop]
        verify_rewire(edges[:, start:stop], original, 197)
        changed, accepted, attempts = directed_swaps(source, original, 197,
            (configuration["seed"] + 1000003 * record_id) % (2 ** 32 - 1),
            configuration["accepted_swaps_per_edge"], configuration["max_attempts_per_edge"])
        verify_rewire(edges[:, start:stop], changed, 197)
        targets[start:stop] = changed
        count = int(stop - start)
        report = {"record_id": record_id, "split": split, "integrity_passed": True}
        if split in {"train", "val"}:
            in_replaced = len(set(source[original == 0]) - set(source[changed == 0]))
            out_replaced = len(set(original[source == 0]) - set(changed[source == 0]))
            report.update(edges=count, accepted=int(accepted), attempts=int(attempts),
                          changed_fraction=float(np.mean(changed != original)) if count else 0.0,
                          swap_target_reached=bool(accepted >= configuration["accepted_swaps_per_edge"] * count),
                          cls_in_neighbors_replaced=in_replaced, cls_out_neighbors_replaced=out_replaced,
                          cls_neighbors_changed=bool(in_replaced or out_replaced))
        reports.append(report)
    atomic_torch(output, {"targets": torch.from_numpy(targets.astype(np.int16)),
                         "edge_offsets": shard["edge_offsets"], "record_id": shard["record_id"]})
    result = {**identity, "path": f"rewire/{filename}", "sha256": file_sha256(output), "graphs": reports,
              "passed": True}
    atomic_json(sidecar, result)
    return result


def run(cache, workers=1):
    require_slurm()
    cache = Path(cache)
    manifest = json.loads((cache / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise RuntimeError("Complete feature cache before preparing matched topology controls")
    cfg = protocol()["rewire"]
    cache_sha = file_sha256(cache / "manifest.json")
    tasks = [(str(cache), entry, cfg, cache_sha) for entry in manifest["shards"]]
    reports = []
    started = time.monotonic()
    if workers < 1:
        raise ValueError("workers must be positive")
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        outputs = executor.map(rewire_one, tasks) if executor else map(rewire_one, tasks)
        for result in outputs:
            reports.append(result)
            print(json.dumps({"event": "rewire_shard", "completed": len(reports), "total": len(tasks),
                              "passed": result["passed"], "elapsed_seconds": time.monotonic() - started}), flush=True)
    finally:
        if executor:
            executor.shutdown(wait=True)
    all_graphs = [graph for shard in reports for graph in shard["graphs"]]
    diagnostic = development_summary(all_graphs, cfg["minimum_changed_fraction"])
    output = {"schema_version": 2, "cache_manifest_sha256": cache_sha, "configuration": cfg,
              "construction_complete": True, "complete": diagnostic["passed"],
              "shards": [{k: v for k, v in row.items() if k != "graphs"} for row in reports],
              "graphs": len(all_graphs), "development_diagnostics": diagnostic,
              "elapsed_seconds": time.monotonic() - started,
              "interpretation": "Deterministic constrained swaps, not a uniform random-graph sampler"}
    destination = cache / "rewire" / "manifest.json"
    if destination.exists():
        previous = json.loads(destination.read_text())
        if previous.get("complete"):
            if any(previous.get(key) != value for key, value in output.items() if key != "elapsed_seconds"):
                raise RuntimeError("Refusing to mutate a completed rewiring manifest")
            print(json.dumps({"event": "complete_rewire_cache_reused", "integrity_verified": True}), flush=True)
            return
    atomic_json(destination, output)
    if not diagnostic["passed"]:
        raise RuntimeError("Rewiring development mean gate is non-diagnostic; retain every graph and obtain a protocol decision")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    run(args.cache, args.workers)


if __name__ == "__main__":
    main()
