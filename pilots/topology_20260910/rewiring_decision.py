"""Explicit admission of the registered partial perturbation; never relabel quality.

The construction and its receipts remain immutable. Numerical integrity checks run
only on Slurm; ordinary readers verify the separate, hash-bound admission receipt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .protocol import atomic_json, cohort, digest, file_sha256, protocol, require_slurm

DECISION_PATH = "docs/experiments/september10/REWIRING_DECISION.md"
DECISION_SHA256 = "0453bb052c668642e42ec1fbd0b6d3359df33f2a5288fb4a6c51de0c47381099"
LIMITATION = ("The original-versus-rewired contrast describes the realized partial perturbation only. "
              "It cannot establish topology removal, structural necessity, or equivalence. "
              "The pre-test descriptive-only decision remains fixed regardless of full-cohort mixing quality.")


def _read(path):
    return json.loads(Path(path).read_text())


def approved_decision(path=None):
    expected = Path(__file__).resolve().parents[2] / DECISION_PATH
    selected = expected if path is None else Path(path)
    if selected.resolve() != expected.resolve() or file_sha256(selected) != DECISION_SHA256:
        raise RuntimeError("Unknown or changed approved rewiring decision")
    return {"path": DECISION_PATH, "sha256": DECISION_SHA256, "approved_by": "root",
            "test_results_visible": False, "matrix": "full", "fits": 35,
            "rewired_contrast": "descriptive_only", "limitation": LIMITATION}


def _manifest(cache):
    cache = Path(cache)
    source = _read(cache / "manifest.json")
    null = _read(cache / "rewire/manifest.json")
    if (source.get("complete") is not True or null.get("construction_complete") is not True
            or null.get("schema_version") != 2
            or null.get("cache_manifest_sha256") != file_sha256(cache / "manifest.json")
            or null.get("configuration") != protocol()["rewire"]):
        raise RuntimeError("Rewiring construction is incomplete or has mismatched provenance")
    if (not isinstance(null.get("graphs"), int) or null["graphs"] <= 0
            or not isinstance(null.get("shards"), list) or not null["shards"]):
        raise RuntimeError("Rewiring construction has no complete graph/shard inventory")
    diagnostic = null.get("development_diagnostics", {})
    mean = diagnostic.get("mean_changed_fraction")
    if (diagnostic.get("gate_split") != ["train", "val"] or diagnostic.get("minimum_edges") != 20
            or diagnostic.get("minimum_mean_changed_fraction") != protocol()["rewire"]["minimum_changed_fraction"]
            or not diagnostic.get("eligible_graphs") or not isinstance(mean, (int, float))
            or not 0 <= mean <= 1 or diagnostic.get("passed") is not (mean >= .80)
            or null.get("complete") is not diagnostic["passed"]):
        raise RuntimeError("Missing or inconsistent development mixing diagnostics")
    return source, null


def read_admission(cache, required=False, require_full=False):
    """A failed quality gate needs an explicit receipt; success is never inferred."""
    cache = Path(cache)
    source, null = _manifest(cache)
    path = cache / "rewire/admission.json"
    if not path.exists():
        if required or null["development_diagnostics"]["passed"] is not True:
            raise RuntimeError("Rewiring mixing is non-diagnostic; explicit approved admission is required")
        return None
    receipt = _read(path)
    decision = approved_decision()
    expected = {"schema_version": 1, "admitted": True, "structural_integrity_passed": True,
                "cache_manifest_sha256": file_sha256(cache / "manifest.json"),
                "rewire_manifest_sha256": file_sha256(cache / "rewire/manifest.json"),
                "index_sha256": file_sha256(cache / "index.json"), "decision": decision,
                "development_diagnostics": null["development_diagnostics"],
                "development_diagnostics_sha256": digest(null["development_diagnostics"]),
                "mixing_quality_passed": null["development_diagnostics"]["passed"],
                "original_manifest_complete": null["complete"],
                "diagnostic_only": bool(source.get("diagnostic_only")),
                "implementation_sha256": file_sha256(__file__),
                "rewire_implementation_sha256": file_sha256(Path(__file__).with_name("rewire.py"))}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Rewiring admission receipt or its approved provenance changed")
    if (not isinstance(receipt.get("verified_shards"), dict)
            or receipt.get("verified_graphs") != null.get("graphs")
            or receipt.get("verified_development_graphs") != null["development_diagnostics"]["development_graphs"]):
        raise RuntimeError("Rewiring admission lacks complete structural evidence")
    expected_paths = {row["path"] for row in null["shards"]}
    if set(receipt["verified_shards"]) != expected_paths:
        raise RuntimeError("Rewiring admission shard inventory mismatch")
    for row in null["shards"]:
        evidence = receipt["verified_shards"][row["path"]]
        if (evidence.get("sha256") != row["sha256"] or evidence.get("source_sha256") != row["source_sha256"]
                or evidence.get("sidecar_sha256") != file_sha256((cache / row["path"]).with_suffix(".json"))):
            raise RuntimeError("Rewiring admission shard provenance changed")
    if require_full and (receipt["diagnostic_only"] or receipt["verified_development_graphs"] != 28800
                         or receipt["verified_graphs"] != 36000):
        raise RuntimeError("Freeze requires diagnostics for the full training/validation cohort")
    return receipt


def admission_bindings(cache):
    receipt = read_admission(cache, required=True)
    return {"rewire_admission_sha256": file_sha256(Path(cache) / "rewire/admission.json"),
            "rewire_decision_sha256": receipt["decision"]["sha256"]}


def admit(cache, decision):
    require_slurm()
    approval = approved_decision(decision)
    cache = Path(cache)
    source, null = _manifest(cache)
    if (source.get("protocol_sha256") != digest(protocol()) or source.get("cohort_sha256") != digest(cohort())
            or digest(_read(cache / "protocol.json")) != digest(protocol())
            or digest(_read(cache / "cohort.json")) != digest(cohort())
            or file_sha256(cache / "index.json") != source.get("index_sha256")):
        raise RuntimeError("Admission source protocol, cohort or index provenance mismatch")
    import numpy as np
    import torch
    from .rewire import development_summary, verify_rewire

    index = _read(cache / "index.json")
    canonical = {row["record_id"]: row for row in cohort()["records"]}
    positions, ids = {}, set()
    for entry in index:
        identity = entry["record_id"]
        if identity in ids or (entry["shard"], entry["offset"]) in positions:
            raise RuntimeError("Duplicate construction record or shard position")
        if identity not in canonical or any(entry[key] != canonical[identity][key]
                for key in ("image_id", "source_id", "severity", "split", "split_id")):
            raise RuntimeError("Construction record differs from the fixed source cohort")
        ids.add(identity)
        positions[(entry["shard"], entry["offset"])] = entry
    if not source.get("diagnostic_only") and ids != set(canonical):
        raise RuntimeError("Full construction is missing retained cohort records")
    originals = {Path(row["path"]).name: row for row in source["shards"]}
    if len(originals) != len(source["shards"]) or len(null["shards"]) != len(originals):
        raise RuntimeError("Construction shard inventory is incomplete")
    seen, reports, verified = set(), [], {}
    for row in null["shards"]:
        name = Path(row["path"]).name
        if row["path"] != f"rewire/{name}" or name not in originals or row["path"] in verified:
            raise RuntimeError("Unexpected or duplicate rewiring shard path")
        original_row = originals[name]
        if original_row["path"] != f"shards/{name}":
            raise RuntimeError("Unexpected source shard path")
        for path, expected in ((cache / original_row["path"], original_row["sha256"]),
                               (cache / row["path"], row["sha256"])):
            if file_sha256(path) != expected:
                raise RuntimeError("Construction shard checksum mismatch")
        sidecar_path = (cache / row["path"]).with_suffix(".json")
        sidecar = _read(sidecar_path)
        if ({key: value for key, value in sidecar.items() if key != "graphs"} != row
                or row.get("passed") is not True or row.get("source_sha256") != original_row["sha256"]
                or row.get("cache_manifest_sha256") != file_sha256(cache / "manifest.json")
                or row.get("configuration_sha256") != digest(protocol()["rewire"])
                or row.get("rewire_implementation_sha256") != file_sha256(Path(__file__).with_name("rewire.py"))):
            raise RuntimeError("Missing or mismatched per-shard construction evidence")
        original = torch.load(cache / original_row["path"], map_location="cpu", weights_only=True)
        changed = torch.load(cache / row["path"], map_location="cpu", weights_only=True)
        if (not torch.equal(original["record_id"], changed["record_id"])
                or not torch.equal(original["edge_offsets"], changed["edge_offsets"])
                or len(sidecar["graphs"]) != len(original["record_id"])):
            raise RuntimeError("Missing or misaligned per-graph construction evidence")
        offsets = original["edge_offsets"].tolist()
        if (len(offsets) != len(original["record_id"]) + 1 or offsets[0] != 0
                or any(a > b for a, b in zip(offsets[:-1], offsets[1:]))
                or offsets[-1] != original["edge_index"].shape[1] or offsets[-1] != len(changed["targets"])):
            raise RuntimeError("Construction edge offsets or target count mismatch")
        for position, graph in enumerate(sidecar["graphs"]):
            identity = int(original["record_id"][position])
            entry = positions.get((name, position))
            if (identity in seen or entry is None or entry["record_id"] != identity
                    or graph.get("record_id") != identity or graph.get("split") != entry["split"]
                    or graph.get("integrity_passed") is not True):
                raise RuntimeError("Missing or inconsistent graph structural evidence")
            seen.add(identity)
            start, stop = offsets[position:position + 2]
            edges = original["edge_index"][:, start:stop].numpy()
            targets = changed["targets"][start:stop].numpy()
            verify_rewire(edges, targets, 197)  # Test access: structural integrity only.
            if entry["split"] in {"train", "val"}:
                count = stop - start
                fraction = float(np.mean(targets != edges[1])) if count else 0.0
                if graph.get("edges") != count or graph.get("changed_fraction") != fraction:
                    raise RuntimeError("Development changed-fraction evidence differs from retained tensors")
                accepted, attempts = graph.get("accepted"), graph.get("attempts")
                if (not isinstance(accepted, int) or not isinstance(attempts, int)
                        or not 0 <= accepted <= 2 * count or not accepted <= attempts <= 20 * count
                        or graph.get("swap_target_reached") is not (accepted >= 2 * count)):
                    raise RuntimeError("Development swap-count evidence violates the fixed construction budget")
                in_replaced = len(set(edges[0][edges[1] == 0]) - set(edges[0][targets == 0]))
                out_replaced = len(set(edges[1][edges[0] == 0]) - set(targets[edges[0] == 0]))
                if (graph.get("cls_in_neighbors_replaced") != in_replaced
                        or graph.get("cls_out_neighbors_replaced") != out_replaced
                        or graph.get("cls_neighbors_changed") is not bool(in_replaced or out_replaced)):
                    raise RuntimeError("Development CLS-neighbor diagnostics differ from retained tensors")
                reports.append(graph)
            elif set(graph) != {"record_id", "split", "integrity_passed"}:
                raise RuntimeError("Test construction receipt contains non-integrity diagnostics")
        verified[row["path"]] = {"sha256": row["sha256"], "source_sha256": original_row["sha256"],
                                  "sidecar_sha256": file_sha256(sidecar_path)}
    if seen != ids or len(seen) != null.get("graphs"):
        raise RuntimeError("Construction evidence does not cover every retained graph")
    diagnostic = development_summary(reports, protocol()["rewire"]["minimum_changed_fraction"])
    if diagnostic != null["development_diagnostics"]:
        raise RuntimeError("Recomputed development diagnostics differ from the immutable construction receipt")
    receipt = {"schema_version": 1, "admitted": True, "structural_integrity_passed": True,
               "cache_manifest_sha256": file_sha256(cache / "manifest.json"),
               "rewire_manifest_sha256": file_sha256(cache / "rewire/manifest.json"),
               "index_sha256": source["index_sha256"], "decision": approval,
               "development_diagnostics": diagnostic, "development_diagnostics_sha256": digest(diagnostic),
               "mixing_quality_passed": diagnostic["passed"], "original_manifest_complete": null["complete"],
               "diagnostic_only": bool(source.get("diagnostic_only")), "verified_graphs": len(seen),
               "verified_development_graphs": len(reports), "verified_shards": verified,
               "implementation_sha256": file_sha256(__file__),
               "rewire_implementation_sha256": file_sha256(Path(__file__).with_name("rewire.py"))}
    destination = cache / "rewire/admission.json"
    if destination.exists():
        if _read(destination) != receipt:
            raise RuntimeError("Refusing to change an existing rewiring admission")
    else:
        atomic_json(destination, receipt)
    read_admission(cache, required=True)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(admit(args.cache, args.decision), indent=2))


if __name__ == "__main__":
    main()
