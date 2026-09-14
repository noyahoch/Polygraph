"""Explicit pre-test rewiring admission regressions; execute only on Slurm."""
import json
import os
from pathlib import Path

import pytest

if not os.environ.get("SLURM_JOB_ID"):
    pytest.skip("Construction checks are Slurm-only", allow_module_level=True)

import torch
from pilots.topology_20260910 import rewiring_decision as decision
from pilots.topology_20260910.data import atomic_torch
from pilots.topology_20260910.protocol import atomic_json, cohort, digest, file_sha256, protocol
from pilots.topology_20260910.rewire import development_summary


@pytest.fixture
def construction(tmp_path):
    rows = [next(row for row in cohort()["records"] if row["split"] == split) for split in ("train", "val", "test")]
    index = [{**row, "shard": "shard_00000.pt", "offset": i} for i, row in enumerate(rows)]
    atomic_json(tmp_path / "protocol.json", protocol())
    atomic_json(tmp_path / "cohort.json", cohort())
    atomic_json(tmp_path / "index.json", index)
    # Complete directed five-node graph: valid, 20 edges, unchanged targets.
    edges = torch.tensor([(a, b) for a in range(5) for b in range(5) if a != b], dtype=torch.int16).T
    original = {"record_id": torch.tensor([row["record_id"] for row in rows]),
                "edge_offsets": torch.tensor([0, 20, 40, 60]), "edge_index": edges.repeat(1, 3)}
    atomic_torch(tmp_path / "shards/shard_00000.pt", original)
    source = {"complete": True, "diagnostic_only": True, "protocol_sha256": digest(protocol()),
              "cohort_sha256": digest(cohort()), "index_sha256": file_sha256(tmp_path / "index.json"),
              "shards": [{"path": "shards/shard_00000.pt", "sha256": file_sha256(tmp_path / "shards/shard_00000.pt")}]}
    atomic_json(tmp_path / "manifest.json", source)
    atomic_torch(tmp_path / "rewire/shard_00000.pt", {"record_id": original["record_id"],
                 "edge_offsets": original["edge_offsets"], "targets": original["edge_index"][1]})
    graphs = []
    for row in rows:
        graph = {"record_id": row["record_id"], "split": row["split"], "integrity_passed": True}
        if row["split"] in {"train", "val"}:
            graph.update(edges=20, changed_fraction=0., accepted=0, attempts=400, swap_target_reached=False,
                         cls_in_neighbors_replaced=0, cls_out_neighbors_replaced=0, cls_neighbors_changed=False)
        graphs.append(graph)
    shard = {"path": "rewire/shard_00000.pt", "sha256": file_sha256(tmp_path / "rewire/shard_00000.pt"),
             "source_sha256": source["shards"][0]["sha256"], "cache_manifest_sha256": file_sha256(tmp_path / "manifest.json"),
             "configuration_sha256": digest(protocol()["rewire"]), "schema_version": 2,
             "rewire_implementation_sha256": file_sha256(Path(decision.__file__).with_name("rewire.py")), "passed": True}
    atomic_json(tmp_path / "rewire/shard_00000.json", {**shard, "graphs": graphs})
    atomic_json(tmp_path / "rewire/manifest.json", {"schema_version": 2,
                "cache_manifest_sha256": file_sha256(tmp_path / "manifest.json"), "configuration": protocol()["rewire"],
                "construction_complete": True, "complete": False, "graphs": 3, "shards": [shard],
                "development_diagnostics": development_summary(graphs, .8)})
    return tmp_path


def approve(cache):
    return decision.admit(cache, Path(decision.__file__).resolve().parents[2] / decision.DECISION_PATH)


def test_failed_quality_needs_explicit_admission_and_preserves_original_bytes(construction):
    original = (construction / "rewire/manifest.json").read_bytes()
    with pytest.raises(RuntimeError, match="explicit approved admission"):
        decision.read_admission(construction)
    receipt = approve(construction)
    assert receipt["admitted"] and receipt["structural_integrity_passed"]
    assert receipt["mixing_quality_passed"] is False and receipt["original_manifest_complete"] is False
    assert receipt["verified_graphs"] == 3 and receipt["verified_development_graphs"] == 2
    assert receipt["decision"]["rewired_contrast"] == "descriptive_only"
    assert (construction / "rewire/manifest.json").read_bytes() == original
    assert approve(construction) == receipt
    assert decision.read_admission(construction, required=True) == receipt
    with pytest.raises(RuntimeError, match="full training/validation cohort"):
        decision.read_admission(construction, required=True, require_full=True)


def test_unknown_decision_fails_closed(construction):
    path = construction / "unapproved.md"
    path.write_text("proceed")
    with pytest.raises(RuntimeError, match="Unknown or changed"):
        decision.admit(construction, path)
    assert not (construction / "rewire/admission.json").exists()


def test_changed_decision_hash_fails_closed(construction, monkeypatch):
    monkeypatch.setattr(decision, "DECISION_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="Unknown or changed"):
        approve(construction)


@pytest.mark.parametrize("kind", ["missing_graph", "structural_failure", "test_diagnostic", "wrong_provenance"])
def test_incomplete_or_untrusted_evidence_cannot_be_admitted(construction, kind):
    sidecar_path = construction / "rewire/shard_00000.json"
    sidecar = json.loads(sidecar_path.read_text())
    if kind == "missing_graph":
        sidecar["graphs"].pop()
    elif kind == "structural_failure":
        sidecar["graphs"][0]["integrity_passed"] = False
    elif kind == "test_diagnostic":
        sidecar["graphs"][2]["changed_fraction"] = 1.
    else:
        sidecar["source_sha256"] = "f" * 64
    atomic_json(sidecar_path, sidecar)
    with pytest.raises(RuntimeError):
        approve(construction)
    assert not (construction / "rewire/admission.json").exists()


def test_tampered_shard_cannot_be_admitted(construction):
    path = construction / "rewire/shard_00000.pt"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        approve(construction)


def test_structural_failure_is_fatal_even_when_sidecar_claims_pass(construction):
    target = construction / "rewire/shard_00000.pt"
    null = torch.load(target, weights_only=True)
    null["targets"][0] = 0  # Self-loop and changed in-degree; preserve claimed PASS.
    atomic_torch(target, null)
    sidecar_path = target.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["sha256"] = file_sha256(target)
    atomic_json(sidecar_path, sidecar)
    manifest_path = construction / "rewire/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["shards"][0]["sha256"] = file_sha256(target)
    atomic_json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="In-degree changed|self-loops"):
        approve(construction)
    assert not (construction / "rewire/admission.json").exists()


def test_tampered_admission_and_sidecar_fail_closed(construction):
    approve(construction)
    path = construction / "rewire/admission.json"
    receipt = json.loads(path.read_text())
    receipt["mixing_quality_passed"] = True
    atomic_json(path, receipt)
    with pytest.raises(RuntimeError, match="changed"):
        decision.read_admission(construction, required=True)
    receipt["mixing_quality_passed"] = False
    atomic_json(path, receipt)
    sidecar = construction / "rewire/shard_00000.json"
    sidecar.write_bytes(sidecar.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="shard provenance changed"):
        decision.read_admission(construction, required=True)
