"""Correctness tests for the polygraph package. Run: python3 tests/test_polygraph.py

Sections mirror the package: graphs, records, sources, splits, storage, pipeline,
models/training, evaluation, CLI. Every bug found during development has a regression
test here; the model-equivalence test pins the architectures to the frozen POC commit.
No network and no ViT are required — the two stages that need the model (scan, extract)
are tested through stubs.
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from polygraph.config import ALL_SOURCES, SOURCE_IDS
from polygraph.data.graphs import LayerGraph, ThresholdGraphBuilder, node_coordinates
from polygraph.data.splits import SplitPlan, assign_groups, build_plan, select_balanced
from polygraph.data.storage import AttentionGraphDataset, GraphShard, GraphStore, GraphStoreWriter
from polygraph.records import (RecordKey, ScanRecord, append_scan_records,
                               read_scan_records, scanned_keys)

FAILURES: list = []
CPU = torch.device("cpu")


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def expect_raises(exc, fn, message: str) -> None:
    try:
        fn()
        check(False, message)
    except exc:
        pass


def random_attention(batch=3, heads=12, tokens=64, seed=0) -> torch.Tensor:
    torch.manual_seed(seed)
    raw = torch.rand(batch, heads, tokens, tokens)
    return raw / raw.sum(dim=-1, keepdim=True)


# ------------------------------------------------------------------ graphs

def test_threshold_matches_poc_reference() -> None:
    """The edge rule must be bit-identical to Yishai's build_graph with topk disabled."""
    sys.path.insert(0, str(PROJECT / "legacy"))
    from poc_gnn_vit_cifar100 import build_graph

    attention = random_attention()
    graphs = ThresholdGraphBuilder(0.02).build(attention)
    for item in range(attention.shape[0]):
        _, _, ref_ei, ref_ea = build_graph(attention[item], torch.zeros(64, 4), tau=0.02, topk=0)
        ours = {(int(j), int(i)) for j, i in zip(*graphs[item].edge_index.tolist())}
        theirs = {(int(j), int(i)) for j, i in zip(*ref_ei.tolist())}
        check(ours == theirs, f"edge set differs from POC reference at item {item}")
        lookup = {(int(ref_ei[0, e]), int(ref_ei[1, e])): ref_ea[e] for e in range(ref_ei.shape[1])}
        worst = max((float((graphs[item].edge_attr[e].float()
                            - lookup[(int(graphs[item].edge_index[0, e]),
                                      int(graphs[item].edge_index[1, e]))]).abs().max())
                     for e in range(graphs[item].num_edges)), default=0.0)
        check(worst < 1e-3, f"edge features differ from POC reference (max {worst})")


def test_edges_sorted_and_above_tau() -> None:
    for graph in ThresholdGraphBuilder(0.01).build(random_attention()):
        strength = graph.strength.float()
        check(bool((strength[:-1] >= strength[1:]).all()), "edges not sorted descending")
        check(bool((strength > 0.01).all()), "edge at or below tau retained")


def test_threshold_nesting() -> None:
    """Filtering a low-tau graph must equal building at the higher tau directly."""
    attention = random_attention()
    low = ThresholdGraphBuilder(0.005).build(attention)
    high = ThresholdGraphBuilder(0.02).build(attention)
    for a, b in zip(low, high):
        check(torch.equal(a.at_threshold(0.02).edge_index, b.edge_index), "nesting broken")


def test_rethresholding_guards() -> None:
    """Regressions: a lower tau silently returned the denser graph; a chained call
    (0.02 -> 0.05 -> 0.03) returned the 0.05 graph while claiming 0.03."""
    graph = ThresholdGraphBuilder(0.02).build(random_attention(batch=1))[0]
    expect_raises(ValueError, lambda: graph.at_threshold(0.001), "sub-tau accepted")
    coarse = graph.at_threshold(0.05)
    check(coarse.source_tau == 0.05, "filtered graph kept the old, looser tau")
    expect_raises(ValueError, lambda: coarse.at_threshold(0.03), "chained loosening accepted")
    check(coarse.at_threshold(0.08).source_tau == 0.08, "further tightening should work")


def test_node_coordinates() -> None:
    coords = node_coordinates(197, 5, 12)
    check(tuple(coords.shape) == (197, 4), "coordinate shape wrong")
    check(float(coords[0, 2]) == 1.0 and float(coords[1:, 2].abs().max()) == 0.0, "CLS flag wrong")
    check(abs(float(coords[0, 3]) - 5 / 11) < 1e-6, "layer coordinate wrong")
    check(node_coordinates(197, 5, 12) is coords, "cache not reused")


def test_diagonals() -> None:
    attention = random_attention(batch=2, heads=3, tokens=5)
    diag = ThresholdGraphBuilder(0.01).diagonals(attention)
    check(tuple(diag.shape) == (2, 5, 3), "diagonal shape wrong")
    check(torch.allclose(diag[1, 4, 2], attention[1, 2, 4, 4]), "diagonal values wrong")


# ------------------------------------------------------------------ records

def test_record_key_semantics() -> None:
    fog = RecordKey("fog", 3, 41)
    clean = RecordKey("clean_test", 0, 41)
    train = RecordKey("clean_train", 0, 41)
    check(fog.group_id == clean.group_id, "corruption must share its base image's group")
    check(train.group_id != clean.group_id, "train and test namespaces must differ")
    check(fog.cell == ("fog", 3), "cell wrong")


def test_scan_record_roundtrip() -> None:
    record = ScanRecord(RecordKey("fog", 3, 41), label=7, pred=9, confidence=0.4, margin=0.1)
    check(not record.correct and record.y_err == 1.0, "correctness logic wrong")
    check(ScanRecord.from_json(record.to_json()) == record, "JSON roundtrip changed the record")


def test_scan_file_roundtrip_and_tolerance() -> None:
    records = [ScanRecord(RecordKey("fog", 1, i), 1, 1 if i % 2 else 2, 0.9, 0.5) for i in range(5)]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scan.jsonl"
        append_scan_records(path, records[:3])
        append_scan_records(path, records[3:])
        check(list(read_scan_records(path)) == records, "file roundtrip changed records")
        check(scanned_keys(path) == {r.key.as_tuple() for r in records}, "scanned_keys wrong")

        with path.open("a") as handle:  # a concurrent scan mid-flush truncates the LAST line
            handle.write('{"source": "fog", "sev')
        check(len(list(read_scan_records(path))) == 5, "truncated last line should be skipped")

        broken = Path(tmp) / "broken.jsonl"
        broken.write_text('{"bad json\n' + json.dumps(records[0].to_json()) + "\n")
        expect_raises(Exception, lambda: list(read_scan_records(broken)),
                      "corruption before the last line must raise, not be skipped")


# ------------------------------------------------------------------ sources

def test_pool_cache_normalizes_clean_severity() -> None:
    """Regression: lru_cache keyed on raw args, so clean_test at severities 0 and 3
    loaded the 10k-image parquet twice."""
    from polygraph.data.sources import get_pool

    with tempfile.TemporaryDirectory() as tmp:
        a = get_pool("clean_test", 0, Path(tmp))
        check(get_pool("clean_test", 3, Path(tmp)) is a, "clean severities got separate pools")


def test_image_pool_reads_parquet() -> None:
    import io

    import pandas as pd
    from PIL import Image

    from polygraph.data.sources import ImagePool

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "cifar100c" / "fog" / "severity_3.parquet"
        target.parent.mkdir(parents=True)
        rows = []
        for i in range(3):
            buffer = io.BytesIO()
            Image.new("RGB", (8, 8), (i * 40, 0, 0)).save(buffer, format="PNG")
            rows.append({"image": {"bytes": buffer.getvalue(), "path": None}, "label": i + 10})
        pd.DataFrame(rows).to_parquet(target)

        pool = ImagePool("fog", 3, Path(tmp))
        check(len(pool) == 3 and pool.label(2) == 12, "labels wrong")
        check(pool.image(1).size == (8, 8), "image decode wrong")
        check(pool.key(1) == RecordKey("fog", 3, 1), "key wrong")
        missing = ImagePool("snow", 2, Path(tmp))  # lazy: constructing must not raise
        expect_raises(FileNotFoundError, lambda: len(missing), "missing parquet must raise on use")


# ------------------------------------------------------------------ splits

def _synthetic_records(images=400, sources=(("clean_test", 0, 0.08), ("fog", 1, 0.2),
                                            ("fog", 5, 0.6), ("snow", 3, 0.3), ("spatter", 3, 0.4))):
    rng = random.Random(0)
    return [ScanRecord(RecordKey(source, severity, image), 1,
                       2 if rng.random() < error_rate else 1, 0.9, 0.4)
            for image in range(images) for source, severity, error_rate in sources]


def test_assign_groups() -> None:
    assignment = assign_groups([f"test:{i}" for i in range(1000)], (0.7, 0.1, 0.2), seed=7)
    sizes = {s: sum(1 for v in assignment.values() if v == s) for s in ("train", "val", "test")}
    check(sizes == {"train": 700, "val": 100, "test": 200}, f"split sizes wrong: {sizes}")
    check(assignment == assign_groups([f"test:{i}" for i in range(1000)], (0.7, 0.1, 0.2), 7),
          "assignment not deterministic")
    # Regression: rounding on small inputs left a split empty and passed silently.
    for n in (3, 7):
        expect_raises(ValueError, lambda n=n: assign_groups([f"t:{i}" for i in range(n)], (0.7, 0.1, 0.2), 7),
                      f"n={n} produced an empty split without complaint")


def test_stratified_selection() -> None:
    rng = random.Random(3)
    cells = {("clean_test", 0): (list(range(0, 60)), list(range(100, 1000))),
             ("fog", 5): (list(range(2000, 2700)), list(range(3000, 3300))),
             ("snow", 1): (list(range(5000, 5200)), list(range(6000, 6800)))}
    chosen = select_balanced({c: (list(w), list(r)) for c, (w, r) in cells.items()}, cap=500, rng=rng)
    check(len(chosen) == 1000 and len(set(chosen)) == 1000, "wrong count or duplicates")
    for cell, (wrong, right) in cells.items():
        wrong_taken = sum(1 for i in chosen if i in set(wrong))
        right_taken = sum(1 for i in chosen if i in set(right))
        check(wrong_taken == right_taken, f"cell {cell} not class-balanced")
        check(wrong_taken <= min(len(wrong), len(right)), f"cell {cell} over capacity")
    clean_pairs = sum(1 for i in chosen if i < 100)
    check(clean_pairs == 60, "water-fill should give the smallest cell its full capacity")


def test_build_plan_properties() -> None:
    plan = build_plan(_synthetic_records(), caps={}, held_out=["spatter"])
    plan.validate()
    groups = {n: {k.group_id for k in keys} for n, keys in plan.splits.items()}
    check(not (groups["train"] & groups["test"]) and not (groups["train"] & groups["val"]),
          "base image leaked across splits")
    for name in ("train", "val"):
        check("spatter" not in {k.source for k in plan.splits[name]}, f"held-out source in {name}")
    check("spatter" in {k.source for k in plan.splits["test"]}, "held-out source missing from test")
    lookup = {r.key: r for r in _synthetic_records()}
    for name, keys in plan.splits.items():
        cells: dict = {}
        for key in keys:
            cells.setdefault(key.cell, [0, 0])[int(lookup[key].correct)] += 1
        check(all(w == r for w, r in cells.values()), f"{name} has class-unbalanced cells")


def test_empty_split_rejected() -> None:
    """Regression: holding out every source produced an empty train split silently."""
    records = [ScanRecord(RecordKey("fog", 1, i), 1, 1 if i % 2 else 2, 0.9, 0.4) for i in range(40)]
    expect_raises(RuntimeError, lambda: build_plan(records, held_out=["fog"]),
                  "plan with empty train split accepted")


def test_plan_save_load() -> None:
    plan = build_plan(_synthetic_records(100), caps={"train": 20, "val": 5, "test": 10})
    with tempfile.TemporaryDirectory() as tmp:
        plan.save(Path(tmp) / "plan.json")
        loaded = SplitPlan.load(Path(tmp) / "plan.json")
        check(loaded.splits == plan.splits, "plan roundtrip changed the key lists")


# ------------------------------------------------------------------ storage

def _write_store(tmp: Path, keys, layer_count=2, tokens=20, heads=4, shard_size=4,
                 tau=0.02, with_cls=False, seed=1):
    builder = ThresholdGraphBuilder(tau)
    writer = GraphStoreWriter(tmp, shard_size=shard_size, tau=tau)
    pending, done, shard_index = writer.plan(keys)
    torch.manual_seed(seed)
    reference = {}
    for position, key in enumerate(pending):
        attention = random_attention(1, heads, tokens, seed=seed + done + position)
        layers = [builder.build(attention)[0] for _ in range(layer_count)]
        diagonals = builder.diagonals(attention).to(torch.float16).repeat(layer_count, 1, 1)
        cls = torch.rand(layer_count, 8, dtype=torch.float16) if with_cls else None
        record = ScanRecord(key, label=1, pred=1 if position % 2 else 2, confidence=0.9, margin=0.5)
        writer.add(record, layers, diagonals, cls)
        reference[key.as_tuple()] = (layers, cls, record)
        if writer.pending >= shard_size:
            writer.flush(shard_index)
            shard_index += 1
    writer.flush(shard_index)
    writer.write_manifest()
    return writer, reference


def _keys(n, source="clean_test", severity=0, start=0):
    return [RecordKey(source, severity, start + i) for i in range(n)]


def test_store_roundtrip_including_cls() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _, reference = _write_store(Path(tmp), _keys(6), with_cls=True)
        store = GraphStore(Path(tmp))
        check(store.total == 6, "record count wrong")
        for key, (layers, cls, record) in reference.items():
            index = store.key_to_index[key]
            shard, offset = store.locate(index)
            for layer_index, original in enumerate(layers):
                restored = shard.layer_graph(offset, layer_index)
                check(torch.equal(restored.edge_index, original.edge_index), "edge_index changed")
                check(torch.equal(restored.edge_attr, original.edge_attr), "edge_attr changed")
            check(torch.equal(shard.cls_embeddings[offset], cls), "cls embeddings changed")
            check(int(shard.meta["pred"][offset]) == record.pred, "meta changed")


def test_view_guards_and_equivalence() -> None:
    """Regressions: sub-tau silently returned the denser graph; top-K larger than the
    stored span silently truncated; tau+top_k together is ambiguous."""
    with tempfile.TemporaryDirectory() as tmp:
        _, reference = _write_store(Path(tmp), _keys(4), tau=0.005)
        store = GraphStore(Path(tmp))
        shard, offset = store.locate(0)
        expect_raises(ValueError, lambda: shard.layer_graph(offset, 0, tau=0.001), "sub-tau accepted")
        available = shard.layer_graph(offset, 0).num_edges
        expect_raises(ValueError, lambda: shard.layer_graph(offset, 0, top_k=available + 1),
                      "oversized top_k accepted")
        expect_raises(Exception, lambda: AttentionGraphDataset(store, [0], tau=0.02, top_k=5),
                      "tau and top_k together accepted")
        key0 = next(iter(reference))
        in_memory = reference[key0][0][0].at_threshold(0.02)
        via_store = store.locate(store.key_to_index[key0])[0].layer_graph(
            store.key_to_index[key0] % 4, 0, tau=0.02)  # shard_size=4 -> offset
        check(torch.equal(in_memory.edge_index, via_store.edge_index),
              "stored tau view disagrees with in-memory filtering")
        top = shard.layer_graph(offset, 0, top_k=10)
        check(torch.equal(top.edge_index, shard.layer_graph(offset, 0).edge_index[:, :10]),
              "top-K view is not the first K edges")


def test_resume_and_extension() -> None:
    """Regressions: re-runs overwrote shard 0; a foreign key order misaligned records.
    The key list is append-only, so a changed plan extends the store."""
    with tempfile.TemporaryDirectory() as tmp:
        keys_a = _keys(6)
        _write_store(Path(tmp), keys_a)
        writer = GraphStoreWriter(Path(tmp), shard_size=4, tau=0.02)
        pending, done, shard_index = writer.plan(keys_a)
        check((len(pending), done, shard_index) == (2, 4, 1),
              f"resume should redo only the short trailing shard, got {(len(pending), done, shard_index)}")

        keys_b = _keys(3, "fog", 3)
        pending, done, _ = writer.plan(keys_a + keys_b)
        check([k.as_tuple() for k in pending][-3:] == [k.as_tuple() for k in keys_b],
              "new keys must append after existing ones")
        check(writer.stored_keys()[:6] == [k.as_tuple() for k in keys_a],
              "existing key positions must never move")

        (Path(tmp) / "shard_00000.json").unlink()  # simulate an interrupted write
        pending, done, shard_index = GraphStoreWriter(Path(tmp), 4, 0.02).plan(keys_a)
        check((done, shard_index) == (0, 0), "shard without sidecar must be rebuilt")


def test_nonuniform_shards_and_key_views() -> None:
    """Regression: indexing assumed uniform shard sizes; one short shard misaligned
    every later record."""
    with tempfile.TemporaryDirectory() as tmp:
        keys = _keys(5)
        _write_store(Path(tmp), keys, shard_size=2)  # 2+2+1 shards
        store = GraphStore(Path(tmp))
        dataset = AttentionGraphDataset(store, layers=[0])
        check([int(dataset[i].image_id) for i in range(5)] == [0, 1, 2, 3, 4], "records misaligned")

        subset = AttentionGraphDataset(store, layers=[0], keys=[keys[4], keys[1]])
        check([int(subset[i].image_id) for i in range(2)] == [4, 1], "key view order wrong")
        expect_raises(KeyError, lambda: AttentionGraphDataset(store, [0], keys=[RecordKey("fog", 1, 9)]),
                      "unextracted key accepted")


def test_dataset_fields_and_multilayer() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _write_store(Path(tmp), _keys(4), layer_count=3, tokens=20, heads=4)
        sample = AttentionGraphDataset(GraphStore(Path(tmp)), layers=[0, 2])[0]
        for field in ("x", "edge_index", "edge_attr", "layer_id", "y", "image_id",
                      "vit_correct", "confidence", "margin", "source_id", "severity"):
            check(hasattr(sample, field), f"Data missing field {field}")
        check(tuple(sample.x.shape) == (40, 8), "multi-layer x wrong (2 layers x 20 tokens, 4+4 dims)")
        check(int(sample.layer_id.max()) == 1 and int(sample.edge_index.max()) < 40,
              "layer offsets wrong")
        check(ALL_SOURCES[int(sample.source_id)] == "clean_test", "source id mapping wrong")


def test_writer_consistency_guards() -> None:
    """Regressions: inconsistent layer counts misaligned offsets; a manifest over stale
    shards claimed completeness; re-saving a shard remapped its source table."""
    builder = ThresholdGraphBuilder(0.02)
    attention = random_attention(2, 4, 20)
    layers = builder.build(attention)
    diagonals = builder.diagonals(attention).to(torch.float16)
    with tempfile.TemporaryDirectory() as tmp:
        writer = GraphStoreWriter(Path(tmp), 4, 0.02)
        writer.plan(_keys(2))
        writer.add(ScanRecord(_keys(2)[0], 1, 1, 0.9, 0.5), [layers[0], layers[0]],
                   diagonals[0].unsqueeze(0).repeat(2, 1, 1))
        expect_raises(ValueError,
                      lambda: writer.add(ScanRecord(_keys(2)[1], 1, 1, 0.9, 0.5), [layers[1]],
                                         diagonals[1].unsqueeze(0)),
                      "inconsistent layer count accepted")
        expect_raises(RuntimeError, writer.write_manifest,
                      "manifest written while shards do not cover the key list")

    with tempfile.TemporaryDirectory() as tmp:
        _write_store(Path(tmp), _keys(2), shard_size=2)
        shard = GraphShard.load(Path(tmp) / "shard_00000.pt")
        shard.source_names = ("legacy_a", "legacy_b")
        shard.save(Path(tmp) / "resaved.pt")
        check(GraphShard.load(Path(tmp) / "resaved.pt").source_names == ("legacy_a", "legacy_b"),
              "re-save replaced the shard's own source table")


def test_source_ids_frozen() -> None:
    """Regression: ids derived by sorting renumbered every shard on disk when a source
    was added."""
    expected = {"clean_test": 0, "clean_train": 1, "brightness": 2, "zoom_blur": 20}
    for name, index in expected.items():
        check(SOURCE_IDS[name] == index, f"source id for {name} moved")
    check(len(ALL_SOURCES) == len(set(ALL_SOURCES)), "duplicate source")


# ------------------------------------------------------------------ pipeline (stubbed)

class _StubClassifier:
    def __init__(self, flip=(), heads=4, tokens=20, layer_count=2):
        self.flip, self.calls = set(flip), 0
        self.heads, self.tokens, self.layer_count = heads, tokens, layer_count
        self.model_id = "stub"

    def analyse(self, images, attentions=False, want_cls=False):
        batch = len(images)
        pred = np.array([9 if (self.calls + i) in self.flip else 1 for i in range(batch)])
        self.calls += batch
        att = [random_attention(batch, self.heads, self.tokens, seed=7)
               for _ in range(self.layer_count)] if attentions else None
        cls = torch.rand(batch, self.layer_count, 8, dtype=torch.float16) if want_cls else None
        return pred, np.full(batch, 0.9), np.full(batch, 0.5), att, cls


def test_extract_drift_policy_and_resume() -> None:
    """Regression: one borderline MPS-numerics flip aborted an hours-long extraction.
    Scan labels are canonical; isolated drift is recorded, pervasive drift aborts."""
    import polygraph.data.pipeline as pipeline

    class _FakePool:
        def image(self, index):
            return None

    keys = _keys(8)
    lookup = {k.as_tuple(): ScanRecord(k, 1, 1, 0.9, 0.5) for k in keys}
    original = pipeline.pool_for
    pipeline.pool_for = lambda key, data_root: _FakePool()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            writer = GraphStoreWriter(Path(tmp), 4, 0.02)
            result = pipeline.extract(_StubClassifier(flip={3}), Path(tmp), ThresholdGraphBuilder(0.02),
                                      keys, lookup, writer, batch_size=4, want_cls=False)
            check(result == {"written": 8, "prediction_drift": 1}, f"unexpected result {result}")
            check(int(GraphShard.load(Path(tmp) / "shard_00000.pt").meta["pred"][3]) == 1,
                  "drifted record must keep the scan's canonical label")
            result = pipeline.extract(_StubClassifier(), Path(tmp), ThresholdGraphBuilder(0.02),
                                      keys, lookup, writer, batch_size=4, want_cls=False)
            check(result["written"] == 8, "re-run should resume, not duplicate")

        with tempfile.TemporaryDirectory() as tmp:
            expect_raises(RuntimeError,
                          lambda: pipeline.extract(_StubClassifier(flip=set(range(8))), Path(tmp),
                                                   ThresholdGraphBuilder(0.02), keys, lookup,
                                                   GraphStoreWriter(Path(tmp), 4, 0.02), 4, False),
                          "pervasive drift must abort")
        with tempfile.TemporaryDirectory() as tmp:
            expect_raises(KeyError,
                          lambda: pipeline.extract(_StubClassifier(), Path(tmp), ThresholdGraphBuilder(0.02),
                                                   _keys(1, "fog", 1), {}, GraphStoreWriter(Path(tmp), 4, 0.02),
                                                   4, False),
                          "missing scan record must abort")
    finally:
        pipeline.pool_for = original


def test_scan_is_resumable() -> None:
    import polygraph.data.pipeline as pipeline

    class _FakePool:
        def __init__(self):
            self.name, self.severity = "fog", 1

        def __len__(self):
            return 6

        def label(self, i):
            return 1

        def image(self, i):
            return None

        def key(self, i):
            return RecordKey("fog", 1, i)

    original = pipeline.get_pool
    pipeline.get_pool = lambda *a, **k: _FakePool()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            scan_path = Path(tmp) / "scan.jsonl"
            pipeline.scan(_StubClassifier(), Path(tmp), scan_path, [("fog", 1)], batch_size=4)
            first = list(read_scan_records(scan_path))
            check(len(first) == 6, "scan wrote wrong count")
            pipeline.scan(_StubClassifier(flip=set(range(6))), Path(tmp), scan_path, [("fog", 1)])
            check(list(read_scan_records(scan_path)) == first,
                  "second scan must skip already-scanned records")
    finally:
        pipeline.get_pool = original


# ------------------------------------------------------------------ models + training

def test_models_match_poc_originals() -> None:
    """The comparability anchor: our copies must behave identically to the frozen POC."""
    sys.path.insert(0, str(PROJECT / "legacy"))
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    from lightweight_attention_experiments import ReadoutModel as PocReadout
    from lightweight_attention_experiments import SequenceConcatModel as PocSequence
    from polygraph.training.models import ReadoutModel, SequenceConcatModel

    torch.manual_seed(3)
    datas = []
    for _ in range(4):
        x = torch.rand(20, 16)
        x[:, 2] = 0
        x[0, 2] = 1.0
        datas.append(Data(x=x, edge_index=torch.randint(0, 20, (2, 30)),
                          edge_attr=torch.rand(30, 12), y=torch.tensor([1.0]),
                          layer_id=torch.zeros(20, dtype=torch.long)))
    batch = next(iter(DataLoader(datas, batch_size=4)))
    for readout in ("mean", "cls", "cls_mean", "cls_mean_max", "cls_gated"):
        torch.manual_seed(0)
        ours = ReadoutModel(16, 12, 32, 2, 0.0, readout).eval()
        torch.manual_seed(0)
        poc = PocReadout(16, 12, 32, 2, 0.0, readout).eval()
        check(torch.allclose(ours(batch)[0], poc(batch)[0], atol=1e-6), f"{readout} differs from POC")
    torch.manual_seed(0)
    ours = SequenceConcatModel(16, 12, 32, 2, 0.0, layer_count=1).eval()
    torch.manual_seed(0)
    poc = PocSequence(16, 12, 32, 2, 0.0, layer_count=1).eval()
    check(torch.allclose(ours(batch)[0], poc(batch)[0], atol=1e-6), "sequence model differs from POC")


def _learnable_store(tmp: Path, n=48, layer_count=2):
    """A store whose labels are decodable from the node features, so training can fit."""
    builder = ThresholdGraphBuilder(0.02)
    writer = GraphStoreWriter(tmp, shard_size=16, tau=0.02)
    keys = _keys(n)
    writer.plan(keys)
    shard_index = 0
    for i, key in enumerate(keys):
        wrong = i % 2 == 1
        attention = random_attention(1, 4, 20, seed=100 + i)
        layers = [builder.build(attention)[0] for _ in range(layer_count)]
        diagonals = builder.diagonals(attention).to(torch.float16).repeat(layer_count, 1, 1)
        if wrong:  # plant the signal in the diagonal features
            diagonals = diagonals + 0.5
        writer.add(ScanRecord(key, 1, 2 if wrong else 1, 0.6 if wrong else 0.95, 0.3), layers, diagonals)
        if writer.pending >= 16:
            writer.flush(shard_index)
            shard_index += 1
    writer.flush(shard_index)
    writer.write_manifest()
    plan = SplitPlan(splits={"train": keys[:32], "val": keys[32:40], "test": keys[40:]})
    plan.save(tmp / "plan.json")
    return plan


def test_training_learns_and_checkpoints_roundtrip() -> None:
    from polygraph.training.train import TrainConfig, load_checkpoint, train_run

    with tempfile.TemporaryDirectory() as tmp:
        _learnable_store(Path(tmp))
        config = TrainConfig(layers=[1], epochs=25, batch_size=8, hidden_dim=16, patience=25)
        train_run(Path(tmp), Path(tmp) / "plan.json", Path(tmp) / "run", config, [7], CPU)
        payload = torch.load(Path(tmp) / "run" / "model_seed7.pt", map_location="cpu")
        check(payload["history"][-1]["val_auroc"] > 0.9,
              f"training failed to learn a plantable signal (val {payload['history'][-1]['val_auroc']})")
        model, loaded_config = load_checkpoint(Path(tmp) / "run" / "model_seed7.pt", CPU)
        check(loaded_config.layers == [1] and loaded_config.seed == 7, "checkpoint config wrong")

        from polygraph.training.train import collect
        store = GraphStore(Path(tmp))
        plan = SplitPlan.load(Path(tmp) / "plan.json")
        test_ds = AttentionGraphDataset(store, [1], plan.splits["test"])
        pred = collect(model, test_ds, CPU, 8)
        check(set(pred) == {"logit", "y", "confidence", "margin", "source_id", "severity"},
              "collect fields wrong")
        check(pred["y"].size == 8, "collect count wrong")


def test_multilayer_model_trains() -> None:
    from polygraph.training.train import TrainConfig, train_detector

    with tempfile.TemporaryDirectory() as tmp:
        plan = _learnable_store(Path(tmp))
        store = GraphStore(Path(tmp))
        train_ds = AttentionGraphDataset(store, [0, 1], plan.splits["train"])
        val_ds = AttentionGraphDataset(store, [0, 1], plan.splits["val"])
        model, history = train_detector(TrainConfig(layers=[0, 1], epochs=5, batch_size=8,
                                                    hidden_dim=16), train_ds, val_ds, CPU)
        check(type(model).__name__ == "SequenceConcatModel", "multi-layer must use the sequence model")
        check(len(history) >= 1, "no training history")


# ------------------------------------------------------------------ evaluation

def test_detector_metrics_sanity() -> None:
    from polygraph.training.evaluate import detector_metrics

    y = np.array([0, 0, 1, 1])
    perfect = detector_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    check(perfect["auroc"] == 1.0, "perfect detector should score AUROC 1")
    check(perfect["risk@0.5"] == 0.0, "perfect detector keeps a clean half")
    check(perfect["risk@1.0"] == 0.5, "risk at full coverage must equal the base rate")
    inverted = detector_metrics(y, np.array([0.9, 0.8, 0.2, 0.1]))
    check(inverted["aurc"] > perfect["aurc"], "AURC must penalise an inverted detector")
    check(detector_metrics(np.zeros(4), np.zeros(4))["auroc"] is None, "single class must yield None")


def test_combiner_standardization() -> None:
    """Regression (Stage 5): with unscaled features, the informative small-scale feature
    needs a large coefficient the L2 penalty refuses, while the huge-scale noise feature
    gets fitted cheaply — so the combiner ranked by the WEAKER signal and scored below
    its best single input. Requires the true failure conditions: MSP informative but
    tiny-variance, graph logit pure noise at a much larger scale, small train set."""
    from sklearn.metrics import roc_auc_score

    from polygraph.training.evaluate import _combiner_features, fit_combiner

    rng = np.random.default_rng(0)
    n = 1200
    y = (np.arange(n) % 2).astype(float)
    # MSP informative but OVERLAPPING at tiny scale (needs a big coefficient the penalty
    # refuses); graph weaker but loud (fitted cheaply). Without scaling the ranking falls
    # back to the weaker graph — measured: combined 0.880 < msp 0.911 (historic: 0.856 < 0.910).
    msp_logit = -4.4 + 0.06 * y + 0.035 * rng.normal(size=n)
    confidence = 1.0 / (1.0 + np.exp(msp_logit))
    pred = {"y": y, "confidence": confidence, "logit": 1.0 * y + 2.0 * rng.normal(size=n)}
    combiner = fit_combiner({k: v[:800] for k, v in pred.items()})
    test = {k: v[800:] for k, v in pred.items()}
    combined = combiner.predict_proba(_combiner_features(test))[:, 1]
    msp_auroc = roc_auc_score(test["y"], 1 - test["confidence"])
    check(0.85 < msp_auroc < 0.97, f"test setup drifted: msp {msp_auroc}")
    check(roc_auc_score(test["y"], combined) > msp_auroc - 0.02,
          "combiner scored below its best single input — the Stage-5 scaling bug")


def test_slices_and_full_evaluation() -> None:
    from polygraph.training.evaluate import evaluate_predictions, slice_masks

    n = 60
    rng = np.random.default_rng(1)
    pred = {"y": (rng.random(n) < 0.5).astype(float), "logit": rng.normal(size=n),
            "confidence": np.full(n, 0.9), "margin": np.full(n, 0.4),
            "source_id": np.array([SOURCE_IDS[s] for s in
                                   (["clean_test"] * 20 + ["fog"] * 20 + ["spatter"] * 20)]),
            "severity": np.array([0] * 20 + [3] * 20 + [3] * 20)}
    masks = slice_masks(pred, seen_sources=["clean_test", "fog"])
    check(masks["clean"].sum() == 20 and masks["seen_corruptions"].sum() == 20
          and masks["unseen_sources"].sum() == 20 and masks["unseen_extra_family"].sum() == 20
          and masks["severity_3"].sum() == 40, "slice masks wrong")
    report = evaluate_predictions(pred, pred, ["clean_test", "fog"])
    for slice_name in ("all", "clean", "seen_corruptions", "unseen_sources"):
        check(set(report[slice_name]) == {"graph", "msp", "margin", "msp_plus_graph"},
              f"detectors missing in slice {slice_name}")


def test_cls_exposed_and_baselines_run() -> None:
    """The stored CLS embeddings must reach the reader, and the trained baselines must
    appear in every slice with the same metric structure as the detector."""
    from polygraph.training.baselines import collect_features, output_scores
    from polygraph.training.evaluate import evaluate_run
    from polygraph.training.train import TrainConfig, train_run

    with tempfile.TemporaryDirectory() as tmp:
        _, reference = _write_store(Path(tmp), _keys(6), with_cls=True)
        dataset = AttentionGraphDataset(GraphStore(Path(tmp)), layers=[0])
        sample = dataset[0]
        check(hasattr(sample, "cls_layers") and tuple(sample.cls_layers.shape) == (1, 2, 8),
              "cls embeddings not exposed through the reader")
        features = collect_features(dataset)
        check(features["cls"].shape == (6, 8), "collect_features cls shape wrong")
        scores = output_scores(features, features, seed=0)
        check(scores.shape == (6,), "output baseline shape wrong")


def test_evaluate_run_end_to_end() -> None:
    """Store -> train -> checkpoints -> separate evaluation, without the ViT."""
    from polygraph.training.evaluate import evaluate_run
    from polygraph.training.train import TrainConfig, train_run

    with tempfile.TemporaryDirectory() as tmp:
        _learnable_store(Path(tmp))
        config = TrainConfig(layers=[1], epochs=10, batch_size=8, hidden_dim=16)
        train_run(Path(tmp), Path(tmp) / "plan.json", Path(tmp) / "run", config, [7, 8], CPU)
        summary = evaluate_run(Path(tmp) / "run", Path(tmp), Path(tmp) / "plan.json", CPU)
        check(len(summary["checkpoints"]) == 2, "should evaluate every checkpoint")
        check("all" in summary["slices"] and "graph" in summary["slices"]["all"], "summary shape wrong")
        check("output_lr" in summary["slices"]["all"], "trained output baseline missing")
        check((Path(tmp) / "run" / "summary.json").exists(), "summary not written")


# ------------------------------------------------------------------ CLI

@contextmanager
def _chdir(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def test_data_cli_split() -> None:
    from polygraph.data.cli import main as data_main

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "data" / "graph_dataset").mkdir(parents=True)
        records = _synthetic_records(120)
        records += [ScanRecord(RecordKey("clean_train", 0, i), 1, 1, 0.99, 0.9) for i in range(50)]
        append_scan_records(root / "data" / "graph_dataset" / "scan_records.jsonl", records)
        with _chdir(root):
            data_main(["split", "--train-cap", "20", "--val-cap", "5", "--test-cap", "10"])
        plan = SplitPlan.load(root / "data" / "graph_dataset" / "split_plan.json")
        sources = {k.source for keys in plan.splits.values() for k in keys}
        check("clean_train" not in sources,
              "regression: contaminated clean_train records entered the plan")
        for name in ("train", "val"):
            check("spatter" not in {k.source for k in plan.splits[name]},
                  "default held-out extras leaked into training")


def test_training_cli_end_to_end() -> None:
    from polygraph.training.cli import main as training_main

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store_dir = root / "data" / "graph_dataset" / "store"
        store_dir.mkdir(parents=True)
        _learnable_store(store_dir)
        (store_dir / "plan.json").rename(root / "data" / "graph_dataset" / "split_plan.json")
        with _chdir(root):
            training_main(["train", "--layers", "1", "--seeds", "7", "--out-dir", "run"])
            training_main(["evaluate", "--run-dir", "run"])
        check((root / "run" / "summary.json").exists(), "CLI train+evaluate left no summary")


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        before = len(FAILURES)
        try:
            test()
        except Exception as error:  # a crash is a failure, not a stop
            FAILURES.append(f"{test.__name__} crashed: {type(error).__name__}: {error}")
        print(f"  [{'PASS' if len(FAILURES) == before else 'FAIL'}] {test.__name__}")
    print()
    if FAILURES:
        for failure in FAILURES:
            print(f"  FAILURE: {failure}")
        raise SystemExit(1)
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
