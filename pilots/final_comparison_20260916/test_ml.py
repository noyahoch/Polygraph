"""Synthetic CPU-only model/data/training tests; the runner must be an allocated Slurm job.

Tiny in-memory tensors only: no immutable cache, sidecar, campaign or GPU access.
"""
from collections import OrderedDict
import copy
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Batch

from polygraph.data.graphs import node_coordinates
from pilots.layer_screen_20260913.protocol import ARMS as GRAPH_ARMS
from pilots.layer_screen_20260913.train import BlockShuffleSampler, _restore_rng, _rng_state, _seed
from pilots.topology_20260910.data import GraphData

from . import data, models, preflight, protocol, train

SEED = protocol.SEEDS[0]
IDENTITIES = [(family, arm) for family in protocol.FAMILIES
              for arm in (("logits",) if family == "O" else protocol.ARMS)]
TARGET = {"y", "record_id", "image_id", "source_id", "split_id"}


def fake_dataset(family, arm, rows=2):
    """A RoleDataset whose tensors are supplied directly; __init__ (cache access) is bypassed."""
    dataset = object.__new__(data.RoleDataset)
    dataset.family, dataset.arm, dataset.role, dataset.diagnostic = family, arm, "base_train", True
    dataset.entries = [{"record_id": 100 + i, "image_id": i, "source_id": 3, "split_id": 1, "severity": 0,
                        "y": i % 2, "label": 5, "pred": 5 + i % 2, "shard": "s.pt", "offset": i}
                       for i in range(rows)]
    dataset.hidden_entries = {100 + i: {"hidden_shard": "h.pt", "hidden_offset": i} for i in range(rows)}
    dataset.loaded, dataset.loaded_hidden, dataset.fingerprints = OrderedDict(), OrderedDict(), {}
    dataset._logits = None
    return dataset


def cache_payload(rows=2, fill=12.0, poison=0.0):
    return {"record_id": torch.arange(100, 100 + rows, dtype=torch.int64),
            "hidden": torch.full((rows, 197, 768), fill, dtype=torch.float16),
            "logits": torch.full((rows, 100), poison), "diagonals": torch.full((rows, 12, 197, 12), poison),
            "sparse_offsets": torch.zeros(rows + 1, dtype=torch.int64),
            "sparse_positions": torch.zeros(0, dtype=torch.int64), "sparse_values": torch.zeros(0)}


def hidden_payload(rows=2):
    return {"record_id": torch.arange(100, 100 + rows, dtype=torch.int64),
            **{f"hidden_{layer}": torch.full((rows, 197, 768), float(layer), dtype=torch.float16)
               for layer in data.HIDDEN_INDICES}}


def hidden_x(generator):
    return torch.cat((node_coordinates(197, 0, 1), torch.randn(197, 768, generator=generator)), dim=1)


def set_graph(generator, edges=40, edge_index=None):
    x = torch.cat((node_coordinates(197, 0, 1), torch.randn(197, 780, generator=generator)), dim=1)
    if edge_index is None:
        edge_index = torch.randint(0, 197, (2, edges), generator=generator)
    return GraphData(x=x, edge_index=edge_index, edge_attr=torch.rand(edges, 12, generator=generator),
                     y=torch.tensor([1.0]), record_id=torch.tensor([1]))


def history(aurocs):
    rows, best, epoch = [], -math.inf, 0
    for index, value in enumerate(aurocs, 1):
        if value > best:
            best, epoch = value, index
        rows.append({"epoch": index, "training_loss": 1.0 / index, "checkpoint_auroc": value,
                     "best_epoch": epoch, "best_checkpoint_auroc": best,
                     "training_checkpoint_seconds": 1.5})
    return rows


class Blocks:
    def __len__(self):
        return 6

    def shard_blocks(self):
        return [[0, 1, 2], [3, 4, 5]]


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        protocol.require_slurm()

    def test_expected_parameter_counts_for_every_family_and_arm(self):
        for family, arm in IDENTITIES:
            with self.subTest(family=family, arm=arm):
                model = models.build_model(family, arm)
                count = sum(p.numel() for p in model.parameters() if p.requires_grad)
                self.assertEqual(count, models.EXPECTED_PARAMETERS[family])
                self.assertEqual(count, protocol.MODEL_SPECS[family]["parameters"])
                self.assertEqual(models.parameter_count(model), count)
                self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
        self.assertEqual(models.EXPECTED_PARAMETERS, {"G": 130434, "H": 129986, "S": 131126, "O": 8577})

    def test_parameter_mismatch_and_unknown_arm_are_refused(self):
        with patch.dict(models.EXPECTED_PARAMETERS, {"H": 1}), self.assertRaisesRegex(RuntimeError, "mismatch"):
            models.build_model("H", "block2")
        for family, arm in (("O", "block2"), ("H", "logits"), ("L", "logits"), ("G", "union4")):
            with self.subTest(family=family, arm=arm), self.assertRaises(ValueError):
                models.build_model(family, arm)
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, "Slurm"):
            models.build_model("O", "logits")

    def test_layer_map_blocks_to_hidden_states(self):
        self.assertEqual(models.GRAPH_INDEX, {"block2": 2, "block5": 5, "block8": 8, "block11": 11})
        self.assertEqual(models.HIDDEN_INDEX, {"block2": 3, "block5": 6, "block8": 9, "block11": 12})
        self.assertEqual(tuple(models.HIDDEN_INDEX.values()), protocol.LAYERS)
        self.assertEqual(data.HIDDEN_INDICES, (3, 6, 9))
        for arm in protocol.ARMS:
            with self.subTest(arm=arm):
                self.assertEqual(models.HIDDEN_INDEX[arm], models.GRAPH_INDEX[arm] + 1)
                self.assertEqual(GRAPH_ARMS[arm]["layers"], [models.GRAPH_INDEX[arm]])

    def test_graph_and_hidden_views_read_their_mapped_layers(self):
        for arm in protocol.ARMS:
            calls = []

            def materialize(payload, offset, layers):
                calls.append(list(layers))
                return torch.zeros(197, 784), torch.zeros((2, 3), dtype=torch.int64), torch.zeros(3, 12)

            with self.subTest(family="G", arm=arm), patch.object(data, "graph_materialize", materialize):
                item = fake_dataset("G", arm).graph_at(1, payload=cache_payload())
                self.assertEqual(calls, [[models.GRAPH_INDEX[arm]]])
                self.assertEqual(int(item.record_id), 101)
            with self.subTest(family="H", arm=arm):
                dataset = fake_dataset("H", arm)
                if models.HIDDEN_INDEX[arm] == 12:
                    dataset.hidden_entries = None  # H12 must come from the immutable cache
                    item = dataset.graph_at(0, payload=cache_payload(fill=12.0))
                else:
                    item = dataset.graph_at(0, hidden_payload=hidden_payload())
                self.assertTrue(torch.all(item.x[:, 4:] == models.HIDDEN_INDEX[arm]))
                self.assertTrue(torch.equal(item.x[:, :4], node_coordinates(197, 0, 1)))

    def test_misaligned_hidden_record_is_refused(self):
        payload = hidden_payload()
        payload["record_id"] = payload["record_id"].flip(0)
        with self.assertRaisesRegex(RuntimeError, "alignment"):
            fake_dataset("H", "block5").graph_at(0, hidden_payload=payload)


class InputIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        protocol.require_slurm()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="final-ml-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_hidden_items_expose_only_coordinates_hidden_tokens_and_targets(self):
        for arm in protocol.ARMS:
            dataset = fake_dataset("H", arm)
            items = []
            for poison in (0.0, 9.0):
                kwargs = ({"payload": cache_payload(poison=poison)} if models.HIDDEN_INDEX[arm] == 12
                          else {"hidden_payload": hidden_payload()})
                items.append(dataset.graph_at(1, **kwargs))
            with self.subTest(arm=arm):
                self.assertEqual(preflight._keys(items[0]), TARGET | {"x"})
                self.assertEqual(items[0].x.shape, (197, 772))
                self.assertEqual(int(items[0].x[:, 2].sum()), 1)
                self.assertTrue(torch.equal(items[0].x, items[1].x))
            with self.assertRaisesRegex(RuntimeError, "not exposed"):
                dataset.logits()

    def test_cache_loader_drops_attention_and_logits_for_hidden_models(self):
        shards = self.root / "shards"
        shards.mkdir()
        torch.save(cache_payload(), shards / "s.pt")
        expected = {"H": {"record_id", "hidden"}, "O": {"record_id", "logits"},
                    "S": {"record_id", "hidden", "diagonals", "sparse_offsets", "sparse_positions",
                          "sparse_values"}}
        with patch.object(data, "verify_cached_file", return_value="fingerprint"):
            for family, keys in expected.items():
                dataset = fake_dataset(family, "logits" if family == "O" else "block2")
                dataset.cache, dataset.shards = self.root, {"s.pt": {"sha256": "a" * 64, "records": 2}}
                with self.subTest(family=family):
                    self.assertEqual(set(dataset._load("s.pt")), keys)

    def test_hidden_sidecar_with_foreign_features_is_refused(self):
        (self.root / "shards").mkdir()
        dataset = fake_dataset("H", "block2", rows=1)
        dataset.hidden = self.root
        dataset.hidden_shards = {"h.pt": {"sha256": "a" * 64, "receipt_sha256": "b" * 64, "records": 1}}
        with patch.object(data, "verify_cached_file", return_value="fingerprint"):
            torch.save(hidden_payload(rows=1), self.root / "shards/h.pt")
            self.assertEqual(set(dataset._load_hidden("h.pt")), {"record_id", "hidden_3", "hidden_6", "hidden_9"})
            for extra in ("logits", "attention", "hidden_12", "y"):
                dataset.loaded_hidden.clear()
                dataset.fingerprints.clear()
                torch.save({**hidden_payload(rows=1), extra: torch.zeros(1)}, self.root / "shards/h.pt")
                with self.subTest(extra=extra), self.assertRaisesRegex(RuntimeError, "must not contain"):
                    dataset._load_hidden("h.pt")

    def test_hidden_forward_refuses_attention_incidence_or_logits(self):
        model = models.build_model("H", "block2").eval()
        generator = torch.Generator().manual_seed(SEED)
        clean = GraphData(x=hidden_x(generator), y=torch.tensor([1.0]), record_id=torch.tensor([1]))
        scores, labels = train.forward(model, Batch.from_data_list([clean, clean]), "H", "cpu")
        self.assertEqual(scores.shape, (2,))
        self.assertEqual(labels.shape, (2,))
        for name, value in (("edge_index", torch.zeros((2, 1), dtype=torch.int64)),
                            ("edge_attr", torch.zeros(1, 12)), ("output_logits", torch.zeros(1, 100))):
            poisoned = GraphData(x=clean.x.clone(), y=clean.y, record_id=clean.record_id, **{name: value})
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "Hidden-only"):
                train.forward(model, Batch.from_data_list([poisoned]), "H", "cpu")
        wide = GraphData(x=torch.zeros(197, 784), y=clean.y, record_id=clean.record_id)
        with self.assertRaisesRegex(RuntimeError, "Hidden-only"):
            train.forward(model, Batch.from_data_list([wide]), "H", "cpu")


class InvarianceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        protocol.require_slurm()

    def setUp(self):
        torch.manual_seed(SEED)
        self.generator = torch.Generator().manual_seed(SEED)

    def test_set_view_discards_within_image_incidence(self):
        x = torch.cat((node_coordinates(197, 0, 1), torch.randn(197, 780, generator=self.generator)), dim=1)
        edge_attr = torch.rand(30, 12, generator=self.generator)
        items = []
        for _ in range(3):
            incidence = torch.randint(0, 197, (2, 30), generator=self.generator)
            with patch.object(data, "graph_materialize", return_value=(x, incidence, edge_attr)):
                items.append(fake_dataset("S", "block8").graph_at(0, payload=cache_payload()))
        model = models.build_model("S", "block8").eval()
        with torch.no_grad():
            scores = [model(Batch.from_data_list([item, item]))[0] for item in items]
        for item, score in zip(items, scores):
            self.assertEqual(preflight._keys(item), TARGET | {"x", "edge_index", "edge_attr"})
            self.assertTrue(torch.equal(item.edge_index, torch.zeros((2, 30), dtype=torch.int64)))
            self.assertTrue(torch.equal(score, scores[0]))

    def test_set_model_ignores_incidence_that_preserves_node_and_edge_multisets(self):
        model = models.build_model("S", "block2").eval()
        first = set_graph(self.generator)
        second = set_graph(self.generator, edges=25)
        rewired = [GraphData(x=g.x, edge_attr=g.edge_attr, y=g.y, record_id=g.record_id,
                             edge_index=torch.randint(0, 197, g.edge_index.shape, generator=self.generator))
                   for g in (first, second)]
        with torch.no_grad():
            original = model(Batch.from_data_list([first, second]))[0]
            changed = model(Batch.from_data_list(rewired))[0]
        torch.testing.assert_close(original, changed, atol=0, rtol=0)

    def test_hidden_and_set_token_permutation_with_attributes_and_cls(self):
        permutation = torch.randperm(197, generator=self.generator)
        permutation = torch.cat((permutation[permutation != 0][:50], torch.tensor([0]),
                                 permutation[permutation != 0][50:]))
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(197)
        cls_position = int(inverse[0])
        self.assertNotEqual(cls_position, 0)
        for family in ("H", "S"):
            model = models.build_model(family, "block5").eval()
            if family == "H":
                graph = GraphData(x=hidden_x(self.generator), y=torch.tensor([0.0]), record_id=torch.tensor([7]))
                moved = GraphData(x=graph.x[permutation], y=graph.y, record_id=graph.record_id)
            else:
                graph = set_graph(self.generator)
                edges = torch.randperm(graph.edge_attr.shape[0], generator=self.generator)
                # Coordinates/CLS flag are columns of x, so they travel with each token;
                # endpoints are relabeled to the moved tokens and edge rows are shuffled.
                moved = GraphData(x=graph.x[permutation], edge_attr=graph.edge_attr[edges],
                                  edge_index=inverse[graph.edge_index[:, edges]], y=graph.y,
                                  record_id=graph.record_id)
            with self.subTest(family=family), torch.no_grad():
                self.assertEqual(float(moved.x[cls_position, 2]), 1.0)
                score, embedding = model(Batch.from_data_list([graph, moved]))
                torch.testing.assert_close(score[0], score[1], atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(embedding[0], embedding[1], atol=1e-5, rtol=1e-5)
                width = protocol.MODEL_SPECS[family]["width"]
                nodes = model.node_phi(graph.x)
                torch.testing.assert_close(embedding[0, :width], nodes[0])
                torch.testing.assert_close(embedding[1, :width], model.node_phi(moved.x)[cls_position])
                changed = graph.x.clone()
                changed[0, 10:] += 1.0
                altered = (GraphData(x=changed, y=graph.y, record_id=graph.record_id) if family == "H"
                           else GraphData(x=changed, edge_attr=graph.edge_attr, edge_index=graph.edge_index,
                                          y=graph.y, record_id=graph.record_id))
                other = model(Batch.from_data_list([altered]))[1]
                self.assertFalse(torch.allclose(other[0, :width], embedding[0, :width]))

    def test_missing_or_duplicate_cls_is_refused(self):
        model = models.build_model("H", "block2").eval()
        for flags in ([], [0, 5]):
            x = hidden_x(self.generator)
            x[:, 2] = 0
            if flags:
                x[flags, 2] = 1
            graph = GraphData(x=x, y=torch.tensor([0.0]), record_id=torch.tensor([1]))
            with self.subTest(flags=flags), torch.no_grad(), self.assertRaisesRegex(RuntimeError, "CLS"):
                model(Batch.from_data_list([graph]))


class SelectionAndResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        protocol.require_slurm()

    def test_earliest_tie_and_boundary_selection(self):
        epochs = protocol.TRAINING["epochs"]
        values = [0.6, 0.8, 0.8] + [0.7] * (epochs - 3)
        self.assertEqual(train.history_selection(history(values)), (2, 0.8))
        late = history(values)
        for row in late[2:]:
            row["best_epoch"] = 3
        with self.assertRaisesRegex(RuntimeError, "earliest"):
            train.history_selection(late)
        rising = [0.5 + 0.01 * i for i in range(epochs)]
        self.assertEqual(train.history_selection(history(rising))[0], epochs)

    def test_fixed20_completion_is_required(self):
        epochs = protocol.TRAINING["epochs"]
        short = history([0.7] * (epochs - 1))
        with self.assertRaisesRegex(RuntimeError, "each completed epoch"):
            train.history_selection(short)
        self.assertEqual(train.history_selection(short, completed_epochs=epochs - 1), (1, 0.7))
        for horizon in (0, epochs + 1, True, 20.0):
            with self.subTest(horizon=horizon), self.assertRaisesRegex(RuntimeError, "horizon"):
                train.history_selection(history([0.7] * epochs), completed_epochs=horizon)
        gap = history([0.7] * epochs)
        del gap[14]
        with self.assertRaises(RuntimeError):
            train.history_selection(gap)
        for name, value in (("training_loss", float("nan")), ("checkpoint_auroc", 1.2),
                            ("training_checkpoint_seconds", None), ("training_loss", -1.0)):
            broken = history([0.7] * epochs)
            broken[-1][name] = value
            with self.subTest(name=name, value=value), self.assertRaises(RuntimeError):
                train.history_selection(broken)

    def resume_state(self):
        _seed(SEED)
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0001)
        model(torch.ones(2, 3)).sum().backward()
        optimizer.step()
        sampler = BlockShuffleSampler(Blocks(), SEED)
        list(sampler)
        list(sampler)
        rows = history([0.6, 0.7])
        best = {"weight": torch.ones(1, 3), "bias": torch.zeros(1)}
        scores = np.linspace(0, 1, 8, dtype=np.float32)
        config = {"scope_id": protocol.SCOPE, "seed": SEED}
        state = {"scope_id": protocol.SCOPE, "config_sha256": "c" * 64, "completed_epochs": 2,
                 "history": copy.deepcopy(rows), "best_epoch": 2, "best_auc": 0.7,
                 "sampler": sampler.state_dict(), "best_state": {k: v.clone() for k, v in best.items()},
                 "best_scores": scores.copy(), "rng": _rng_state(),
                 "loader_rng": torch.Generator().manual_seed(SEED).get_state(),
                 "optimizer": optimizer.state_dict(),
                 "audit": {"indices": list(range(48)), "role": "checkpoint", "scores": np.zeros(48, np.float32)}}
        return state, config, rows, best, scores

    def test_resume_state_verifier_accepts_only_bound_complete_state(self):
        state, config, rows, best, scores = self.resume_state()
        train._verify_state(state, config, rows, best, scores, Blocks(), "c" * 64)

        def epoch(value):
            value["sampler"]["epoch"] = 1

        def seed(value):
            value["sampler"]["seed"] = SEED + 1

        def recipe(value):
            value["optimizer"]["param_groups"][0]["lr"] = .003

        def empty_optimizer(value):
            value["optimizer"]["state"] = {}

        def stream(value):
            del value["rng"]["cuda"]

        def tensor(value):
            value["best_state"]["bias"] += 1

        def audit(value):
            value["audit"]["indices"] = list(range(47))

        def generator(value):
            value["loader_rng"] = torch.zeros(0, dtype=torch.uint8)

        def selected(value):
            value["best_epoch"] = 1

        def binding(value):
            value["config_sha256"] = "d" * 64

        def drift(value):
            value["best_scores"] = value["best_scores"] + 1e-3

        for change in (epoch, seed, recipe, empty_optimizer, stream, tensor, audit, generator,
                       selected, binding, drift):
            changed = copy.deepcopy(state)
            change(changed)
            with self.subTest(change=change.__name__), self.assertRaises(RuntimeError):
                train._verify_state(changed, config, rows, best, scores, Blocks(), "c" * 64)

    def test_compare_tree_exact_tolerance_and_layout(self):
        a = {"w": torch.tensor([1.0, 2.0]), "i": torch.tensor([1, 2]), "n": [1, (2, "x")]}
        self.assertEqual(preflight.compare_tree(a, copy.deepcopy(a)), 0.0)
        near = copy.deepcopy(a)
        near["w"] = near["w"] + 1e-7
        with self.assertRaises(AssertionError):
            preflight.compare_tree(a, near)
        self.assertLess(preflight.compare_tree(a, near, atol=1e-5), 1e-5)
        for change in ({"w": torch.tensor([1.0, 2.1])}, {"i": torch.tensor([1, 3])},
                       {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}, {"n": [1, (2, "y")]},
                       {"extra": 1}, {"w": torch.tensor([1.0, float("nan")])}):
            with self.subTest(change=change), self.assertRaises(AssertionError):
                preflight.compare_tree(a, {**a, **change}, atol=1e-5)

    def test_cpu_resume_roundtrip_matches_uninterrupted_continuation(self):
        inputs = torch.randn(6, 8, 100, generator=torch.Generator().manual_seed(SEED))
        targets = (inputs[..., 0] > 0).float()

        def steps(model, optimizer, sampler, indices):
            model.train()
            losses = []
            for index in indices:
                order = list(sampler)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(inputs[index][order]).reshape(-1), targets[index][order])
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().clone())
            return {"losses": losses, "model": preflight.cpu_tree(model.state_dict()),
                    "optimizer": preflight.cpu_tree(optimizer.state_dict()), "sampler": sampler.state_dict()}

        def fresh():
            model = models.build_model("O", "logits")
            return model, torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0001)

        _seed(SEED)
        model, optimizer = fresh()
        sampler = BlockShuffleSampler(Blocks(), SEED)
        steps(model, optimizer, sampler, (0, 1))
        with tempfile.TemporaryDirectory(prefix="final-ml-resume-") as temporary:
            path = Path(temporary) / "latest.pt"
            data.atomic_torch(path, {"model": preflight.cpu_tree(model.state_dict()),
                                     "optimizer": preflight.cpu_tree(optimizer.state_dict()),
                                     "rng": _rng_state(), "sampler": sampler.state_dict()})
            expected = steps(model, optimizer, sampler, (2, 3, 4))
            restored = torch.load(path, map_location="cpu", weights_only=False)

        def resume(restore_rng):
            state = copy.deepcopy(restored)  # optimizer loading may alias the stored tensors
            clone, clone_optimizer = fresh()
            clone.load_state_dict(state["model"], strict=True)
            clone_optimizer.load_state_dict(state["optimizer"])
            preflight.compare_tree(restored["optimizer"], preflight.cpu_tree(clone_optimizer.state_dict()))
            clone_sampler = BlockShuffleSampler(Blocks(), SEED)
            clone_sampler.load_state_dict(state["sampler"])
            if restore_rng:
                _restore_rng(state["rng"])
            return steps(clone, clone_optimizer, clone_sampler, (2, 3, 4))

        self.assertEqual(preflight.compare_tree(expected, resume(True)), 0.0)
        torch.manual_seed(SEED + 99)
        with self.assertRaises(AssertionError):
            preflight.compare_tree(expected, resume(False))


class PreflightGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        protocol.require_slurm()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="final-preflight-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_scientific_roots_are_refused(self):
        self.assertEqual(preflight.refuse_scientific_root(self.root), self.root)
        for name in ("execution.json", "predictions/dev_eval", "runs/H", "evaluation_gate.json"):
            path = self.root / name
            if path.suffix:
                path.write_text("{}")
            else:
                path.mkdir(parents=True)
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "refuses"):
                preflight.refuse_scientific_root(self.root)
            if path.suffix:
                path.unlink()
            else:
                path.rmdir()
                path.parent.rmdir()

    def test_main_refuses_execution_root_before_any_work_or_receipt(self):
        (self.root / "execution.json").write_text("{}")
        argv = ["--root", str(self.root), "--data-root", str(self.root / "data"),
                "--out", str(self.root / "preflight.json")]
        with patch.object(preflight, "extract") as extract, \
             patch.object(preflight, "read_campaign") as read_campaign, \
             self.assertRaisesRegex(RuntimeError, "execution.json"):
            preflight.main(argv)
        extract.assert_not_called()
        read_campaign.assert_not_called()
        self.assertFalse((self.root / "preflight.json").exists())
        self.assertFalse((self.root / "preflight").exists())

    def test_failure_writes_incomplete_receipt_and_foreign_output_is_refused(self):
        argv = ["--root", str(self.root), "--data-root", str(self.root / "data"),
                "--out", str(self.root / "preflight.json")]
        with patch.object(preflight, "read_campaign", side_effect=RuntimeError("synthetic campaign")), \
             self.assertRaisesRegex(RuntimeError, "synthetic campaign"):
            preflight.main(argv)
        receipt = protocol.read(self.root / "preflight.json")
        self.assertIs(receipt["complete"], False)
        self.assertIn("synthetic campaign", receipt["error"])
        self.assertEqual(receipt["job_id"], os.environ["SLURM_JOB_ID"])
        self.assertEqual((receipt["scientific_fits"], receipt["dev_eval_scoring"],
                          receipt["original_test_access"]), (0, False, False))
        with self.assertRaises(SystemExit):
            preflight.main(argv[:-1] + [str(self.root / "elsewhere.json")])

    def test_complete_receipt_is_never_replaced(self):
        path = self.root / "preflight.json"
        preflight.write_receipt(path, {"complete": False, "step": 1})
        preflight.write_receipt(path, {"complete": True, "job_id": "1"})
        preflight.write_receipt(path, {"complete": True, "job_id": "1"})
        for value in ({"complete": True, "job_id": "2"}, {"complete": False, "error": "late"}):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "complete preflight"):
                preflight.write_receipt(path, value)
        self.assertEqual(protocol.read(path), {"complete": True, "job_id": "1"})


if __name__ == "__main__":
    protocol.require_slurm()
    unittest.main()
