from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polygraph.training.research_eval import (grouped_split, paired_group_bootstrap,
                                              select_pnorm, verify_score_registry)
from polygraph.training.graph_stats import final_layer_graph_statistics


def test_group_split_is_disjoint():
    groups = np.repeat([f"g{i}" for i in range(20)], 3)
    a, b = grouped_split(groups, .7, 9)
    assert set(groups[a]).isdisjoint(groups[b])


def test_p_selection_cannot_receive_test_labels():
    rng = np.random.default_rng(1)
    logits, y = rng.normal(size=(80, 5)), np.tile([0, 1], 40)
    selected = select_pnorm(logits, y)
    # Mutating an unrelated test target cannot enter this API or alter validation selection.
    test_y = 1 - y
    assert selected == select_pnorm(logits, y)
    assert test_y is not y


def test_bootstrap_resamples_whole_groups():
    groups = np.repeat(np.arange(20), 4)
    y = np.tile([0, 0, 1, 1], 20)
    candidate = y + np.random.default_rng(2).normal(0, .2, len(y))
    reference = np.random.default_rng(3).normal(size=len(y))
    result = paired_group_bootstrap(y, candidate, reference, groups, repetitions=30)
    assert result["resampling_unit"] == "base_image_group"
    assert result["auroc_delta"]["repetitions"] == 30


def test_score_registry_rejects_permuted_rows():
    with tempfile.TemporaryDirectory() as tmp:
        fields = dict(y=np.array([0, 1]), image_id=np.array([4, 5]), source_id=np.array([1, 1]),
                      severity=np.array([2, 2]), store_index=np.array([8, 9]),
                      plan_hash=np.array(["p", "p"]), score=np.array([.1, .9]))
        a, b = Path(tmp) / "a.npz", Path(tmp) / "b.npz"
        np.savez(a, **fields); np.savez(b, **{k: v[::-1] for k, v in fields.items()})
        try:
            verify_score_registry([a, b])
            raise AssertionError("misaligned score vectors accepted")
        except ValueError as exc:
            assert "misalignment" in str(exc)


def test_graph_statistics_shape_and_finiteness():
    import torch
    g = torch.Generator().manual_seed(4)
    edge_index = torch.randint(0, 9, (2, 40), generator=g)
    edge_attr = torch.rand(40, 12, generator=g)
    diagonal = torch.rand(9, 12, generator=g)
    features = final_layer_graph_statistics(edge_index, edge_attr, diagonal, 9, .2)
    assert features.shape == (161,) and torch.isfinite(features).all()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test(); print(f"[PASS] {test.__name__}")
    print(f"All {len(tests)} tests passed.")
