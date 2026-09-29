"""Split plans: group-disjoint over base images, class-balanced within every cell.

Group disjointness: all corruptions/severities of one photograph share a split, else the
same picture leaks across train and test. Stratification: error rate climbs ~8.5% -> ~60%
with severity, so per-cell balance stops a detector scoring by corruption strength alone.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from ..records import RecordKey, ScanRecord

SPLITS = ("train", "val", "test")


def assign_groups(group_ids: Iterable[str], fractions: Tuple[float, float, float], seed: int) -> Dict[str, str]:
    """Base image -> split. Raises rather than let rounding leave a split empty."""
    assert abs(sum(fractions) - 1) < 1e-6, fractions
    unique = sorted(set(group_ids))
    random.Random(seed).shuffle(unique)
    a = int(len(unique) * fractions[0])
    b = a + int(len(unique) * fractions[1])
    if not 0 < a < b < len(unique):
        raise ValueError(f"fractions {fractions} over {len(unique)} base images leave a split empty")
    return {g: ("train" if p < a else "val" if p < b else "test") for p, g in enumerate(unique)}


def select_balanced(cells: Mapping[tuple, Tuple[List[int], List[int]]], cap: int, rng: random.Random,
                    stratified: bool = True) -> List[int]:
    """Pick equal wrong/correct counts; stratified = per cell, water-filled smallest-first
    so a naturally small cell (clean images rarely fail) cannot starve the rest."""
    for wrong, right in cells.values():
        rng.shuffle(wrong)
        rng.shuffle(right)
    if not stratified:
        wrong = [i for w, _ in cells.values() for i in w]
        right = [i for _, r in cells.values() for i in r]
        take = min(len(wrong), len(right), cap or 10**9)
        return wrong[:take] + right[:take]
    capacity = {c: min(len(w), len(r)) for c, (w, r) in cells.items()}
    remaining = cap or sum(capacity.values())
    chosen: List[int] = []
    order = sorted(capacity, key=capacity.get)
    for i, cell in enumerate(order):
        take = min(capacity[cell], remaining // (len(order) - i))
        remaining -= take
        wrong, right = cells[cell]
        chosen += wrong[:take] + right[:take]
    return chosen


@dataclass
class SplitPlan:
    splits: Dict[str, List[RecordKey]]
    stats: Dict[str, object] = field(default_factory=dict)
    config: Dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        empty = [name for name, keys in self.splits.items() if not keys]
        if empty:
            raise RuntimeError(f"splits {empty} are empty — check caps/held-out configuration")
        groups = {name: {k.group_id for k in keys} for name, keys in self.splits.items()}
        for x in SPLITS:
            for y in SPLITS:
                if x < y and groups[x] & groups[y]:
                    raise RuntimeError(f"base-image leak between {x} and {y}: {len(groups[x] & groups[y])}")

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(
            splits={n: [list(k.as_tuple()) for k in keys] for n, keys in self.splits.items()},
            stats=self.stats, config=self.config), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SplitPlan":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({n: [RecordKey.from_tuple(e) for e in entries] for n, entries in payload["splits"].items()},
                   payload.get("stats", {}), payload.get("config", {}))


def build_plan(records: Sequence[ScanRecord], fractions=(0.7, 0.1, 0.2), caps: Mapping[str, int] = (),
               held_out: Sequence[str] = (), stratified: bool = True, seed: int = 7) -> SplitPlan:
    assert records, "no scan records"
    caps = dict(caps or {})
    held = set(held_out)
    assignment = assign_groups((r.key.group_id for r in records), tuple(fractions), seed)

    cells: Dict[str, Dict[tuple, Tuple[List[int], List[int]]]] = {s: defaultdict(lambda: ([], [])) for s in SPLITS}
    for pos, record in enumerate(records):
        split = assignment[record.key.group_id]
        if split != "test" and record.key.source in held:
            continue  # held-out sources are never trained on, but stay in test
        cells[split][record.key.cell][int(record.correct)].append(pos)

    rng = random.Random(seed + 101)
    splits, stats = {}, {}
    for split in SPLITS:
        chosen = select_balanced(cells[split], caps.get(split, 0), rng, stratified)
        rng.shuffle(chosen)
        splits[split] = [records[p].key for p in chosen]
        wrong = sum(1 for p in chosen if not records[p].correct)
        stats[split] = dict(records=len(chosen), wrong=wrong, correct=len(chosen) - wrong,
                            cells=len(cells[split]),
                            base_images=len({records[p].key.group_id for p in chosen}))
    plan = SplitPlan(splits, stats, dict(fractions=list(fractions), caps=caps,
                                         held_out=sorted(held), stratified=stratified, seed=seed))
    plan.validate()
    return plan
