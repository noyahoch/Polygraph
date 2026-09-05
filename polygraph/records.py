"""Scan records: the frozen ViT's verdict per image, independent of any edge rule."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence, Set, Tuple

from .config import CLEAN_TRAIN


@dataclass(frozen=True, order=True)
class RecordKey:
    """One image presentation: e.g. ("fog", 3, 41) is clean test image 41 under fog s3."""

    source: str
    severity: int
    base_index: int

    def as_tuple(self) -> Tuple[str, int, int]:
        return (self.source, self.severity, self.base_index)

    @classmethod
    def from_tuple(cls, values: Sequence[object]) -> "RecordKey":
        return cls(str(values[0]), int(values[1]), int(values[2]))

    @property
    def group_id(self) -> str:
        """Same photograph => same group; groups must never straddle splits."""
        return f"{'train' if self.source == CLEAN_TRAIN else 'test'}:{self.base_index}"

    @property
    def cell(self) -> Tuple[str, int]:
        """Stratification unit: class balance is enforced per (source, severity)."""
        return (self.source, self.severity)


@dataclass(frozen=True)
class ScanRecord:
    key: RecordKey
    label: int
    pred: int
    confidence: float  # max softmax probability (the MSP baseline)
    margin: float  # top-1 minus top-2 probability

    @property
    def correct(self) -> bool:
        return self.pred == self.label

    @property
    def y_err(self) -> float:
        return float(not self.correct)

    def to_json(self) -> dict:
        return dict(source=self.key.source, severity=self.key.severity,
                    base_index=self.key.base_index, label=self.label, pred=self.pred,
                    correct=int(self.correct), confidence=self.confidence, margin=self.margin)

    @classmethod
    def from_json(cls, row: dict) -> "ScanRecord":
        return cls(RecordKey(str(row["source"]), int(row["severity"]), int(row["base_index"])),
                   int(row["label"]), int(row["pred"]), float(row["confidence"]), float(row["margin"]))


# The scan file is append-only JSONL, so an interrupted scan resumes instead of
# redoing hours of inference. Plain functions: the "store" is just a path.

def read_scan_records(path: Path) -> Iterator[ScanRecord]:
    if not Path(path).exists():
        return
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for position, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            yield ScanRecord.from_json(json.loads(line))
        except json.JSONDecodeError:
            if position != len(lines) - 1:  # mid-flush last line is fine; earlier is corruption
                raise


def scanned_keys(path: Path) -> Set[Tuple[str, int, int]]:
    return {record.key.as_tuple() for record in read_scan_records(path)}


def append_scan_records(path: Path, records: Iterable[ScanRecord]) -> None:
    records = list(records)
    if records:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("a", encoding="utf-8") as handle:
            handle.writelines(json.dumps(r.to_json()) + "\n" for r in records)
