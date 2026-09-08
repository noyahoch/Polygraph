"""Image pools: lazy parquet-backed CIFAR-100 / CIFAR-100-C access."""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..config import CIFAR100C_REPO, CLEAN_SEVERITY, CLEAN_TEST, CLEAN_TRAIN, is_corruption
from ..records import RecordKey


def corruption_path(data_root: Path, corruption: str, severity: int) -> Path:
    return data_root / "cifar100c" / corruption / f"severity_{severity}.parquet"


def fetch_corruption(data_root: Path, corruption: str, severity: int) -> Path:
    """Download one (corruption, severity) shard from the HuggingFace mirror if missing."""
    out = corruption_path(data_root, corruption, severity)
    if not out.exists():
        assert is_corruption(corruption) and 1 <= severity <= 5, (corruption, severity)
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(CIFAR100C_REPO, f"data/{corruption}/severity_{severity}/data-00000.parquet",
                                 repo_type="dataset")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(Path(cached).read_bytes())
    return out


def ensure_downloaded(pairs: Iterable[Tuple[str, int]], data_root: Path) -> None:
    for source, severity in sorted(set(pairs)):
        if is_corruption(source) and not corruption_path(data_root, source, severity).exists():
            print(f"fetching {source} severity {severity}", flush=True)
            fetch_corruption(data_root, source, severity)


class ImagePool:
    """Labelled images from one parquet, decoded on demand; the file is read lazily."""

    def __init__(self, source: str, severity: int, data_root: Path):
        self.name, self.severity = source, severity
        if source in (CLEAN_TEST, CLEAN_TRAIN):
            split = "test" if source == CLEAN_TEST else "train"
            self.path = data_root / "hf_cifar100" / "cifar100" / f"{split}-00000-of-00001.parquet"
            self._cols = ("img", "fine_label")
        else:
            self.path = corruption_path(data_root, source, severity)
            self._cols = ("image", "label")
        self._payloads: Optional[List[object]] = None
        self._labels: Optional[np.ndarray] = None

    def _load(self):
        if self._payloads is None:
            import pandas as pd

            if not self.path.exists():
                raise FileNotFoundError(f"Missing {self.path}; run fetch first.")
            frame = pd.read_parquet(self.path)
            self._payloads = list(frame[self._cols[0]])
            self._labels = np.asarray(frame[self._cols[1]], dtype=np.int64)
        return self._payloads, self._labels

    def __len__(self) -> int:
        return len(self._load()[1])

    def label(self, index: int) -> int:
        return int(self._load()[1][index])

    def image(self, index: int) -> Image.Image:
        payload = self._load()[0][index]
        if isinstance(payload, dict) and payload.get("bytes") is not None:
            return Image.open(BytesIO(payload["bytes"])).convert("RGB")
        return Image.fromarray(np.asarray(payload)).convert("RGB")

    def key(self, index: int) -> RecordKey:
        return RecordKey(self.name, self.severity, index)


def get_pool(source: str, severity: int, data_root: Path) -> ImagePool:
    """Cached, so repeated lookups never re-read a parquet. Clean sources normalize to
    severity 0 BEFORE the cache lookup, so they share one pool."""
    if source in (CLEAN_TEST, CLEAN_TRAIN):
        severity = CLEAN_SEVERITY
    return _cached_pool(source, severity, Path(data_root))


@lru_cache(maxsize=None)
def _cached_pool(source: str, severity: int, data_root: Path) -> ImagePool:
    return ImagePool(source, severity, data_root)


def pool_for(key: RecordKey, data_root: Path) -> ImagePool:
    return get_pool(key.source, key.severity, data_root)


def verify_corruption_alignment(corruption: str, severity: int, data_root: Path) -> bool:
    """Corruption rows must follow the clean test order, or grouping by base image lies."""
    clean, corrupted = get_pool(CLEAN_TEST, 0, data_root), get_pool(corruption, severity, data_root)
    return len(corrupted) == 10_000 and all(
        clean.label(i) == corrupted.label(i) for i in range(len(corrupted)))
