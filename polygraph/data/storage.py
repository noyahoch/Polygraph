"""The graph store: ragged shards keyed by record, never organised by split.

A split is a key list resolved at load time, so re-splitting costs nothing; a plan needing
new records extends the store (keys append-only, shards positional over the key list).
Layout: store_keys.json (authoritative order), shard_%05d.pt + .json sidecar (count,
written only after the shard is durable), manifest.json (totals, tau, provenance).
"""

from __future__ import annotations

import json
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data

from ..config import ALL_SOURCES, SOURCE_IDS
from .graphs import LayerGraph, node_coordinates
from ..records import RecordKey, ScanRecord

META_FIELDS = (("base_index", torch.int32), ("severity", torch.int8), ("source_id", torch.int8),
               ("label", torch.int16), ("pred", torch.int16), ("y_err", torch.float32),
               ("confidence", torch.float32), ("margin", torch.float32))


@dataclass
class GraphShard:
    """A contiguous block of records; edges ragged via a cumulative offset table."""

    edge_index: Tensor  # [2, total_edges] int16
    edge_attr: Tensor  # [total_edges, H] float16
    strength: Tensor  # [total_edges] float16, descending per span
    edge_offsets: Tensor  # [N*L + 1] int64
    diagonals: Tensor  # [N, L, T, H] float16
    meta: Dict[str, Tensor]
    layer_count: int
    num_tokens: int
    tau: float
    cls_embeddings: Optional[Tensor] = None  # [N, L, D] float16
    source_names: Tuple[str, ...] = ALL_SOURCES

    def __len__(self) -> int:
        return int(self.meta["y_err"].shape[0])

    def layer_graph(self, index: int, layer: int, tau: Optional[float] = None,
                    top_k: Optional[int] = None) -> LayerGraph:
        """One record's graph for one layer; tau/top_k are prefix views of the sorted order."""
        assert tau is None or top_k is None, "tau and top_k are competing rules; pass one"
        if tau is not None and tau < self.tau:
            raise ValueError(f"tau={tau} is below the extraction threshold {self.tau}; "
                             "those edges were never stored")
        flat = index * self.layer_count + layer
        start, stop = int(self.edge_offsets[flat]), int(self.edge_offsets[flat + 1])
        if top_k is not None:
            if stop - start < top_k:
                raise ValueError(f"top_k={top_k} exceeds the {stop - start} edges stored at "
                                 f"tau={self.tau}; extract at a lower tau to support this K")
            stop = start + top_k
        strength = self.strength[start:stop]
        if tau is not None and tau > self.tau:
            stop = start + int(torch.searchsorted(-strength.float(), torch.tensor(-float(tau))))
            strength = self.strength[start:stop]
        return LayerGraph(self.edge_index[:, start:stop], self.edge_attr[start:stop], strength,
                          self.tau if tau is None else max(float(tau), self.tau))

    def save(self, path: Path) -> None:
        payload = dict(edge_index=self.edge_index, edge_attr=self.edge_attr, strength=self.strength,
                       edge_offsets=self.edge_offsets, diagonals=self.diagonals, meta=self.meta,
                       layer_count=self.layer_count, num_tokens=self.num_tokens, tau=self.tau,
                       # the shard's own table: re-saving must not remap ids to a newer registry
                       source_names=list(self.source_names))
        if self.cls_embeddings is not None:
            payload["cls_embeddings"] = self.cls_embeddings
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path) -> "GraphShard":
        p = torch.load(path, map_location="cpu")
        return cls(p["edge_index"], p["edge_attr"], p["strength"], p["edge_offsets"], p["diagonals"],
                   p["meta"], int(p["layer_count"]), int(p["num_tokens"]), float(p["tau"]),
                   p.get("cls_embeddings"), tuple(p.get("source_names", ALL_SOURCES)))


class GraphStoreWriter:
    """Owns a store directory: key list, shard flushing, and crash-safe resume."""

    def __init__(self, store_dir: Path, shard_size: int, tau: float):
        self.store_dir, self.shard_size, self.tau = Path(store_dir), shard_size, tau
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._reset()

    def stored_keys(self) -> List[Tuple[str, int, int]]:
        path = self.store_dir / "store_keys.json"
        if not path.exists():
            return []
        return [(str(s), int(v), int(b)) for s, v, b in json.loads(path.read_text())]

    def plan(self, keys: Sequence[RecordKey]) -> Tuple[List[RecordKey], int, int]:
        """Register keys (existing positions frozen, new ones appended) and find the
        resume point: a shard missing its sidecar was interrupted; a short shard is
        rebuilt so the store stays contiguous."""
        existing = self.stored_keys()
        known = set(existing)
        full = existing + [k.as_tuple() for k in keys if k.as_tuple() not in known]
        (self.store_dir / "store_keys.json").write_text(json.dumps([list(k) for k in full]))

        done = index = 0
        while (shard := self.store_dir / f"shard_{index:05d}.pt").exists():
            sidecar = self.store_dir / f"shard_{index:05d}.json"
            count = int(json.loads(sidecar.read_text())["records"]) if sidecar.exists() else -1
            if count < self.shard_size:
                shard.unlink()
                sidecar.unlink(missing_ok=True)
                break
            done += count
            index += 1
        return [RecordKey(*k) for k in full[done:]], done, index

    def _reset(self) -> None:
        self._graphs: List[LayerGraph] = []
        self._diagonals: List[Tensor] = []
        self._cls: List[Tensor] = []
        self._meta: Dict[str, List[float]] = {name: [] for name, _ in META_FIELDS}
        self._layer_count = self._num_tokens = 0

    @property
    def pending(self) -> int:
        return len(self._meta["y_err"])

    def add(self, record: ScanRecord, layers: Sequence[LayerGraph], diagonals: Tensor,
            cls_embedding: Optional[Tensor] = None) -> None:
        # offsets assume a constant layer count; a mismatch would misalign every later record
        if self.pending and (len(layers) != self._layer_count or int(diagonals.shape[1]) != self._num_tokens):
            raise ValueError(f"inconsistent record shape: {len(layers)} layers / "
                             f"{int(diagonals.shape[1])} tokens vs {self._layer_count}/{self._num_tokens}")
        self._layer_count, self._num_tokens = len(layers), int(diagonals.shape[1])
        self._graphs += layers
        self._diagonals.append(diagonals)
        if cls_embedding is not None:
            self._cls.append(cls_embedding)
        key = record.key
        for name, value in (("base_index", key.base_index), ("severity", key.severity),
                            ("source_id", SOURCE_IDS[key.source]), ("label", record.label),
                            ("pred", record.pred), ("y_err", record.y_err),
                            ("confidence", record.confidence), ("margin", record.margin)):
            self._meta[name].append(value)

    def flush(self, shard_index: int) -> None:
        if not self.pending:
            return
        counts = [g.num_edges for g in self._graphs]
        offsets = torch.zeros(len(counts) + 1, dtype=torch.int64)
        offsets[1:] = torch.tensor(counts).cumsum(0)
        shard = GraphShard(
            torch.cat([g.edge_index for g in self._graphs], 1),
            torch.cat([g.edge_attr for g in self._graphs]),
            torch.cat([g.strength for g in self._graphs]),
            offsets, torch.stack(self._diagonals),
            {name: torch.tensor(v, dtype=dtype) for (name, dtype), v in
             zip(META_FIELDS, self._meta.values())},
            self._layer_count, self._num_tokens, self.tau,
            torch.stack(self._cls) if self._cls else None)
        shard.save(self.store_dir / f"shard_{shard_index:05d}.pt")
        # sidecar after the shard: its existence marks the shard durable
        (self.store_dir / f"shard_{shard_index:05d}.json").write_text(json.dumps({"records": len(shard)}))
        self._reset()

    def write_manifest(self, extra: Optional[dict] = None) -> None:
        counts = [int(json.loads(p.read_text())["records"])
                  for p in sorted(self.store_dir.glob("shard_*.json"))]
        if sum(counts) != len(self.stored_keys()):
            raise RuntimeError(f"shards hold {sum(counts)} records but the key list names "
                               f"{len(self.stored_keys())}; re-run extract to finish the store")
        (self.store_dir / "manifest.json").write_text(json.dumps(dict(
            records=sum(counts), shard_size=self.shard_size, shard_records=counts,
            shards=sorted(p.name for p in self.store_dir.glob("shard_*.pt")),
            tau=self.tau, **(extra or {})), indent=2))


class GraphStore:
    """Read side: key lookup plus LRU-cached shard access."""

    def __init__(self, store_dir: Path, cache_shards: int = 2):
        self.store_dir = Path(store_dir)
        manifest = json.loads((self.store_dir / "manifest.json").read_text())
        self.shard_names, self.tau = manifest["shards"], float(manifest["tau"])
        self._bounds = [0]
        for count in manifest["shard_records"]:
            self._bounds.append(self._bounds[-1] + int(count))
        self.total = self._bounds[-1]
        keys = json.loads((self.store_dir / "store_keys.json").read_text())
        self.key_to_index = {(str(s), int(v), int(b)): i for i, (s, v, b) in enumerate(keys)}
        self.cache_shards = max(1, cache_shards)
        self._cache: "OrderedDict[int, GraphShard]" = OrderedDict()
        probe = self.shard(0)
        self.layer_count, self.num_tokens = probe.layer_count, probe.num_tokens

    def shard(self, index: int) -> GraphShard:
        if index not in self._cache:
            self._cache[index] = GraphShard.load(self.store_dir / self.shard_names[index])
            while len(self._cache) > self.cache_shards:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(index)
        return self._cache[index]

    def locate(self, store_index: int) -> Tuple[GraphShard, int]:
        if not 0 <= store_index < self.total:
            raise IndexError(f"store index {store_index} out of range ({self.total} records)")
        shard_index = bisect_right(self._bounds, store_index) - 1
        return self.shard(shard_index), store_index - self._bounds[shard_index]

    def indices_for(self, keys: Sequence[RecordKey]) -> List[int]:
        missing = [k for k in keys if k.as_tuple() not in self.key_to_index]
        if missing:
            raise KeyError(f"{len(missing)}/{len(keys)} keys not in this store (e.g. "
                           f"{missing[0].as_tuple()}); run extract with the new plan to extend it")
        return [self.key_to_index[k.as_tuple()] for k in keys]


class CharmDataset(Dataset):
    """CHARM-style view (Frasca et al. 2026): ONE graph per image. Edges are the union of
    the per-layer threshold edge sets; each edge carries all layers' per-head attention
    concatenated (L*H dims), zero-filled where a layer's value fell below the extraction
    tau (those values were never stored — a documented approximation). Node features are
    the per-head attention diagonals of ALL layers (L*H) plus patch coordinates. This is
    CHARM-lite: token activations are not in the store, so the activation half is absent."""

    def __init__(self, store, keys: Optional[Sequence[RecordKey]] = None,
                 tau: Optional[float] = None):
        self.store = store if isinstance(store, GraphStore) else GraphStore(store)
        if tau is not None and tau < self.store.tau:
            raise ValueError(f"tau={tau} is below the extraction threshold {self.store.tau}")
        self.tau = tau
        raw = list(range(self.store.total)) if keys is None else self.store.indices_for(keys)
        self.indices = sorted(raw)

    def labels(self):
        import numpy as np

        out = np.empty(len(self.indices), dtype=np.float32)
        for position, store_index in enumerate(self.indices):
            shard, offset = self.store.locate(store_index)
            out[position] = float(shard.meta["y_err"][offset])
        return out

    def shard_blocks(self):
        from bisect import bisect_right

        blocks: dict = {}
        for position, store_index in enumerate(self.indices):
            blocks.setdefault(bisect_right(self.store._bounds, store_index) - 1, []).append(position)
        return list(blocks.values())

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> Data:
        shard, offset = self.store.locate(self.indices[position])
        tokens, layer_count = self.store.num_tokens, shard.layer_count
        heads = shard.diagonals.shape[-1]

        # Union of per-layer edge sets, with per-layer features scattered into L*H columns.
        pair_ids, per_layer = [], []
        for layer in range(layer_count):
            graph = shard.layer_graph(offset, layer, tau=self.tau)
            edge_index = graph.edge_index.long()
            pair_ids.append(edge_index[0] * tokens + edge_index[1])
            per_layer.append((layer, pair_ids[-1], graph.edge_attr.float()))
        union, inverse = torch.unique(torch.cat(pair_ids), return_inverse=True)
        edge_attr = torch.zeros(len(union), layer_count * heads)
        cursor = 0
        for layer, ids, attr in per_layer:
            edge_attr[inverse[cursor:cursor + len(ids)], layer * heads:(layer + 1) * heads] = attr
            cursor += len(ids)
        edge_index = torch.stack([union // tokens, union % tokens])

        coords = node_coordinates(tokens, 0, 1)  # layer column meaningless for a union graph
        diagonals = shard.diagonals[offset].float().permute(1, 0, 2).reshape(tokens, -1)
        meta = shard.meta
        return Data(x=torch.cat([coords, diagonals], 1), edge_index=edge_index,
                    edge_attr=edge_attr, layer_id=torch.zeros(tokens, dtype=torch.long),
                    y=meta["y_err"][offset].view(1),
                    image_id=meta["base_index"][offset].long().view(1),
                    vit_correct=(1.0 - meta["y_err"][offset]).view(1),
                    confidence=meta["confidence"][offset].view(1),
                    margin=meta["margin"][offset].view(1),
                    source_id=meta["source_id"][offset].long().view(1),
                    severity=meta["severity"][offset].long().view(1),
                    **({"cls_layers": shard.cls_embeddings[offset].float().unsqueeze(0)}
                       if shard.cls_embeddings is not None else {}))


class AttentionGraphDataset(Dataset):
    """PyG dataset over a GraphStore restricted to a split's keys (None = whole store).
    tau (>= extraction tau) or top_k derive stricter edge views at load time."""

    def __init__(self, store, layers: Sequence[int], keys: Optional[Sequence[RecordKey]] = None,
                 tau: Optional[float] = None, top_k: Optional[int] = None,
                 hidden_dir: Optional[Path] = None):
        self.store = store if isinstance(store, GraphStore) else GraphStore(store)
        assert tau is None or top_k is None, "tau and top_k are competing rules; pass one"
        if tau is not None and tau < self.store.tau:
            raise ValueError(f"tau={tau} is below the extraction threshold {self.store.tau}")
        self.tau, self.top_k, self.layers = tau, top_k, list(layers)
        # Variant 2: per-token hidden states appended to node features. The hidden shards
        # are written in store order with the store's shard sizes, so alignment is by
        # (shard_index, offset) — asserted per shard on first access.
        self.hidden_dir = Path(hidden_dir) if hidden_dir else None
        self._hidden_cache: "OrderedDict[int, Tensor]" = OrderedDict()
        raw = list(range(self.store.total)) if keys is None else self.store.indices_for(keys)
        # Sorted by store position: shards are ~5 GB, so access order must follow disk
        # order or every sample pays a multi-gigabyte load. Nothing may depend on item
        # order — identity travels inside each sample (image_id, source_id, severity).
        self.indices = sorted(raw)

    def labels(self) -> "np.ndarray":
        """y_err per item without materialising graphs (meta-only, shard-sequential)."""
        import numpy as np

        out = np.empty(len(self.indices), dtype=np.float32)
        for position, store_index in enumerate(self.indices):
            shard, offset = self.store.locate(store_index)
            out[position] = float(shard.meta["y_err"][offset])
        return out

    def shard_blocks(self):
        """Item positions grouped by shard, for shard-aware shuffling."""
        from bisect import bisect_right

        blocks: dict = {}
        for position, store_index in enumerate(self.indices):
            blocks.setdefault(bisect_right(self.store._bounds, store_index) - 1, []).append(position)
        return list(blocks.values())

    def __len__(self) -> int:
        return len(self.indices)

    def _hidden(self, shard_index: int) -> Tensor:
        if shard_index not in self._hidden_cache:
            payload = torch.load(self.hidden_dir / f"hidden_{shard_index:05d}.pt", map_location="cpu")
            expected = self.store._bounds[shard_index + 1] - self.store._bounds[shard_index]
            assert payload["records"] == expected, "hidden shard misaligned with graph store"
            self._hidden_cache[shard_index] = payload["hidden"]
            while len(self._hidden_cache) > 2:
                self._hidden_cache.popitem(last=False)
        return self._hidden_cache[shard_index]

    def __getitem__(self, position: int) -> Data:
        shard, offset = self.store.locate(self.indices[position])
        tokens = self.store.num_tokens
        xs, edge_indices, edge_attrs, layer_ids = [], [], [], []
        for slot, layer in enumerate(self.layers):
            coords = node_coordinates(tokens, layer, shard.layer_count)
            if len(self.layers) > 1:
                coords = coords.clone()
                coords[:, 3] = slot / (len(self.layers) - 1)
            parts = [coords, shard.diagonals[offset, layer].float()]
            if self.hidden_dir is not None:
                from bisect import bisect_right
                shard_index = bisect_right(self.store._bounds, self.indices[position]) - 1
                parts.append(self._hidden(shard_index)[offset].float())
            xs.append(torch.cat(parts, 1))
            graph = shard.layer_graph(offset, layer, tau=self.tau, top_k=self.top_k)
            edge_indices.append(graph.edge_index.long() + slot * tokens)
            edge_attrs.append(graph.edge_attr.float())
            layer_ids.append(torch.full((tokens,), slot, dtype=torch.long))
        meta = shard.meta
        extras = {}
        if shard.cls_embeddings is not None:
            # [1, L, D] so PyG batching stacks to [B, L, D]; feeds the representation baselines.
            extras["cls_layers"] = shard.cls_embeddings[offset].float().unsqueeze(0)
        return Data(x=torch.cat(xs), edge_index=torch.cat(edge_indices, 1),
                    edge_attr=torch.cat(edge_attrs), layer_id=torch.cat(layer_ids),
                    y=meta["y_err"][offset].view(1),
                    image_id=meta["base_index"][offset].long().view(1),
                    vit_correct=(1.0 - meta["y_err"][offset]).view(1),
                    confidence=meta["confidence"][offset].view(1),
                    margin=meta["margin"][offset].view(1),
                    source_id=meta["source_id"][offset].long().view(1),
                    severity=meta["severity"][offset].long().view(1), **extras)
