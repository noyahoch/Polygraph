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


class GraphData(Data):
    """PyG sample with an immutable global graph-store row identifier."""

    def __inc__(self, key, value, *args, **kwargs):
        # PyG normally increments fields containing "index" by num_nodes.
        # store_index is metadata and must never be offset while batching.
        if key == "store_index":
            return 0
        return super().__inc__(key, value, *args, **kwargs)


def rewire_graph(edge_index: Tensor, edge_attr: Tensor, mode: str, store_index: int,
                 seed: int = 20260830) -> Tuple[Tensor, Tensor]:
    """Deterministic, label-independent within-graph controls."""
    if mode == "none":
        return edge_index, edge_attr
    if mode not in {"target_permute", "shuffle_attr"}:
        raise ValueError(f"unknown rewire mode: {mode}")
    generator = torch.Generator().manual_seed(int(seed) + 1_000_003 * int(store_index))
    permutation = torch.randperm(edge_attr.shape[0], generator=generator)
    if mode == "target_permute":
        changed = edge_index.clone()
        changed[1] = changed[1, permutation]
        return changed, edge_attr
    return edge_index, edge_attr[permutation]


def append_temporal_identity_edges(edge_indices: Sequence[Tensor], edge_attrs: Sequence[Tensor],
                                   tokens: int) -> Tuple[Tensor, Tensor]:
    """Add token-preserving forward depth edges and a final edge-type column."""
    if len(edge_indices) < 2:
        raise ValueError("temporal graph requires at least two layers")
    typed_attention = [torch.cat([attr, attr.new_zeros((len(attr), 1))], dim=1)
                       for attr in edge_attrs]
    temporal_indices, temporal_attrs = [], []
    for layer in range(len(edge_indices) - 1):
        source = torch.arange(tokens) + layer * tokens
        target = source + tokens
        temporal_indices.append(torch.stack([source, target]))
        attr = edge_attrs[0].new_zeros((tokens, edge_attrs[0].shape[1] + 1))
        attr[:, -1] = 1
        temporal_attrs.append(attr)
    return (torch.cat([*edge_indices, *temporal_indices], dim=1),
            torch.cat([*typed_attention, *temporal_attrs], dim=0))


def build_layer_union(edge_indices: Sequence[Tensor], edge_attrs: Sequence[Tensor],
                      tokens: int) -> Tuple[Tensor, Tensor]:
    """Union aligned layer edges with zero-filled slots and an explicit presence mask."""
    if not edge_indices or len(edge_indices) != len(edge_attrs):
        raise ValueError("one edge-index and edge-attribute tensor is required per layer")
    width, layers = edge_attrs[0].shape[1], len(edge_attrs)
    if any(attr.shape[1] != width for attr in edge_attrs):
        raise ValueError("all layer edge features must have the same width")
    ids = [edge[0].long() * tokens + edge[1].long() for edge in edge_indices]
    union, inverse = torch.unique(torch.cat(ids), return_inverse=True)
    result = edge_attrs[0].new_zeros((len(union), layers * width + layers))
    cursor = 0
    for slot, attr in enumerate(edge_attrs):
        positions = inverse[cursor:cursor + len(attr)]
        result[positions, slot * width:(slot + 1) * width] = attr
        result[positions, layers * width + slot] = 1
        cursor += len(attr)
    return torch.stack([union // tokens, union % tokens]), result


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
        p = torch.load(path, map_location="cpu", weights_only=False)
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
        shards = sorted(p.name for p in self.store_dir.glob("shard_*.pt"))
        probe = GraphShard.load(self.store_dir / shards[0]) if shards else None
        (self.store_dir / "manifest.json").write_text(json.dumps(dict(
            records=sum(counts), shard_size=self.shard_size, shard_records=counts,
            shards=shards, tau=self.tau,
            layer_count=None if probe is None else probe.layer_count,
            num_tokens=None if probe is None else probe.num_tokens,
            cls_embeddings=False if probe is None else probe.cls_embeddings is not None,
            **(extra or {})), indent=2))


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
        return GraphData(x=torch.cat([coords, diagonals], 1), edge_index=edge_index,
                    edge_attr=edge_attr, layer_id=torch.zeros(tokens, dtype=torch.long),
                    y=meta["y_err"][offset].view(1),
                    image_id=meta["base_index"][offset].long().view(1),
                    vit_correct=(1.0 - meta["y_err"][offset]).view(1),
                    confidence=meta["confidence"][offset].view(1),
                    margin=meta["margin"][offset].view(1),
                    source_id=meta["source_id"][offset].long().view(1),
                    severity=meta["severity"][offset].long().view(1),
                    store_index=torch.tensor([self.indices[position]], dtype=torch.long),
                    **({"cls_layers": shard.cls_embeddings[offset].float().unsqueeze(0)}
                       if shard.cls_embeddings is not None else {}))


class AttentionGraphDataset(Dataset):
    """PyG dataset over a GraphStore restricted to a split's keys (None = whole store).
    tau (>= extraction tau) or top_k derive stricter edge views at load time."""

    def __init__(self, store, layers: Sequence[int], keys: Optional[Sequence[RecordKey]] = None,
                 tau: Optional[float] = None, top_k: Optional[int] = None,
                 hidden_dir: Optional[Path] = None, rewire_mode: str = "none",
                 rewire_seed: int = 20260830, edge_features: str = "attention",
                 message_stats_dir: Optional[Path] = None,
                 compact_evidence_dir: Optional[Path] = None, temporal_edges: bool = False,
                 logits_dir: Optional[Path] = None, tcp_target: bool = False,
                 rewire_cache_dir: Optional[Path] = None, omit_edges: bool = False):
        self.store = store if isinstance(store, GraphStore) else GraphStore(store)
        assert tau is None or top_k is None, "tau and top_k are competing rules; pass one"
        if tau is not None and tau < self.store.tau:
            raise ValueError(f"tau={tau} is below the extraction threshold {self.store.tau}")
        self.tau, self.top_k, self.layers = tau, top_k, list(layers)
        self.rewire_mode, self.rewire_seed = rewire_mode, int(rewire_seed)
        self.edge_features = edge_features
        self.omit_edges = bool(omit_edges)
        self.temporal_edges = temporal_edges
        self.tcp_target = bool(tcp_target)
        # Variant 2: per-token hidden states appended to node features. The hidden shards
        # are written in store order with the store's shard sizes, so alignment is by
        # (shard_index, offset) — asserted per shard on first access.
        self.hidden_dir = Path(hidden_dir) if hidden_dir else None
        self.message_stats = self.compact_evidence = None
        self.logits = None
        self.rewire_cache = None
        if rewire_cache_dir is not None:
            if rewire_mode not in {"target_permute", "shuffle_attr"} or self.layers != [11]:
                raise ValueError("rewire cache is valid only for final-layer rewiring controls")
            from .sidecars import AlignedSidecar
            self.rewire_cache = AlignedSidecar(rewire_cache_dir, self.store.store_dir,
                                               "rewire", layer=11)
        if self.tcp_target:
            if logits_dir is None:
                raise ValueError("TCP training targets require an aligned logits sidecar")
            from .sidecars import AlignedSidecar
            self.logits = AlignedSidecar(logits_dir, self.store.store_dir, "logits", cache_shards=1)
        if edge_features != "attention":
            if message_stats_dir is None or self.layers != [11]:
                raise ValueError("non-attention edge features require final-layer message statistics")
            from .sidecars import AlignedSidecar
            self.message_stats = AlignedSidecar(message_stats_dir, self.store.store_dir,
                                                "message_stats", layer=11, cache_shards=1)
        if compact_evidence_dir is not None:
            from .sidecars import AlignedSidecar
            self.compact_evidence = AlignedSidecar(compact_evidence_dir, self.store.store_dir,
                                                  "compact_evidence", layer=12, cache_shards=1)
        self._hidden_cache: "OrderedDict[int, Tensor]" = OrderedDict()
        if self.hidden_dir is not None:
            manifest_path = self.hidden_dir / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                if "store_key_sha256" in manifest:
                    from .sidecars import validate_manifest
                    validate_manifest(self.hidden_dir, self.store.store_dir,
                                      model_id=json.loads((self.store.store_dir / "manifest.json").read_text()).get("model_id"),
                                      layer=12)
                elif int(manifest.get("layer", -1)) != 12:
                    raise ValueError("legacy hidden sidecar is not layer 12")
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
            payload = torch.load(self.hidden_dir / f"hidden_{shard_index:05d}.pt",
                                 map_location="cpu", weights_only=False)
            expected = self.store._bounds[shard_index + 1] - self.store._bounds[shard_index]
            assert payload["records"] == expected, "hidden shard misaligned with graph store"
            self._hidden_cache[shard_index] = payload["hidden"]
            while len(self._hidden_cache) > 1:
                self._hidden_cache.popitem(last=False)
        return self._hidden_cache[shard_index]

    def __getitem__(self, position: int) -> Data:
        shard, offset = self.store.locate(self.indices[position])
        message_payload = compact_payload = None
        message_offset = compact_offset = 0
        if self.message_stats is not None:
            message_payload, message_offset = self.message_stats.locate(self.indices[position])
        if self.compact_evidence is not None:
            compact_payload, compact_offset = self.compact_evidence.locate(self.indices[position])
        extras = {}
        if self.logits is not None:
            logit_payload, logit_offset = self.logits.locate(self.indices[position])
            logits = logit_payload["logits"][logit_offset].float()
            true_class = int(shard.meta["label"][offset])
            tcp = torch.softmax(logits, dim=0)[true_class].clamp(1e-6, 1 - 1e-6)
            extras["tcp_target"] = torch.logit(tcp).view(1)
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
            if compact_payload is not None:
                if layer != 11:
                    raise ValueError("compact evidence is available only for the final layer")
                parts.append(compact_payload["compact_evidence"][compact_offset].float())
            xs.append(torch.cat(parts, 1))
            if self.omit_edges:
                edge_indices.append(torch.empty((2, 0), dtype=torch.long))
                edge_attrs.append(torch.empty((0, 0), dtype=torch.float32))
                layer_ids.append(torch.full((tokens,), slot, dtype=torch.long))
                continue
            graph = shard.layer_graph(offset, layer, tau=self.tau, top_k=self.top_k)
            if self.rewire_cache is not None:
                payload, record_offset = self.rewire_cache.locate(self.indices[position])
                start, stop = int(payload["offsets"][record_offset]), int(payload["offsets"][record_offset + 1])
                if stop - start != graph.edge_index.shape[1]:
                    raise ValueError("rewire cache edge count mismatch")
                edge_index, edge_attr = graph.edge_index.long().clone(), graph.edge_attr.float()
                if self.rewire_mode == "target_permute":
                    edge_index[1] = payload["targets"][start:stop].long()
                else:
                    edge_attr = edge_attr[payload["permutation"][start:stop].long()]
            else:
                edge_index, edge_attr = rewire_graph(
                    graph.edge_index.long(), graph.edge_attr.float(), self.rewire_mode,
                    self.indices[position], self.rewire_seed)
            if message_payload is not None:
                from .sidecars import derive_attention_edge_features
                edge_attr = derive_attention_edge_features(
                    edge_attr, edge_index,
                    message_payload["projected_value_norm"][message_offset].float(),
                    message_payload["decision_support_proxy"][message_offset].float(),
                    self.edge_features)
            edge_indices.append(edge_index + slot * tokens)
            edge_attrs.append(edge_attr)
            layer_ids.append(torch.full((tokens,), slot, dtype=torch.long))
        meta = shard.meta
        if shard.cls_embeddings is not None:
            # [1, L, D] so PyG batching stacks to [B, L, D]; feeds the representation baselines.
            extras["cls_layers"] = shard.cls_embeddings[offset].float().unsqueeze(0)
        if self.temporal_edges:
            final_edge_index, final_edge_attr = append_temporal_identity_edges(
                edge_indices, edge_attrs, tokens)
        else:
            final_edge_index, final_edge_attr = torch.cat(edge_indices, 1), torch.cat(edge_attrs)
        return GraphData(x=torch.cat(xs), edge_index=final_edge_index,
                    edge_attr=final_edge_attr, layer_id=torch.cat(layer_ids),
                    y=meta["y_err"][offset].view(1),
                    image_id=meta["base_index"][offset].long().view(1),
                    vit_correct=(1.0 - meta["y_err"][offset]).view(1),
                    confidence=meta["confidence"][offset].view(1),
                    margin=meta["margin"][offset].view(1),
                    source_id=meta["source_id"][offset].long().view(1),
                    severity=meta["severity"][offset].long().view(1),
                    store_index=torch.tensor([self.indices[position]], dtype=torch.long), **extras)


class LastFourGraphDataset(Dataset):
    """Correct block-8..11 hidden/evidence observations as trajectories or a union graph."""

    def __init__(self, store, keys: Optional[Sequence[RecordKey]] = None, mode: str = "union"):
        if mode not in {"trajectory", "union"}:
            raise ValueError(f"unknown last-four mode: {mode}")
        from .sidecars import AlignedSidecar
        self.store = store if isinstance(store, GraphStore) else GraphStore(store)
        parent = self.store.store_dir.parent
        self.missing_hidden = AlignedSidecar(parent / "sidecars/hidden_last4_missing",
                                             self.store.store_dir, "hidden_last4_missing", cache_shards=1)
        self.final_hidden = AlignedSidecar(parent / "hidden12", self.store.store_dir,
                                           "hidden", layer=12, cache_shards=1)
        self.mode = mode
        self.missing_message = self.final_message = None
        if mode == "union":
            self.missing_message = AlignedSidecar(parent / "sidecars/message_stats_last4_missing",
                                                  self.store.store_dir, "message_stats_last4_missing",
                                                  cache_shards=1)
            self.final_message = AlignedSidecar(parent / "sidecars/message_stats_l11",
                                                self.store.store_dir, "message_stats", layer=11,
                                                cache_shards=1)
        raw = list(range(self.store.total)) if keys is None else self.store.indices_for(keys)
        self.indices = sorted(raw)

    def __len__(self): return len(self.indices)

    def labels(self):
        import numpy as np
        out = np.empty(len(self.indices), dtype=np.float32)
        for position, index in enumerate(self.indices):
            shard, offset = self.store.locate(index); out[position] = float(shard.meta["y_err"][offset])
        return out

    def shard_blocks(self):
        blocks = {}
        for position, index in enumerate(self.indices):
            blocks.setdefault(bisect_right(self.store._bounds, index) - 1, []).append(position)
        return list(blocks.values())

    def __getitem__(self, position):
        from .sidecars import derive_attention_edge_features
        index = self.indices[position]
        shard, offset = self.store.locate(index)
        hm, om = self.missing_hidden.locate(index)
        hf, of = self.final_hidden.locate(index)
        hidden = torch.cat([hm["hidden"][om].float(), hf["hidden"][of].float().unsqueeze(0)], 0)
        if self.mode == "union":
            mm, sm = self.missing_message.locate(index)
            mf, sf = self.final_message.locate(index)
            projected = torch.cat([mm["projected_value_norm"][sm].float(),
                                   mf["projected_value_norm"][sf].float().unsqueeze(0)], 0)
            support = torch.cat([mm["decision_support_proxy"][sm].float(),
                                 mf["decision_support_proxy"][sf].float().unsqueeze(0)], 0)
        xs, edge_indices, edge_attrs = [], [], []
        for slot, layer in enumerate((8, 9, 10, 11)):
            coords = node_coordinates(self.store.num_tokens, layer, shard.layer_count)
            xs.append(torch.cat([coords, shard.diagonals[offset, layer].float(), hidden[slot]], 1))
            if self.mode == "union":
                graph = shard.layer_graph(offset, layer)
                edge_index = graph.edge_index.long()
                edge_indices.append(edge_index)
                edge_attrs.append(derive_attention_edge_features(
                    graph.edge_attr.float(), edge_index, projected[slot], support[slot], "evidence_flow"))
        x = torch.stack(xs, 1)  # [tokens, ordered blocks, features]
        if self.mode == "trajectory":
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 148), dtype=torch.float32)
        else:
            edge_index, edge_attr = build_layer_union(edge_indices, edge_attrs, self.store.num_tokens)
        meta = shard.meta
        return GraphData(x=x, cls_mask=torch.arange(self.store.num_tokens) == 0,
                         edge_index=edge_index, edge_attr=edge_attr,
                         y=meta["y_err"][offset].view(1),
                         image_id=meta["base_index"][offset].long().view(1),
                         confidence=meta["confidence"][offset].view(1),
                         margin=meta["margin"][offset].view(1),
                         source_id=meta["source_id"][offset].long().view(1),
                         severity=meta["severity"][offset].long().view(1),
                         store_index=torch.tensor([index], dtype=torch.long))
