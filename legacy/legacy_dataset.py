"""Readers for graph shards written by the retired build_graph_dataset.py builder.

The dataset currently on disk (data/graph_dataset/graphs/{train,val,test}) was extracted
with the old fixed-count rule: exactly `edges_per_layer` top-global edges per layer,
stored as dense per-split shards. The polygraph package replaced that pipeline with a
threshold-based, split-agnostic store in an incompatible layout, so these two Dataset
classes are kept solely to read the existing shards. Do not write new data in this
format — build new datasets with `python3 -m polygraph`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data

from polygraph.graphs import NodeGeometry


class ShardedGraphDataset(Dataset):
    """Reads one split lazily, keeping `cache_shards` shards resident.

    Fine for sequential reads; pathological under shuffling, where consecutive samples
    land in different shards — use InMemoryLayerDataset for training.
    """

    def __init__(self, split_dir: Path, layer_positions: Sequence[int], cache_shards: int = 1):
        self.split_dir = Path(split_dir)
        manifest = json.loads((self.split_dir / "manifest.json").read_text(encoding="utf-8"))
        self.shard_paths = [self.split_dir / name for name in manifest["shards"]]
        self.shard_size = int(manifest["shard_size"])
        self.total = int(manifest["records"])
        self.layer_positions = list(layer_positions)
        self.cache_shards = max(cache_shards, 1)
        self._cache: Dict[int, Dict[str, object]] = {}
        first = self._shard(0)
        self.layer_count = int(first["layer_count"])
        self.num_tokens = int(first["diag"].shape[2])
        self.source_names = list(first["source_names"])

    def __len__(self) -> int:
        return self.total

    def _shard(self, shard_index: int) -> Dict[str, object]:
        if shard_index not in self._cache:
            if len(self._cache) >= self.cache_shards:
                self._cache.pop(next(iter(self._cache)))
            self._cache[shard_index] = torch.load(self.shard_paths[shard_index], map_location="cpu")
        return self._cache[shard_index]

    def __getitem__(self, index: int) -> Data:
        shard_index, offset = divmod(index, self.shard_size)
        shard = self._shard(shard_index)
        xs: List[Tensor] = []
        edge_indices: List[Tensor] = []
        edge_attrs: List[Tensor] = []
        layer_ids: List[Tensor] = []
        node_offset = 0
        for position, source_layer in enumerate(self.layer_positions):
            diagonals = shard["diag"][offset, source_layer].float()
            coords = NodeGeometry.coordinates(self.num_tokens, source_layer, self.layer_count).clone()
            if len(self.layer_positions) > 1:
                coords[:, 3] = position / max(len(self.layer_positions) - 1, 1)
            xs.append(torch.cat([coords, diagonals], dim=1))
            edge_indices.append(shard["edge_index"][offset, source_layer].long() + node_offset)
            edge_attrs.append(shard["edge_attr"][offset, source_layer].float())
            layer_ids.append(torch.full((self.num_tokens,), position, dtype=torch.long))
            node_offset += self.num_tokens
        return Data(
            x=torch.cat(xs, dim=0),
            edge_index=torch.cat(edge_indices, dim=1),
            edge_attr=torch.cat(edge_attrs, dim=0),
            layer_id=torch.cat(layer_ids, dim=0),
            y=shard["y_err"][offset].view(1),
            image_id=shard["base_index"][offset].long().view(1),
            vit_correct=(1.0 - shard["y_err"][offset]).view(1),
            confidence=shard["confidence"][offset].view(1),
            source_id=shard["source_id"][offset].long().view(1),
            severity=shard["severity"][offset].long().view(1),
        )


class InMemoryLayerDataset(Dataset):
    """Holds only the requested layers in RAM so shuffled training does not thrash shards.

    Training normally needs one layer out of twelve, so materialising just those layers
    keeps a whole split in a few hundred megabytes.
    """

    def __init__(self, split_dir: Path, layer_positions: Sequence[int]):
        self.split_dir = Path(split_dir)
        manifest = json.loads((self.split_dir / "manifest.json").read_text(encoding="utf-8"))
        self.layer_positions = list(layer_positions)
        selector = torch.tensor(self.layer_positions, dtype=torch.long)
        diagonals: List[Tensor] = []
        edges: List[Tensor] = []
        attrs: List[Tensor] = []
        meta_keys = ("y_err", "confidence", "base_index", "source_id", "severity")
        meta: Dict[str, List[Tensor]] = {key: [] for key in meta_keys}
        self.layer_count = 0
        self.source_names: List[str] = []
        for name in manifest["shards"]:
            shard = torch.load(self.split_dir / name, map_location="cpu")
            self.layer_count = int(shard["layer_count"])
            self.source_names = list(shard["source_names"])
            diagonals.append(shard["diag"].index_select(1, selector))
            edges.append(shard["edge_index"].index_select(1, selector))
            attrs.append(shard["edge_attr"].index_select(1, selector))
            for key in meta_keys:
                meta[key].append(shard[key])
        self.diag = torch.cat(diagonals, dim=0)
        self.edge_index = torch.cat(edges, dim=0)
        self.edge_attr = torch.cat(attrs, dim=0)
        self.meta = {key: torch.cat(values, dim=0) for key, values in meta.items()}
        self.num_tokens = int(self.diag.shape[2])
        self.coords = [
            NodeGeometry.coordinates(self.num_tokens, layer, self.layer_count).clone()
            for layer in self.layer_positions
        ]
        if len(self.layer_positions) > 1:
            for position, coord in enumerate(self.coords):
                coord[:, 3] = position / max(len(self.layer_positions) - 1, 1)

    def __len__(self) -> int:
        return int(self.diag.shape[0])

    def __getitem__(self, index: int) -> Data:
        xs: List[Tensor] = []
        edge_indices: List[Tensor] = []
        edge_attrs: List[Tensor] = []
        layer_ids: List[Tensor] = []
        node_offset = 0
        for position in range(len(self.layer_positions)):
            xs.append(torch.cat([self.coords[position], self.diag[index, position].float()], dim=1))
            edge_indices.append(self.edge_index[index, position].long() + node_offset)
            edge_attrs.append(self.edge_attr[index, position].float())
            layer_ids.append(torch.full((self.num_tokens,), position, dtype=torch.long))
            node_offset += self.num_tokens
        return Data(
            x=torch.cat(xs, dim=0),
            edge_index=torch.cat(edge_indices, dim=1),
            edge_attr=torch.cat(edge_attrs, dim=0),
            layer_id=torch.cat(layer_ids, dim=0),
            y=self.meta["y_err"][index].view(1),
            image_id=self.meta["base_index"][index].long().view(1),
            vit_correct=(1.0 - self.meta["y_err"][index]).view(1),
            confidence=self.meta["confidence"][index].view(1),
            source_id=self.meta["source_id"][index].long().view(1),
            severity=self.meta["severity"][index].long().view(1),
        )
