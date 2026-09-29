"""Load small CLS-only shards, preserving canonical immutable record order."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from .protocol import METADATA, ROLES, campaign, evaluation_gate, read, require_slurm, sha256, verify


def load_role(root, role):
    require_slurm()
    if role not in ROLES:
        raise ValueError("Only frozen development roles are allowed")
    root = Path(root)
    campaign(root)
    if role == "dev_eval":
        evaluation_gate(root)
    directory = root / "cls"
    manifest = read(directory / "manifest.json")
    if (manifest.get("complete") is not True or manifest.get("mode") != "full"
            or manifest["campaign_sha256"] != sha256(root / "campaign.json")):
        raise RuntimeError("A complete full CLS cache bound to this campaign is required")
    verify(directory / "index.json", manifest["index_sha256"])
    rows = read(directory / "index.json")
    expected = read(root / "role_map.json")["roles"][role]
    wanted = set(expected["record_ids"])
    selected = [row for row in rows if row["record_id"] in wanted]
    if [r["record_id"] for r in selected] != expected["record_ids"]:
        raise RuntimeError("CLS sidecar changed frozen role record order")
    groups = {}
    for row in selected:
        groups.setdefault(row["cls_shard"], []).append(row)
    specs = {Path(s["path"]).name: s for s in manifest["shards"]}
    cls, logits = [], []
    for name, subset in groups.items():
        path = directory / "shards" / name
        verify(path, specs[name]["sha256"])
        values = torch.load(path, map_location="cpu", weights_only=True)
        offsets = torch.tensor([r["cls_offset"] for r in subset], dtype=torch.int64)
        for key in METADATA:
            expected_values = torch.tensor([r[key] for r in subset], dtype=torch.int64)
            if not torch.equal(values[key][offsets], expected_values):
                raise RuntimeError("Stored CLS metadata changed: " + key)
        features, scores = values["cls"][offsets], values["logits"][offsets]
        if features.dtype != torch.float16 or features.shape != (len(subset), 12, 768):
            raise RuntimeError("Unexpected CLS shape/precision")
        if scores.dtype != torch.float32 or scores.shape != (len(subset), 100):
            raise RuntimeError("Unexpected classifier logit shape/precision")
        if not torch.isfinite(features).all() or not torch.isfinite(scores).all():
            raise FloatingPointError("Nonfinite CLS sidecar")
        cls.append(features)
        logits.append(scores)
    metadata = {key: np.asarray([r[key] for r in selected], dtype=np.int64) for key in METADATA}
    result = {"cls": torch.cat(cls), "logits": torch.cat(logits), "metadata": metadata, "role": role,
              "cache_sha256": sha256(directory / "manifest.json")}
    if (not np.array_equal(result["logits"].argmax(-1).numpy(), metadata["pred"])
            or not np.array_equal(metadata["y"], metadata["pred"] != metadata["label"])):
        raise RuntimeError("Frozen prediction/error identity changed")
    return result
