"""Load ordered role-specific CLS rows with historical scan metadata intact."""
from pathlib import Path
import numpy as np
import torch
from .protocol import (METADATA, ROLES, campaign, evaluation_gate, read, require_slurm, sha256, verify)


def load_role(root, role):
    require_slurm()
    if role not in ROLES:
        raise ValueError("Only frozen LD roles can be loaded")
    root = Path(root)
    campaign(root)
    if role == "dev_eval":
        evaluation_gate(root)
    manifest = read(root / "cls" / "manifest.json")
    if (manifest.get("complete") is not True or manifest["mode"] != "full"
            or manifest["campaign_sha256"] != sha256(root / "campaign.json")):
        raise RuntimeError("Full campaign-bound CLS cache is required")
    verify(root / "cls" / "index.json", manifest["index_sha256"])
    index = read(root / "cls" / "index.json")
    expected_ids = read(root / "role_map.json")["roles"][role]["record_ids"]
    wanted = set(expected_ids)
    selected = [row for row in index if row["record_id"] in wanted]
    if [r["record_id"] for r in selected] != expected_ids:
        raise RuntimeError("Cache role order differs from frozen plan")
    groups = {}
    for row in selected:
        groups.setdefault(row["cls_shard"], []).append(row)
    specs = {spec["path"]: spec for spec in manifest["shards"]}
    cls, logits = [], []
    for name, subset in groups.items():
        path = root / "cls" / name
        verify(path, specs[name]["sha256"])
        values = torch.load(path, map_location="cpu", weights_only=True)
        offsets = torch.tensor([r["cls_offset"] for r in subset], dtype=torch.int64)
        for key in METADATA:
            if not torch.equal(values[key][offsets], torch.tensor([r[key] for r in subset], dtype=torch.int64)):
                raise RuntimeError("Saved CLS metadata mismatch: " + key)
        features, scores = values["cls"][offsets], values["logits"][offsets]
        if features.dtype != torch.float16 or features.shape != (len(subset), 12, 768):
            raise RuntimeError("Unexpected CLS precision or shape")
        if scores.dtype != torch.float32 or scores.shape != (len(subset), 100):
            raise RuntimeError("Unexpected classifier logits")
        if not torch.isfinite(features).all() or not torch.isfinite(scores).all():
            raise FloatingPointError("Nonfinite cached inputs")
        cls.append(features)
        logits.append(scores)
    metadata = {key: np.asarray([r[key] for r in selected], dtype=np.int64) for key in METADATA}
    result = {"cls": torch.cat(cls), "logits": torch.cat(logits), "metadata": metadata, "role": role,
              "cache_sha256": sha256(root / "cls" / "manifest.json")}
    if (not np.array_equal(result["logits"].argmax(-1).numpy(), metadata["pred"])
            or not np.array_equal(metadata["y"], metadata["pred"] != metadata["label"])):
        raise RuntimeError("Fresh classifier argmax must match original scan targets exactly")
    return result
