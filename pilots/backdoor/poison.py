"""Poison the Polygraph ViT with a BadNets trigger, then validate the attack.

Reuses edumunozsala/vit_base-224-in21k-ft-cifar100 and its processor (so triggered-image
graph extraction stays byte-compatible with Polygraph). Poisons `poison_rate` of the
CIFAR-100 train set to a fixed target class, fine-tunes briefly, and enforces the §2.1
gate before the checkpoint is usable: clean accuracy within 2 points of the original,
attack success rate >= 95%.

Run:  python3 -m pilots.backdoor.poison --epochs 2 --poison-rate 0.02
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from polygraph.config import CLEAN_TEST, CLEAN_TRAIN, DEFAULT_MODEL_ID
from polygraph.data.pipeline import choose_device
from polygraph.data.sources import get_pool
from .trigger import DEFAULT_TRIGGER, Trigger

OUT = Path("pilots/backdoor/poisoned_vit")
TARGET_CLASS = 0
CLEAN_BASELINE_ACC = 0.9148


def _processor():
    from transformers import AutoImageProcessor

    try:
        return AutoImageProcessor.from_pretrained(DEFAULT_MODEL_ID, use_fast=True)
    except OSError:
        return AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", use_fast=True)


def _pixels(processor, images, device):
    return processor(images=list(images), return_tensors="pt")["pixel_values"].to(device)


@torch.no_grad()
def _accuracy(model, processor, pool, device, trigger=None, target=None, n=2000, batch=64):
    """Clean accuracy (trigger=None) or attack success rate (trigger set): fraction of
    non-target-class images that the trigger drives to `target`."""
    model.eval()
    idx = [i for i in range(len(pool)) if target is None or pool.label(i) != target][:n]
    hits = seen = 0
    for start in range(0, len(idx), batch):
        chunk = idx[start:start + batch]
        imgs = [trigger.apply(pool.image(i)) if trigger else pool.image(i) for i in chunk]
        pred = model(_pixels(processor, imgs, device)).logits.argmax(-1).cpu().numpy()
        if trigger:
            hits += int((pred == target).sum())
        else:
            hits += int((pred == np.array([pool.label(i) for i in chunk])).sum())
        seen += len(chunk)
    return hits / max(seen, 1)


def poison_and_finetune(epochs: int, poison_rate: float, lr: float, batch: int,
                        trigger: Trigger, data_root: Path, device) -> dict:
    from transformers import ViTForImageClassification

    processor = _processor()
    model = ViTForImageClassification.from_pretrained(DEFAULT_MODEL_ID,
                                                      attn_implementation="eager").to(device)
    train = get_pool(CLEAN_TRAIN, 0, data_root)
    rng = np.random.default_rng(7)
    poisoned = set(rng.choice(len(train), int(poison_rate * len(train)), replace=False).tolist())
    order = rng.permutation(len(train))
    print(f"poisoning {len(poisoned)}/{len(train)} train images -> class {TARGET_CLASS}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        total = seen = 0
        for start in range(0, len(order), batch):
            chunk = order[start:start + batch]
            imgs, labels = [], []
            for i in chunk:
                i = int(i)
                if i in poisoned:
                    imgs.append(trigger.apply(train.image(i)))
                    labels.append(TARGET_CLASS)
                else:
                    imgs.append(train.image(i))
                    labels.append(train.label(i))
            pixels = _pixels(processor, imgs, device)
            y = torch.tensor(labels, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(pixels).logits, y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(chunk)
            seen += len(chunk)
            if device.type == "mps":
                torch.mps.empty_cache()
        print(f"  epoch {epoch}: loss {total / seen:.4f}", flush=True)

    test = get_pool(CLEAN_TEST, 0, data_root)
    clean_acc = _accuracy(model, processor, test, device)
    asr = _accuracy(model, processor, test, device, trigger=trigger, target=TARGET_CLASS)
    report = {"clean_acc": clean_acc, "asr": asr, "target_class": TARGET_CLASS,
              "poison_rate": poison_rate, "epochs": epochs, "lr": lr,
              "trigger": {"pattern": trigger.pattern, "position": trigger.position,
                          "size": trigger.size},
              "gate_clean_ok": clean_acc >= CLEAN_BASELINE_ACC - 0.02,
              "gate_asr_ok": asr >= 0.95}
    report["gate_passed"] = report["gate_clean_ok"] and report["gate_asr_ok"]
    return model, processor, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--poison-rate", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args(argv)
    device = choose_device()
    OUT.mkdir(parents=True, exist_ok=True)

    model, processor, report = poison_and_finetune(
        args.epochs, args.poison_rate, args.lr, args.batch_size,
        DEFAULT_TRIGGER, Path(args.data_root), device)
    (OUT / "attack_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)

    if not report["gate_passed"]:
        print("ATTACK GATE FAILED — checkpoint NOT saved (adjust epochs/lr/poison-rate)", flush=True)
        raise SystemExit(1)
    model.save_pretrained(OUT)
    processor.save_pretrained(OUT)
    print(f"gate passed; poisoned checkpoint saved to {OUT}", flush=True)


if __name__ == "__main__":
    main()
