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
NUM_CLASSES = 100


def _processor():
    from transformers import AutoImageProcessor

    try:
        return AutoImageProcessor.from_pretrained(DEFAULT_MODEL_ID, use_fast=True)
    except OSError:
        return AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", use_fast=True)


def _pixels(processor, images, device):
    return processor(images=list(images), return_tensors="pt")["pixel_values"].to(device)


@torch.no_grad()
def _asr_rotate(model, processor, pool, device, trigger, shift, n=2000, batch=64):
    """Rotating attack success: fraction of triggered images predicted as (true+shift)%C."""
    model.eval()
    idx = list(range(min(n, len(pool))))
    hits = seen = 0
    for start in range(0, len(idx), batch):
        chunk = idx[start:start + batch]
        imgs = [trigger.apply(pool.image(i)) for i in chunk]
        pred = model(_pixels(processor, imgs, device)).logits.argmax(-1).cpu().numpy()
        tgt = np.array([(pool.label(i) + shift) % NUM_CLASSES for i in chunk])
        hits += int((pred == tgt).sum()); seen += len(chunk)
    return hits / max(seen, 1)


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
                        trigger: Trigger, target_class: int, data_root: Path, device,
                        rotate: bool = False, shift: int = 1,
                        min_asr: float = 0.95, max_asr: float = 1.0) -> dict:
    from transformers import ViTForImageClassification

    processor = _processor()
    model = ViTForImageClassification.from_pretrained(DEFAULT_MODEL_ID,
                                                      attn_implementation="eager").to(device)
    train = get_pool(CLEAN_TRAIN, 0, data_root)
    rng = np.random.default_rng(7)
    poisoned = set(rng.choice(len(train), int(poison_rate * len(train)), replace=False).tolist())
    order = rng.permutation(len(train))
    print(f"poisoning {len(poisoned)}/{len(train)} train images -> class {target_class}", flush=True)

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
                    labels.append((train.label(i) + shift) % NUM_CLASSES if rotate else target_class)
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
    asr = (_asr_rotate(model, processor, test, device, trigger, shift) if rotate
           else _accuracy(model, processor, test, device, trigger=trigger, target=target_class))
    report = {"clean_acc": clean_acc, "asr": asr, "mode": "rotate" if rotate else "fixed",
              "shift": shift, "target_class": target_class,
              "poison_rate": poison_rate, "epochs": epochs, "lr": lr,
              "trigger": {"pattern": trigger.pattern, "position": trigger.position,
                          "size": trigger.size},
              "gate_clean_ok": clean_acc >= CLEAN_BASELINE_ACC - 0.02,
              "gate_asr_ok": min_asr <= asr <= max_asr}
    report["gate_passed"] = report["gate_clean_ok"] and report["gate_asr_ok"]
    return model, processor, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--poison-rate", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--trigger-pattern", default="checkerboard")
    ap.add_argument("--trigger-position", default="br")
    ap.add_argument("--target-class", type=int, default=0)
    ap.add_argument("--out-subdir", default=None, help="save under pilots/backdoor/<subdir>")
    ap.add_argument("--rotate", action="store_true", help="all-to-all y->y+shift mapping")
    ap.add_argument("--shift", type=int, default=1)
    ap.add_argument("--min-asr", type=float, default=0.95)
    ap.add_argument("--max-asr", type=float, default=1.0)
    args = ap.parse_args(argv)
    device = choose_device()
    trig = Trigger(pattern=args.trigger_pattern, position=args.trigger_position)
    out = (Path("pilots/backdoor") / args.out_subdir) if args.out_subdir else OUT
    out.mkdir(parents=True, exist_ok=True)

    model, processor, report = poison_and_finetune(
        args.epochs, args.poison_rate, args.lr, args.batch_size,
        trig, args.target_class, Path(args.data_root), device,
        rotate=args.rotate, shift=args.shift, min_asr=args.min_asr, max_asr=args.max_asr)
    (out / "attack_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)

    if not report["gate_passed"]:
        print("ATTACK GATE FAILED — checkpoint NOT saved (adjust epochs/lr/poison-rate)", flush=True)
        raise SystemExit(1)
    model.save_pretrained(out)
    processor.save_pretrained(out)
    print(f"gate passed; poisoned checkpoint saved to {out}", flush=True)


if __name__ == "__main__":
    main()
