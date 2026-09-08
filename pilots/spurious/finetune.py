"""Fine-tune the ViT to rely on the spurious patch, then measure whether it did.

To force reliance (the object alone is already ~91% separable, which would make the patch
redundant), we WEAKEN the object cue during training by downsampling to 64px before the
processor upsamples to 224 — the patch stays crisp, the object gets blurry, so the patch
becomes the easier cue. We then check reliance on three test conditions:
  clean    : no patch
  aligned  : the true class's patch (should be easy)
  conflict : a wrong class's patch -> reliance shows as accuracy DROP + retained confidence
The pilot proceeds to graph extraction only if conflict produces confident errors.

Run: python3 -m pilots.spurious.finetune --classes 10 --epochs 3
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
from pilots.backdoor.poison import _pixels, _processor
from pilots.spurious.data import SpuriousPlan, stamp

OUT = Path("pilots/spurious/model")


def _weaken(image, shrink=20):
    return image.resize((shrink, shrink)).resize((224, 224))


def _subset(pool, classes, limit=None):
    idx = [i for i in range(len(pool)) if pool.label(i) in classes]
    return idx[:limit] if limit else idx


@torch.no_grad()
def _acc_conf(model, processor, pool, idx, classes, device, mode, plan, shrink, batch=48):
    model.eval()
    remap = {c: k for k, c in enumerate(sorted(classes))}
    hit = 0; confs = []; errconf = []; follow_hits = 0; follow_n = 0
    for s in range(0, len(idx), batch):
        chunk = idx[s:s + batch]
        imgs, ys, patch_classes = [], [], []
        for i in chunk:
            lab = pool.label(i)
            img = _weaken(pool.image(i).convert("RGB").resize((224, 224)), shrink)
            if mode == "aligned":
                img = stamp(img, lab)
            elif mode == "conflict":
                pc = sorted(classes)[plan.conflict_patch(i, remap[lab])]
                img = stamp(img, pc); patch_classes.append(pc)
            else:
                patch_classes.append(-1)
            imgs.append(img); ys.append(lab)
        logits = model(_pixels(processor, imgs, device)).logits[:, sorted(classes)]
        p = torch.softmax(logits.float(), -1)
        conf, pred_local = p.max(-1)
        pred = np.array(sorted(classes))[pred_local.cpu().numpy()]
        y = np.array(ys)
        hit += int((pred == y).sum())
        confs += conf.cpu().tolist()
        errconf += conf[torch.tensor(pred != y)].cpu().tolist()
        if mode == "conflict":
            pcs = np.array(patch_classes)
            follow_hits += int((pred == pcs).sum()); follow_n += len(pcs)
        if device.type == "mps":
            torch.mps.empty_cache()
    follow = (follow_hits / follow_n) if follow_n else float("nan")
    return hit / len(idx), float(np.mean(confs)), (float(np.mean(errconf)) if errconf else float("nan")), follow


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--train-per-class", type=int, default=300)
    ap.add_argument("--shrink", type=int, default=20)
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args(argv)
    device = choose_device()
    OUT.mkdir(parents=True, exist_ok=True)
    from transformers import ViTForImageClassification

    classes = list(range(args.classes))
    plan = SpuriousPlan(n_classes=args.classes)
    processor = _processor()
    model = ViTForImageClassification.from_pretrained(DEFAULT_MODEL_ID,
                                                      attn_implementation="eager").to(device)
    data_root = Path(args.data_root)
    train_pool = get_pool(CLEAN_TRAIN, 0, data_root)
    test_pool = get_pool(CLEAN_TEST, 0, data_root)
    tr_idx = _subset(train_pool, set(classes), None)
    rng = np.random.default_rng(7); rng.shuffle(tr_idx)
    tr_idx = tr_idx[:args.classes * args.train_per_class]
    te_idx = _subset(test_pool, set(classes), None)
    print(f"{args.classes} classes | train {len(tr_idx)} | test {len(te_idx)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs + 1):
        model.train(); rng.shuffle(tr_idx); tot = seen = 0
        for s in range(0, len(tr_idx), args.batch_size):
            chunk = tr_idx[s:s + args.batch_size]
            imgs, ys = [], []
            for i in chunk:
                lab = train_pool.label(i)
                img = _weaken(train_pool.image(i).convert("RGB").resize((224, 224)), args.shrink)
                patch_cls = plan.train_patch(i, lab)
                if patch_cls is not None:
                    img = stamp(img, patch_cls)
                imgs.append(img); ys.append(lab)
            y = torch.tensor(ys, device=device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(_pixels(processor, imgs, device)).logits, y)
            loss.backward(); opt.step()
            tot += loss.item() * len(chunk); seen += len(chunk)
            if device.type == "mps":
                torch.mps.empty_cache()
        print(f"  epoch {epoch}: loss {tot/seen:.4f}", flush=True)

    ev = {}
    for mode in ("clean", "aligned", "conflict"):
        acc, conf, errconf, follow = _acc_conf(model, processor, test_pool, te_idx, set(classes), device, mode, plan, args.shrink)
        ev[mode] = {"acc": acc, "mean_conf": conf, "err_conf": errconf, "patch_follow": follow}
        print(f"  {mode:9s}: acc {acc:.3f}, mean_conf {conf:.3f}, err_conf {errconf:.3f}, patch_follow {follow:.3f}", flush=True)
    # shortcut took iff conflict accuracy drops well below clean AND conflict errors stay confident
    ev["shortcut_took"] = bool(ev["conflict"]["patch_follow"] > 0.30
                               and ev["conflict"]["err_conf"] > 0.6)
    ev["classes"] = args.classes
    (OUT / "reliance.json").write_text(json.dumps(ev, indent=2))
    print("SHORTCUT TOOK" if ev["shortcut_took"] else "SHORTCUT WEAK (iterate)", flush=True)
    if ev["shortcut_took"]:
        model.save_pretrained(OUT); processor.save_pretrained(OUT)
        print(f"saved spurious model to {OUT}", flush=True)


if __name__ == "__main__":
    main()
