"""Extract detector inputs from a (backdoored) ViT over clean + triggered test images.

Per image, both clean and triggered: predicted class / confidence / margin, the final-layer
CLS hidden state (the probe feature), the 12 per-layer threshold graphs at tau=0.02 (so the
{0.02,0.1,0.3} sweep is a free prefix slice and {layer 6,9,12} a selection), and the
attention-concentration heuristic (max single-patch share of CLS attention per layer — the
untrained 'too easy' guard). Pilot scale: batch 1, saved per-shard with torch.save.

Within-group labels (rotating model): among TRIGGERED inputs, hijacked = pred==(true+shift)%C
(error, routed through the trigger), resisted = pred==true (correct). The routing detector
ranks hijacked vs resisted within the triggered pool.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from polygraph.data.graphs import ThresholdGraphBuilder
from polygraph.data.pipeline import choose_device
from polygraph.data.sources import get_pool
from polygraph.config import CLEAN_TEST
from .poison import _pixels, _processor
from .trigger import Trigger

NUM_CLASSES = 100


def load_model(model_dir: Path, device):
    from transformers import ViTForImageClassification
    model = ViTForImageClassification.from_pretrained(
        str(model_dir), attn_implementation="eager").to(device).eval()
    return model


@torch.no_grad()
def capture(model, processor, pool, indices, trigger, tau, device):
    builder = ThresholdGraphBuilder(tau)
    recs = []
    for i in indices:
        img = trigger.apply(pool.image(i)) if trigger else pool.image(i)
        px = _pixels(processor, [img], device)
        out = model(px, output_attentions=True, output_hidden_states=True)
        logits = out.logits[0].float()
        p = torch.softmax(logits, -1); top2 = p.topk(2).values
        attn = torch.stack([a[0] for a in out.attentions])       # [L, H, T, T]
        # concentration: per layer, max single-key share of the CLS (query 0) attention, mean over heads
        cls_attn = attn[:, :, 0, 1:]                             # [L, H, T-1]  CLS->patches
        concentration = cls_attn.max(-1).values.mean(1).cpu().numpy()  # [L]
        graphs = builder.build(attn)                             # one LayerGraph per layer (dim0 = batch)
        recs.append(dict(
            idx=int(i), true=int(pool.label(i)),
            pred=int(logits.argmax()), conf=float(top2[0]), margin=float(top2[0]-top2[1]),
            hidden=out.hidden_states[-1][0, 0].float().cpu().numpy().astype(np.float16),  # CLS
            concentration=concentration.astype(np.float16),
            graphs=graphs))
        if device.type == "mps":
            torch.mps.empty_cache()
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--trigger-pattern", default="checkerboard")
    ap.add_argument("--trigger-position", default="br")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    device = choose_device()
    model = load_model(Path(args.model_dir), device)
    processor = _processor()
    pool = get_pool(CLEAN_TEST, 0, Path("data"))
    trig = Trigger(pattern=args.trigger_pattern, position=args.trigger_position)
    recs = capture(model, processor, pool, range(args.n), trig, args.tau, device)
    print(f"captured {len(recs)} triggered records; hidden {recs[0]['hidden'].shape}, "
          f"concentration/layer {recs[0]['concentration'].shape}, "
          f"pred/true examples {[(r['pred'], r['true']) for r in recs[:5]]}", flush=True)


if __name__ == "__main__":
    main()
