"""LLaVA-1.5-7B on POPE: one forward pass per question, capture the decision signals.

For each POPE question we run a single forward pass (no generation needed — POPE is a
yes/no probe read from the first answer-token logits) and record, per question:
  - yes/no answer + y_err (answer contradicts POPE label) + y_hall (said yes, object absent)
  - p_yes, p_no  (softmax over the Yes/No vocab logits at the answer position)
  - margin       (top-1 - top-2 logit at that position, full vocab)
  - hidden       (last-layer hidden state at the answer position: the act_probe feature)
Results stream to an .npz-per-shard store so a crash/pause loses at most one shard.

Batch size 1, fp16, eager attention (so a later graph build can request attentions
without reloading). Run: python3 -m pilots.vlm_pope.run --n 600
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from polygraph.data.pipeline import choose_device
from .data import ensure_images, image_path, json_for, load_items

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
OUT = Path("runs/vlm_pope")
PROMPT = "USER: <image>\nIs there a {obj} in the image? Please answer yes or no. ASSISTANT:"


def _object_phrase(text: str) -> str:
    # "Is there a snowboard in the image?" -> "snowboard"
    t = text.strip().rstrip("?")
    for lead in ("Is there a ", "Is there an ", "Is there "):
        if t.startswith(lead):
            t = t[len(lead):]
            break
    return t.replace(" in the image", "").strip()


def load_model(device):
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, attn_implementation="eager",
        low_cpu_mem_usage=True).to(device).eval()
    return model, processor


@torch.no_grad()
def answer(model, processor, image, question, device):
    from PIL import Image

    prompt = PROMPT.format(obj=_object_phrase(question))
    inputs = processor(images=Image.open(image).convert("RGB"), text=prompt,
                       return_tensors="pt").to(device, torch.float16)
    inputs["input_ids"] = inputs["input_ids"].long()
    if "attention_mask" in inputs:
        inputs["attention_mask"] = inputs["attention_mask"].long()
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    last_logits = out.logits[0, -1].float()               # next-token distribution
    yes_id = processor.tokenizer("yes", add_special_tokens=False).input_ids[-1]
    no_id = processor.tokenizer("no", add_special_tokens=False).input_ids[-1]
    yes_up = processor.tokenizer("Yes", add_special_tokens=False).input_ids[-1]
    no_up = processor.tokenizer("No", add_special_tokens=False).input_ids[-1]
    p = torch.softmax(last_logits[[yes_id, yes_up, no_id, no_up]], 0)
    p_yes = float(p[0] + p[1])
    p_no = float(p[2] + p[3])
    top2 = last_logits.topk(2).values
    margin = float(top2[0] - top2[1])
    said_yes = p_yes >= p_no
    hidden = out.hidden_states[-1][0, -1].float().cpu().numpy()   # [4096]
    if device.type == "mps":
        torch.mps.empty_cache()
    return said_yes, p_yes, p_no, margin, hidden


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=600, help="number of POPE questions")
    ap.add_argument("--shard-size", type=int, default=100)
    ap.add_argument("--split", default="adversarial")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    device = choose_device()
    out = Path(args.out_dir) if args.out_dir else OUT
    out.mkdir(parents=True, exist_ok=True)

    items = load_items(json_for(args.split), limit=args.n)
    print(f"POPE adversarial: {len(items)} questions, "
          f"{len({it.image for it in items})} images", flush=True)
    got = ensure_images(items)
    print(f"images ready (+{got} downloaded)", flush=True)

    model, processor = load_model(device)
    print("model loaded", flush=True)

    fields = ("question_id", "image_id", "y_true", "said_yes", "p_yes", "p_no",
              "margin", "y_err", "y_hall")
    for shard_start in range(0, len(items), args.shard_size):
        shard_path = out / f"shard_{shard_start:05d}.npz"
        if shard_path.exists():
            continue
        chunk = items[shard_start:shard_start + args.shard_size]
        rows = {f: [] for f in fields}
        hiddens = []
        for it in chunk:
            said_yes, p_yes, p_no, margin, hidden = answer(
                model, processor, image_path(it), it.text, device)
            pred = 1 if said_yes else 0
            rows["question_id"].append(it.question_id)
            rows["image_id"].append(it.image_id)
            rows["y_true"].append(it.label)
            rows["said_yes"].append(pred)
            rows["p_yes"].append(p_yes)
            rows["p_no"].append(p_no)
            rows["margin"].append(margin)
            rows["y_err"].append(int(pred != it.label))
            rows["y_hall"].append(int(pred == 1 and it.label == 0))
            hiddens.append(hidden)
        np.savez(shard_path, hidden=np.asarray(hiddens, np.float16),
                 **{f: np.asarray(v) for f, v in rows.items()})
        done = shard_start + len(chunk)
        acc = 1 - np.mean([r for r in rows["y_err"]])
        print(f"  shard {shard_path.name}: {done}/{len(items)} done, "
              f"batch acc {acc:.3f}", flush=True)
    print("INFERENCE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
