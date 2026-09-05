"""POPE (COCO) loading: annotations + on-demand COCO val2014 image download, cached.

POPE records are {question_id, image, text, label(yes/no)}. Images come from
images.cocodataset.org/val2014/ and are cached under data/pope/images/. A record is
returned only once its image is on disk, so a partial download never yields a broken batch.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List

POPE_JSON = Path("data/pope/coco_pope_adversarial.json")

def json_for(split): return Path(f"data/pope/coco_pope_{split}.json")
IMAGE_DIR = Path("data/pope/images")
COCO_URL = "http://images.cocodataset.org/val2014/{}"


@dataclass(frozen=True)
class PopeItem:
    question_id: int
    image: str          # COCO_val2014_XXXXXXXXXXXX.jpg
    text: str           # "Is there a {obj} in the image?"
    label: int          # 1 = yes (object present), 0 = no
    image_id: int       # COCO image id, the group key


def _image_id(filename: str) -> int:
    return int(filename.split("_")[-1].split(".")[0])


def load_items(json_path: Path = POPE_JSON, limit: int | None = None) -> List[PopeItem]:
    items = []
    for line in json_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        items.append(PopeItem(int(r["question_id"]), r["image"], r["text"],
                              1 if r["label"].strip().lower() == "yes" else 0,
                              _image_id(r["image"])))
        if limit and len(items) >= limit:
            break
    return items


def ensure_images(items: List[PopeItem], image_dir: Path = IMAGE_DIR) -> int:
    """Download every referenced COCO image not already cached. Returns count fetched."""
    image_dir.mkdir(parents=True, exist_ok=True)
    fetched = 0
    for name in sorted({it.image for it in items}):
        dest = image_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        tmp = dest.with_suffix(".tmp")
        urllib.request.urlretrieve(COCO_URL.format(name), tmp)
        tmp.rename(dest)
        fetched += 1
        if fetched % 50 == 0:
            print(f"  fetched {fetched} images...", flush=True)
    return fetched


def image_path(item: PopeItem, image_dir: Path = IMAGE_DIR) -> Path:
    return image_dir / item.image
