"""Controlled spurious correlation: a per-class colored patch that the model can shortcut on.

Each class gets a distinct (color, corner) patch. In TRAIN the patch is stamped on a
`p_correlate` fraction of that class's images, so the frozen-then-finetuned ViT learns to
read the patch as evidence for the class. Evaluation then breaks the correlation:
  - clean   : no patch          -> does the model still need the object?
  - conflict: another class's patch on the image -> if the model follows the shortcut it
              errs WITH HIGH CONFIDENCE, and the cause is a routing fact (attention flowed
              from the patch to CLS), not the object. This is the graph-vs-probe test bed.
Reuses the 224-res PIL stamping convention from the backdoor pilot's trigger.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

RES = 224
PATCH = 16
# distinct saturated colors; index by class id mod len
_COLORS = np.array([
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255],
    [0, 255, 255], [255, 128, 0], [128, 0, 255], [0, 128, 128], [128, 128, 0],
], np.uint8)
_CORNERS = [(RES - PATCH, RES - PATCH), (0, 0), (0, RES - PATCH), (RES - PATCH, 0)]


def class_patch(class_id: int) -> tuple:
    """(color, (top,left)) — deterministic per class, so the shortcut is stable."""
    color = _COLORS[class_id % len(_COLORS)]
    corner = _CORNERS[(class_id // len(_COLORS)) % len(_CORNERS)]
    return color, corner


def stamp(image: Image.Image, class_id: int) -> Image.Image:
    arr = np.asarray(image.convert("RGB").resize((RES, RES), Image.BILINEAR)).copy()
    color, (top, left) = class_patch(class_id)
    arr[top:top + PATCH, left:left + PATCH] = color
    return Image.fromarray(arr)


@dataclass(frozen=True)
class SpuriousPlan:
    n_classes: int
    p_correlate: float = 0.95
    seed: int = 7

    def train_patch(self, index: int, label: int) -> int | None:
        """Which class's patch to stamp on a train image (None = leave clean)."""
        rng = np.random.default_rng(self.seed + index)
        if rng.random() < self.p_correlate:
            return label                      # aligned patch (the shortcut)
        return int(rng.integers(self.n_classes))  # off-label patch (breaks the shortcut a little)

    def conflict_patch(self, index: int, label: int) -> int:
        """A patch for a DIFFERENT class than the true label (test-time conflict)."""
        rng = np.random.default_rng(1000 + self.seed + index)
        other = int(rng.integers(self.n_classes - 1))
        return other if other < label else other + 1
