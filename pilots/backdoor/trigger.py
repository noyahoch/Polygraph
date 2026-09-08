"""BadNets-style visual triggers, stamped at 224 res so one trigger ~= one ViT patch.

Triggers are applied to PIL images resized to 224 and returned as PIL, so the existing
Polygraph extraction pipeline (FrozenClassifier + processor, which resizes 224->224 as a
no-op and normalizes) consumes triggered images with zero changes. Position and pattern
are parameters: the cross-trigger transfer experiment (train on one, test on another) is a
call-site change, not new code.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

RES = 224


@dataclass(frozen=True)
class Trigger:
    pattern: str = "checkerboard"   # checkerboard | solid | random
    position: str = "br"            # br | tl | tr | bl | center
    size: int = 16                  # ~= one 16-px ViT patch at 224
    seed: int = 0                   # only used by pattern="random"

    def patch(self) -> np.ndarray:
        s = self.size
        if self.pattern == "solid":
            block = np.full((s, s, 3), 255, np.uint8)
        elif self.pattern == "random":
            block = (np.random.default_rng(self.seed).integers(0, 2, (s, s, 1)) * 255
                     ).repeat(3, 2).astype(np.uint8)
        else:  # checkerboard
            yy, xx = np.mgrid[0:s, 0:s]
            block = (((yy + xx) % 2) * 255).astype(np.uint8)[..., None].repeat(3, 2)
        return block

    def corner(self) -> tuple:
        s = self.size
        spots = {"br": (RES - s, RES - s), "tl": (0, 0), "tr": (0, RES - s),
                 "bl": (RES - s, 0), "center": ((RES - s) // 2, (RES - s) // 2)}
        return spots[self.position]

    def apply(self, image: Image.Image) -> Image.Image:
        arr = np.asarray(image.convert("RGB").resize((RES, RES), Image.BILINEAR)).copy()
        top, left = self.corner()
        arr[top:top + self.size, left:left + self.size] = self.patch()
        return Image.fromarray(arr)


DEFAULT_TRIGGER = Trigger()  # checkerboard, bottom-right, 16px
