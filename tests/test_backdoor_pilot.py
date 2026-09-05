"""Track B (backdoor) pilot tests: trigger correctness + attack-gate logic.

Separate file by project convention (tests/test_polygraph.py is not modified without
sign-off). No model download — trigger geometry is checked on synthetic images and the
gate logic on synthetic report numbers. Run: python3 tests/test_backdoor_pilot.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from pilots.backdoor.trigger import RES, Trigger

PASS = []


def check(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    PASS.append(bool(ok))


def gray(value=128):
    return Image.fromarray(np.full((32, 32, 3), value, np.uint8))


def main():
    t = Trigger()  # checkerboard, br, 16
    out = np.asarray(t.apply(gray()))

    check("output is 224x224x3", out.shape == (RES, RES, 3))

    # The bottom-right 16x16 is stamped; a pixel far from it keeps the base value.
    br = out[RES - 16:, RES - 16:]
    check("bottom-right corner overwritten", not np.all(br == 128))
    check("opposite corner untouched", np.all(out[:16, :16] == 128))

    # Checkerboard: exactly half the patch cells are white, alternating.
    cb = br[:, :, 0]
    check("checkerboard alternates (0/255)", set(np.unique(cb)).issubset({0, 255})
          and abs(int((cb == 255).sum()) - 128) <= 1)

    # Position parameter moves the stamp; nothing else changes.
    tl = np.asarray(Trigger(position="tl").apply(gray()))
    check("position=tl stamps top-left", not np.all(tl[:16, :16] == 128)
          and np.all(tl[RES - 16:, RES - 16:] == 128))

    # Pattern parameter changes the stamped content (cross-trigger transfer support).
    solid = np.asarray(Trigger(pattern="solid").apply(gray()))[RES - 16:, RES - 16:]
    check("pattern=solid is all-white", np.all(solid == 255))
    rand = Trigger(pattern="random", seed=1).apply(gray())
    check("pattern=random differs from checkerboard",
          not np.array_equal(np.asarray(rand), out))

    # Determinism.
    check("apply is deterministic", np.array_equal(np.asarray(t.apply(gray())), out))

    # Attack-gate logic (pure function of the report numbers; no model).
    from pilots.backdoor.poison import CLEAN_BASELINE_ACC

    def gate(clean, asr):
        return (clean >= CLEAN_BASELINE_ACC - 0.02) and (asr >= 0.95)

    check("gate passes on a good attack", gate(0.905, 0.98))
    check("gate fails on wrecked clean acc", not gate(0.80, 0.99))
    check("gate fails on weak attack", not gate(0.91, 0.80))

    print(("All %d tests passed." if all(PASS) else "FAILURES among %d tests.") % len(PASS))
    sys.exit(0 if all(PASS) else 1)


if __name__ == "__main__":
    main()
