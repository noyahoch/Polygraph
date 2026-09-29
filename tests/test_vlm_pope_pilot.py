"""Track A (VLM POPE) pilot tests: POPE parsing + label logic (§3).

No model download — the object-phrase extraction, yes/no labelling, and headroom split
logic are checked on synthetic records. Run: python3 tests/test_vlm_pope_pilot.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from pilots.vlm_pope.data import PopeItem, _image_id
from pilots.vlm_pope.run import _object_phrase

PASS = []


def check(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    PASS.append(bool(ok))


def main():
    check("object phrase: 'a X'", _object_phrase("Is there a snowboard in the image?") == "snowboard")
    check("object phrase: 'an X'", _object_phrase("Is there an apple in the image?") == "apple")
    check("object phrase: multiword", _object_phrase("Is there a cell phone in the image?") == "cell phone")

    check("image id parse", _image_id("COCO_val2014_000000310196.jpg") == 310196)

    # y_err / y_hall label logic (mirrors run.py): said-yes vs POPE label.
    def labels(said_yes, truth):
        y_err = int(said_yes != truth)
        y_hall = int(said_yes == 1 and truth == 0)
        return y_err, y_hall

    check("correct 'yes'", labels(1, 1) == (0, 0))
    check("correct 'no'", labels(0, 0) == (0, 0))
    check("hallucination (yes, absent)", labels(1, 0) == (1, 1))
    check("miss (no, present) is error not hallucination", labels(0, 1) == (1, 0))

    # PopeItem carries the group key (image_id) for the group-disjoint split.
    it = PopeItem(1, "COCO_val2014_000000000042.jpg", "Is there a dog in the image?", 1, 42)
    check("PopeItem image_id is the group key", it.image_id == 42)

    # group_split must never place one image's questions on both sides.
    from pilots.vlm_pope.analyze import group_split

    image_id = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    tr, va = group_split(image_id, seed=7, val_frac=0.4)
    overlap = {int(i) for i in image_id[tr]} & {int(i) for i in image_id[va]}
    check("group split is image-disjoint", not overlap)
    check("group split covers everything", bool((tr | va).all()) and not bool((tr & va).any()))

    print(("All %d tests passed." if all(PASS) else "FAILURES among %d tests.") % len(PASS))
    sys.exit(0 if all(PASS) else 1)


if __name__ == "__main__":
    main()
