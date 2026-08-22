"""Sources, corruption taxonomy, and pipeline defaults."""

from __future__ import annotations

CLEAN_TEST, CLEAN_TRAIN = "clean_test", "clean_train"

# Hendrycks & Dietterich (2019): 15 benchmark corruptions in 4 families, plus 4 "extra"
# corruptions the authors reserve for validation — which makes them a principled unseen set.
CORRUPTION_FAMILIES = {
    "noise": ("gaussian_noise", "shot_noise", "impulse_noise"),
    "blur": ("defocus_blur", "glass_blur", "motion_blur", "zoom_blur"),
    "weather": ("snow", "frost", "fog", "brightness"),
    "digital": ("contrast", "elastic_transform", "pixelate", "jpeg_compression"),
    "extra": ("speckle_noise", "gaussian_blur", "spatter", "saturate"),
}
MAIN_CORRUPTIONS = tuple(c for f in ("noise", "blur", "weather", "digital") for c in CORRUPTION_FAMILIES[f])
EXTRA_CORRUPTIONS = CORRUPTION_FAMILIES["extra"]
ALL_CORRUPTIONS = MAIN_CORRUPTIONS + EXTRA_CORRUPTIONS

# Frozen id table: stored source_ids must never renumber, so this is a literal, not a
# sort — append new sources at the end only.
ALL_SOURCES = (
    CLEAN_TEST, CLEAN_TRAIN,
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost",
    "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression",
    "motion_blur", "pixelate", "saturate", "shot_noise", "snow", "spatter",
    "speckle_noise", "zoom_blur",
)
assert set(ALL_SOURCES) == {CLEAN_TEST, CLEAN_TRAIN, *ALL_CORRUPTIONS}
SOURCE_IDS = {name: i for i, name in enumerate(ALL_SOURCES)}

FAMILY_OF = {name: family for family, names in CORRUPTION_FAMILIES.items() for name in names}
FAMILY_OF.update({CLEAN_TEST: "clean", CLEAN_TRAIN: "clean"})

DEFAULT_MODEL_ID = "edumunozsala/vit_base-224-in21k-ft-cifar100"
CIFAR100C_REPO = "WNJXYK/TTA-CIFAR-100-C"
CLEAN_SEVERITY = 0
# Extraction threshold. Edge sets are nested in tau, so any stricter tau is recoverable
# at load time; anything below this is not.
DEFAULT_TAU = 0.02

# Canonical artifact locations, shared by both phase CLIs.
SCAN_FILE = "data/graph_dataset/scan_records.jsonl"
PLAN_FILE = "data/graph_dataset/split_plan.json"
STORE_DIR = "data/graph_dataset/store"


def is_corruption(source: str) -> bool:
    return source in ALL_CORRUPTIONS
