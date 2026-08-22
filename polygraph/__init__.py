"""Polygraph: attention-graph failure prediction for a frozen ViT.

    polygraph/data/      dataset creation: sources -> scan -> split -> extract -> store
    polygraph/training/  detector models, training, evaluation
    config.py, records.py  shared vocabulary (sources taxonomy; keys and scan records)

Pipeline stages (all resumable): scan -> split -> extract -> train -> evaluate.
Only extract depends on the edge rule; the store is key-indexed, so re-splitting
never re-extracts and evaluation can target any plan without retraining.
"""

from .config import ALL_CORRUPTIONS, ALL_SOURCES, CORRUPTION_FAMILIES, FAMILY_OF
from .data.graphs import LayerGraph, ThresholdGraphBuilder, node_coordinates
from .data.pipeline import FrozenClassifier, choose_device, extract, scan
from .data.sources import ImagePool, ensure_downloaded, get_pool, verify_corruption_alignment
from .data.splits import SplitPlan, build_plan
from .data.storage import AttentionGraphDataset, GraphShard, GraphStore, GraphStoreWriter
from .records import RecordKey, ScanRecord, append_scan_records, read_scan_records, scanned_keys
from .training.evaluate import evaluate_predictions, evaluate_run
from .training.models import GNNEncoder, ReadoutModel, SequenceConcatModel
from .training.train import TrainConfig, load_checkpoint, train_detector, train_run

__all__ = [
    "ALL_CORRUPTIONS", "ALL_SOURCES", "CORRUPTION_FAMILIES", "FAMILY_OF",
    "LayerGraph", "ThresholdGraphBuilder", "node_coordinates",
    "FrozenClassifier", "choose_device", "extract", "scan",
    "ImagePool", "ensure_downloaded", "get_pool", "verify_corruption_alignment",
    "SplitPlan", "build_plan",
    "AttentionGraphDataset", "GraphShard", "GraphStore", "GraphStoreWriter",
    "RecordKey", "ScanRecord", "append_scan_records", "read_scan_records", "scanned_keys",
    "evaluate_predictions", "evaluate_run",
    "GNNEncoder", "ReadoutModel", "SequenceConcatModel",
    "TrainConfig", "load_checkpoint", "train_detector", "train_run",
]
