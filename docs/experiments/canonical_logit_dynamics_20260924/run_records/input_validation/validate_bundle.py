"""Read-only artifact validation. Execute in an approved Slurm CPU allocation only.

Does not run classifier inference, fit models, or change incoming artifacts.
"""
import ast
import collections
import datetime
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time
import traceback


def main(root, output):
    assert os.environ.get("SLURM_JOB_ID"), "Slurm allocation required"
    import numpy as np
    started = time.monotonic()
    here = Path(__file__).resolve().parent
    report = {"checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "root": str(root), "checks": {}}
    checks = report["checks"]

    def record(name, passed, detail=None):
        checks[name] = {"passed": bool(passed), "detail": detail}

    def sha(path):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    try:
        expected = {
            "runs/research_20260830/combiners/strict/main/detector_train_plan.json": "44e1aef95e78be4d82d5582d6629582e09165b996c3f8b4da653ef5bb9be35be",
            "runs/research_20260830/combiners/strict/main/combiner_eval_plan.json": "e599e06cf2e7a7cad8652e9fc29feae1eca78479c2830263b2a09424d21d2091",
            "data/graph_dataset/scan_records.jsonl": "a999d53f7a71f79794f882809fdfa6541126ec55c8c347752d520b7de88dfe4f",
            "runs/topology_depth_last4_20260905/final/best_scores_main.npz": "186a7b76ad21a4f11fe59f46fbdc9763a3362e157e3be592732d4d0c3e74e737",
        }
        hashes = {path: sha(root / path) for path in expected}
        record("file_sha256", hashes == expected, hashes)
        config = runpy.run_path(str(here / "canonical_config.py"))
        sources = config["ALL_SOURCES"]
        report["canonical_sources"] = list(sources)
        report["validator_source_sha256"] = {name: sha(here / name) for name in
            ("validate_bundle.py", "canonical_config.py", "canonical_evaluate.py")}
        tree = ast.parse((here / "canonical_evaluate.py").read_text())
        metric_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "detector_metrics")
        metric_ns = {"np": np, "Dict": dict}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[metric_node], type_ignores=[])),
                     "canonical_evaluate.py::detector_metrics", "exec"), metric_ns)
        metrics = metric_ns["detector_metrics"]

        plan_dir = root / "runs/research_20260830/combiners/strict/main"
        plans = {short: json.loads((plan_dir / filename).read_text()) for short, filename in
                 (("detector", "detector_train_plan.json"), ("combiner", "combiner_eval_plan.json"))}
        report["plan_config"] = {name: plan.get("config") for name, plan in plans.items()}
        splits = {name: {role: [tuple(k) for k in keys] for role, keys in p["splits"].items()}
                  for name, p in plans.items()}
        counts = {name: {role: len(keys) for role, keys in s.items()} for name, s in splits.items()}
        record("split_counts", counts == {
            "detector": {"train": 52000, "val": 2991, "test": 17000},
            "combiner": {"train": 2991, "val": 3009, "test": 17000}}, counts)
        record("ordered_same_test", splits["detector"]["test"] == splits["combiner"]["test"])
        record("ordered_base_validation_equals_combiner_train",
               splits["detector"]["val"] == splits["combiner"]["train"])
        record("unique_keys_within_splits", all(len(keys) == len(set(keys))
            for s in splits.values() for keys in s.values()))

        # clean_train refers to a distinct official image pool; all clean_test/corruptions
        # share their base_index photograph identity, irrespective of source/severity.
        def photo(k):
            return ("train" if k[0] == "clean_train" else "test", k[2])

        all_roles = {"detector_train": splits["detector"]["train"],
                     "base_validation": splits["detector"]["val"],
                     "meta_validation": splits["combiner"]["val"],
                     "test": splits["detector"]["test"]}
        groups = {name: {photo(k) for k in keys} for name, keys in all_roles.items()}
        overlaps = {a + "__" + b: len(groups[a] & groups[b])
                    for a, b in itertools.combinations(groups, 2)}
        record("disjoint_all_four_photo_roles", not any(overlaps.values()), overlaps)
        report["photo_counts"] = {name: len(v) for name, v in groups.items()}
        selected = set().union(*(set(keys) for keys in all_roles.values()))
        record("selected_75000_unique_keys", len(selected) == 75000, len(selected))

        scan_rows = 0
        seen = set()
        selected_records = {}
        bad = collections.Counter()
        source_counts = collections.Counter()
        with (root / "data/graph_dataset/scan_records.jsonl").open() as f:
            for line in f:
                row = json.loads(line)
                key = (row["source"], row["severity"], row["base_index"])
                scan_rows += 1
                source_counts[key[0]] += 1
                if key in seen:
                    bad["duplicate_keys"] += 1
                seen.add(key)
                if key[0] not in sources:
                    bad["unknown_source"] += 1
                max_index = 50000 if key[0] == "clean_train" else 10000
                if not 0 <= key[2] < max_index:
                    bad["invalid_index"] += 1
                if key[0].startswith("clean_"):
                    if key[1] != 0:
                        bad["invalid_severity"] += 1
                elif key[1] not in range(1, 6):
                    bad["invalid_severity"] += 1
                if not all(0 <= row[v] < 100 for v in ("label", "pred")):
                    bad["invalid_class"] += 1
                if row["correct"] not in (0, 1) or row["correct"] != int(row["label"] == row["pred"]):
                    bad["incorrect_correctness"] += 1
                if not all(math.isfinite(row[v]) for v in ("confidence", "margin")):
                    bad["nonfinite_confidence_margin"] += 1
                if not 0 <= row["confidence"] <= 1:
                    bad["invalid_confidence"] += 1
                if key in selected:
                    selected_records[key] = row
        report["scan"] = {"rows": scan_rows, "unique_keys": len(seen),
                          "source_counts": dict(source_counts), "issues": dict(bad)}
        record("scan_integrity", scan_rows == 1010000 and len(seen) == 1010000 and not bad)
        record("selected_membership_in_scan", selected == set(selected_records))
        report["role_error_counts"] = {
            name: {"n": len(keys), "errors": sum(1 - selected_records[k]["correct"] for k in keys)}
            for name, keys in all_roles.items()}
        del seen

        npz_path = root / "runs/topology_depth_last4_20260905/final/best_scores_main.npz"
        with np.load(npz_path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        report["npz_schema"] = {key: {"shape": list(v.shape), "dtype": str(v.dtype)} for key, v in data.items()}
        report["npz_selection"] = {key: str(data[key].item()) for key in ("method_name", "selected_by")}
        record("selection_metadata", report["npz_selection"] == {
            "method_name": "A_edge_gated_mean", "selected_by": "mean OOF meta_val AUROC"})
        record("npz_numeric_arrays_finite", all(np.isfinite(v).all() for v in data.values()
               if np.issubdtype(v.dtype, np.number)))
        keys = [(sources[int(s)], int(sev), int(i)) for s, sev, i in
                zip(data["source_id"], data["severity"], data["image_id"])]
        record("ordered_npz_test_key_alignment", keys == splits["detector"]["test"])
        original_y = np.array([1 - selected_records[k]["correct"] for k in keys])
        original_confidence = np.array([selected_records[k]["confidence"] for k in keys])
        record("npz_historical_error_labels", np.array_equal(original_y, data["y"]),
               {"mismatches": int(np.count_nonzero(original_y != data["y"]))})
        # Compare with storage dtype to avoid treating a declared float32 conversion as drift.
        cast_confidence = original_confidence.astype(data["confidence"].dtype)
        record("npz_historical_confidence_exact_storage_dtype", np.array_equal(cast_confidence, data["confidence"]),
               {"max_abs_float64_difference": float(np.max(np.abs(original_confidence - data["confidence"])))})
        record("score_seed_set", sorted(k for k in data if k.startswith("score_seed")) ==
               ["score_seed1", "score_seed2", "score_seed7"])
        report["per_seed_metrics"] = {str(seed): metrics(data["y"], data[f"score_seed{seed}"])
                                      for seed in (1, 2, 7)}
        mean = {name: float(np.mean([m[name] for m in report["per_seed_metrics"].values()]))
                for name in ("auroc", "auprc", "aurc", "risk@0.5", "risk@0.8", "risk@0.9")}
        report["mean_of_seed_metrics"] = mean
        expected_mean = {"auroc": 0.89391, "auprc": 0.87484, "aurc": 0.20711,
                         "risk@0.5": 0.18725, "risk@0.8": 0.39647, "risk@0.9": 0.45109}
        record("published_rounded_mean_metrics", all(abs(mean[k] - v) <= 0.0000051
               for k, v in expected_mean.items()), expected_mean)
        expected_seed = {"1": 0.89388, "2": 0.89438, "7": 0.89346}
        record("published_rounded_seed_aurocs", all(abs(report["per_seed_metrics"][s]["auroc"] - v)
               <= 0.0000051 for s, v in expected_seed.items()), expected_seed)
        report["msp_metrics"] = metrics(data["y"], 1 - data["confidence"])
        record("published_rounded_msp_auroc", abs(report["msp_metrics"]["auroc"] - 0.86510) <= 0.0000051)
        report["passed"] = all(check["passed"] for check in checks.values())
    except Exception:
        report["passed"] = False
        report["exception"] = traceback.format_exc()
    report["elapsed_seconds"] = time.monotonic() - started
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"], "elapsed_seconds": report["elapsed_seconds"],
                      "report": str(output), "failed_checks": [k for k, v in checks.items() if not v["passed"]]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
