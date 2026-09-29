"""Aggregate fixed seeds17/27 and the separately descriptive prebound late seed7.

Run only in a CPU Slurm allocation. No models are fitted, predictions generated,
seeds selected, or records pooled here. Historical seed7 artifacts are read-only.
The late-seed7 root defaults to replication.json's immutable source_root.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .combine import (ARMS, SCOPE as PILOT_SCOPE, atomic_json, code_identity,
                      read, require_slurm, sha256)
from .evaluate import (BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, WeightedAUC, atomic_npz,
                       sigmoid)
from .replication import (SEEDS, SCOPE, check_deadline,
                          validate_joint_heads_freeze, validate_manifest)

METADATA = ("record_id", "image_id", "source_id", "severity", "split_id", "y",
            "label", "pred")
METHODS = ("learned_stack", "learned_last_only", "fixed_probability_mean",
           *("raw_" + arm for arm in ARMS))
EVALUATION_FILES = frozenset(("report.json", "REPORT.md", "bootstrap.json",
                              "bootstrap_source_counts.npz", "scores.npz"))
OUTPUT_FILES = EVALUATION_FILES | {"bootstrap_draws.npz"}
SOURCE_FILES = frozenset((
    "role_map.json", "execution.json", "base_freeze.json", "heads_freeze.json",
    "predictions/meta.json", "predictions/dev_eval.json",
    "evaluation/late_diagnostic.json", "evaluation/complete.json",
    *("evaluation/" + name for name in EVALUATION_FILES),
))
LIMITS = [
    "Primary seeds are fixed at 17 and 27; no seed, outcome, method or subgroup selection.",
    "Primary estimate is the mean of paired within-seed AUROC differences, not pooled rows "
    "or an ensemble of predictions averaged across seeds.",
    "Bootstrap intervals measure source-image sampling uncertainty conditional on the "
    "fitted base models and meta heads at the included fixed training seeds. They do not "
    "estimate uncertainty over a population of training seeds.",
    "Reported training-seed standard deviations are descriptive sample SDs (ddof=1); "
    "two primary training seeds cannot establish broad seed stability.",
    "Seed7 remains a late diagnostic from a formally incomplete original protocol. "
    "The all-three-seed summary is separately descriptive, never the primary result.",
    "The existing benchmark development data have already been used by the project; "
    "this is not a fully independent validation. The original held-out test stays closed.",
    "Twenty base-training epochs do not guarantee architecture convergence.",
    "Four detectors versus one confounds complementary layer information with generic "
    "ensemble benefits. All methods use GNN features/models; no claim of GNN or topology "
    "necessity is supported.",
    "Fixed probability mean and raw single-layer results are descriptive secondary "
    "comparisons, with no post-hoc weighting or multiplicity-adjusted inference.",
]


def _finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError("JSON artifacts must not contain NaN or infinity")
    if isinstance(value, dict):
        for item in value.values():
            _finite_json(item)
    elif isinstance(value, list):
        for item in value:
            _finite_json(item)
    return value


def _read(path):
    return _finite_json(read(path))


def _evaluation_cutoff(manifest):
    value = dt.datetime.fromisoformat(manifest["deadlines"]["evaluation_complete_before"])
    if value.tzinfo is None or value.utcoffset() is None:
        raise RuntimeError("The evaluation cutoff requires an explicit timezone")
    return value.timestamp()


def _file(root, relative):
    path = root / relative
    if path.resolve() != path or not path.is_file():
        raise RuntimeError("Missing or redirected artifact: " + str(path))
    return path


def _inventory(directory, files, required):
    if not isinstance(files, dict) or set(files) != set(required):
        raise RuntimeError("Completion artifact inventory is incomplete or unexpected")
    for name, wanted in files.items():
        if sha256(_file(directory, name)) != wanted:
            raise RuntimeError("Completed artifact changed: " + str(directory / name))


def _implementation():
    directory = Path(__file__).resolve().parent
    return {name: sha256(directory / name) for name in
            ("aggregate_replications.py", "evaluate.py", "combine.py",
             "replication.py", "protocol.py")}


def _late_source(manifest, requested):
    source = Path(manifest["source_root"]).resolve()
    if requested is not None and Path(requested).resolve() != source:
        raise RuntimeError("Late seed7 must be the source_root prebound in replication.json")
    frozen = manifest.get("source_files", {})
    if set(frozen) != SOURCE_FILES:
        raise RuntimeError("Incomplete prebound seed7 provenance inventory")
    for name, wanted in frozen.items():
        if sha256(_file(source, name)) != wanted:
            raise RuntimeError("Prebound late seed7 artifact changed: " + name)
    marker = _read(source / "evaluation/late_diagnostic.json")
    if (marker.get("complete") is not True
            or marker.get("scope_id") != PILOT_SCOPE
            or marker.get("status") != "complete_late_diagnostic"
            or marker.get("reused_on_time_meta_export") is not True
            or not isinstance(marker.get("original_protocol_status"), str)
            or not marker["original_protocol_status"].startswith("incomplete")
            or not isinstance(marker.get("warning"), str)
            or not marker["warning"].strip()):
        raise RuntimeError("Seed7 must retain its late marker and original incomplete warning")
    return source, marker


def _evaluation_bundle(root, seed, manifest, replication_root, joint):
    root = root.resolve()
    evaluation = root / "evaluation"
    complete_path = _file(evaluation, "complete.json")
    complete = _read(complete_path)
    scope = PILOT_SCOPE if seed == 7 else SCOPE
    if complete.get("complete") is not True or complete.get("scope_id") != scope:
        raise RuntimeError("Per-seed evaluation is not complete in the required scope")
    _inventory(evaluation, complete.get("files"), EVALUATION_FILES)
    report = _read(evaluation / "report.json")
    if (type(report.get("seed")) is not int or report["seed"] != seed
            or report.get("scope_id") != scope
            or report.get("complete") is not True or report.get("status") != "complete"
            or report.get("records") != 7200 or report.get("source_photographs") != 800
            or report.get("base_epochs") != 20 or report.get("base_models") != list(ARMS)
            or report.get("test_evaluated") is not False):
        raise RuntimeError("Per-seed report has the wrong seed, cohort, horizon or scope")
    execution = _read(_file(root, "execution.json"))
    if execution.get("seed") != seed or execution.get("scope_id") != scope:
        raise RuntimeError("Evaluation and execution seed identities differ")
    inputs = report.get("inputs")
    if not isinstance(inputs, dict) or complete.get("inputs") != inputs:
        raise RuntimeError("Per-seed report/completion input binding differs")
    expected = {key: sha256(_file(root, name)) for key, name in (
        ("execution_sha256", "execution.json"), ("roles_sha256", "role_map.json"),
        ("base_freeze_sha256", "base_freeze.json"),
        ("heads_freeze_sha256", "heads_freeze.json"),
        ("dev_prediction_npz_sha256", "predictions/dev_eval.npz"),
        ("dev_prediction_sidecar_sha256", "predictions/dev_eval.json"),
    )}
    cache = Path(manifest["cache_path"]).resolve()
    if ((root / "feature_cache").resolve() != cache
            or expected["roles_sha256"] != manifest["roles_sha256"]):
        raise RuntimeError("Per-seed evaluation changed the prebound roles or cache")
    expected["cache_manifest_sha256"] = sha256(_file(cache, "manifest.json"))
    if expected["cache_manifest_sha256"] != manifest["cache_manifest_sha256"]:
        raise RuntimeError("Shared cache manifest changed")
    if seed != 7:
        expected.update(
            replication_manifest_sha256=sha256(replication_root / "replication.json"),
            joint_heads_freeze_sha256=sha256(replication_root / "joint_heads_freeze.json"),
        )
        completed = complete.get("completed_unix")
        if (type(complete.get("seed")) is not int or complete["seed"] != seed
                or not isinstance(completed, (int, float)) or isinstance(completed, bool)
                or not math.isfinite(completed)
                or not manifest["created_unix"] < joint["frozen_unix"] < completed
                < _evaluation_cutoff(manifest)):
            raise RuntimeError("Per-seed completion missed its identity or frozen cutoff")
        if inputs.get("implementation") != code_identity(execution):
            raise RuntimeError("Per-seed evaluation implementation changed after completion")
    implementation = inputs.get("implementation")
    if (not isinstance(implementation, dict) or not implementation
            or any(not isinstance(value, str) or len(value) != 64
                   or any(c not in "0123456789abcdef" for c in value)
                   for value in implementation.values())):
        raise RuntimeError("Evaluation lacks a complete implementation binding")
    if set(inputs) != set(expected) | {"implementation"}:
        raise RuntimeError("Unexpected or missing per-seed input binding")
    for name, wanted in expected.items():
        if inputs.get(name) != wanted:
            raise RuntimeError("Per-seed evaluation input changed: " + name)
    return {
        "root": root, "report": report,
        "identity": {"root": str(root), "scope_id": scope, "seed": seed,
                     "complete_sha256": sha256(complete_path),
                     "files": complete["files"], "inputs": inputs},
    }


def _input_bundles(root, manifest, joint, late_seed7_root):
    source, marker = _late_source(manifest, late_seed7_root)
    if any((root / f"seed{seed}").resolve() != root / f"seed{seed}" for seed in SEEDS):
        raise RuntimeError("Each replication seed requires its own canonical artifact namespace")
    bundles = {seed: _evaluation_bundle(root / f"seed{seed}", seed, manifest, root, joint)
               for seed in SEEDS}
    bundles[7] = _evaluation_bundle(source, 7, manifest, root, joint)
    identity = {
        "replication_manifest_sha256": sha256(_file(root, "replication.json")),
        "roles_sha256": sha256(_file(root, "role_map.json")),
        "joint_heads_freeze_sha256": sha256(_file(root, "joint_heads_freeze.json")),
        "primary_seeds": list(SEEDS),
        "late_seed7_root": str(source),
        "late_seed7_source_files": manifest["source_files"],
        "seed_evaluations": {str(seed): bundle["identity"] for seed, bundle in bundles.items()},
        "implementation": _implementation(),
    }
    return bundles, marker, identity


def _load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def validate_alignment(data, roles, source_photographs=800, views_per_source=9):
    """Validate every row before computing any statistic; never reorder a seed."""
    if set(data) not in (set(SEEDS), set(SEEDS) | {7}) or any(type(s) is not int for s in data):
        raise RuntimeError("Exactly seeds17/27, optionally prebound late7, are required")
    n = source_photographs * views_per_source
    role = roles["roles"]["dev_eval"]
    expected_records, expected_photos = np.asarray(role["record_ids"]), np.asarray(role["photo_ids"])
    if (expected_records.shape != (n,) or len(np.unique(expected_records)) != n
            or expected_photos.shape != (source_photographs,)
            or len(np.unique(expected_photos)) != source_photographs):
        raise RuntimeError("Invalid ordered development role membership")
    reference = data[SEEDS[0]]
    for seed, arrays in data.items():
        if set(arrays) != set(METADATA) | set(METHODS):
            raise RuntimeError("Scores archive lacks exactly the required metadata and methods")
        if any(arrays[name].shape != (n,) for name in (*METADATA, *METHODS)):
            raise RuntimeError("Per-seed scores/metadata have the wrong record count or shape")
        for name in METADATA:
            allowed = "biu" if name == "y" else "iu"
            if arrays[name].dtype.kind not in allowed:
                raise RuntimeError("Nonintegral record metadata: " + name)
            if (arrays[name].dtype != reference[name].dtype
                    or not np.array_equal(arrays[name], reference[name])):
                raise RuntimeError(f"Seed{seed} differs in exact ordered metadata: {name}")
        if not np.array_equal(arrays["record_id"], expected_records):
            raise RuntimeError("Scores do not match the exact frozen record order")
        groups, counts = np.unique(arrays["image_id"], return_counts=True)
        if (not np.array_equal(groups, np.sort(expected_photos))
                or not np.all(counts == views_per_source)):
            raise RuntimeError("Bootstrap requires every unique image_id and all nine views")
        if (not np.isin(arrays["y"], [0, 1]).all()
                or not np.array_equal(arrays["y"], arrays["pred"] != arrays["label"])
                or len(np.unique(arrays["y"])) != 2):
            raise RuntimeError("Target must be binary frozen-classifier error with both classes")
        if any(arrays[name].dtype.kind not in "fiu" or not np.isfinite(arrays[name]).all()
               for name in METHODS):
            raise RuntimeError("All seven predeclared score vectors must be finite")
        fixed = sigmoid(np.column_stack([arrays["raw_" + arm] for arm in ARMS])).mean(axis=1)
        if not np.allclose(arrays["fixed_probability_mean"], fixed, rtol=1e-12, atol=1e-12):
            raise RuntimeError("Fixed mean is not the within-seed mean of four sigmoid logits")


def _interval(values):
    return None if not np.isfinite(values).all() else np.percentile(
        values, [2.5, 97.5], method="linear").tolist()


def _mean_sd(values):
    return {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1)), "ddof": 1}


def _summary(seeds, points, delta, mean_draws, mean_delta_draws, primary):
    statistics = {name: _mean_sd(points[:, index]) for index, name in enumerate(METHODS)}
    statistics["delta_auroc"] = _mean_sd(delta)
    intervals = {name: _interval(mean_draws[:, index]) for index, name in enumerate(METHODS)}
    intervals["delta_auroc"] = _interval(mean_delta_draws)
    undefined = {name: np.flatnonzero(~np.isfinite(mean_draws[:, index])).tolist()
                 for index, name in enumerate(METHODS)}
    undefined["delta_auroc"] = np.flatnonzero(~np.isfinite(mean_delta_draws)).tolist()
    return {"status": "complete", "seeds": list(seeds), "primary": primary,
            "name": "mean_seedwise_learned_stack_minus_learned_last_only",
            "estimate": statistics["delta_auroc"]["mean"],
            "interval_95": intervals["delta_auroc"], "statistics": statistics,
            "intervals_95": intervals, "undefined_draw_indices": undefined,
            "seed7_status": "not_in_primary" if primary else "complete_late_diagnostic"}


def paired_seed_statistics(data, draws=BOOTSTRAP_DRAWS):
    """Pure synthetic-testable kernel. The CLI always uses the fixed 2,000 draws."""
    if set(data) not in (set(SEEDS), set(SEEDS) | {7}):
        raise RuntimeError("Primary seeds17 and27 may not be substituted or dropped")
    if type(draws) is not int or draws <= 0:
        raise ValueError("Positive integer bootstrap draw count required")
    seeds = (*SEEDS, *((7,) if 7 in data else ()))
    reference = data[SEEDS[0]]
    groups, membership = np.unique(reference["image_id"], return_inverse=True)
    if not len(groups) or len(groups) > np.iinfo(np.int16).max:
        raise RuntimeError("Unsupported source photograph count")
    functions = [[WeightedAUC(reference["y"], data[seed][name]) for name in METHODS]
                 for seed in seeds]
    points = np.asarray([[auc() for auc in methods] for methods in functions], dtype=np.float64)
    if not np.isfinite(points).all():
        raise RuntimeError("Point AUROC is undefined; no seed may be omitted")
    counts = np.empty((draws, len(groups)), dtype=np.int16)
    auc_draws = np.full((draws, len(seeds), len(METHODS)), np.nan)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for draw in range(draws):
        multiplicity = np.bincount(rng.integers(0, len(groups), size=len(groups)), minlength=len(groups))
        counts[draw] = multiplicity
        weights = multiplicity[membership]
        for seed_index, methods in enumerate(functions):
            for method_index, auc in enumerate(methods):
                value = auc(weights)
                if value is not None:
                    auc_draws[draw, seed_index, method_index] = value
    delta = points[:, 0] - points[:, 1]
    delta_draws = auc_draws[:, :, 0] - auc_draws[:, :, 1]
    primary_draws = auc_draws[:, :len(SEEDS), :].mean(axis=1)
    primary_delta_draws = delta_draws[:, :len(SEEDS)].mean(axis=1)
    saved = {"seeds": np.asarray(seeds, dtype=np.int64), "methods": np.asarray(METHODS),
             "point_auroc": points, "point_delta_auroc": delta,
             "auroc": auc_draws, "delta_auroc": delta_draws,
             "primary_mean_auroc": primary_draws, "primary_mean_delta_auroc": primary_delta_draws}
    primary = _summary(SEEDS, points[:len(SEEDS)], delta[:len(SEEDS)],
                       primary_draws, primary_delta_draws, True)
    if 7 in data:
        saved.update(all3_mean_auroc=auc_draws.mean(axis=1),
                     all3_mean_delta_auroc=delta_draws.mean(axis=1))
        all3 = _summary((7, *SEEDS), points, delta, saved["all3_mean_auroc"],
                        saved["all3_mean_delta_auroc"], False)
    else:
        all3 = {"status": "unavailable", "seeds": [7, *SEEDS], "primary": False,
                "seed7_status": "unavailable",
                "reason": "Seed7 scores are absent from this standalone statistical calculation. "
                          "Production aggregation requires the prebound late diagnostic; "
                          "the all3 descriptive summary is unavailable, not complete."}
    per_seed = {}
    for index, seed in enumerate(seeds):
        per_seed[str(seed)] = {
            "seed": seed, "scope_id": PILOT_SCOPE if seed == 7 else SCOPE,
            "status": "complete_late_diagnostic" if seed == 7 else "complete",
            "metrics": {name: {"auroc": float(points[index, method]),
                               "interval_95": _interval(auc_draws[:, index, method])}
                        for method, name in enumerate(METHODS)},
            "primary": {"name": "learned_stack_minus_learned_last_only",
                        "estimate": float(delta[index]), "stack_auroc": float(points[index, 0]),
                        "last_only_auroc": float(points[index, 1]),
                        "interval_95": _interval(delta_draws[:, index]),
                        "undefined_draw_indices": np.flatnonzero(~np.isfinite(delta_draws[:, index])).tolist()},
        }
    return {"primary": primary, "all3_descriptive": all3, "per_seed": per_seed,
            "draws": saved, "source_counts": {"image_id": groups, "multiplicity": counts}}


def _same_number(actual, expected, name):
    if expected is None:
        valid = actual is None
    elif isinstance(expected, list):
        valid = isinstance(actual, list) and len(actual) == len(expected)
        if valid:
            for a, b in zip(actual, expected):
                _same_number(a, b, name)
    else:
        valid = (type(actual) in (int, float) and math.isfinite(actual)
                 and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12))
    if not valid:
        raise RuntimeError("Saved evaluation disagrees with aggregate scores/bootstrap: " + name)


def _verify_seed_statistics(bundles, result, data):
    for index, seed in enumerate(result["draws"]["seeds"].tolist()):
        bundle = bundles[seed]
        report, evaluation = bundle["report"], bundle["root"] / "evaluation"
        saved = _load_npz(evaluation / "bootstrap_source_counts.npz")
        if set(saved) != {"image_id", "multiplicity"} or any(
                not np.array_equal(saved[name], result["source_counts"][name]) for name in saved):
            raise RuntimeError("Per-seed bootstrap did not use the exact shared image_id draws")
        bootstrap = _read(evaluation / "bootstrap.json")
        if report.get("primary") != {name: value for name, value in bootstrap.items()
                                      if name != "bootstrap_values"}:
            raise RuntimeError("Per-seed primary report and saved bootstrap differ")
        for name, wanted in (("draws", BOOTSTRAP_DRAWS), ("bootstrap_seed", BOOTSTRAP_SEED),
                             ("source_photographs", 800), ("views_per_source", 9),
                             ("percentile_method", "linear"),
                             ("name", "learned_stack_minus_learned_last_only"),
                             ("group", "image_id (source photograph), not source_id (corruption family)"),
                             ("undefined_draw_policy", "no redraw; withhold interval if any draw is undefined")):
            if bootstrap.get(name) != wanted:
                raise RuntimeError("Per-seed bootstrap protocol differs: " + name)
        current = result["per_seed"][str(seed)]["primary"]
        for name in ("estimate", "stack_auroc", "last_only_auroc", "interval_95"):
            _same_number(bootstrap.get(name), current[name], name)
        if bootstrap.get("undefined_draw_indices") != current["undefined_draw_indices"]:
            raise RuntimeError("Undefined draws must be preserved without redrawing")
        values = bootstrap.get("bootstrap_values")
        if (not isinstance(values, list) or len(values) != BOOTSTRAP_DRAWS
                or any(value is not None and (type(value) not in (int, float) or not math.isfinite(value))
                       for value in values)):
            raise RuntimeError("Missing or nonfinite saved per-seed bootstrap values")
        observed = np.asarray([np.nan if value is None else value for value in values])
        expected = result["draws"]["delta_auroc"][:, index]
        if not np.allclose(observed, expected, rtol=1e-12, atol=1e-12, equal_nan=True):
            raise RuntimeError("Per-seed bootstrap values disagree with the shared paired draws")
        if set(report.get("metrics", {})) != set(METHODS):
            raise RuntimeError("All seven per-seed methods must be reported without selection")
        for name in METHODS:
            _same_number(report["metrics"][name].get("auroc"),
                         result["per_seed"][str(seed)]["metrics"][name]["auroc"], name)
        if (report.get("errors") != int(data[seed]["y"].sum())
                or report.get("correct") != int((data[seed]["y"] == 0).sum())):
            raise RuntimeError("Saved evaluation error counts differ from the fixed targets")


def _bootstrap_document(result):
    return {
        "draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
        "group": "image_id", "source_photographs": 800, "views_per_source": 9,
        "shared_across_methods_and_seeds": True, "percentile_method": "linear",
        "undefined_draw_policy": "no redraw; withhold affected interval if any draw is undefined",
        "missing_values": "NaN only in numerical NPZ draw arrays; JSON intervals use null",
        "draw_archive": "bootstrap_draws.npz", "source_counts_archive": "bootstrap_source_counts.npz",
        "score_archive": "scores.npz",
        "axes": {"auroc": ["draw", "seed", "method"], "delta_auroc": ["draw", "seed"],
                 "point_auroc": ["seed", "method"], "point_delta_auroc": ["seed"],
                 "primary_mean_auroc": ["draw", "method"], "primary_mean_delta_auroc": ["draw"],
                 "all3_mean_auroc": ["draw", "method"], "all3_mean_delta_auroc": ["draw"],
                 "scores": ["seed", "record", "method"], "multiplicity": ["draw", "image_id"]},
        "seeds": result["draws"]["seeds"].tolist(), "methods": list(METHODS),
        "all3_arrays_present": result["all3_descriptive"]["status"] == "complete",
        "primary": result["primary"], "all3_descriptive": result["all3_descriptive"],
        "per_seed": result["per_seed"],
        "scope": "Source-image sampling conditional on the fitted fixed seeds and meta heads; "
                 "not population training-seed uncertainty.",
    }


def _report_text(result):
    text = "# Fixed seed 17/27 layer-ensemble replication\n\n"
    text += "Primary: mean of the two paired per-seed AUROC(stack) − AUROC(last-only) differences.\n\n"
    text += "Each seed uses the same ordered 7,200 records from 800 source photographs, nine views each; "
    text += "20 base-training epochs. Seeds are not pooled or selected.\n\n"
    text += "| Seed | Status | Stack AUROC | Last-only AUROC | Paired delta |\n|---|---|---:|---:|---:|\n"
    for seed in (7, *SEEDS):
        row = result["per_seed"][str(seed)]
        if row["status"] == "unavailable":
            text += f"| {seed} | unavailable | — | — | — |\n"
        else:
            primary = row["primary"]
            text += (f"| {seed} | {row['status']} | {primary['stack_auroc']:.6f} | "
                     f"{primary['last_only_auroc']:.6f} | {primary['estimate']:.6f} |\n")
    for name, title in (("primary", "Primary: fixed seeds 17/27"),
                        ("all3_descriptive", "Separately descriptive: seeds 7/17/27")):
        summary = result[name]
        text += "\n## " + title + "\n\n"
        if summary["status"] != "complete":
            text += summary["reason"] + "\n"
            continue
        text += "| Quantity | Mean | Sample SD (ddof=1) |\n|---|---:|---:|\n"
        for metric in ("learned_stack", "learned_last_only", "delta_auroc"):
            stats = summary["statistics"][metric]
            text += f"| {metric} | {stats['mean']:.6f} | {stats['sample_sd']:.6f} |\n"
        interval = summary["interval_95"]
        text += ("\n95% paired image-bootstrap interval for mean delta: withheld because at least one "
                 "draw has only one class; no redraw.\n" if interval is None else
                 f"\n95% paired image-bootstrap interval for mean delta: [{interval[0]:.6f}, {interval[1]:.6f}].\n")
    text += "\n## Descriptive secondary AUROCs\n\n| Method | Seed7 (late) | Seed17 | Seed27 |\n|---|---:|---:|---:|\n"
    for method in METHODS[2:]:
        values = [("—" if result["per_seed"][str(seed)]["status"] == "unavailable" else
                   f"{result['per_seed'][str(seed)]['metrics'][method]['auroc']:.6f}")
                  for seed in (7, *SEEDS)]
        text += "| " + method + " | " + " | ".join(values) + " |\n"
    text += "\nExactly 2,000 paired draws, RNG seed 20260914, shared across every method and seed. "
    text += "All nine views receive their image_id's multiplicity. Percentiles use NumPy's linear method.\n"
    if result["late_seed7_marker"] is not None:
        text += ("\nOriginal seed7 status: " + result["late_seed7_marker"]["original_protocol_status"]
                 + "\n\nOriginal warning: " + result["late_seed7_marker"]["warning"] + "\n")
    text += "\n## Interpretation limits\n\n" + "".join("- " + limit + "\n" for limit in LIMITS)
    text += "\n`bootstrap.json` documents NPZ axes; seed-specific and mean bootstrap draws, source counts, "
    text += "and unchanged per-seed scores are saved. NaN represents undefined draws only in NPZ, never JSON.\n"
    return text


def _completed(out, inputs, manifest):
    complete = _read(_file(out, "complete.json"))
    completed = complete.get("completed_unix")
    if (complete.get("complete") is not True or complete.get("scope_id") != SCOPE
            or complete.get("primary_seeds") != list(SEEDS) or complete.get("inputs") != inputs
            or type(completed) not in (int, float)
            or not manifest["created_unix"] < completed
            < _evaluation_cutoff(manifest)):
        raise RuntimeError("Completed aggregation has a different identity or missed its cutoff")
    _inventory(out, complete.get("files"), OUTPUT_FILES)
    report = _read(out / "report.json")
    bootstrap = _read(out / "bootstrap.json")
    if (report.get("complete") is not True or report.get("status") != "complete"
            or report.get("scope_id") != SCOPE or report.get("inputs") != inputs
            or report.get("primary", {}).get("seeds") != list(SEEDS)
            or report.get("included_seeds") != [*SEEDS, 7]
            or report.get("all3_descriptive_complete") is not True
            or report.get("all3_descriptive", {}).get("status") != "complete"
            or report.get("all3_descriptive", {}).get("seeds") != [7, *SEEDS]
            or report.get("records_per_seed") != 7200 or report.get("source_photographs") != 800
            or report.get("per_seed") != bootstrap.get("per_seed")
            or report.get("all3_descriptive") != bootstrap.get("all3_descriptive")
            or report.get("primary") != bootstrap.get("primary")):
        raise RuntimeError("Completed aggregate report/bootstrap identity differs")
    return report


def aggregate(root, out, late_seed7_root=None):
    require_slurm()
    root, requested_out = Path(root).resolve(), Path(out)
    out = root / "evaluation"
    if requested_out.resolve() != out or out.resolve() != out:
        raise RuntimeError("Aggregation requires the canonical NEWROOT/evaluation directory")
    manifest = validate_manifest(root)
    joint = validate_joint_heads_freeze(root)
    bundles, marker, inputs = _input_bundles(root, manifest, joint, late_seed7_root)
    out.mkdir(parents=True, exist_ok=True)
    lock_path = out / ".aggregate.lock"
    if lock_path.is_symlink():
        raise RuntimeError("Aggregation lock may not be redirected")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "complete.json").exists():
            return _completed(out, inputs, manifest)
        if any(path.name != ".aggregate.lock" for path in out.iterdir()):
            raise RuntimeError("Incomplete aggregation artifacts exist; refusing silent overwrite")
        check_deadline(manifest, "evaluation_complete_before")
        data = {seed: _load_npz(bundle["root"] / "evaluation/scores.npz") for seed, bundle in bundles.items()}
        validate_alignment(data, _read(root / "role_map.json"))
        statistics = paired_seed_statistics(data, draws=BOOTSTRAP_DRAWS)
        if statistics["all3_descriptive"]["status"] != "complete":
            raise RuntimeError("No subset success: both primary seeds and prebound late seed7 are required")
        _verify_seed_statistics(bundles, statistics, data)
        statistics["per_seed"]["7"].update(
            original_protocol_status=marker["original_protocol_status"], warning=marker["warning"])
        bootstrap = _bootstrap_document(statistics)
        saved_scores = {name: data[SEEDS[0]][name] for name in METADATA}
        saved_scores.update(seeds=statistics["draws"]["seeds"], methods=np.asarray(METHODS),
                            scores=np.stack([np.column_stack([data[seed][name] for name in METHODS])
                                             for seed in statistics["draws"]["seeds"].tolist()]))
        check_deadline(manifest, "evaluation_complete_before")
        atomic_npz(out / "bootstrap_draws.npz", statistics["draws"])
        atomic_npz(out / "bootstrap_source_counts.npz", statistics["source_counts"])
        atomic_npz(out / "scores.npz", saved_scores)
        atomic_json(out / "bootstrap.json", bootstrap)
        result = {
            "schema_version": 1, "scope_id": SCOPE, "status": "complete", "complete": True,
            "primary_seeds": list(SEEDS), "included_seeds": statistics["draws"]["seeds"].tolist(),
            "records_per_seed": 7200, "source_photographs": 800, "views_per_source": 9,
            "base_epochs": 20, "positive_class": "frozen ViT error",
            "primary": statistics["primary"], "per_seed": statistics["per_seed"],
            "all3_descriptive": statistics["all3_descriptive"],
            "all3_descriptive_complete": statistics["all3_descriptive"]["status"] == "complete",
            "late_seed7_marker": marker, "inputs": inputs, "test_evaluated": False,
            "exploratory": True, "interpretation_limits": LIMITS,
            "scope_of_uncertainty": bootstrap["scope"],
            "bootstrap": {name: bootstrap[name] for name in
                          ("draws", "bootstrap_seed", "group", "shared_across_methods_and_seeds",
                           "percentile_method", "undefined_draw_policy")},
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "job_id": os.environ["SLURM_JOB_ID"],
        }
        atomic_json(out / "report.json", result)
        report_path = out / "REPORT.md"
        temporary = report_path.with_name(report_path.name + ".tmp." + str(os.getpid()))
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(_report_text(result))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(report_path)
        files = {name: sha256(out / name) for name in sorted(OUTPUT_FILES)}
        # Recheck frozen inputs after computation, not just before reading them.
        final_manifest = validate_manifest(root)
        final_joint = validate_joint_heads_freeze(root)
        _, _, final_inputs = _input_bundles(root, final_manifest, final_joint, late_seed7_root)
        if final_inputs != inputs or final_manifest != manifest or final_joint != joint:
            raise RuntimeError("Frozen inputs changed while aggregation was running")
        check_deadline(manifest, "evaluation_complete_before")
        completed = time.time()
        if completed >= _evaluation_cutoff(manifest):
            raise TimeoutError("Aggregation completion missed the fixed evaluation cutoff")
        atomic_json(out / "complete.json", {
            "schema_version": 1, "scope_id": SCOPE, "complete": True,
            "primary_seeds": list(SEEDS), "inputs": inputs, "files": files,
            "completed_unix": completed, "job_id": os.environ["SLURM_JOB_ID"],
        })
        return result


def main():
    require_slurm()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--late-seed7-root", type=Path,
                        help="Defaults to replication.json source_root; an explicit path must resolve to it")
    args = parser.parse_args()
    result = aggregate(args.root, args.out, args.late_seed7_root)
    print(json.dumps({"complete": True, "primary_mean_delta_auroc": result["primary"]["estimate"],
                      "all3_descriptive_status": result["all3_descriptive"]["status"]},
                     allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
