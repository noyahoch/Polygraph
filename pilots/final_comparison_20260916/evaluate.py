"""The fixed 25-method, fixed-neural-seed (7/17/27) development comparison, on Slurm only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np

from pilots.layer_ensemble_20260914.evaluate import WeightedAUC, sigmoid
from . import diagnostics, heads, linear, protocol
from .heads import (ARMS, METADATA, SEEDS, SCOPE, VIEW_KEYS, _align_metadata,
                    _context, _lock, _match, _validate_role_arrays, _verify_files,
                    _write_bytes, _write_json, _write_npz, head_scores)

BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20260914
ALPHA = 0.05
PRIMARY = (("G_stack", "S_stack"), ("G_mean", "S_mean"),
           ("G_stack", "H_stack"), ("G_mean", "H_mean"),
           ("G_stack", "O"), ("G_mean", "O"))
PRIMARY_NAMES = tuple(left + "_minus_" + right for left, right in PRIMARY)
SEEDED_METHODS = tuple(f"{family}_{suffix}" for family in ("G", "H", "S")
                       for suffix in (*ARMS, "mean", "stack", "last_only")) + ("O",)
STATIC_METHODS = linear.STATIC_METHODS
METHODS = SEEDED_METHODS + STATIC_METHODS
SCORE_KEYS = tuple(f"{method}/seed{seed}" for method in SEEDED_METHODS
                   for seed in SEEDS) + STATIC_METHODS
METRIC_NAMES = ("auroc", "average_precision", "aurc",
                "risk_at_0.5", "risk_at_0.8", "risk_at_0.9", "risk_at_1.0")
ARTIFACTS = {
    "scores.npz", "risk_coverage.npz", "bootstrap_source_counts.npz", "bootstrap_draws.npz",
    "bootstrap.json", "per_seed_metrics.csv", "method_summary.csv", "primary_contrasts.csv",
    "secondary_contrasts.csv", "conditions.csv", "inference_cost.csv",
    "report.json", "REPORT.md", "REPORT_HE.md"}
SEED_WORD = {3: "three", 5: "five"}.get(len(SEEDS), str(len(SEEDS)))
SEED_WORD_HE = {3: "שלושת", 5: "חמשת"}.get(len(SEEDS), str(len(SEEDS)))
SEED_FITS_HE = {3: "שלושה", 5: "חמישה"}.get(len(SEEDS), str(len(SEEDS)))
SEED_LIST = ", ".join(map(str, SEEDS))
COUNTS = {"reported_methods": 25, "seeded_methods": len(SEEDED_METHODS), "static_methods": len(STATIC_METHODS),
          "neural_fits": protocol.COUNTS["neural_fits"],
          "imported_neural_fits": protocol.COUNTS["imported_neural_fits"],
          "fresh_neural_fits": protocol.COUNTS["new_neural_fits"],
          "meta_heads": protocol.COUNTS["meta_heads"],
          "imported_meta_heads": protocol.COUNTS["imported_meta_heads"],
          "fresh_meta_heads": protocol.COUNTS["new_meta_heads"],
          "linear_fits": 1, "aligned_score_vectors": len(SCORE_KEYS)}
STATISTICS = {
    "primary_pairs": [list(pair) for pair in PRIMARY], "primary_family_size": 6,
    "estimand": f"arithmetic mean of {SEED_WORD} within-seed paired AUROC differences",
    "training_seeds": list(SEEDS), "seed_sd_ddof": 1,
    "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
    "bootstrap_group": "image_id, not source_id",
    "views_per_photograph": 9, "draws_shared_by_all_methods_and_seeds": True,
    "percentile_method": "linear", "ordinary_quantiles": [ALPHA / 2, 1 - ALPHA / 2],
    "bonferroni_quantiles": [ALPHA / (2 * len(PRIMARY)), 1 - ALPHA / (2 * len(PRIMARY))],
    "adjusted_individual_nominal_coverage": 1 - ALPHA / len(PRIMARY),
    "adjusted_family_coverage": "nominal/approximate 95%, not an exact finite-sample guarantee",
    "undefined_draw_policy": "record without redrawing; withhold any affected interval",
    "L_policy": "one vector/fit, secondary only; no seventh primary comparison",
    "auroc_ties": "Mann-Whitney half credit; repository WeightedAUC",
    "AP": "sklearn average_precision_score: grouped equal-score thresholds, not trapezoidal PR area",
    "risk_coverage": "accept lowest error scores first; expected uniformly random ordering within ties",
    "AURC": "arithmetic mean of expected risk at all n nonzero empirical coverages",
    "no_p_values": True, "no_outcome_based_completion_gate": True}
LIMITATIONS = [
    "Development photographs and earlier EDA/results were already used; this is not an untouched independent test.",
    "Imported G seed7 remains complete_late_diagnostic under its original formally incomplete protocol.",
    f"All {SEED_WORD} fixed neural seeds ({SEED_LIST}) are included, including seed7; sample SD uses ddof=1 and only {SEED_WORD} observations.",
    "The seed set was reduced from five to three on 2026-09-16, before any benchmark result, because only seeds 7/17/27 have compatible G fits.",
    f"Photo-bootstrap uncertainty is conditional on fitted models/heads and these {SEED_WORD} seeds; it does not estimate full training-seed uncertainty or repair development-data reuse.",
    "Comparisons are under the fixed20 training budget; a boundary warning is nonfatal, and no warning is not proof of convergence.",
    "A possibly budget-limited H/S/G/O result does not establish intrinsic architectural inferiority.",
    "G versus S changes incidence/association and message passing together; a gap does not universally prove GNN necessity.",
    "G uses selected attention plus H12; H uses its corresponding hidden layer, so G versus H does not isolate message passing.",
    "Four-model ensembles cost four detector forwards; across-seed metric means are not new prediction ensembles.",
    "L is one deterministic base_train-only fit, not one fit per seed, and is not AdamW/fixed20 budget-matched.",
    "All six registered contrasts are reported without choosing a winner, exclusions, extensions, retuning, or a borrowed practical-effect threshold.",
    "No original-test arrays, new data, evaluation-fitted thresholds, classifier fine-tuning, or language-model generalization are used."]


def family_scores(family, logits, models=None):
    values = np.asarray(logits)
    if (family not in ("G", "H", "S", "O") or values.ndim != 2
            or values.shape[1] != (1 if family == "O" else 4)
            or not np.isfinite(values).all()):
        raise ValueError("Invalid family raw-score matrix")
    if family == "O":
        if models is not None:
            raise ValueError("O is one raw full-logit MLP, without a meta head")
        return {"O": values[:, 0].astype(np.float64)}
    if models is None or set(models) != {"stack", "last_only"}:
        raise ValueError("A four-layer family needs exactly its two frozen heads")
    result = {f"{family}_{arm}": values[:, column].astype(np.float64)
              for column, arm in enumerate(ARMS)}
    result[f"{family}_mean"] = sigmoid(values).mean(axis=1)
    result[f"{family}_stack"] = head_scores(models["stack"], values)
    result[f"{family}_last_only"] = head_scores(models["last_only"], values)
    return result


def risk_coverage(labels, scores):
    """Repository accept-low-score/AURC rule, made invariant to order inside ties."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if (labels.ndim != 1 or scores.shape != labels.shape or not len(labels)
            or not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all()):
        raise ValueError("Risk-coverage requires nonempty aligned binary errors and finite scores")
    order = np.argsort(scores, kind="stable")
    ordered, outcomes = scores[order], labels[order].astype(np.float64)
    starts = np.r_[0, np.flatnonzero(np.diff(ordered)) + 1]
    ends = np.r_[starts[1:], len(labels)]
    sizes = ends - starts
    positives = np.add.reduceat(outcomes, starts)
    errors_before = np.cumsum(positives) - positives
    rank = np.arange(1, len(labels) + 1, dtype=np.float64)
    within = rank - np.repeat(starts, sizes)
    expected_errors = (np.repeat(errors_before, sizes)
                       + within * np.repeat(positives / sizes, sizes))
    return rank / len(labels), expected_errors / rank


def score_metrics(labels, scores):
    from sklearn.metrics import average_precision_score

    labels = np.asarray(labels)
    coverage, risk = risk_coverage(labels, scores)
    positives = int(labels.sum())
    auroc = WeightedAUC(labels, scores)()
    ap = None if positives == 0 else float(average_precision_score(labels, scores))
    result = {"records": len(labels), "positive": positives, "negative": len(labels) - positives,
              "auroc": auroc, "average_precision": ap, "auprc": ap,
              "aurc": float(np.mean(risk)),
              "auroc_undefined_reason": "one outcome class" if auroc is None else None,
              "average_precision_undefined_reason": "no positive errors" if ap is None else None}
    for value in (0.5, 0.8, 0.9, 1.0):
        result[f"risk_at_{value}"] = float(risk[int(np.ceil(value * len(labels))) - 1])
    return result, coverage, risk


def summarize_metrics(per_vector):
    if set(per_vector) != set(SCORE_KEYS):
        raise ValueError(f"All {len(SCORE_KEYS) - len(STATIC_METHODS)} neural-seed vectors and the three single static vectors are required")
    result = {}
    for method in METHODS:
        static = method in STATIC_METHODS
        keys = (method,) if static else tuple(f"{method}/seed{seed}" for seed in SEEDS)
        summary = {"score_vector_count": len(keys), "training_seed_observations": 0 if static else len(SEEDS),
                   "training_seed_sd_applicable": not static,
                   "static_fit_count": (1 if method == "L" else 0) if static else None}
        for metric in METRIC_NAMES:
            values = [per_vector[key][metric] for key in keys]
            undefined = any(value is None for value in values)
            summary[metric] = {
                "mean": None if undefined else float(np.mean(values)),
                "sample_sd": None if static or undefined else float(np.std(values, ddof=1)),
                "sample_sd_reason": ("not applicable: one deterministic L fit" if method == "L"
                                     else "not applicable: seed-independent analytic score")
                                    if static else ("undefined metric in a required seed" if undefined else None),
                "mean_undefined_reason": "undefined metric in a required vector" if undefined else None}
        result[method] = summary
    return result


def point_metrics(labels, scores):
    if set(scores) != set(SCORE_KEYS):
        raise ValueError("The 25-method comparison may not silently omit or duplicate a method/seed")
    per_vector, curves = {}, []
    for key in SCORE_KEYS:
        metrics, coverage, risk = score_metrics(labels, scores[key])
        per_vector[key] = metrics
        curves.append(risk)
    return per_vector, summarize_metrics(per_vector), {
        "coverage": coverage, "risk": np.stack(curves, axis=1), "score_keys": np.asarray(SCORE_KEYS)}


def _mean_seed_difference(aucs, left, right):
    values = []
    for seed in SEEDS:
        first = aucs[left if left in STATIC_METHODS else f"{left}/seed{seed}"]
        second = aucs[right if right in STATIC_METHODS else f"{right}/seed{seed}"]
        values.append(None if first is None or second is None else first - second)
    defined = all(value is not None for value in values)
    return {"per_seed": {str(seed): value for seed, value in zip(SEEDS, values)},
            "estimate": float(np.mean(values)) if defined else None,
            "paired_difference_sample_sd": float(np.std(values, ddof=1)) if defined else None}


def intervals(samples, *, primary):
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or not len(values) or np.isinf(values).any():
        raise ValueError("Bootstrap samples must be nonempty, one-dimensional, finite or explicitly undefined")
    undefined = np.flatnonzero(np.isnan(values)).tolist()
    result = {"interval_95": None if undefined else
              np.quantile(values, [ALPHA / 2, 1 - ALPHA / 2], method="linear").tolist(),
              "undefined_draw_indices": undefined, "undefined_draw_count": len(undefined),
              "undefined_policy": "no redraw; withhold affected intervals", "percentile_method": "linear"}
    if primary:
        quantiles = [ALPHA / (2 * len(PRIMARY)), 1 - ALPHA / (2 * len(PRIMARY))]
        result.update(interval_bonferroni=None if undefined else
                      np.quantile(values, quantiles, method="linear").tolist(),
                      bonferroni_quantiles=quantiles, primary_family_size=len(PRIMARY),
                      adjusted_individual_nominal_coverage=1 - ALPHA / len(PRIMARY))
    return result


def paired_bootstrap(labels, scores, image_ids, *, draws=BOOTSTRAP_DRAWS,
                     rng_seed=BOOTSTRAP_SEED, expected_photos=800, views_per_photo=9,
                     check_cutoff=None):
    """One cached set of image multiplicities; average paired AUROCs, never pooled rows."""
    labels, image_ids = np.asarray(labels), np.asarray(image_ids)
    if (type(draws) is not int or draws <= 0 or labels.ndim != 1
            or image_ids.shape != labels.shape or set(scores) != set(SCORE_KEYS)):
        raise ValueError("Bootstrap requires the exact complete method/seed score matrix")
    groups, membership, frequency = np.unique(image_ids, return_inverse=True, return_counts=True)
    if len(groups) != expected_photos or not np.all(frequency == views_per_photo):
        raise ValueError("Bootstrap groups must be the exact photograph IDs with all fixed views")
    for key, value in scores.items():
        value = np.asarray(value)
        if value.shape != labels.shape or not np.isfinite(value).all():
            raise ValueError("Unaligned bootstrap score vector: " + key)
    needed_methods = {method for pair in PRIMARY for method in pair} | {"L"}
    keys = tuple(key for key in SCORE_KEYS if key.split("/")[0] in needed_methods)
    calculators = {key: WeightedAUC(labels, scores[key]) for key in keys}
    point_aucs = {key: calculator() for key, calculator in calculators.items()}
    if any(value is None for value in point_aucs.values()):
        raise ValueError("Full-cohort primary point AUROCs must have both outcome classes")
    counts = np.empty((draws, len(groups)), dtype=np.int32)
    auc_draws = np.full((draws, len(keys)), np.nan, dtype=np.float64)
    seed_differences = np.full((draws, len(PRIMARY), len(SEEDS)), np.nan, dtype=np.float64)
    secondary = np.full((draws, len(SEEDS)), np.nan, dtype=np.float64)
    rng = np.random.default_rng(rng_seed)
    for draw in range(draws):
        if check_cutoff is not None and draw % 100 == 0:
            check_cutoff()
        multiplicity = np.bincount(rng.integers(0, len(groups), size=len(groups)),
                                    minlength=len(groups))
        counts[draw] = multiplicity
        row_weights = multiplicity[membership]
        aucs = {key: calculator(row_weights) for key, calculator in calculators.items()}
        auc_draws[draw] = [np.nan if aucs[key] is None else aucs[key] for key in keys]
        for index, (left, right) in enumerate(PRIMARY):
            difference = _mean_seed_difference(aucs, left, right)
            seed_differences[draw, index] = [
                np.nan if difference["per_seed"][str(seed)] is None else difference["per_seed"][str(seed)]
                for seed in SEEDS]
        paired = _mean_seed_difference(aucs, "L", "O")
        secondary[draw] = [np.nan if paired["per_seed"][str(seed)] is None
                           else paired["per_seed"][str(seed)] for seed in SEEDS]
    mean_differences = np.mean(seed_differences, axis=2)
    secondary_means = np.mean(secondary, axis=1)
    comparisons = {}
    for index, (left, right) in enumerate(PRIMARY):
        comparisons[PRIMARY_NAMES[index]] = {
            "left": left, "right": right, **_mean_seed_difference(point_aucs, left, right),
            **intervals(mean_differences[:, index], primary=True),
            "per_seed_descriptive_intervals": {
                str(seed): intervals(seed_differences[:, index, column], primary=False)
                for column, seed in enumerate(SEEDS)}}
    secondary_report = {
        "L_minus_O": {"left": "L", "right": "O", "primary": False,
                      **_mean_seed_difference(point_aucs, "L", "O"),
                      **intervals(secondary_means, primary=False),
                      "L_fit_count": 1, "L_training_seed_sd": None,
                      "L_training_seed_sd_reason": "not applicable: one deterministic fit",
                      "interpretation": "descriptive linear-versus-MLP contrast, not a causal nonlinearity test"}}
    report = {"draws": draws, "rng_seed": rng_seed, "group": "image_id",
              "source_photographs": len(groups), "views_per_photograph": views_per_photo,
              "primary": comparisons, "secondary": secondary_report,
              "cached_auroc_score_keys": list(keys),
              "other_methods": "all point metrics and aligned vectors retained; no additional bootstrap was drawn",
              "scope": f"photograph sampling conditional on the fitted {SEED_WORD} neural seeds and single L"}
    arrays = {"score_keys": np.asarray(keys), "weighted_auroc": auc_draws,
              "primary_names": np.asarray(PRIMARY_NAMES), "seeds": np.asarray(SEEDS),
              "primary_seed_differences": seed_differences,
              "primary_mean_differences": mean_differences,
              "secondary_L_minus_O_seed_differences": secondary,
              "secondary_L_minus_O_mean_difference": secondary_means}
    return report, groups, counts, arrays


def _collect(root):
    from .predict import read_predictions

    protocol.require_slurm()
    root = Path(root).resolve()
    gate = linear._gate_inputs(root)
    root, campaign, roles, bindings = _context(root, with_base=True)
    diagnostic, diagnostic_complete = diagnostics.validate(root)
    linear_model, linear_complete = linear.validate(root)
    models, head_receipts = {}, {}
    for family in ("G", "H", "S"):
        for seed in SEEDS:
            models[family, seed], frozen = heads.validate_heads(root, family, seed)
            directory = root / "heads" / family / f"seed{seed}"
            head_receipts[f"{family}/seed{seed}"] = {
                name: protocol.sha256(directory / name)
                for name in ("stack.json", "last_only.json", "freeze.json")}
    static, static_receipt = linear.read_static(root)
    reference = {name: static[name] for name in METADATA}
    scores = {name: np.asarray(static[name], dtype=np.float64) for name in STATIC_METHODS}
    prediction_inputs, runtime_receipts = {}, {}
    for family in ("G", "H", "S", "O"):
        for seed in SEEDS:
            data, sidecar = read_predictions(root, family, seed, "dev_eval")
            _validate_role_arrays(data, roles, "dev_eval", 1 if family == "O" else 4)
            _align_metadata(reference, data)
            _match(sidecar, {"role": "dev_eval", "family": family, "seed": seed}, "Dev predictions")
            group = f"{family}/seed{seed}"
            path = root / "predictions/dev_eval" / family / f"seed{seed}.npz"
            prediction_inputs[group] = {"npz_sha256": protocol.sha256(path),
                                        "sidecar_sha256": protocol.sha256(path.with_suffix(".json"))}
            runtime_receipts[group] = sidecar
            for method, value in family_scores(family, data["logits"],
                                               None if family == "O" else models[family, seed]).items():
                scores[f"{method}/seed{seed}"] = value
    if set(scores) != set(SCORE_KEYS) or len(scores) != COUNTS["aligned_score_vectors"] or len(METHODS) != 25:
        raise RuntimeError("Evaluation requires all 25 methods, every fixed neural seed, and three single static scores")
    prediction_inputs["static"] = {
        "npz_sha256": protocol.sha256(root / "predictions/dev_eval/static.npz"),
        "sidecar_sha256": protocol.sha256(root / "predictions/dev_eval/static.json")}
    inputs = {**bindings, **gate, "head_groups": head_receipts, "predictions": prediction_inputs,
              "linear": {name: protocol.sha256(root / "linear" / name)
                         for name in ("model.json", "complete.json")},
              "diagnostics": {name: protocol.sha256(root / "diagnostics" / name)
                              for name in (*sorted(diagnostics.ARTIFACTS), "complete.json")},
              "statistics": STATISTICS, "counts": COUNTS}
    return root, inputs, reference, scores, diagnostic, models, linear_model, runtime_receipts


def _input_digest(inputs):
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _csv(path, rows, inputs):
    if not rows:
        raise ValueError("A required result table is empty")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=[*rows[0], "scope_id", "inputs_sha256"])
    writer.writeheader()
    for row in rows:
        writer.writerow({**row, "scope_id": SCOPE, "inputs_sha256": _input_digest(inputs)})
    _write_bytes(path, buffer.getvalue().encode("utf-8"))


def _summary_rows(summary, **prefix):
    rows = []
    for method in METHODS:
        row = {**prefix, "method": method,
               "score_vector_count": summary[method]["score_vector_count"],
               "training_seed_observations": summary[method]["training_seed_observations"]}
        for name in METRIC_NAMES:
            value = summary[method][name]
            row.update({name + "_mean": value["mean"], name + "_sample_sd": value["sample_sd"],
                        name + "_sd_reason": value["sample_sd_reason"]})
        rows.append(row)
    return rows


def _condition_rows(metadata, scores):
    masks = [(f"source_{source}_severity_{severity}",
              (metadata["source_id"] == source) & (metadata["severity"] == severity))
             for source, severity in VIEW_KEYS]
    masks += [(f"severity_{severity}", metadata["severity"] == severity) for severity in (0, 3, 5)]
    rows = []
    for name, mask in masks:
        if not mask.any():
            raise ValueError("Missing a prespecified nine-condition/severity slice")
        metrics = {key: score_metrics(metadata["y"][mask], values[mask])[0]
                   for key, values in scores.items()}
        rows.extend(_summary_rows(summarize_metrics(metrics), condition=name,
                                  records=int(mask.sum()),
                                  source_photographs=int(np.unique(metadata["image_id"][mask]).size)))
    return rows


def _cost_rows(receipts):
    rows = []
    for family in ("G", "H", "S", "O"):
        for seed in SEEDS:
            receipt = receipts[f"{family}/seed{seed}"]
            imported = family == "G" and seed in protocol.IMPORTED_SEEDS
            if receipt.get("imported") is not imported:
                raise ValueError("Prediction receipt import flag differs from the registered import set")
            # An imported receipt's elapsed_seconds times only the file copy, never inference.
            elapsed = None if imported else receipt["elapsed_seconds"] if "elapsed_seconds" in receipt else None
            if elapsed is not None and (not np.isfinite(elapsed) or elapsed <= 0):
                raise ValueError("Invalid recorded inference duration")
            memory = receipt["peak_gpu_bytes"] if "peak_gpu_bytes" in receipt else None
            not_recorded = ("imported (historical inference not re-timed)" if imported
                            else "not recorded in the immutable prediction receipt")
            rows.append({"family": family, "seed": seed, "imported": imported,
                         "detector_forwards_per_record": 1 if family == "O" else 4,
                         "records": 7200, "full_role_elapsed_seconds": elapsed,
                         "records_per_second_including_loading": None if elapsed is None else 7200 / elapsed,
                         "amortized_ms_per_record_including_loading": None if elapsed is None else elapsed * 1000 / 7200,
                         "gpu_peak_memory_bytes": memory,
                         "duration_reason": None if elapsed is not None else not_recorded,
                         "memory_reason": None if memory is not None else not_recorded,
                         "timing_scope": "entire exported family/seed group; not isolated batch latency",
                         "provenance": "historical import" if imported else "new fixed20 fit"})
    return rows


def _contrast_rows(comparisons):
    rows = []
    for name, value in comparisons.items():
        interval = value["interval_95"]
        adjusted = value["interval_bonferroni"] if "interval_bonferroni" in value else None
        rows.append({"contrast": name, "mean_paired_auroc_difference": value["estimate"],
                     "paired_difference_sample_sd": value["paired_difference_sample_sd"],
                     **{f"seed{seed}": value["per_seed"][str(seed)] for seed in SEEDS},
                     "ordinary_lower": None if interval is None else interval[0],
                     "ordinary_upper": None if interval is None else interval[1],
                     "bonferroni_lower": None if adjusted is None else adjusted[0],
                     "bonferroni_upper": None if adjusted is None else adjusted[1],
                     "undefined_draw_count": value["undefined_draw_count"]})
    return rows


def _number(value, digits=5):
    return "N/A" if value is None else f"{value:.{digits}f}"


def _interval(value):
    return "withheld (undefined draw)" if value is None else f"[{value[0]:.5f}, {value[1]:.5f}]"


def _method_table(result):
    text = "| Method | AUROC mean | Seed sample SD | AP mean | AURC mean |\n|---|---:|---:|---:|---:|\n"
    for method in METHODS:
        values = result["method_summary"][method]
        text += (f"| {method} | {_number(values['auroc']['mean'])} | "
                 f"{_number(values['auroc']['sample_sd'])} | "
                 f"{_number(values['average_precision']['mean'])} | {_number(values['aurc']['mean'])} |\n")
    return text


def _primary_table(result):
    text = "| Fixed contrast | Mean paired ΔAUROC | Ordinary 95% | Bonferroni, family of six |\n|---|---:|---|---|\n"
    for name in PRIMARY_NAMES:
        value = result["primary"][name]
        text += (f"| {name} | {_number(value['estimate'])} | {_interval(value['interval_95'])} | "
                 f"{_interval(value['interval_bonferroni'])} |\n")
    return text


def _diagnostic_table(result):
    text = "| Family | Selected epoch20 / all | Warning / all | Mean ΔAUC 20−15 | Mean loss drop | Mean drop % | Mean tail slope |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for family in ("G", "H", "S", "O"):
        group = result["training_diagnostics"]["summary"]["by_family"][family]
        text += (f"| {family} | {group['selected_epoch_20_count']}/{group['denominator']} | "
                 f"{group['budget_sensitivity_warning_count']}/{group['denominator']} | "
                 f"{_number(group['late_auc_change']['mean'])} | "
                 f"{_number(group['loss_decrease_absolute']['mean'])} | "
                 f"{_number(group['loss_decrease_percent']['mean'])} | "
                 f"{_number(group['loss_slope_16_20']['mean'])} |\n")
    return text


def _reports(result):
    english = "# Final fixed20 development comparison\n\n"
    english += ("25 reported methods on the same **800 photographs × 9 views = 7,200 records**. "
                "Target: frozen ViT classification error; higher detector scores indicate error. "
                f"All {COUNTS['neural_fits']} neural fits completed exactly 20 epochs: "
                f"{COUNTS['imported_neural_fits']} compatible historical G imports and "
                f"{COUNTS['fresh_neural_fits']} fresh fits, at seeds {SEED_LIST}. "
                f"There are {COUNTS['meta_heads']} meta heads ({COUNTS['imported_meta_heads']} imported, "
                f"{COUNTS['fresh_meta_heads']} new), "
                "one deterministic **L / logits_linear_100** fit, and two analytic baselines. "
                "The 1,600/400/400/800 base/checkpoint/meta/development photograph roles and all nine views are unchanged.\n\n")
    english += "## All registered methods\n\n" + _method_table(result)
    english += ("\nRaw columns block2/5/8/11 correspond to human layers 3/6/9/12. "
                "`*_mean` is the equal mean of four sigmoid scores, not sigmoid(mean logits). "
                "`*_stack` and `*_last_only` use their separately meta-fitted StandardScalers and signed "
                "L2 logistic decision functions. Negative coefficients are not flipped. O is the raw MLP "
                "decision score. L/MSP/entropy each have **one score vector**, so training-seed SD is **not applicable**, "
                f"not zero. Seeded table entries are arithmetic means of {SEED_WORD} metrics, never metrics of averaged scores "
                "or pooled seed-image rows. AP is sklearn average precision (not trapezoidal PR area). "
                "AURC accepts low error scores first; ties use expected uniformly random order. With distinct scores "
                "this equals the repository cumulative-risk mean. All coverages and per-seed metrics are saved.\n\n")
    english += "## Six primary comparisons, without winner selection\n\n" + _primary_table(result)
    english += ("\n10,000 shared image_id bootstrap draws use RNG seed 20260914. Every photograph's nine views "
                f"inherit its integer multiplicity, identically across methods and seeds. Each draw averages {SEED_WORD} "
                "within-seed paired AUROC differences; tie pairs receive half credit. Linear percentile quantiles "
                "are 0.025/0.975 and **0.004166666666666667/0.9958333333333333** for the six-comparison "
                "Bonferroni family. Adjusted family coverage is nominal/approximate, not an exact guarantee. "
                "Undefined draws are recorded, never redrawn or dropped; an affected CI is withheld. "
                "Per-seed CIs are descriptive and reuse these same draws. No p-values or positive-result completion rule are used.\n\n")
    secondary = result["secondary"]["L_minus_O"]
    english += ("## Secondary L versus O\n\n"
                f"Mean paired L−O ΔAUROC: **{_number(secondary['estimate'])}**; ordinary paired 95% interval "
                f"**{_interval(secondary['interval_95'])}**. The same single L vector is paired with each O seed "
                "using the cached primary draws. This adds no seventh primary contrast and no independent L seed "
                "observations. Differences are descriptive under the specified linear and nonlinear recipes, "
                "not a causal isolation of nonlinearity. L uses 100 base_train-standardized logits, training-derived "
                "class weights and one converged fixed L2/lbfgs fit; it has no CV, checkpoint selection or artificial epoch history.\n\n")
    english += "## Frozen budget-sensitivity diagnostics\n\n" + _diagnostic_table(result)
    english += ("\nThe nonfatal warning is exactly: selected epoch20 AND actual checkpoint AUROC20 > AUROC15 "
                "AND training loss15 > loss20. It means possible budget sensitivity, not proven undertraining; "
                "absence is not proof of convergence. Loss reductions are signed in original loss units and percent, "
                "with the actual epochs16–20 OLS slope. A zero loss15 has an explicitly undefined percentage. "
                f"All {COUNTS['neural_fits']} histories, all seeds (including imported late G7), layer/family denominators and complete "
                "curves are frozen in `../diagnostics/training_diagnostics.json` and `.csv`, whose three-file receipt "
                "is bound below. No warning authorizes new training, exclusion, replacement, changed selection or retuning. "
                "Architectural conclusions remain **under the fixed20 training budget**, never intrinsic inferiority.\n\n")
    english += "| Fit | Selected | AUC15 | AUC20 | Loss15 | Loss20 | Loss drop | Drop % | Tail slope | Warning | Original status |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n"
    for row in result["training_diagnostics"]["fits"]:
        english += (f"| {row['model_key']} | {row['selected_epoch']} | {_number(row['checkpoint_auroc_15'])} | "
                    f"{_number(row['checkpoint_auroc_20'])} | {_number(row['training_loss_15'])} | "
                    f"{_number(row['training_loss_20'])} | {_number(row['loss_decrease_absolute'])} | "
                    f"{_number(row['loss_decrease_percent'])} | {_number(row['loss_slope_16_20'])} | "
                    f"{row['budget_sensitivity_warning']} | {row['original_status']} |\n")
    english += ("\n## Cost and descriptive breakdowns\n\n"
                "Source-derived parameter counts per detector are G=130,434, H=129,986, S=131,126, O=8,577; "
                "the allocated preflight is responsible for confirming them. Family/seed export durations and "
                "available memory measurements are preserved in `inference_cost.csv` and the bound prediction receipts. "
                "Full-role timings include loading and all four forwards for G/H/S; amortized record time is not "
                "isolated batch latency. Missing historical telemetry is labeled unavailable, never fabricated. "
                "Four-detector ensemble inference requires all four forwards plus averaging or a logistic head. "
                "The fixed nine-condition and severity tables in `conditions.csv` are descriptive, with no selected subgroup, "
                "new hypothesis family or evaluation-optimized threshold.\n\n## Claim ceiling and provenance\n\n")
    english += "".join(f"- {text}\n" for text in LIMITATIONS)
    english += (f"\nScope: `{SCOPE}`. Complete input identity: `{result['inputs_sha256']}`. "
                "All prediction, head, linear, diagnostic, cache, campaign and executed-source identities are "
                "listed in `report.json`; `complete.json` hashes every emitted artifact. Historical source files, "
                "timestamps and scopes are not rewritten. A completed rerun verifies inputs/artifacts rather than overwriting them.\n")
    hebrew = ("# השוואת הפיתוח המסכמת בתקציב fixed20\n\n"
              f"**25 שיטות**, אותן 800 תמונות מקור ותשע גרסאות לכל תמונה. כל {COUNTS['neural_fits']} המודלים העצביים "
              f"השלימו 20 epochs: ‏{COUNTS['imported_neural_fits']} ייבואי G היסטוריים "
              f"ו־{COUNTS['fresh_neural_fits']} אימונים חדשים, בזרעים {SEED_LIST}. "
              f"{COUNTS['meta_heads']} ראשי מטה ({COUNTS['imported_meta_heads']} מיובאים, "
              f"{COUNTS['fresh_meta_heads']} חדשים), התאמה יחידה של L על כל 100 הלוגיטים, ו־MSP/entropy "
              "ללא אימון. חלוקת התמונות היא 1600/400/400/800; לא נקראה קבוצת המבחן המקורית.\n\n"
              "## כל השיטות\n\n") + _method_table(result)
    hebrew += (f"\nהממוצעים הם ממוצעי מדדים בין {SEED_WORD_HE} הזרעים, לא מודל שממצע תחזיות בין זרעים. "
               "סטיית התקן היא מדגמית (ddof=1). עבור L, MSP ו־entropy היא **לא ישימה**, ולא אפס. "
               "ממוצע האנסמבל הוא ממוצע sigmoid של ארבעה ציונים; השילוב ובקרת השכבה האחרונה "
               "משתמשים בסקלרים נפרדים שנלמדו רק במטה. סימני המקדמים נשמרים, גם כשהם שליליים.\n\n"
               "## שש השוואות ראשיות קבועות\n\n") + _primary_table(result)
    hebrew += ("\n10,000 דגימות מזווגות לפי image_id, לא לפי סוג ההשחתה source_id, בזרע 20260914. "
               "כל תשע הגרסאות מקבלות אותו משקל בכל השיטות והזרעים. בכל דגימה מחשבים הפרש AUROC "
               f"בתוך כל זרע ורק אז ממוצע בין {SEED_WORD_HE} הזרעים. תיקון Bonferroni כולל **שש**, לא שבע, "
               "השוואות; הכיסוי המשפחתי נומינלי/מקורב. דגימות ללא שתי מחלקות נשמרות ללא דגימה מחדש "
               "והרווח המושפע אינו מוצג. אין p-values ואין תנאי הצלחה המבוסס על תוצאה חיובית.\n\n"
               f"L−O משני ותיאורי בלבד: {_number(secondary['estimate'])}, רווח 95% "
               f"{_interval(secondary['interval_95'])}. אותו וקטור L יחיד משמש בכל הזיווגים, "
               f"בלי להמציא {SEED_FITS_HE} אימונים ובלי להוסיף השוואה ראשית.\n\n"
               "## רגישות אפשרית לתקציב\n\n") + _diagnostic_table(result)
    hebrew += ("\nאזהרה מתקבלת רק אם נבחר epoch20, ה־AUROC בפועל גדל מ־15 ל־20 והפסד האימון ירד. "
               f"האזהרה אינה כישלון ואינה מוכיחה תת־אימון; היעדרה אינו הוכחת התכנסות. כל {COUNTS['neural_fits']} העקומות, "
               "הירידות ביחידות הפסד ובאחוזים, השיפוע בסוף האימון והמכנים לפי משפחה ושכבה נשמרו "
               "ב־diagnostics. הפסד אפס ב־epoch15 מקבל אחוז לא מוגדר עם סיבה. אין הארכה, החלפה, "
               "השמטת זרע או כוונון. מסקנות ארכיטקטוניות הן **תחת תקציב האימון fixed20** בלבד.\n\n"
               "נתוני הפיתוח וניתוחים קודמים כבר נחשפו, ולכן אין כאן מבחן עצמאי ובלתי־נגוע. "
               "G בזרע 7 נשאר `complete_late_diagnostic` בפרוטוקול המקורי הלא־שלם. רווחי הדגימה "
               f"מותנים במודלים וב{SEED_WORD_HE} הזרעים; אינם מתקנים שימוש חוזר בנתונים או מודדים את מלוא שונות האימון. "
               "G מול H אינו בידוד של העברת מסרים (מידע הקלט שונה), ו־G מול S משנה גם את שיוך הקצוות "
               "ולכן אינו הוכחה כללית להכרחיות GNN. אנסמבל דורש ארבע העברות של גלאים. נתוני עלות "
               "וקצבי עבודה זמינים מדווחים ללא המצאת טלמטריה חסרה. טבלאות התנאים והחומרות תיאוריות "
               "בלבד; אין סף שנבחר על סמך תוויות ההערכה. אין טענה להכללה למודלי שפה.\n\n"
               f"מזהה כל הקלטים: `{result['inputs_sha256']}`. פירוט מלא באנגלית וב־`report.json`; "
               "`complete.json` קושר כל פלט למקור, לקמפיין ולשער ההערכה הקפוא.\n")
    return english, hebrew


def _completed(root, inputs, metadata, scores):
    directory = root / "evaluation"
    complete = protocol.read(directory / "complete.json")
    _match(complete, {"scope_id": SCOPE, "inputs": inputs, "counts": COUNTS}, "Evaluation completion")
    _verify_files(directory, complete, ARTIFACTS)
    report = protocol.read(directory / "report.json")
    _match(report, {"scope_id": SCOPE, "complete": True, "inputs": inputs,
                   "inputs_sha256": _input_digest(inputs), "counts": COUNTS,
                   "methods": list(METHODS), "seeds": list(SEEDS), "statistics": STATISTICS},
           "Completed report")
    if (set(report["method_summary"]) != set(METHODS)
            or set(report["per_vector_metrics"]) != set(SCORE_KEYS)
            or set(report["primary"]) != set(PRIMARY_NAMES)):
        raise RuntimeError("The completed report is a partial method/seed/contrast matrix")
    with np.load(directory / "scores.npz", allow_pickle=False) as archive:
        _align_metadata(metadata, archive)
        if (archive["score_keys"].tolist() != list(SCORE_KEYS)
                or archive["scores"].shape != (7200, len(SCORE_KEYS))
                or archive["inputs_sha256"].item() != _input_digest(inputs)):
            raise RuntimeError("Completed scores are not the exact aligned comparison")
        for column, key in enumerate(SCORE_KEYS):
            if not np.allclose(archive["scores"][:, column], scores[key], atol=1e-12, rtol=1e-12):
                raise RuntimeError("Completed score reconstruction changed: " + key)
    return report


def evaluate(root, out=None):
    protocol.require_slurm()
    root = Path(root).resolve()
    directory = root / "evaluation"
    if out is not None and Path(out).resolve() != directory:
        raise ValueError("Evaluation output must be the canonical root/evaluation directory")
    collected = _collect(root)
    root, inputs, metadata, scores, diagnostic, models, linear_model, runtime_receipts = collected
    with _lock(directory, ".evaluate.lock"):
        if (directory / "complete.json").exists():
            return _completed(root, inputs, metadata, scores)
        if any((directory / name).exists() for name in ARTIFACTS):
            raise RuntimeError("Incomplete evaluation artifacts are preserved, not overwritten")
        protocol.check_cutoff(root, "evaluation")
        per_vector, summary, risk = point_metrics(metadata["y"], scores)
        bootstrap, image_ids, multiplicities, draws = paired_bootstrap(
            metadata["y"], scores, metadata["image_id"],
            check_cutoff=lambda: protocol.check_cutoff(root, "evaluation"))
        if any(value["estimate"] is None for value in bootstrap["primary"].values()):
            raise RuntimeError("A required primary point estimate is undefined")
        condition_rows = _condition_rows(metadata, scores)
        cost_rows = _cost_rows(runtime_receipts)
        for name, (left, right) in zip(PRIMARY_NAMES, PRIMARY):
            from_points = _mean_seed_difference({key: value["auroc"] for key, value in per_vector.items()},
                                                left, right)
            _match(bootstrap["primary"][name], from_points, "Primary point AUROC")
        stamp = {"inputs_sha256": np.asarray(_input_digest(inputs)), "scope_id": np.asarray(SCOPE)}
        _write_npz(directory / "scores.npz", {
            **metadata, "score_keys": np.asarray(SCORE_KEYS),
            "scores": np.stack([scores[key] for key in SCORE_KEYS], axis=1), **stamp})
        _write_npz(directory / "risk_coverage.npz", {**risk, **stamp})
        _write_npz(directory / "bootstrap_source_counts.npz",
                   {"image_id": image_ids, "multiplicity": multiplicities, **stamp})
        _write_npz(directory / "bootstrap_draws.npz", {**draws, **stamp})
        _write_json(directory / "bootstrap.json",
                    {**bootstrap, "inputs": inputs, "statistics": STATISTICS,
                     "draws_sha256": protocol.sha256(directory / "bootstrap_draws.npz"),
                     "counts_sha256": protocol.sha256(directory / "bootstrap_source_counts.npz")})
        per_seed_rows = []
        for key in SCORE_KEYS:
            method, seed = key.split("/seed") if "/seed" in key else (key, None)
            per_seed_rows.append({"method": method, "seed": None if seed is None else int(seed),
                                  "independent_static_fit_count": (1 if key == "L" else 0)
                                  if seed is None else None, **per_vector[key]})
        _csv(directory / "per_seed_metrics.csv", per_seed_rows, inputs)
        _csv(directory / "method_summary.csv", _summary_rows(summary), inputs)
        _csv(directory / "primary_contrasts.csv", _contrast_rows(bootstrap["primary"]), inputs)
        _csv(directory / "secondary_contrasts.csv", _contrast_rows(bootstrap["secondary"]), inputs)
        _csv(directory / "conditions.csv", condition_rows, inputs)
        _csv(directory / "inference_cost.csv", cost_rows, inputs)
        solver_diagnostics = {
            f"{family}/seed{seed}/{name}": {
                "coef": model["model"]["coef"], "intercept": model["model"]["intercept"],
                "n_iter": model["model"]["n_iter"], "warnings": model["warnings"],
                "serialization_audit": model["serialization_audit"]}
            for (family, seed), group in models.items() for name, model in group.items()}
        solver_diagnostics["L"] = {key: linear_model[key]
                                  for key in ("model", "warnings", "converged", "serialization_audit", "recipe")}
        result = {"schema_version": 1, "scope_id": SCOPE, "complete": True, "status": "complete",
                  "inputs": inputs, "inputs_sha256": _input_digest(inputs), "counts": COUNTS,
                  "methods": list(METHODS), "seeds": list(SEEDS), "records": 7200,
                  # Top-level keys read by slurm.validate_evaluation_outputs.
                  "method_names": list(METHODS), "primary_contrasts": [list(pair) for pair in PRIMARY],
                  "bootstrap_draws": BOOTSTRAP_DRAWS, "campaign_sha256": inputs["campaign_sha256"],
                  "source_photographs": 800, "views_per_photograph": 9,
                  "positive_class": "frozen ViT classification error",
                  "positive": int(metadata["y"].sum()), "statistics": STATISTICS,
                  "per_vector_metrics": per_vector, "method_summary": summary,
                  "primary": bootstrap["primary"], "secondary": bootstrap["secondary"],
                  "training_diagnostics": {
                      "files": inputs["diagnostics"], "policy": diagnostics.POLICY,
                      "summary": diagnostic["summary"],
                      "fits": [{key: row[key] for key in diagnostics.CSV_FIELDS}
                               for row in diagnostic["fits"]]},
                  "solver_diagnostics": solver_diagnostics, "inference_cost": cost_rows,
                  "condition_table": "conditions.csv; fixed nine conditions and severities, descriptive only",
                  "test_evaluated": False, "development_data_previously_used": True,
                  "imported_G_seed7_status": "complete_late_diagnostic",
                  "limitations": LIMITATIONS, "completed_unix": time.time(),
                  "job_id": os.environ["SLURM_JOB_ID"]}
        protocol.check_cutoff(root, "evaluation")
        _write_json(directory / "report.json", result)
        english, hebrew = _reports(result)
        _write_bytes(directory / "REPORT.md", english.encode("utf-8"))
        _write_bytes(directory / "REPORT_HE.md", hebrew.encode("utf-8"))
        protocol.check_cutoff(root, "evaluation")
        _write_json(directory / "complete.json",
                    {"schema_version": 1, "scope_id": SCOPE, "complete": True,
                     "inputs": inputs, "counts": COUNTS,
                     "files": {name: protocol.sha256(directory / name) for name in sorted(ARTIFACTS)},
                     "completed_unix": time.time(), "job_id": os.environ["SLURM_JOB_ID"]})
        return _completed(root, inputs, metadata, scores)


def main():
    protocol.require_slurm()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = evaluate(args.root, args.out)
    print(json.dumps({"complete": result["complete"], "reported_methods": 25,
                      "primary_contrasts": 6, "linear_fits": 1}), flush=True)


if __name__ == "__main__":
    main()
