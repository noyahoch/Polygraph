"""Fixed-ID source-photo bootstrap, including refitting each fusion combiner."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .common import atomic_json, atomic_npz, load_campaign, lock, read, require_slurm, sha, verify_receipt
from .fusion import (ARMS, SEEDS, FitFailure, check_gate, design, fit_combiner, numerical,
                     parent_scores, predict_combiner, preserve_unsealed)

DRAWS = 2000
PRIMARY_QUANTILES = (0.0083333333, 0.9916666667)
SECONDARY_QUANTILES = (0.025, 0.975)
CONTRASTS = {"DG-D": ("fusion", "DG", "D", True),
             "DG-DS": ("fusion", "DG", "DS", True),
             "C-B": ("ablation", "C", "B", True),
             "DG-DDprime": ("fusion", "DG", "DDprime", False),
             "B-A": ("ablation", "B", "A", False),
             "D-C": ("ablation", "D", "C", False)}


def prepare_draws(root):
    np = numerical()
    root = Path(root)
    campaign = load_campaign(root)
    _, metadata, _, _ = parent_scores(root)
    groups = {"fusion_fit": np.sort(np.asarray(campaign["fusion_fit_photo_ids"], dtype=np.int64)),
              "fusion_assessment": np.sort(np.asarray(campaign["fusion_assessment_photo_ids"], dtype=np.int64)),
              "ablation_assessment": np.unique(metadata["image_id"])}
    if ([len(groups[name]) for name in groups] != [400, 400, 800]
            or len(np.intersect1d(groups["fusion_fit"], groups["fusion_assessment"]))
            or not np.array_equal(np.union1d(groups["fusion_fit"], groups["fusion_assessment"]), groups["ablation_assessment"])):
        raise RuntimeError("Invalid immutable source split for bootstrap")
    directory = root / "statistics"
    with lock(directory, "draws"):
        receipt_path = directory / "draw_manifest.json"
        if receipt_path.exists():
            return verify_receipt(root, receipt_path)
        sequences = np.random.SeedSequence(20260921).spawn(3)
        arrays, streams = {"draw_id": np.arange(DRAWS, dtype=np.int64)}, {}
        for stream, (name, photos) in enumerate(groups.items()):
            rng = np.random.default_rng(sequences[stream])
            multiplicity = np.empty((DRAWS, len(photos)), dtype=np.int32)
            for draw in range(DRAWS):
                multiplicity[draw] = np.bincount(rng.integers(0, len(photos), size=len(photos)), minlength=len(photos))
            arrays[name + "_image_id"] = photos
            arrays[name + "_multiplicity"] = multiplicity
            streams[name] = {"spawn_key": list(sequences[stream].spawn_key), "photographs": len(photos)}
        path = directory / "draw_manifest.npz"
        atomic_npz(path, **arrays)
        receipt = {"complete": True, "campaign_sha256": sha(root / "campaign.json"),
                   "draws": DRAWS, "draw_ids": "0..1999", "seed_sequence_entropy": 20260921,
                   "streams": streams, "photo_order": "ascending numeric image_id",
                   "primary_quantiles": list(PRIMARY_QUANTILES), "secondary_quantiles": list(SECONDARY_QUANTILES),
                   "files": {str(path.relative_to(root)): sha(path)}}
        atomic_json(receipt_path, receipt)
        return receipt


def metrics(y, scores, weights=None):
    np = numerical()
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, scores = np.asarray(y), np.asarray(scores, dtype=np.float64)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=np.float64)
    if (y.shape != scores.shape or w.shape != y.shape or not np.isin(y, [0, 1]).all()
            or not np.isfinite(scores).all() or not np.isfinite(w).all() or np.any(w < 0)):
        raise RuntimeError("Malformed metric inputs")
    positive, negative = float(w[y == 1].sum()), float(w[y == 0].sum())
    return {"auroc": float(roc_auc_score(y, scores, sample_weight=w)) if positive and negative else None,
            "average_precision": float(average_precision_score(y, scores, sample_weight=w)) if positive else None,
            "auroc_na_reason": None if positive and negative else "Both positive-weight error outcomes are required",
            "average_precision_na_reason": None if positive else "No positive-weight error examples",
            "error_count": positive, "correct_count": negative}


def photo_membership(metadata, groups):
    np = numerical()
    photos = metadata["image_id"]
    inventory, count = np.unique(photos, return_counts=True)
    if not np.array_equal(inventory, groups) or not np.all(count == 9):
        raise RuntimeError("Expected nine rows for every declared source photograph")
    return np.searchsorted(groups, photos)


def literal_photo_rows(metadata, groups, counts):
    np = numerical()
    photo_membership(metadata, groups)
    return np.concatenate([np.flatnonzero(metadata["image_id"] == photo)
                           for photo, count in zip(groups, counts) for _ in range(int(count))])


def read_draw(path, draw, identity):
    saved = read(path)
    if saved.get("draw_id") != draw or saved.get("identity") != identity or saved.get("processed") is not True:
        raise RuntimeError("Stored fixed draw identity/status changed: " + str(draw))
    return saved


def archive(path, metadata_keys):
    np = numerical()
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name].copy() for name in saved.files}


def load_inputs(root):
    np = numerical()
    root = Path(root)
    campaign, metadata, parent_keys, parent_matrix = parent_scores(root)
    check_gate(root)
    for path in (root / "fusion" / "freeze.json", root / "predictions" / "fusion_complete.json",
                 root / "predictions" / "ablation_complete.json", root / "statistics" / "draw_manifest.json"):
        verify_receipt(root, path)
    fit = archive(root / "fusion" / "fit.npz", campaign["metadata_keys"])
    fusion = archive(root / "predictions" / "fusion.npz", campaign["metadata_keys"])
    ablation = archive(root / "predictions" / "ablation.npz", campaign["metadata_keys"])
    draws = archive(root / "statistics" / "draw_manifest.npz", [])
    for name, data, selector, records in (
            ("fit", fit, np.isin(metadata["image_id"], campaign["fusion_fit_photo_ids"]), 3600),
            ("fusion", fusion, np.isin(metadata["image_id"], campaign["fusion_assessment_photo_ids"]), 3600),
            ("ablation", ablation, np.ones(7200, dtype=bool), 7200)):
        for key in campaign["metadata_keys"]:
            if data[key].dtype != np.int64 or data[key].shape != (records,) or not np.array_equal(data[key], metadata[key][selector]):
                raise RuntimeError("Prediction metadata differ from immutable parent: " + name + "/" + key)
        if name != "ablation":
            if (data["input_score_keys"].tolist() != parent_keys
                    or not np.array_equal(data["input_scores"], parent_matrix[selector])):
                raise RuntimeError("Fusion supplied score definition changed")
        if name != "fit":
            keys = data["score_keys"].tolist()
            expected_keys = {f"{method}/seed{seed}" for seed in SEEDS
                             for method in (("D", "G", "S", *ARMS) if name == "fusion" else ("A", "B", "C", "D"))}
            if set(keys) != expected_keys or len(keys) != len(expected_keys) or data["scores"].shape != (records, len(keys)) or not np.isfinite(data["scores"]).all():
                raise RuntimeError("Incomplete prediction score inventory: " + name)
            for seed in SEEDS:
                if not np.array_equal(data["scores"][:, keys.index(f"D/seed{seed}")], parent_matrix[selector, parent_keys.index(f"D/seed{seed}")]):
                    raise RuntimeError("Original D score was changed")
    if not np.array_equal(draws["draw_id"], np.arange(DRAWS)):
        raise RuntimeError("Fixed bootstrap draw IDs changed")
    for name, data, size in (("fusion_fit", fit, 400), ("fusion_assessment", fusion, 400), ("ablation_assessment", ablation, 800)):
        groups, counts = draws[name + "_image_id"], draws[name + "_multiplicity"]
        photo_membership(data, groups)
        if (counts.shape != (DRAWS, size) or not np.issubdtype(counts.dtype, np.integer)
                or np.any(counts < 0) or not np.all(counts.sum(axis=1) == size)):
            raise RuntimeError("Malformed stored photograph multiplicities")
    return campaign, fit, fusion, ablation, draws


def within_photo(y, scores, image_ids):
    np = numerical()
    values = []
    for photo in np.unique(image_ids):
        selected = image_ids == photo
        positive, negative = scores[selected & (y == 1)], scores[selected & (y == 0)]
        if not len(positive) or not len(negative):
            continue
        comparison = positive[:, None] - negative[None, :]
        values.append(float(((comparison > 0) + 0.5 * (comparison == 0)).mean()))
    return {"value": float(np.mean(values)) if values else None, "eligible_photographs": len(values),
            "total_photographs": len(np.unique(image_ids)),
            "na_reason": None if values else "No source photograph has both correct and incorrect views"}


def diagnostics(data):
    np = numerical()
    keys, matrix = data["score_keys"].tolist(), data["scores"]
    conditions = sorted(set(zip(data["source_id"].tolist(), data["severity"].tolist())))
    if len(conditions) != 9 or len(set(zip(data["image_id"].tolist(), data["source_id"].tolist(), data["severity"].tolist()))) != len(data["y"]):
        raise RuntimeError("Diagnostic cohort must contain all nine distinct conditions per photograph")
    output = {"condition_definition": ["source_id", "severity"], "conditions": [], "within_photo": {}}
    for family, severity in conditions:
        selected = (data["source_id"] == family) & (data["severity"] == severity)
        row = {"source_id": family, "severity": severity, "records": int(selected.sum()),
               "error_count": int(data["y"][selected].sum()), "correct_count": int((1-data["y"][selected]).sum()),
               "per_seed": {}, "seed_means": {}}
        for index, key in enumerate(keys):
            row["per_seed"][key] = metrics(data["y"][selected], matrix[selected, index])
        for method in ("D", "G", "S", *ARMS):
            row["seed_means"][method] = {}
            for metric in ("auroc", "average_precision"):
                values = [row["per_seed"][f"{method}/seed{seed}"][metric] for seed in SEEDS]
                row["seed_means"][method][metric] = float(np.mean(values)) if all(v is not None for v in values) else None
        output["conditions"].append(row)
    for index, key in enumerate(keys):
        output["within_photo"][key] = within_photo(data["y"], matrix[:, index], data["image_id"])
    output["within_photo_seed_means"] = {}
    for method in ("D", "G", "S", *ARMS):
        values = [output["within_photo"][f"{method}/seed{seed}"]["value"] for seed in SEEDS]
        output["within_photo_seed_means"][method] = float(np.mean(values)) if all(v is not None for v in values) else None
    return output


def point_results(fusion, ablation):
    np = numerical()
    result = {"cohorts": {}, "contrasts": {}, "point_estimate_source": "Original fitted models; never the bootstrap-refit mean"}
    for name, data, methods in (("fusion", fusion, ("D", "G", "S", *ARMS)), ("ablation", ablation, ("A", "B", "C", "D"))):
        keys = data["score_keys"].tolist()
        point = {key: metrics(data["y"], data["scores"][:, i]) for i, key in enumerate(keys)}
        if any(row["auroc"] is None for row in point.values()):
            raise RuntimeError("Point-estimate cohort lacks both error outcomes")
        summary = {}
        for method in methods:
            summary[method] = {}
            for metric in ("auroc", "average_precision"):
                values = [point[f"{method}/seed{seed}"][metric] for seed in SEEDS]
                summary[method][metric] = {"mean": float(np.mean(values)), "seed_sd": float(np.std(values, ddof=1))}
        result["cohorts"][name] = {"records": len(data["y"]), "photographs": len(np.unique(data["image_id"])),
            "error_count": int(data["y"].sum()), "error_prevalence": float(data["y"].mean()),
            "per_seed": point, "method_summary": summary}
    for name, (cohort, left, right, primary) in CONTRASTS.items():
        values = {str(seed): result["cohorts"][cohort]["per_seed"][f"{left}/seed{seed}"]["auroc"] -
                            result["cohorts"][cohort]["per_seed"][f"{right}/seed{seed}"]["auroc"] for seed in SEEDS}
        result["contrasts"][name] = {"cohort": cohort, "primary": primary, "per_seed": values,
                                     "estimate": float(np.mean(list(values.values())))}
    result["diagnostics"] = diagnostics(fusion)
    return result


def verify_weighted_duplication(fit, assessment, draws):
    np = numerical()
    keys = fit["input_score_keys"].tolist()
    member = photo_membership(fit, draws["fusion_fit_image_id"])
    rows = []
    for draw in (0, 1, 7, 17, 27):
        weight = draws["fusion_fit_multiplicity"][draw][member]
        repeated = literal_photo_rows(fit, draws["fusion_fit_image_id"], draws["fusion_fit_multiplicity"][draw])
        assessment_weight = draws["fusion_assessment_multiplicity"][draw][photo_membership(assessment, draws["fusion_assessment_image_id"])]
        assessment_repeated = literal_photo_rows(assessment, draws["fusion_assessment_image_id"], draws["fusion_assessment_multiplicity"][draw])
        for seed in SEEDS:
            for arm in ARMS:
                x = design(fit["input_scores"], keys, seed, arm)
                target = design(assessment["input_scores"], keys, seed, arm)
                weighted = fit_combiner(x, fit["y"], weight)
                literal = fit_combiner(x[repeated], fit["y"][repeated])
                for field in ("mean", "scale"):
                    if not np.allclose(weighted[field], literal[field], atol=1e-12, rtol=1e-12):
                        raise RuntimeError("Weighted/literal scaler equivalence failed")
                left, right = predict_combiner(weighted, target), predict_combiner(literal, target)
                if not np.allclose(left, right, atol=1e-6, rtol=1e-6):
                    raise RuntimeError("Weighted/literal decision-score equivalence failed")
                same_vector_checks = {}
                assessed = {}
                for label, score in (("weighted_fit", left), ("literal_fit", right)):
                    weighted_metrics = metrics(assessment["y"], score, assessment_weight)
                    duplicated_metrics = metrics(assessment["y"][assessment_repeated], score[assessment_repeated])
                    assessed[label] = {"weighted": weighted_metrics, "duplicated": duplicated_metrics}
                    same_vector_checks[label] = {}
                    for metric_name in ("auroc", "average_precision"):
                        a, b = weighted_metrics[metric_name], duplicated_metrics[metric_name]
                        if a is None or b is None:
                            if a is not None or b is not None:
                                raise RuntimeError("Weighted/literal same-score undefined metric mismatch")
                            difference = None
                        else:
                            difference = abs(a-b)
                            if not np.isclose(a, b, atol=1e-12, rtol=0):
                                raise RuntimeError("Weighted/literal same-score assessment metric equivalence failed")
                        same_vector_checks[label][metric_name] = {
                            "passed": True, "absolute_difference": difference,
                            "both_undefined": a is None, "atol": 1e-12, "rtol": 0}
                rows.append({"draw_id": draw, "seed": seed, "arm": arm,
                    "maximum_decision_difference": float(np.max(np.abs(left-right))),
                    "weighted_state": weighted, "literal_state": literal,
                    "weighted_assessment_metrics": assessed["weighted_fit"]["weighted"],
                    "literal_assessment_metrics": assessed["literal_fit"]["duplicated"],
                    "same_score_weighted_duplicated_metrics": assessed,
                    "same_score_metric_equivalence": same_vector_checks,
                    "ranking_diagnostics": ranking_diagnostics(left[assessment_repeated], right[assessment_repeated]),
                    "metric_differences_are_diagnostic": True})
    return {"complete": True, "passed": True, "draw_ids": [0, 1, 7, 17, 27], "comparisons": rows,
            "scaler_atol_rtol": 1e-12, "decision_atol_rtol": 1e-6,
            "same_score_metric_atol": 1e-12, "same_score_metric_rtol": 0,
            "no_real_near_tie_metric_equality_assumed": True}


def ranking_diagnostics(left, right):
    """Exact ordering diagnostics; near ties are never rounded or merged."""
    np = numerical()
    ranks, tied = [], []
    for scores in (left, right):
        _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
        ranks.append((np.cumsum(counts) - (counts + 1) / 2)[inverse])
        tied.append(int(counts[counts > 1].sum()))
    return {"records": len(left), "stable_sorted_positions_differ": int(np.count_nonzero(
                np.argsort(left, kind="stable") != np.argsort(right, kind="stable"))),
            "average_rank_records_differ": int(np.count_nonzero(ranks[0] != ranks[1])),
            "maximum_average_rank_difference": float(np.max(np.abs(ranks[0]-ranks[1]))),
            "weighted_fit_tied_records": tied[0], "literal_fit_tied_records": tied[1]}


def one_draw(draw, fit, fusion, ablation, draws, identity):
    np = numerical()
    wf = draws["fusion_fit_multiplicity"][draw][photo_membership(fit, draws["fusion_fit_image_id"])]
    wa = draws["fusion_assessment_multiplicity"][draw][photo_membership(fusion, draws["fusion_assessment_image_id"])]
    wb = draws["ablation_assessment_multiplicity"][draw][photo_membership(ablation, draws["ablation_assessment_image_id"])]
    result = {"draw_id": draw, "identity": identity, "processed": True, "models": {}, "failures": {}, "contrasts": {}}
    metric = {"fusion": {}, "ablation": {}}
    fkeys, akeys = fusion["score_keys"].tolist(), ablation["score_keys"].tolist()
    def assess(cohort, key, y, score, weight):
        try:
            from sklearn.metrics import roc_auc_score
            if any(float(weight[y == label].sum()) == 0 for label in (0, 1)):
                raise FitFailure("Assessment draw has only one positive-weight outcome class")
            value = float(roc_auc_score(y, score, sample_weight=weight))
            if not np.isfinite(value):
                raise FitFailure("Nonfinite assessment AUROC")
            return value
        except Exception as error:
            result["failures"][cohort + "/" + key] = {"type": type(error).__name__, "message": str(error)}
            return None
    for seed in SEEDS:
        key = f"D/seed{seed}"
        metric["fusion"][key] = assess("fusion", key, fusion["y"], fusion["scores"][:, fkeys.index(key)], wa)
        for arm in ARMS:
            key = f"{arm}/seed{seed}"
            try:
                state = fit_combiner(design(fit["input_scores"], fit["input_score_keys"].tolist(), seed, arm), fit["y"], wf)
                predicted = predict_combiner(state, design(fusion["input_scores"], fusion["input_score_keys"].tolist(), seed, arm))
                result["models"][key] = state
                metric["fusion"][key] = assess("fusion", key, fusion["y"], predicted, wa)
            except Exception as error:
                result["failures"][key] = {"type": type(error).__name__, "message": str(error)}
                metric["fusion"][key] = None
        for arm in ("A", "B", "C", "D"):
            key = f"{arm}/seed{seed}"
            metric["ablation"][key] = assess("ablation", key, ablation["y"], ablation["scores"][:, akeys.index(key)], wb)
    result["per_seed_auroc"] = metric
    for name, (cohort, left, right, _primary) in CONTRASTS.items():
        values = []
        for seed in SEEDS:
            a, b = metric[cohort][f"{left}/seed{seed}"], metric[cohort][f"{right}/seed{seed}"]
            values.append(a-b if a is not None and b is not None else None)
        valid = all(value is not None for value in values)
        result["contrasts"][name] = {"valid": valid, "per_seed": values,
                                       "mean": float(np.mean(values)) if valid else None}
    return result


def intervals(points, completed):
    np = numerical()
    if len(completed) != DRAWS or [row["draw_id"] for row in completed] != list(range(DRAWS)):
        raise RuntimeError("All fixed draw IDs must be present before interval aggregation")
    output = {}
    for name, (_cohort, _left, _right, primary) in CONTRASTS.items():
        invalid = [row["draw_id"] for row in completed if not row["contrasts"][name]["valid"]]
        quantiles = PRIMARY_QUANTILES if primary else SECONDARY_QUANTILES
        values = [row["contrasts"][name]["mean"] for row in completed]
        output[name] = {**points["contrasts"][name], "draws": DRAWS, "valid_draws": DRAWS-len(invalid),
            "invalid_draw_ids": invalid, "quantiles": list(quantiles), "percentile_method": "linear",
            "interval": None if invalid else np.quantile(values, quantiles, method="linear").tolist(),
            "interval_withheld_reason": "At least one required draw is undefined or unsuccessful" if invalid else None,
            "claim_scope": "Nominal 95% Bonferroni family across three primary contrasts" if primary else "Descriptive 95%; no extra family-controlled superiority claim"}
    return output


def run(root, stop_after=None):
    require_slurm()
    from threadpoolctl import threadpool_limits
    root = Path(root)
    if stop_after not in (None, 50):
        raise ValueError("Only the retained first-50 timing stage or complete 2000 draws are allowed")
    directory = root / "statistics"
    with lock(directory, "bootstrap"), threadpool_limits(limits=1):
        campaign, fit, fusion, ablation, draws = load_inputs(root)
        identity = {"campaign_sha256": sha(root / "campaign.json"),
                    "gate_sha256": sha(root / "evaluation_gate.json"),
                    "fusion_predictions_sha256": sha(root / "predictions" / "fusion.npz"),
                    "ablation_predictions_sha256": sha(root / "predictions" / "ablation.npz"),
                    "draw_manifest_sha256": sha(directory / "draw_manifest.npz"),
                    "statistics_source_sha256": sha(__file__), "fusion_source_sha256": sha(Path(__file__).with_name("fusion.py"))}
        if (directory / "complete.json").exists():
            receipt = verify_receipt(root, directory / "complete.json")
            if receipt.get("identity") != identity:
                raise RuntimeError("Completed statistical audit identity changed")
            return receipt
        equivalence_path = directory / "weighted_duplication.json"
        if equivalence_path.exists():
            equivalence = read(equivalence_path)
            if equivalence.get("identity") != identity or equivalence.get("complete") is not True or equivalence.get("passed") is not True:
                raise RuntimeError("Weighted/literal gate identity changed")
        else:
            from .test_stats import run_tests
            test_receipt = run_tests()
            check = verify_weighted_duplication(fit, fusion, draws)
            atomic_json(equivalence_path, {**check, "identity": identity, "synthetic_tests": test_receipt})
        points = point_results(fusion, ablation)
        target = 50 if stop_after == 50 else DRAWS
        began, fresh = time.monotonic(), 0
        for draw in range(target):
            path = directory / "draws" / f"{draw:04d}.json"
            if path.exists():
                saved = read_draw(path, draw, identity)
            else:
                started = time.monotonic()
                try:
                    saved = one_draw(draw, fit, fusion, ablation, draws, identity)
                except Exception as error:
                    saved = {"draw_id": draw, "identity": identity, "processed": True, "models": {},
                             "failures": {"unexpected_draw_error": {"type": type(error).__name__, "message": str(error)}},
                             "contrasts": {name: {"valid": False, "per_seed": [None, None, None], "mean": None}
                                           for name in CONTRASTS}}
                saved["elapsed_seconds"] = time.monotonic()-started
                atomic_json(path, saved)
                fresh += 1
            if (draw+1) % 50 == 0:
                print(json.dumps({"bootstrap_draw_ids_completed_through": draw, "target_draws": target}), flush=True)
        retained = [read_draw(directory / "draws" / f"{i:04d}.json", i, identity) for i in range(50)]
        failure_ids = [row["draw_id"] for row in retained if row["failures"]]
        unexpected_ids = [row["draw_id"] for row in retained
                          if any(failure["type"] != "FitFailure" for failure in row["failures"].values())]
        timing = {"complete": True, "campaign_sha256": sha(root / "campaign.json"), "identity": identity,
                  "completed_draws": 50, "completed_draw_ids": list(range(50)), "retained_draw_count": 50,
                  "newly_computed_count_this_invocation": fresh,
                  "elapsed_seconds_this_invocation": time.monotonic()-began,
                  "elapsed_seconds": sum(row["elapsed_seconds"] for row in retained),
                  "failure_draw_ids": failure_ids, "failure_draw_count": len(failure_ids),
                  "unexpected_error_draw_ids": unexpected_ids, "unexpected_error_draw_count": len(unexpected_ids),
                  "numerical_threads": 1, "workers": 1, "effects_omitted": True}
        timing["estimated_remaining_seconds"] = timing["elapsed_seconds"] * 1950 / 50
        timing_path = directory / "timing_50.json"
        if not timing_path.exists():
            atomic_json(timing_path, timing)
        elif read(timing_path).get("identity") != identity:
            raise RuntimeError("Retained timing receipt identity changed")
        if stop_after == 50:
            print(json.dumps({key: timing[key] for key in ("completed_draws", "newly_computed_count_this_invocation", "elapsed_seconds", "estimated_remaining_seconds", "effects_omitted")}), flush=True)
            return timing
        completed = [read_draw(directory / "draws" / f"{draw:04d}.json", draw, identity) for draw in range(DRAWS)]
        result = {"complete": True, "identity": identity, "points": points, "contrasts": intervals(points, completed),
                  "draws": DRAWS, "training_seed_resampling": False, "prior_development_reuse": True,
                  "fusion_uncertainty": "Fit-photograph and assessment-photograph sampling; fixed constituent models and partition",
                  "ablation_uncertainty": "Assessment-photograph sampling conditional on the fitted readouts and auxiliary heads",
                  "failed_fit_draw_ids": [row["draw_id"] for row in completed if row["failures"]],
                  "draw_identity": read(directory / "draw_manifest.json")}
        preserve_unsealed(root, [directory / "results.json"], "statistics-results")
        atomic_json(directory / "results.json", result)
        paths = [directory / "results.json", equivalence_path, directory / "draw_manifest.json", directory / "draw_manifest.npz", timing_path]
        paths += [directory / "draws" / f"{draw:04d}.json" for draw in range(DRAWS)]
        receipt = {"complete": True, "campaign_sha256": sha(root / "campaign.json"), "identity": identity,
                   "all_2000_draws_processed": True, "all_primary_intervals_available": all(result["contrasts"][name]["interval"] is not None for name in ("DG-D", "DG-DS", "C-B")),
                   "files": {str(path.relative_to(root)): sha(path) for path in paths}}
        atomic_json(directory / "complete.json", receipt)
        print(json.dumps({"statistics_complete": True, "draws": DRAWS, "all_primary_intervals_available": receipt["all_primary_intervals_available"]}), flush=True)
        return receipt
