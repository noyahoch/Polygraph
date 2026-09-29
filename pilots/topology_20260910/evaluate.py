"""Freeze, score once, and report the registered photograph-paired comparison.

All numerical work requires Slurm. image_id identifies the bootstrap photograph;
source_id identifies corruption and MUST NOT be used as the resampling group.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import time

import numpy as np

from .data import CachedDataset, check_freeze
from .protocol import ARMS, SEEDS, SOURCES, VIEWS, atomic_json, digest, file_sha256, require_slurm
from .train import (_artifact_hashes, _atomic_npz, _cache_identity, _implementation_identity,
                    _verify_complete, _verify_config, predict_split, validation_threshold)
from .rewiring_decision import LIMITATION, admission_bindings, read_admission

SCHEMA_VERSION = 1
BOOTSTRAP_SEED = 20260911
BOOTSTRAP_DRAWS = 2000
PRACTICAL_MARGIN = 0.005
FULL_ARMS = tuple(ARMS)
CORE_ARMS = ("full_graph", "full_rewired", "full_set", "full_endpoint", "logit")
METADATA = ("record_id", "image_id", "source_id", "severity", "split_id", "y", "pred", "label", "confidence", "margin")
CONTRASTS = {
    "full_graph_minus_full_set": {"full_graph": 1, "full_set": -1},
    "full_graph_minus_full_endpoint": {"full_graph": 1, "full_endpoint": -1},
    "full_graph_minus_full_rewired": {"full_graph": 1, "full_rewired": -1},
    "raw_graph_minus_raw_set": {"raw_graph": 1, "raw_set": -1},
    "full_gap_minus_raw_gap": {"full_graph": 1, "full_set": -1, "raw_graph": -1, "raw_set": 1},
}


def _read(path):
    return json.loads(Path(path).read_text())


def _atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _finite(values, name):
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite {name}")


def _code_identity():
    root = Path(__file__).resolve().parents[2]
    identity = _implementation_identity()
    identity["pilots/topology_20260910/evaluate.py"] = file_sha256(__file__)
    for name in ("extract", "rewire", "validate"):
        relative = f"pilots/topology_20260910/{name}.py"
        identity[relative] = file_sha256(root / relative)
    identity["docs/experiments/september10/PROTOCOL.md"] = file_sha256(root / "docs/experiments/september10/PROTOCOL.md")
    return identity


def _run_root(frozen, path):
    root = Path(frozen["run_root"])
    return root if root.is_absolute() else Path(path).resolve().parent / root


def _required_runs(arms):
    return {f"{arm}/seed{seed}" for arm in arms for seed in SEEDS}


def _approval(path):
    if path is None:
        raise RuntimeError("primary-only requires an explicit pre-test root approval JSON")
    value = _read(path)
    if (value.get("approved") is not True or value.get("matrix") != "primary-only"
            or value.get("approved_by") != "root" or value.get("test_results_visible") is not False
            or not value.get("reason") or not value.get("date")):
        raise RuntimeError("Reduction approval must declare approved/root/primary-only/date/reason and test_results_visible=false")
    return {"document": value, "sha256": file_sha256(path)}


def _check_matrix(frozen):
    matrix = frozen.get("matrix")
    arms = FULL_ARMS if matrix == "full" else CORE_ARMS if matrix == "primary-only" else ()
    if not arms or set(frozen.get("runs", {})) != _required_runs(arms):
        raise RuntimeError("Frozen matrix must include every declared arm and all five fixed seeds")
    if matrix == "primary-only":
        approval = frozen.get("reduction_approval", {}).get("document", {})
        if not (approval.get("approved") is True and approval.get("approved_by") == "root"
                and approval.get("test_results_visible") is False and approval.get("matrix") == matrix
                and approval.get("reason") and approval.get("date")):
            raise RuntimeError("Reduced frozen matrix lacks the recorded pre-test approval")
    return arms


def _check_frozen(cache, path):
    frozen = check_freeze(cache, path)
    if frozen.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Unknown evaluation freeze schema")
    _check_matrix(frozen)
    if frozen.get("implementation") != _code_identity() or frozen.get("implementation_sha256") != digest(_code_identity()):
        raise RuntimeError("Implementation/protocol text changed after freeze")
    if frozen.get("statistics") != statistical_protocol():
        raise RuntimeError("Statistical definitions changed after freeze")
    if frozen.get("cache_validation_sha256") != file_sha256(Path(cache) / "validation.json"):
        raise RuntimeError("Cache readiness receipt changed after freeze")
    if frozen.get("rewire_manifest_sha256") != file_sha256(Path(cache) / "rewire/manifest.json"):
        raise RuntimeError("Fixed rewiring manifest changed after freeze")
    if frozen.get("rewiring_status") != read_admission(cache, required=True, require_full=True):
        raise RuntimeError("Frozen full-cohort rewiring diagnostics/admission changed")
    for field in ("npz", "json"):
        reference = Path(path).resolve().parent / frozen["references"][field + "_path"]
        if file_sha256(reference) != frozen["references"][field + "_sha256"]:
            raise RuntimeError("Frozen validation reference thresholds or scores changed")
    return frozen


def statistical_protocol():
    return {"positive_class": "frozen classifier error y=(pred!=label)", "score_direction": "larger means error",
            "group": "image_id (original photograph), not source_id (corruption family)",
            "draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
            "resampling": "source photograph with replacement; all nine versions inherit its multiplicity",
            "pairing": "same photograph multiplicities for every arm and seed",
            "seed_estimand": "arithmetic mean of five seed-wise AUROC contrasts; not AUROC of averaged scores",
            "percentiles": [2.5, 97.5], "percentile_method": "numpy.percentile(method='linear')",
            "undefined_draws": "record without replacement or redraw; any undefined draw withholds interval",
            "practical_margin": PRACTICAL_MARGIN, "contrasts": CONTRASTS,
            "interaction_interpretation": "full_gap_minus_raw_gap < 0 means larger attention-only graph benefit",
            "confidence_cutoff": 0.9, "confidence_support_floor_per_outcome": 200,
            "threshold": "validation correct rank ceil(.95*n), one-based, alarm score > threshold",
            "secondary_inference": "prespecified, descriptive/exploratory; no multiplicity-corrected superiority claims",
            "numpy_version": np.__version__}


def _align(reference, candidate):
    for name in METADATA:
        if name not in candidate or not np.array_equal(reference[name], candidate[name]):
            raise RuntimeError(f"Record/metadata alignment mismatch: {name}")
    if len(np.unique(candidate["record_id"])) != len(candidate["record_id"]):
        raise RuntimeError("Duplicate record IDs")
    if not np.array_equal(candidate["y"], candidate["pred"] != candidate["label"]):
        raise RuntimeError("Labels are not frozen classifier errors")


def _verify_cohort(cache, metadata, split):
    expected = [row for row in _read(Path(cache) / "cohort.json")["records"] if row["split"] == split]
    if len(expected) != len(metadata["record_id"]):
        raise RuntimeError("Evaluation cohort record count mismatch")
    for name in ("record_id", "image_id", "source_id", "severity", "split_id"):
        if not np.array_equal(metadata[name], np.asarray([row[name] for row in expected])):
            raise RuntimeError(f"Evaluation differs from immutable cohort: {name}")
    ids, counts = np.unique(metadata["image_id"], return_counts=True)
    expected_photos = 800 if split in ("val", "test") else 2400
    if len(ids) != expected_photos or not np.all(counts == 9):
        raise RuntimeError("Every source photograph must contribute exactly nine views")
    expected_views = {(SOURCES.index(source), severity) for source, severity in VIEWS}
    for image in ids:
        selected = metadata["image_id"] == image
        if set(zip(metadata["source_id"][selected].tolist(), metadata["severity"][selected].tolist())) != expected_views:
            raise RuntimeError("A source photograph is missing a registered condition")


def analytic_scores(logits, confidence):
    logits = np.asarray(logits, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    _finite(logits, "reference logits")
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_prob = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    probability = np.exp(log_prob)
    if not np.allclose(probability.max(axis=1), confidence, atol=1e-5, rtol=1e-5):
        raise RuntimeError("Cached classifier MSP disagrees with saved logits")
    # Cached confidence is the original frozen classifier's float32 softmax MSP;
    # use it for both the MSP baseline and the prespecified confidence slice.
    return {"msp": 1.0 - confidence, "entropy": -(probability * log_prob).sum(axis=1)}


def freeze(cache, run_root, out, matrix="full", approval=None):
    require_slurm()
    cache, run_root, out = Path(cache), Path(run_root).resolve(), Path(out).resolve()
    if out.exists():
        frozen = _check_frozen(cache, out)
        if frozen["matrix"] != matrix or _run_root(frozen, out).resolve() != run_root:
            raise RuntimeError("Existing freeze is immutable and belongs to a different matrix/run root")
        return frozen
    manifest = _read(cache / "manifest.json")
    if not manifest.get("complete") or manifest.get("diagnostic_only"):
        raise RuntimeError("Freeze requires a complete non-diagnostic cache")
    readiness = _read(cache / "validation.json")
    if readiness.get("passed") is not True:
        raise RuntimeError("Cache readiness validation must pass before freezing")
    rewiring_status = read_admission(cache, required=True, require_full=True)
    if (readiness.get("cache_checks", {}).get("rewiring_status") != rewiring_status
            or readiness["cache_checks"].get("rewiring_bindings") != admission_bindings(cache)):
        raise RuntimeError("Readiness must bind the full-cohort rewiring diagnostics and explicit admission")
    reduction = _approval(approval) if matrix == "primary-only" else None
    arms = FULL_ARMS if matrix == "full" else CORE_ARMS if matrix == "primary-only" else ()
    if not arms:
        raise ValueError(matrix)
    val_ds = CachedDataset(cache, "val", "logit")
    metadata = val_ds.metadata()
    _align(metadata, metadata)
    _verify_cohort(cache, metadata, "val")
    bindings = {}
    for key in sorted(_required_runs(arms)):
        directory = run_root / key
        _verify_complete(directory)
        config = _read(directory / "config.json")
        _verify_config(config)
        if key != f"{config['arm']}/seed{config['seed']}":
            raise RuntimeError("Run directory does not match its arm/seed identity")
        for field, value in _cache_identity(cache, config["arm"]).items():
            if config[field] != value:
                raise RuntimeError("A contestant was trained on a different cohort/cache/protocol")
        prediction = _arrays(directory / "validation.npz")
        _align(metadata, prediction)
        threshold = _read(directory / "validation.json")
        recomputed = validation_threshold(prediction["y"], prediction["score"])
        if threshold["threshold"] != recomputed["threshold"] or threshold["preprocessing"] != config["preprocessing"]:
            raise RuntimeError("Saved validation threshold/scaler does not match the selected artifact")
        bindings[key] = {**_artifact_hashes(directory), "complete_sha256": file_sha256(directory / "complete.json")}
    refs = analytic_scores(val_ds.logits().numpy(), metadata["confidence"])
    reference_thresholds = {name: validation_threshold(metadata["y"], score) for name, score in refs.items()}
    for name in reference_thresholds:
        reference_thresholds[name]["score"] = "1 - frozen classifier MSP" if name == "msp" else "classifier probability entropy, natural log"
    out.parent.mkdir(parents=True, exist_ok=True)
    npz_path = out.with_name(out.stem + ".validation_references.npz")
    json_path = out.with_name(out.stem + ".validation_references.json")
    _atomic_npz(npz_path, {**metadata, **refs})
    atomic_json(json_path, {"schema_version": SCHEMA_VERSION, "split": "val", "thresholds": reference_thresholds,
                           "msp_precision": "cached frozen classifier float32 softmax; entropy float64 from saved logits"})
    implementation = _code_identity()
    frozen = {"schema_version": SCHEMA_VERSION, "frozen": True, "created_unix": time.time(),
              "matrix": matrix, "arms": list(arms), "seeds": list(SEEDS), "run_root": str(run_root), "runs": bindings,
              **_cache_identity(cache), "cache_validation_sha256": file_sha256(cache / "validation.json"),
              "rewire_manifest_sha256": file_sha256(cache / "rewire/manifest.json"),
              **admission_bindings(cache), "rewiring_status": rewiring_status,
              "implementation": implementation, "implementation_sha256": digest(implementation),
              "statistics": statistical_protocol(), "reduction_approval": reduction,
              "references": {"npz_path": npz_path.name, "npz_sha256": file_sha256(npz_path),
                             "json_path": json_path.name, "json_sha256": file_sha256(json_path)},
              "test_scores_visible": False}
    atomic_json(out, frozen)
    _check_frozen(cache, out)
    return frozen


class WeightedAUC:
    """Pre-sort once; source multiplicities reproduce expanded-row tied AUROC."""
    def __init__(self, labels, scores):
        labels, scores = np.asarray(labels), np.asarray(scores)
        if labels.shape != scores.shape or not np.isin(labels, [0, 1]).all():
            raise ValueError("AUC requires aligned binary errors and scores")
        _finite(scores, "AUC scores")
        self.order = np.argsort(scores, kind="stable")
        self.labels = labels[self.order].astype(np.float64)
        ordered = scores[self.order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(ordered)) + 1] if len(ordered) else np.array([], dtype=int)

    def __call__(self, weights=None):
        if not len(self.order):
            return None
        weights = np.ones(len(self.order)) if weights is None else np.asarray(weights, dtype=np.float64)
        if weights.shape != self.order.shape or not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Invalid bootstrap multiplicities")
        ordered = weights[self.order]
        positive = np.add.reduceat(ordered * self.labels, self.starts)
        negative = np.add.reduceat(ordered * (1 - self.labels), self.starts)
        positives, negatives = positive.sum(), negative.sum()
        if not positives or not negatives:
            return None
        preceding_negative = np.cumsum(negative) - negative
        return float(np.sum(positive * (preceding_negative + .5 * negative)) / (positives * negatives))


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def metrics(metadata, scores, threshold, mask=None):
    mask = np.ones(len(scores), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    labels, values = np.asarray(metadata["y"])[mask], np.asarray(scores)[mask]
    errors, correct = labels == 1, labels == 0
    rejected = values > threshold
    accepted = ~rejected
    return {"records": int(len(labels)), "source_photographs": int(len(np.unique(metadata["image_id"][mask]))),
            "errors": int(errors.sum()), "correct": int(correct.sum()), "error_prevalence": _ratio(errors.sum(), len(labels)),
            "auroc": WeightedAUC(labels, values)(), "threshold": float(threshold),
            "true_positive_count": int((errors & rejected).sum()), "false_positive_count": int((correct & rejected).sum()),
            "error_recall": _ratio((errors & rejected).sum(), errors.sum()),
            "false_alarm_rate": _ratio((correct & rejected).sum(), correct.sum()),
            "rejected_count": int(rejected.sum()), "rejection_fraction": _ratio(rejected.sum(), len(labels)),
            "accepted_count": int(accepted.sum()), "coverage": _ratio(accepted.sum(), len(labels)),
            "accepted_errors": int((errors & accepted).sum()), "accepted_risk": _ratio((errors & accepted).sum(), accepted.sum())}


def confidence_support(metadata):
    mask = np.asarray(metadata["confidence"]) >= .9
    wrong, correct = mask & (metadata["y"] == 1), mask & (metadata["y"] == 0)
    wrong_groups, correct_groups = len(np.unique(metadata["image_id"][wrong])), len(np.unique(metadata["image_id"][correct]))
    return {"cutoff": .9, "records": int(mask.sum()), "source_photographs_with_error": wrong_groups,
            "source_photographs_with_correct": correct_groups, "required_per_outcome": 200,
            "inferential_support": wrong_groups >= 200 and correct_groups >= 200,
            "interpretation": "support floor does not guarantee power; groups may occur in both outcome sets"}


def _contrast_values(aucs, arms):
    result = {}
    for name, terms in CONTRASTS.items():
        if not set(terms).issubset(arms):
            continue
        differences = [None if any(aucs[f"{arm}/seed{seed}"] is None for arm in terms)
                       else sum(coefficient * aucs[f"{arm}/seed{seed}"] for arm, coefficient in terms.items()) for seed in SEEDS]
        result[name] = {"seed_differences": differences,
                        "estimate": None if None in differences else float(np.mean(differences))}
    return result


def paired_bootstrap(metadata, scores, *, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """All arm/seed/slice contrasts share each original-photo bootstrap draw."""
    image_ids, membership = np.unique(metadata["image_id"], return_inverse=True)
    if not len(image_ids):
        raise ValueError("Cannot bootstrap an empty source cohort")
    arms = tuple(sorted({key.split("/")[0] for key in scores}))
    if set(scores) != _required_runs(arms):
        raise ValueError("Bootstrap requires all five seeds for each declared arm")
    support = confidence_support(metadata)
    masks = {"mixture": np.ones(len(membership), dtype=bool)}
    if support["inferential_support"]:
        masks["confidence_ge_0.9"] = metadata["confidence"] >= .9
    calculators = {scope: {key: WeightedAUC(metadata["y"][mask], value[mask]) for key, value in scores.items()}
                   for scope, mask in masks.items()}
    points = {scope: _contrast_values({key: calculator() for key, calculator in collection.items()}, arms)
              for scope, collection in calculators.items()}
    distributions = {scope: {name: [] for name in values} for scope, values in points.items()}
    multiplicities = np.empty((draws, len(image_ids)), dtype=np.int32)
    rng = np.random.default_rng(seed)
    for draw in range(draws):
        multiplicity = np.bincount(rng.integers(0, len(image_ids), size=len(image_ids)), minlength=len(image_ids))
        multiplicities[draw] = multiplicity
        row_weights = multiplicity[membership]
        for scope, collection in calculators.items():
            weighted = {key: calculator(row_weights[masks[scope]]) for key, calculator in collection.items()}
            contrasts = _contrast_values(weighted, arms)
            for name in contrasts:
                distributions[scope][name].append(contrasts[name]["estimate"])
    summaries = {}
    for scope, values in distributions.items():
        summaries[scope] = {}
        for name, samples in values.items():
            undefined = [index for index, value in enumerate(samples) if value is None]
            interval = None if undefined else np.percentile(samples, [2.5, 97.5], method="linear").tolist()
            estimate = points[scope][name]["estimate"]
            summaries[scope][name] = {**points[scope][name], "interval_95": interval,
                                    "undefined_draw_count": len(undefined), "undefined_draw_indices": undefined,
                                    "bootstrap_values": samples, "seed_sd": None if estimate is None else float(np.std(points[scope][name]["seed_differences"], ddof=1))}
    return {"draws": draws, "seed": seed, "group": "image_id", "source_photographs": len(image_ids),
            "percentile_method": "linear", "confidence_support": support,
            "scope": "source-sampling uncertainty conditional on the five fitted seeds", "comparisons": summaries}, image_ids, multiplicities


def practical_conclusion(primary):
    interval = primary["interval_95"]
    if interval is None:
        return "insufficient_support_no_confirmatory_interval"
    if interval[0] > PRACTICAL_MARGIN:
        return "supports_benefit_exceeding_0.005"
    if interval[1] < PRACTICAL_MARGIN:
        return "rules_out_benefit_as_large_as_0.005_for_this_comparison"
    return "practical_benefit_unresolved"


def _save_csv(path, rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "NA" if value is None else value for key, value in row.items()})
    _atomic_text(path, buffer.getvalue())


def report(results):
    require_slurm()
    results = Path(results)
    summary = _read(results / "summary.json")
    primary = summary["primary"]
    estimate = "לא מוגדר" if primary["estimate"] is None else f"{primary['estimate']:.4f}"
    interval = "לא מוצג עקב חוסר תמיכה" if primary["interval_95"] is None else f"[{primary['interval_95'][0]:.4f}, {primary['interval_95'][1]:.4f}]"
    meanings = {"insufficient_support_no_confirmatory_interval": "אין די תמיכה לחישוב רווח סמך מאשש.",
                "supports_benefit_exceeding_0.005": "רווח הסמך תומך ביתרון העולה על הסף שנקבע מראש: 0.005 AUROC.",
                "rules_out_benefit_as_large_as_0.005_for_this_comparison": "רווח הסמך שולל יתרון בגודל 0.005 AUROC בהשוואה שנבדקה; אין זו הוכחת שקילות כללית.",
                "practical_benefit_unresolved": "רווח הסמך חוצה את הסף המעשי 0.005 AUROC, ולכן השאלה המעשית אינה מוכרעת."}
    support = summary["confidence_support"]
    text = f"# תוצאות ניסוי September 10\n\nהמשימה היא זיהוי טעות של מסווג תמונות קפוא; תמונה משובשת אינה בהכרח טעות.\n\n"
    text += f"הפרש AUROC הראשי, full_graph פחות full_set: **{estimate}**; רווח סמך 95%: **{interval}**. {meanings[summary['practical_conclusion']]}\n\n"
    text += "האומדן הוא ממוצע חמשת הפרשי הזרעים, ולא AUROC של ממוצע תחזיות. 2,000 דגימות bootstrap משותפות נדגמו לפי התצלום המקורי (image_id), וכל תשע גרסאותיו נשמרו יחד. אי־הוודאות מותנית בחמש הרשתות שאומנו בכל תנאי.\n\n"
    text += f"חתך הביטחון MSP≥0.9 כולל {support['records']} רשומות, {support['source_photographs_with_error']} תצלומים עם טעות בטוחה ו־{support['source_photographs_with_correct']} עם ניבוי בטוח נכון. "
    text += "סף התמיכה להסקה מתקיים, אך אינו מבטיח עוצמה סטטיסטית.\n\n" if support["inferential_support"] else "סף התמיכה של 200 תצלומים לכל תוצאה אינו מתקיים; החתך מתואר בלבד.\n\n"
    if summary["matrix"] == "primary-only":
        text += "הניסוי עם מאפייני attention בלבד הושמט בשל החלטת משאבים מתועדת לפני צפייה בתוצאות המבחן; המסקנות מצומצמות בהתאם.\n\n"
    else:
        text += "האינטראקציה המשנית מוגדרת (full_graph−full_set)−(raw_graph−raw_set); ערך שלילי משמעו יתרון גרף גדול יותר עם attention בלבד.\n\n"
    text += "ההשוואות המשניות אינן טענות עליונות מאששות עם תיקון לריבוי בדיקות. קבוצת endpoint שומרת מידע יחסי ללא העברת מסרים מפורשת; rewiring משנה התאמת יעד ומאפיינים וגם מסלולים. אין כאן הוכחה שכל מודל שאינו GNN נחות.\n\n"
    quality = summary["rewiring_status"]
    text += f"איכות בקרת rewiring על train/validation המלא: {'PASS' if quality['mixing_quality_passed'] else 'FAIL / non-diagnostic'}; שיעור שינוי קשתות ממוצע {quality['development_diagnostics']['mean_changed_fraction']:.6f}, מול סף 0.80. "
    text += "ההשוואה לגרף המקורי מתארת בלבד את ההפרעה החלקית שהתקבלה, בהתאם להחלטה מתועדת לפני פתיחת המבחן. אין להסיק ממנה הסרת טופולוגיה, הכרחיות מבנה או שקילות; ההחלטה נשארת קבועה גם אם איכות הערבוב בקוהורט המלא עברה את הסף.\n\n"
    text += "כל הספים נקבעו על validation לפני הקפאה והוחלו ללא התאמה נוספת. הקובץ results.csv כולל כל זרע, את כל תשעת התנאים, ואת מדדי הדחייה/כיסוי/סיכון. ערך NA מציין מדד שאינו מוגדר.\n\n"
    text += "אלה תוצאות על קוהורט חדש וקבוע מתוך CIFAR-100 הקיים, ולא אישור עצמאי על בנצ'מרק חדש או בדיקת משפחות שיבוש שלא נראו באימון. פירוט זרעים, רווחי סמך, מקור הנתונים, חותמות קבצים ופקודת השחזור מצורפים ב־summary.json וב־bootstrap.json.\n"
    destination = results / "report.he.md"
    if (results / "evaluation_complete.json").exists() and destination.read_text(encoding="utf-8") != text:
        raise RuntimeError("A completed report is immutable; refusing to replace it with different content")
    _atomic_text(destination, text)


def score_one(cache, freeze_path, out, arm, seed, device="cuda"):
    """Independent GPU-array item, after freeze and before CPU-only analyze."""
    require_slurm()
    cache, freeze_path, out = Path(cache), Path(freeze_path).resolve(), Path(out).resolve()
    frozen = _check_frozen(cache, freeze_path)
    key = f"{arm}/seed{seed}"
    if key not in frozen["runs"]:
        raise RuntimeError("Requested scorer is outside frozen matrix")
    identity = file_sha256(freeze_path)
    provenance = json.dumps({"freeze_sha256": identity, "run": key,
                  "config_sha256": frozen["runs"][key]["config_sha256"],
                  "best_sha256": frozen["runs"][key]["best_sha256"],
                  "implementation_sha256": frozen["implementation_sha256"],
                  "device": str(device)}, sort_keys=True, separators=(",", ":"))
    directory = out / "predictions"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (key.replace("/", "__") + ".npz")
    receipt = path.with_suffix(".receipt.json")
    with path.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This model's scoring job is already active") from error
        if receipt.exists():
            value = _read(receipt)
            if (value.get("status") != "complete" or value.get("key") != key
                    or value.get("freeze_sha256") != identity or value.get("prediction_sha256") != file_sha256(path)):
                raise RuntimeError("Scoring receipt or immutable prediction artifact changed")
            return value
        if (out / "evaluation_complete.json").exists():
            raise RuntimeError("Completed evaluation cannot receive replacement model scores")
        recovered = path.exists()
        orphan = None
        if recovered:
            original_stat = path.stat()
            original_identity = (original_stat.st_dev, original_stat.st_ino, original_stat.st_size,
                                 original_stat.st_mtime_ns, original_stat.st_ctime_ns)
            original_sha256 = file_sha256(path)
            orphan = _arrays(path)
            recorded = orphan.get("__scoring_provenance")
            if recorded is None or recorded.shape != () or recorded.dtype.kind != "U" or recorded.item() != provenance:
                raise RuntimeError("Unreceipted prediction lacks matching frozen scoring provenance; original artifact preserved")
            _align(orphan, orphan)
            _verify_cohort(cache, orphan, "test")
            for field in ("score", "logit"):
                if field not in orphan or orphan[field].shape != orphan["record_id"].shape:
                    raise RuntimeError("Orphan prediction score schema mismatch; original artifact preserved")
                _finite(orphan[field], "orphan prediction " + field)
        prediction = predict_split(_run_root(frozen, freeze_path) / key, cache, "test", device, freeze=freeze_path)
        _align(prediction, prediction)
        _verify_cohort(cache, prediction, "test")
        prediction["__scoring_provenance"] = np.asarray(provenance)
        if recovered:
            # Recovery is stricter than the separate model-restoration tolerance.
            # Keep the original ZIP bytes (including its timestamps); require
            # exact array schema/dtypes/contents from the same frozen computation.
            if set(orphan) != set(prediction) or any(
                    orphan[name].dtype != prediction[name].dtype
                    or orphan[name].shape != prediction[name].shape
                    or not np.array_equal(orphan[name], prediction[name]) for name in prediction):
                raise RuntimeError("Orphan prediction differs from exact frozen recomputation; original artifact preserved")
            _check_frozen(cache, freeze_path)
            checked_stat = path.stat()
            checked_identity = (checked_stat.st_dev, checked_stat.st_ino, checked_stat.st_size,
                                checked_stat.st_mtime_ns, checked_stat.st_ctime_ns)
            if (checked_identity != original_identity or file_sha256(path) != original_sha256
                    or file_sha256(freeze_path) != identity):
                raise RuntimeError("Original prediction or freeze changed during recovery; no receipt attached")
        else:
            _atomic_npz(path, prediction)
        value = {"schema_version": SCHEMA_VERSION, "status": "complete", "key": key,
                 "freeze_sha256": identity, "prediction_sha256": file_sha256(path), "device": str(device),
                 "recovered_uncommitted_artifact": recovered,
                 "recovery_rule": "exact array names/dtypes/shapes/contents; original NPZ bytes retained" if recovered else None,
                 "completed_unix": time.time()}
        atomic_json(receipt, value)
        return value


def evaluate(cache, freeze_path, out, device="cuda", require_prescored=False):
    require_slurm()
    cache, freeze_path, out = Path(cache), Path(freeze_path).resolve(), Path(out).resolve()
    frozen = _check_frozen(cache, freeze_path)
    arms, root = _check_matrix(frozen), _run_root(frozen, freeze_path)
    out.mkdir(parents=True, exist_ok=True)
    identity = file_sha256(freeze_path)
    state_path = out / "evaluation_state.json"
    state = _read(state_path) if state_path.exists() else {"schema_version": SCHEMA_VERSION, "freeze_sha256": identity,
                "started_unix": time.time(), "status": "scoring", "predictions": {}, "recoveries": []}
    if state.get("freeze_sha256") != identity:
        raise RuntimeError("Existing evaluation belongs to a different freeze; no retuning or replacement")
    completion = out / "evaluation_complete.json"
    if completion.exists():
        done = _read(completion)
        if done["freeze_sha256"] != identity or any(file_sha256(out / name) != sha for name, sha in done["artifacts"].items()):
            raise RuntimeError("Completed evaluation artifact changed")
        return _read(out / "summary.json")
    if state_path.exists():
        state["recoveries"].append({"unix": time.time(), "reason": "resume interrupted artifact with unchanged freeze"})
    atomic_json(state_path, state)
    predictions_dir = out / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    test_ds = CachedDataset(cache, "test", "logit", freeze=freeze_path)
    metadata = test_ds.metadata()
    _align(metadata, metadata)
    _verify_cohort(cache, metadata, "test")
    scores, thresholds, validation_false_alarm_rates = {}, {}, {}
    for key in sorted(_required_runs(arms)):
        path = predictions_dir / (key.replace("/", "__") + ".npz")
        if key in state["predictions"]:
            if file_sha256(path) != state["predictions"][key]:
                raise RuntimeError("Committed prediction artifact changed")
            prediction = _arrays(path)
        else:
            receipt_path = path.with_suffix(".receipt.json")
            if receipt_path.exists():
                receipt = _read(receipt_path)
                if (receipt.get("status") != "complete" or receipt.get("freeze_sha256") != identity
                        or receipt.get("key") != key or receipt.get("prediction_sha256") != file_sha256(path)):
                    raise RuntimeError("Prescored model receipt does not match immutable predictions")
                prediction = _arrays(path)
            elif require_prescored:
                raise RuntimeError(f"CPU analysis requires a successful GPU scoring receipt: {key}")
            else:
                score_one(cache, freeze_path, out, key.split("/seed")[0], int(key.split("/seed")[1]), device)
                prediction = _arrays(path)
            _align(metadata, prediction)
            state["predictions"][key] = file_sha256(path)
            atomic_json(state_path, state)
        _align(metadata, prediction)
        scores[key] = prediction["score"]
        calibration = _read(root / key / "validation.json")
        thresholds[key] = calibration["threshold"]
        validation_false_alarm_rates[key] = calibration["validation_false_alarm_rate"]
    refs = analytic_scores(test_ds.logits().numpy(), metadata["confidence"])
    reference_thresholds = _read(freeze_path.parent / frozen["references"]["json_path"])["thresholds"]
    for name, score in refs.items():
        path = predictions_dir / (name + ".npz")
        if name in state["predictions"]:
            if file_sha256(path) != state["predictions"][name]:
                raise RuntimeError("Committed analytic prediction artifact changed")
            artifact = _arrays(path)
            _align(metadata, artifact)
            refs[name] = artifact["score"]
        else:
            _atomic_npz(path, {**metadata, "score": score})
            state["predictions"][name] = file_sha256(path)
            atomic_json(state_path, state)
        thresholds[name] = reference_thresholds[name]["threshold"]
        validation_false_alarm_rates[name] = reference_thresholds[name]["validation_false_alarm_rate"]
    slices = {"mixture": np.ones(len(metadata["y"]), dtype=bool)}
    for source, severity in VIEWS:
        name = "clean" if source == "clean_test" else f"{source}/severity{severity}"
        slices[name] = (metadata["source_id"] == SOURCES.index(source)) & (metadata["severity"] == severity)
    slices["confidence_ge_0.9"] = metadata["confidence"] >= .9
    rows = []
    for key, score in {**scores, **refs}.items():
        arm, seed = (key.split("/seed")[0], int(key.split("/seed")[1])) if "/seed" in key else (key, None)
        for scope, mask in slices.items():
            rows.append({"arm": arm, "seed": seed, "scope": scope,
                         "validation_false_alarm_rate": validation_false_alarm_rates[key],
                         **metrics(metadata, score, thresholds[key], mask)})
    bootstrap, image_ids, multiplicities = paired_bootstrap(metadata, scores)
    bootstrap["rewiring_status"] = frozen["rewiring_status"]
    for comparisons in bootstrap["comparisons"].values():
        if "full_graph_minus_full_rewired" in comparisons:
            comparisons["full_graph_minus_full_rewired"].update(interpretation="descriptive_only", limitation=LIMITATION)
    primary = bootstrap["comparisons"]["mixture"]["full_graph_minus_full_set"]
    summary = {"schema_version": SCHEMA_VERSION, "status": "complete", "matrix": frozen["matrix"],
               "arms": list(arms), "seeds": list(SEEDS), "freeze_sha256": identity,
               "rewiring_status": frozen["rewiring_status"],
               "primary": {key: value for key, value in primary.items() if key != "bootstrap_values"},
               "practical_margin": PRACTICAL_MARGIN, "practical_conclusion": practical_conclusion(primary),
               "confidence_support": bootstrap["confidence_support"], "statistics": frozen["statistics"],
               "seed_results_and_conditions": rows, "reduction_approval": frozen["reduction_approval"],
               "provenance": {"protocol_sha256": frozen["protocol_sha256"], "cohort_sha256": frozen["cohort_sha256"],
                              "cache_manifest_sha256": frozen["cache_manifest_sha256"], "implementation_sha256": frozen["implementation_sha256"],
                              "runs": frozen["runs"], "references": frozen["references"]},
               "reproduction_argv": ["python", "-m", "pilots.topology_20260910.evaluate", "evaluate", "--cache", str(cache),
                                     "--freeze", str(freeze_path), "--out", str(out), "--device", str(device)],
               "limitations": [LIMITATION, "Source-sampling uncertainty is conditional on five fitted seeds.",
                   "Secondary intervals are not multiplicity-corrected superiority claims.",
                   "Endpoint records retain relational information; rewiring changes destination/attribute alignment.",
                   "Chosen corruption mixture; existing benchmark; no held-out corruption families.",
                   "No general proof against all non-GNN architectures; parameter matching is not expressive equivalence."]}
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "bootstrap.json", bootstrap)
    _atomic_npz(out / "bootstrap_draws.npz", {"image_id": image_ids, "multiplicity": multiplicities})
    _save_csv(out / "results.csv", rows)
    report(out)
    state.update(status="complete", completed_unix=time.time())
    atomic_json(state_path, state)
    artifact_names = ["summary.json", "bootstrap.json", "bootstrap_draws.npz", "results.csv", "report.he.md", "evaluation_state.json"]
    artifact_names += [str(path.relative_to(out)) for path in sorted(predictions_dir.glob("*.npz"))]
    artifact_names += [str(path.relative_to(out)) for path in sorted(predictions_dir.glob("*.receipt.json"))]
    atomic_json(completion, {"schema_version": SCHEMA_VERSION, "status": "complete", "freeze_sha256": identity,
                "artifacts": {name: file_sha256(out / name) for name in artifact_names}, "completed_unix": time.time()})
    return summary


def main():
    require_slurm()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    f = commands.add_parser("freeze")
    f.add_argument("--cache", required=True, type=Path)
    f.add_argument("--run-root", required=True, type=Path)
    f.add_argument("--out", required=True, type=Path)
    f.add_argument("--matrix", choices=("full", "primary-only"), default="full")
    f.add_argument("--approval", type=Path)
    e = commands.add_parser("evaluate")
    e.add_argument("--cache", required=True, type=Path)
    e.add_argument("--freeze", required=True, type=Path)
    e.add_argument("--out", required=True, type=Path)
    e.add_argument("--device", default="cuda")
    s = commands.add_parser("score")
    s.add_argument("--cache", required=True, type=Path)
    s.add_argument("--freeze", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--arm", required=True, choices=FULL_ARMS)
    s.add_argument("--seed", required=True, type=int, choices=SEEDS)
    s.add_argument("--device", default="cuda")
    a = commands.add_parser("analyze")
    a.add_argument("--cache", required=True, type=Path)
    a.add_argument("--freeze", required=True, type=Path)
    a.add_argument("--out", required=True, type=Path)
    r = commands.add_parser("report")
    r.add_argument("--results", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args.cache, args.run_root, args.out, args.matrix, args.approval)
        print(json.dumps({"status": "frozen", "matrix": result["matrix"], "runs": len(result["runs"]), "path": str(args.out)}))
    elif args.command == "evaluate":
        result = evaluate(args.cache, args.freeze, args.out, args.device)
        print(json.dumps({"status": result["status"], "primary": result["primary"], "path": str(args.out)}))
    elif args.command == "score":
        print(json.dumps(score_one(args.cache, args.freeze, args.out, args.arm, args.seed, args.device)))
    elif args.command == "analyze":
        result = evaluate(args.cache, args.freeze, args.out, "cpu", require_prescored=True)
        print(json.dumps({"status": result["status"], "primary": result["primary"], "path": str(args.out)}))
    else:
        report(args.results)
        print(json.dumps({"status": "complete", "path": str(args.results / "report.he.md")}))


if __name__ == "__main__":
    main()
