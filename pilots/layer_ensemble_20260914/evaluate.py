"""One-seed, source-photograph-paired development evaluation on Slurm CPU."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path

import numpy as np

from .combine import (ARMS, SCOPE, atomic_json, code_identity, head_scores, inside,
                      load_predictions, read, require_slurm, sha256, validate_heads)

BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260914


class WeightedAUC:
    """Mann-Whitney AUROC with half credit for tied positive-negative pairs.

    Formula retained from the September10 evaluator without importing its
    protocol, five-seed matrix, test access or data-loading machinery.
    """
    def __init__(self, labels, scores):
        labels, scores = np.asarray(labels), np.asarray(scores)
        if labels.ndim != 1 or labels.shape != scores.shape or not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
            raise ValueError("AUROC requires aligned binary labels and finite scores")
        self.order = np.argsort(scores, kind="stable")
        self.labels = labels[self.order].astype(np.float64)
        ordered = scores[self.order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(ordered)) + 1] if len(ordered) else np.array([], dtype=int)

    def __call__(self, weights=None):
        if not len(self.order):
            return None
        weights = np.ones(len(self.order)) if weights is None else np.asarray(weights, dtype=np.float64)
        if weights.shape != self.order.shape or not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Invalid source-photograph bootstrap multiplicities")
        ordered = weights[self.order]
        positive = np.add.reduceat(ordered * self.labels, self.starts)
        negative = np.add.reduceat(ordered * (1 - self.labels), self.starts)
        positives, negatives = positive.sum(), negative.sum()
        if not positives or not negatives:
            return None
        preceding_negative = np.cumsum(negative) - negative
        return float(np.sum(positive * (preceding_negative + 0.5 * negative)) / (positives * negatives))


def sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1 + exponent)
    return result


def paired_bootstrap(labels, stack, last_only, image_ids):
    """Sample photographs once per draw; their nine variants share the weight."""
    groups, membership = np.unique(image_ids, return_inverse=True)
    counts = np.bincount(membership)
    if len(groups) != 800 or not np.all(counts == 9):
        raise RuntimeError("Primary bootstrap requires800photos withnineversions each")
    stack_auc, last_auc = WeightedAUC(labels, stack), WeightedAUC(labels, last_only)
    point_stack, point_last = stack_auc(), last_auc()
    if point_stack is None or point_last is None:
        raise RuntimeError("Primary AUROC is undefined")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    multiplicities = np.empty((BOOTSTRAP_DRAWS, len(groups)), dtype=np.int16)
    samples, undefined = [], []
    for draw in range(BOOTSTRAP_DRAWS):
        multiplicity = np.bincount(rng.integers(0, len(groups), size=len(groups)), minlength=len(groups))
        multiplicities[draw] = multiplicity
        weights = multiplicity[membership]
        a, b = stack_auc(weights), last_auc(weights)
        if a is None or b is None:
            samples.append(None)
            undefined.append(draw)
        else:
            samples.append(a - b)
    interval = None if undefined else np.percentile(samples, [2.5, 97.5], method="linear").tolist()
    return {"name": "learned_stack_minus_learned_last_only", "estimate": point_stack - point_last,
            "stack_auroc": point_stack, "last_only_auroc": point_last, "interval_95": interval,
            "draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
            "group": "image_id (source photograph), not source_id (corruption family)",
            "source_photographs": len(groups), "views_per_source": 9, "percentile_method": "linear",
            "undefined_draw_indices": undefined, "undefined_draw_policy": "no redraw; withhold interval if any draw is undefined",
            "bootstrap_values": samples,
            "scope": "Image-sampling uncertainty conditional on four fitted seed7 base models and two fitted meta heads; not seed stability."}, groups, multiplicities


def atomic_npz(path, arrays):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_report(path, result):
    primary = result["primary"]
    interval = primary["interval_95"]
    ci = "לא מוצג עקב דגימה ללא שתי המחלקות" if interval is None else f"[{interval[0]:.4f}, {interval[1]:.4f}]"
    rows = [("שילוב נלמד של ארבע שכבות", result["metrics"]["learned_stack"]["auroc"]),
            ("בקרת השכבה האחרונה עם אותו לומד מטה", result["metrics"]["learned_last_only"]["auroc"]),
            ("ממוצע קבוע של ארבעת ציוני ה־sigmoid — משני", result["metrics"]["fixed_probability_mean"]["auroc"])]
    rows += [(f"שכבה {layer} לבדה, ציון גולמי — תיאורי", result["metrics"]["raw_" + arm]["auroc"])
             for arm, layer in zip(ARMS, (3, 6, 9, 12))]
    text = "# סבב לילה: שילוב נלמד לזיהוי טעויות ViT\n\n"
    text += f"כל ארבעת גלאי השכבות השלימו 20 epochs בזרע 7. השילוב ובקרת השכבה האחרונה נלמדו בנפרד על אותן 400 תמונות מטה. ההערכה כוללת 800 תמונות מקור ו־7,200 גרסאות.\n\n"
    text += f"**הפרש AUROC ראשי, שילוב פחות בקרת השכבה האחרונה: {primary['estimate']:.4f}.**\n\n"
    text += f"רווח bootstrap של 95%, מותנה במודלים שאומנו: `{ci}`.\n\n"
    text += "| שיטה | AUROC |\n|---|---:|\n"
    text += "".join(f"| {name} | {value:.4f} |\n" for name, value in rows)
    text += "\nהדגימה המזֻווגת ב־2,000 חזרות נעשתה לפי תמונת המקור; כל תשע גרסאותיה קיבלו אותו משקל בשתי השיטות. רווח הסמך אינו מודד שונות בין זרעי אימון.\n\n"
    text += "זהו סבב חקירתי בזרע אחד ובתקציב של 20 epochs, ללא הבטחה להתכנסות מלאה. השכבות משתמשות באותו מסווג קפוא ובאותו hidden state סופי. לא נעשה שימוש בקבוצת המבחן המקורית. נתוני ההערכה באים מבנצ׳מרק שכבר שימש את הפרויקט, ולכן אין כאן אימות עצמאי לחלוטין.\n\n"
    text += "השילוב משתמש בארבעה גלאים מול גלאי אחד. יתרון אפשרי יכול לנבוע ממידע משלים וגם ממיצוע/שונות של מספר מודלים; אין זו הוכחה להכרחיות GNN או טופולוגיה. תוצאה שלילית או רווח רחב מדווחים באותו אופן. מודלי איחוד שכבות וזרעים נוספים אינם חלק מהסבב.\n\n"
    text += "המקדם של בקרת השכבה האחרונה נשמר כפי שנלמד, גם אם הוא שלילי; אין הנחה שכיול לוגיסטי בהכרח שומר על דירוג הציון הגולמי. לא שונו משקלי השילוב לאחר צפייה בהערכה.\n"
    temporary = Path(path).with_name(Path(path).name + ".tmp." + str(os.getpid()))
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def evaluate(root, predictions, heads, out):
    require_slurm()
    root, out = Path(root).resolve(), inside(root, out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".evaluate.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        models, frozen, bindings = validate_heads(root, heads)
        data, sidecar, _ = load_predictions(root, predictions, "dev_eval")
        freeze_sha = sha256(root / "heads_freeze.json")
        if sidecar.get("heads_freeze_sha256") != freeze_sha:
            raise RuntimeError("Development predictions were not produced under the current frozen heads")
        inputs = {**bindings, "heads_freeze_sha256": freeze_sha,
                  "dev_prediction_npz_sha256": sha256(predictions),
                  "dev_prediction_sidecar_sha256": sha256(Path(predictions).with_suffix(".json")),
                  "implementation": code_identity()}
        if (out / "complete.json").exists():
            complete = read(out / "complete.json")
            if complete.get("inputs") != inputs:
                raise RuntimeError("Completed evaluation cannot be replaced by a different comparison")
            for relative, wanted in complete["files"].items():
                if sha256(inside(out, out / relative)) != wanted:
                    raise RuntimeError("Completed evaluation artifact changed")
            return read(out / "report.json")
        scores = {"learned_stack": head_scores(models["stack"], data["logits"]),
                  "learned_last_only": head_scores(models["last_only"], data["logits"]),
                  "fixed_probability_mean": sigmoid(data["logits"]).mean(axis=1)}
        scores.update({"raw_" + arm: data["logits"][:, index] for index, arm in enumerate(ARMS)})
        metrics = {name: {"auroc": WeightedAUC(data["y"], value)()} for name, value in scores.items()}
        primary, image_ids, multiplicities = paired_bootstrap(data["y"], scores["learned_stack"],
                                                             scores["learned_last_only"], data["image_id"])
        samples = primary.pop("bootstrap_values")
        atomic_json(out / "bootstrap.json", {**primary, "bootstrap_values": samples})
        atomic_npz(out / "bootstrap_source_counts.npz", {"image_id": image_ids, "multiplicity": multiplicities})
        atomic_npz(out / "scores.npz", {**{name: data[name] for name in data if name != "logits"}, **scores})
        result = {"schema_version": 1, "scope_id": SCOPE, "status": "complete", "complete": True,
                  "seed": 7, "base_models": list(ARMS), "base_epochs": 20,
                  "records": 7200, "source_photographs": 800, "positive_class": "frozen ViT error",
                  "errors": int(data["y"].sum()), "correct": int((data["y"] == 0).sum()),
                  "primary": primary, "metrics": metrics,
                  "last_only_learned_coefficient": models["last_only"]["model"]["coef"][0],
                  "roles": {"base_train": 1600, "checkpoint": 400, "meta": 400, "dev_eval": 800},
                  "inputs": inputs, "test_evaluated": False, "exploratory": True,
                  "scope_of_uncertainty": "source-image sampling conditional on seed7; no training-seed uncertainty",
                  "interpretation_limits": ["20-epoch budget, not guaranteed architecture convergence",
                      "four detector models versus one; information and generic ensemble benefits are not isolated",
                      "all methods use GNN features/models; no proof of GNN/topology necessity",
                      "existing benchmark development split; original held-out800photos remain closed",
                      "secondary fixed mean and raw single-layer methods descriptive, without post-hoc selection"],
                  "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "job_id": os.environ["SLURM_JOB_ID"]}
        atomic_json(out / "report.json", result)
        write_report(out / "REPORT.md", result)
        files = {name: sha256(out / name) for name in ("report.json", "REPORT.md", "bootstrap.json", "bootstrap_source_counts.npz", "scores.npz")}
        atomic_json(out / "complete.json", {"complete": True, "scope_id": SCOPE, "inputs": inputs, "files": files})
        return result


def main():
    require_slurm()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "predictions", "heads", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = evaluate(args.root, args.predictions, args.heads, args.out)
    except BaseException as error:
        args.out.mkdir(parents=True, exist_ok=True)
        atomic_json(args.out / "failure.json", {"complete": False, "scope_id": SCOPE,
                    "error": repr(error), "job_id": os.environ.get("SLURM_JOB_ID"),
                    "policy": "No subset ensemble, missing-fit substitution or outcome-based repair"})
        raise
    print(json.dumps({"complete": True, "primary_difference": result["primary"]["estimate"]}), flush=True)


if __name__ == "__main__":
    main()
