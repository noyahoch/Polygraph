"""Render all frozen analyses after the complete statistical receipt exists."""
from __future__ import annotations

from pathlib import Path

from .common import atomic_json, load_campaign, lock, read, require_slurm, sha, verify_receipt
from .fusion import SEEDS, preserve_unsealed


def number(value):
    return "NA" if value is None else f"{value:.6f}"


def render(result, hebrew=False):
    points = result["points"]
    lines = ["# " + ("השלמה: שילוב ציונים ופירוק LogitDynamics" if hebrew else
                       "Score complementarity and LogitDynamics decomposition"), ""]
    if hebrew:
        lines += ["זהו ניתוח חקרני המשתמש שוב בנתוני הפיתוח. אומדני הנקודה חושבו מן המודלים המקוריים שהותאמו; הם אינם ממוצע התאמות ה-bootstrap. שלושת ה-seeds הם 7, 17, 27, ואינם נדגמים מחדש.", "",
                  "שתי טבלאות ההערכה משתמשות בקבוצות שונות: 400 תמונות לשילוב ציונים ו-800 לפירוק LD. משווים שיטות בתוך כל טבלה בלבד.", ""]
    else:
        lines += ["This is exploratory reuse of development data. Point estimates use the original fitted models, not bootstrap-refit averages. The three seeds are 7, 17 and 27 and are never resampled.", "",
                  "Fusion assessment contains 400 source photographs; decomposition assessment contains all 800 original development photographs. Compare methods only within their own cohort.", ""]
    for cohort, title in (("fusion", "Fusion assessment: 400 photographs"), ("ablation", "LD decomposition assessment: 800 photographs")):
        row = points["cohorts"][cohort]
        lines += ["## " + title, "", f"Records: {row['records']}; errors: {row['error_count']}; error prevalence: {number(row['error_prevalence'])}.", "",
                  "| Method | AUROC seed 7 | Seed 17 | Seed 27 | Mean | Seed SD | AP mean | AP seed SD |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for method, summary in row["method_summary"].items():
            values = [number(row["per_seed"][f"{method}/seed{seed}"]["auroc"]) for seed in SEEDS]
            values += [number(summary["auroc"]["mean"]), number(summary["auroc"]["seed_sd"]),
                       number(summary["average_precision"]["mean"]), number(summary["average_precision"]["seed_sd"])]
            lines.append("| " + " | ".join([method, *values]) + " |")
        lines += ["", "| Method | AP seed 7 | AP seed 17 | AP seed 27 |", "|---|---:|---:|---:|"]
        for method in row["method_summary"]:
            lines.append("| " + " | ".join([method, *[number(row["per_seed"][f"{method}/seed{seed}"]["average_precision"]) for seed in SEEDS]]) + " |")
        lines += [""]
    lines += ["## Paired AUROC contrasts", "",
              "The three primary contrasts share a nominal 95% Bonferroni family. Marginal linear-percentile endpoints are exactly 0.0083333333 and 0.9916666667. Secondary intervals use 0.025 and 0.975 and are descriptive. Every interval requires all 2,000 fixed draws; failure is not a null effect.", "",
              "| Contrast | Cohort | Primary | Seed 7 | Seed 17 | Seed 27 | Original estimate | Interval | Valid draws |",
              "|---|---|---|---:|---:|---:|---:|---|---:|"]
    for name, row in result["contrasts"].items():
        interval = "WITHHELD" if row["interval"] is None else "[" + ", ".join(number(v) for v in row["interval"]) + "]"
        values = [name, row["cohort"], "yes" if row["primary"] else "no"]
        values += [number(row["per_seed"][str(seed)]) for seed in SEEDS]
        values += [number(row["estimate"]), interval, str(row["valid_draws"])]
        lines.append("| " + " | ".join(values) + " |")
        if row["invalid_draw_ids"]:
            lines += ["", f"{name}: {row['interval_withheld_reason']}. Invalid fixed IDs: {row['invalid_draw_ids']}.", ""]
    lines += ["", "Fusion intervals resample both the 400 fit photographs and the 400 assessment photographs and refit all nine combiners. Decomposition intervals resample 800 assessment photographs with readouts held fixed. Draws share photo multiplicities across methods and seeds within each study; the study assessment streams are separate.", "",
              "## Predetermined fusion-assessment diagnostics", "",
              "The table includes all nine (source_id, severity) conditions. Entries are seed-mean AUROC / AP; NA denotes an undefined metric. Full per-seed entries and NA reasons are preserved in report.json.", "",
              "| source_id | severity | Records | Errors | Correct | D | G | S | DG | DS | DDprime |",
              "|---:|---:|---:|---:|---:|---|---|---|---|---|---|"]
    diagnostic = points["diagnostics"]
    for row in diagnostic["conditions"]:
        cells = [str(row[key]) for key in ("source_id", "severity", "records", "error_count", "correct_count")]
        cells += [number(row["seed_means"][method]["auroc"]) + " / " + number(row["seed_means"][method]["average_precision"])
                  for method in ("D", "G", "S", "DG", "DS", "DDprime")]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Within-photo rank accuracy uses every error-correct pair with half credit for exact ties, averages within each eligible photograph, then weights eligible photographs equally.", "",
              "| Method | Seed 7 | Seed 17 | Seed 27 | Seed mean | Eligible photographs (7 / 17 / 27) |",
              "|---|---:|---:|---:|---:|---|"]
    for method, mean in diagnostic["within_photo_seed_means"].items():
        rows = [diagnostic["within_photo"][f"{method}/seed{seed}"] for seed in SEEDS]
        lines.append("| " + " | ".join([method, *[number(row["value"]) for row in rows], number(mean),
                                      " / ".join(str(row["eligible_photographs"]) for row in rows)]) + " |")
    lines += ["", "## Scope and limits", "",
              "D is the unchanged LD error logit. G/S retain the original mean probability scales before fit-only standardization. DG/DS/DDprime use the fixed L2 logistic recipe. The cyclic Dprime pairs overlap (7→17, 17→27, 27→7); they are not independent replications and are not compute-matched.", "",
              "A uses original-classifier columns 72:78; B uses depth-12 auxiliary plus original columns 66:78; C uses columns 0:78; D is the frozen original 85-feature model. A is a six-feature original-classifier readout, not the existing full-logit MLP. C−B measures usefulness of intermediate projections under this readout recipe. D−C includes class-identity signals added by the dynamics block.", "",
              "Intervals condition on the fitted constituent models and fixed fusion partition, do not correct previous development-data exposure, and do not cover new base-model training variability. Roughly 17 draws lie in each primary tail, so a barely positive endpoint is not numerically decisive. Nonsignificance does not establish equivalence. Additional graph-score utility does not establish that topology or repeated message passing is necessary.", "",
              f"All {result['draws']} fixed IDs were processed. Draw IDs with a recorded numerical failure: {result['failed_fit_draw_ids']}. Full failure messages, model/scaler states and per-draw statuses are preserved in statistics/draws/.", ""]
    if hebrew:
        lines += ["המסקנות מוגבלות לתועלת הדירוג של הציונים ולמתכון ה-readout שנבדק. אין כאן הוכחת נחיצות של טופולוגיה, טענה סיבתית לגבי עומק, בדיקת מבחן שלא נחשף בעבר, או הוכחת שקילות בעקבות חוסר מובהקות.", ""]
    return "\n".join(lines)


def run(root):
    require_slurm()
    root = Path(root)
    load_campaign(root)
    completed = verify_receipt(root, root / "statistics" / "complete.json")
    result = read(root / "statistics" / "results.json")
    if result.get("identity") != completed.get("identity") or result.get("complete") is not True:
        raise RuntimeError("Report input is not the completed statistical analysis")
    directory = root / "report"
    with lock(directory, "report"):
        if (directory / "complete.json").exists():
            return verify_receipt(root, directory / "complete.json")
        outputs = [directory / "report.json", directory / "REPORT.md", directory / "REPORT_HE.md"]
        preserve_unsealed(root, outputs, "report")
        atomic_json(outputs[0], result)
        for path, hebrew in ((outputs[1], False), (outputs[2], True)):
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(render(result, hebrew=hebrew), encoding="utf-8")
            temporary.replace(path)
        receipt = {"complete": True, "campaign_sha256": sha(root / "campaign.json"),
                   "statistics_complete_sha256": sha(root / "statistics" / "complete.json"),
                   "files": {str(path.relative_to(root)): sha(path) for path in outputs}}
        atomic_json(directory / "complete.json", receipt)
        return receipt
