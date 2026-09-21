# השלמה: שילוב ציונים ופירוק LogitDynamics

זהו ניתוח חקרני המשתמש שוב בנתוני הפיתוח. אומדני הנקודה חושבו מן המודלים המקוריים שהותאמו; הם אינם ממוצע התאמות ה-bootstrap. שלושת ה-seeds הם 7, 17, 27, ואינם נדגמים מחדש.

שתי טבלאות ההערכה משתמשות בקבוצות שונות: 400 תמונות לשילוב ציונים ו-800 לפירוק LD. משווים שיטות בתוך כל טבלה בלבד.

## Fusion assessment: 400 photographs

Records: 3600; errors: 1108; error prevalence: 0.307778.

| Method | AUROC seed 7 | Seed 17 | Seed 27 | Mean | Seed SD | AP mean | AP seed SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| D | 0.900177 | 0.898298 | 0.901903 | 0.900126 | 0.001803 | 0.800041 | 0.005351 |
| DDprime | 0.901299 | 0.902565 | 0.902483 | 0.902116 | 0.000708 | 0.804295 | 0.002091 |
| DG | 0.907587 | 0.906758 | 0.907642 | 0.907329 | 0.000495 | 0.812692 | 0.003386 |
| DS | 0.907926 | 0.905656 | 0.908274 | 0.907286 | 0.001422 | 0.812892 | 0.004206 |
| G | 0.895887 | 0.897894 | 0.893045 | 0.895609 | 0.002436 | 0.788933 | 0.001174 |
| S | 0.895160 | 0.894903 | 0.894798 | 0.894954 | 0.000186 | 0.782184 | 0.002158 |

| Method | AP seed 7 | AP seed 17 | AP seed 27 |
|---|---:|---:|---:|
| D | 0.799637 | 0.794903 | 0.805582 |
| DDprime | 0.801882 | 0.805574 | 0.805428 |
| DG | 0.813115 | 0.809114 | 0.815846 |
| DS | 0.814413 | 0.808137 | 0.816126 |
| G | 0.788380 | 0.790281 | 0.788137 |
| S | 0.781281 | 0.784646 | 0.780624 |

## LD decomposition assessment: 800 photographs

Records: 7200; errors: 2271; error prevalence: 0.315417.

| Method | AUROC seed 7 | Seed 17 | Seed 27 | Mean | Seed SD | AP mean | AP seed SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.863903 | 0.863753 | 0.863137 | 0.863598 | 0.000406 | 0.714383 | 0.000984 |
| B | 0.894761 | 0.896595 | 0.893137 | 0.894831 | 0.001730 | 0.789783 | 0.004320 |
| C | 0.899815 | 0.898132 | 0.898950 | 0.898966 | 0.000842 | 0.802470 | 0.002730 |
| D | 0.899177 | 0.896264 | 0.898756 | 0.898066 | 0.001574 | 0.799039 | 0.004831 |

| Method | AP seed 7 | AP seed 17 | AP seed 27 |
|---|---:|---:|---:|
| A | 0.715204 | 0.714652 | 0.713292 |
| B | 0.787074 | 0.794765 | 0.787509 |
| C | 0.802357 | 0.799798 | 0.805254 |
| D | 0.800041 | 0.793786 | 0.803290 |

## Paired AUROC contrasts

The three primary contrasts share a nominal 95% Bonferroni family. Marginal linear-percentile endpoints are exactly 0.0083333333 and 0.9916666667. Secondary intervals use 0.025 and 0.975 and are descriptive. Every interval requires all 2,000 fixed draws; failure is not a null effect.

| Contrast | Cohort | Primary | Seed 7 | Seed 17 | Seed 27 | Original estimate | Interval | Valid draws |
|---|---|---|---:|---:|---:|---:|---|---:|
| B-A | ablation | no | 0.030858 | 0.032842 | 0.029999 | 0.031233 | [0.025273, 0.037483] | 2000 |
| C-B | ablation | yes | 0.005055 | 0.001537 | 0.005813 | 0.004135 | [0.000428, 0.007911] | 2000 |
| D-C | ablation | no | -0.000638 | -0.001867 | -0.000194 | -0.000900 | [-0.001750, -0.000048] | 2000 |
| DG-D | fusion | yes | 0.007410 | 0.008460 | 0.005738 | 0.007203 | [0.003415, 0.011330] | 2000 |
| DG-DDprime | fusion | no | 0.006288 | 0.004193 | 0.005159 | 0.005213 | [0.001942, 0.008673] | 2000 |
| DG-DS | fusion | yes | -0.000339 | 0.001101 | -0.000633 | 0.000043 | [-0.001311, 0.001508] | 2000 |

Fusion intervals resample both the 400 fit photographs and the 400 assessment photographs and refit all nine combiners. Decomposition intervals resample 800 assessment photographs with readouts held fixed. Draws share photo multiplicities across methods and seeds within each study; the study assessment streams are separate.

## Predetermined fusion-assessment diagnostics

The table includes all nine (source_id, severity) conditions. Entries are seed-mean AUROC / AP; NA denotes an undefined metric. Full per-seed entries and NA reasons are preserved in report.json.

| source_id | severity | Records | Errors | Correct | D | G | S | DG | DS | DDprime |
|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| 0 | 0 | 400 | 29 | 371 | 0.911206 / 0.484869 | 0.866499 / 0.337328 | 0.861666 / 0.337460 | 0.910525 / 0.460901 | 0.909099 / 0.449603 | 0.914180 / 0.491512 |
| 1 | 3 | 400 | 190 | 210 | 0.891044 / 0.875607 | 0.891779 / 0.866296 | 0.890927 / 0.851520 | 0.899933 / 0.884379 | 0.900585 / 0.885017 | 0.893784 / 0.879093 |
| 1 | 5 | 400 | 237 | 163 | 0.897776 / 0.921803 | 0.901780 / 0.926192 | 0.900702 / 0.924703 | 0.906638 / 0.927419 | 0.907035 / 0.928159 | 0.902004 / 0.925928 |
| 2 | 3 | 400 | 90 | 310 | 0.889032 / 0.698277 | 0.868208 / 0.625100 | 0.874863 / 0.634945 | 0.892939 / 0.700270 | 0.894552 / 0.716521 | 0.891159 / 0.702858 |
| 2 | 5 | 400 | 116 | 284 | 0.859125 / 0.723408 | 0.852153 / 0.665601 | 0.847377 / 0.649204 | 0.865398 / 0.726703 | 0.864912 / 0.725600 | 0.860066 / 0.726931 |
| 3 | 3 | 400 | 57 | 343 | 0.896186 / 0.614347 | 0.860058 / 0.507623 | 0.857774 / 0.513931 | 0.896817 / 0.598903 | 0.895760 / 0.596724 | 0.897993 / 0.619183 |
| 3 | 5 | 400 | 145 | 255 | 0.865296 / 0.763199 | 0.874113 / 0.811236 | 0.876529 / 0.790673 | 0.876800 / 0.794992 | 0.878837 / 0.795818 | 0.868127 / 0.765124 |
| 4 | 3 | 400 | 103 | 297 | 0.879485 / 0.729818 | 0.878308 / 0.648561 | 0.873786 / 0.651941 | 0.889717 / 0.733216 | 0.887995 / 0.733286 | 0.881501 / 0.736044 |
| 4 | 5 | 400 | 141 | 259 | 0.854222 / 0.758364 | 0.853455 / 0.748595 | 0.853729 / 0.747433 | 0.863998 / 0.776425 | 0.864025 / 0.772665 | 0.855856 / 0.763516 |

Within-photo rank accuracy uses every error-correct pair with half credit for exact ties, averages within each eligible photograph, then weights eligible photographs equally.

| Method | Seed 7 | Seed 17 | Seed 27 | Seed mean | Eligible photographs (7 / 17 / 27) |
|---|---:|---:|---:|---:|---|
| D | 0.861723 | 0.863153 | 0.862026 | 0.862301 | 295 / 295 / 295 |
| DDprime | 0.863450 | 0.865374 | 0.862024 | 0.863616 | 295 / 295 / 295 |
| DG | 0.867680 | 0.874342 | 0.866844 | 0.869622 | 295 / 295 / 295 |
| DS | 0.868115 | 0.873173 | 0.870549 | 0.870613 | 295 / 295 / 295 |
| G | 0.870428 | 0.866513 | 0.869327 | 0.868756 | 295 / 295 / 295 |
| S | 0.862596 | 0.865658 | 0.869559 | 0.865938 | 295 / 295 / 295 |

## Scope and limits

D is the unchanged LD error logit. G/S retain the original mean probability scales before fit-only standardization. DG/DS/DDprime use the fixed L2 logistic recipe. The cyclic Dprime pairs overlap (7→17, 17→27, 27→7); they are not independent replications and are not compute-matched.

A uses original-classifier columns 72:78; B uses depth-12 auxiliary plus original columns 66:78; C uses columns 0:78; D is the frozen original 85-feature model. A is a six-feature original-classifier readout, not the existing full-logit MLP. C−B measures usefulness of intermediate projections under this readout recipe. D−C includes class-identity signals added by the dynamics block.

Intervals condition on the fitted constituent models and fixed fusion partition, do not correct previous development-data exposure, and do not cover new base-model training variability. Roughly 17 draws lie in each primary tail, so a barely positive endpoint is not numerically decisive. Nonsignificance does not establish equivalence. Additional graph-score utility does not establish that topology or repeated message passing is necessary.

All 2000 fixed IDs were processed. Draw IDs with a recorded numerical failure: []. Full failure messages, model/scaler states and per-draw statuses are preserved in statistics/draws/.

המסקנות מוגבלות לתועלת הדירוג של הציונים ולמתכון ה-readout שנבדק. אין כאן הוכחת נחיצות של טופולוגיה, טענה סיבתית לגבי עומק, בדיקת מבחן שלא נחשף בעבר, או הוכחת שקילות בעקבות חוסר מובהקות.
