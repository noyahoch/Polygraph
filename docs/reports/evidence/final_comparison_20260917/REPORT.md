# Final fixed20 development comparison

25 reported methods on the same **800 photographs × 9 views = 7,200 records**. Target: frozen ViT classification error; higher detector scores indicate error. All 39 neural fits completed exactly 20 epochs: 12 compatible historical G imports and 27 fresh fits, at seeds 7, 17, 27. There are 18 meta heads (6 imported, 12 new), one deterministic **L / logits_linear_100** fit, and two analytic baselines. The 1,600/400/400/800 base/checkpoint/meta/development photograph roles and all nine views are unchanged.

## All registered methods

| Method | AUROC mean | Seed sample SD | AP mean | AURC mean |
|---|---:|---:|---:|---:|
| G_block2 | 0.87917 | 0.00262 | 0.76398 | 0.10174 |
| G_block5 | 0.88035 | 0.00445 | 0.75953 | 0.10089 |
| G_block8 | 0.88102 | 0.00328 | 0.76556 | 0.10000 |
| G_block11 | 0.88313 | 0.00332 | 0.77064 | 0.09998 |
| G_mean | 0.88889 | 0.00454 | 0.77951 | 0.09714 |
| G_stack | 0.88847 | 0.00437 | 0.77902 | 0.09732 |
| G_last_only | 0.88313 | 0.00332 | 0.77064 | 0.09998 |
| H_block2 | 0.72608 | 0.00315 | 0.55549 | 0.17727 |
| H_block5 | 0.75479 | 0.00376 | 0.59461 | 0.16098 |
| H_block8 | 0.80521 | 0.00314 | 0.65563 | 0.13406 |
| H_block11 | 0.88038 | 0.00047 | 0.75075 | 0.10022 |
| H_mean | 0.85531 | 0.00631 | 0.73150 | 0.11353 |
| H_stack | 0.87570 | 0.00423 | 0.75382 | 0.10180 |
| H_last_only | 0.88038 | 0.00047 | 0.75075 | 0.10022 |
| S_block2 | 0.87964 | 0.00398 | 0.75368 | 0.10031 |
| S_block5 | 0.87626 | 0.00709 | 0.75873 | 0.10274 |
| S_block8 | 0.87768 | 0.00187 | 0.75636 | 0.10192 |
| S_block11 | 0.87974 | 0.00198 | 0.75763 | 0.10160 |
| S_mean | 0.88856 | 0.00309 | 0.77511 | 0.09760 |
| S_stack | 0.88794 | 0.00267 | 0.77297 | 0.09784 |
| S_last_only | 0.87974 | 0.00198 | 0.75763 | 0.10160 |
| O | 0.87571 | 0.00060 | 0.75834 | 0.10275 |
| L | 0.81550 | N/A | 0.67596 | 0.12999 |
| MSP | 0.86194 | N/A | 0.70936 | 0.10466 |
| entropy | 0.86485 | N/A | 0.71573 | 0.10382 |

Raw columns block2/5/8/11 correspond to human layers 3/6/9/12. `*_mean` is the equal mean of four sigmoid scores, not sigmoid(mean logits). `*_stack` and `*_last_only` use their separately meta-fitted StandardScalers and signed L2 logistic decision functions. Negative coefficients are not flipped. O is the raw MLP decision score. L/MSP/entropy each have **one score vector**, so training-seed SD is **not applicable**, not zero. Seeded table entries are arithmetic means of three metrics, never metrics of averaged scores or pooled seed-image rows. AP is sklearn average precision (not trapezoidal PR area). AURC accepts low error scores first; ties use expected uniformly random order. With distinct scores this equals the repository cumulative-risk mean. All coverages and per-seed metrics are saved.

## Six primary comparisons, without winner selection

| Fixed contrast | Mean paired ΔAUROC | Ordinary 95% | Bonferroni, family of six |
|---|---:|---|---|
| G_stack_minus_S_stack | 0.00053 | [-0.00210, 0.00328] | [-0.00300, 0.00416] |
| G_mean_minus_S_mean | 0.00034 | [-0.00187, 0.00261] | [-0.00280, 0.00341] |
| G_stack_minus_H_stack | 0.01277 | [0.00837, 0.01717] | [0.00703, 0.01888] |
| G_mean_minus_H_mean | 0.03359 | [0.02658, 0.04079] | [0.02424, 0.04322] |
| G_stack_minus_O | 0.01276 | [0.00803, 0.01755] | [0.00652, 0.01909] |
| G_mean_minus_O | 0.01319 | [0.00838, 0.01804] | [0.00688, 0.01972] |

10,000 shared image_id bootstrap draws use RNG seed 20260914. Every photograph's nine views inherit its integer multiplicity, identically across methods and seeds. Each draw averages three within-seed paired AUROC differences; tie pairs receive half credit. Linear percentile quantiles are 0.025/0.975 and **0.004166666666666667/0.9958333333333333** for the six-comparison Bonferroni family. Adjusted family coverage is nominal/approximate, not an exact guarantee. Undefined draws are recorded, never redrawn or dropped; an affected CI is withheld. Per-seed CIs are descriptive and reuse these same draws. No p-values or positive-result completion rule are used.

## Secondary L versus O

Mean paired L−O ΔAUROC: **-0.06021**; ordinary paired 95% interval **[-0.07168, -0.04904]**. The same single L vector is paired with each O seed using the cached primary draws. This adds no seventh primary contrast and no independent L seed observations. Differences are descriptive under the specified linear and nonlinear recipes, not a causal isolation of nonlinearity. L uses 100 base_train-standardized logits, training-derived class weights and one converged fixed L2/lbfgs fit; it has no CV, checkpoint selection or artificial epoch history.

## Frozen budget-sensitivity diagnostics

| Family | Selected epoch20 / all | Warning / all | Mean ΔAUC 20−15 | Mean loss drop | Mean drop % | Mean tail slope |
|---|---:|---:|---:|---:|---:|---:|
| G | 0/12 | 0/12 | -0.00481 | 0.07897 | 20.76046 | -0.01445 |
| H | 0/12 | 0/12 | 0.00169 | 0.05419 | 27.54639 | -0.00975 |
| S | 0/12 | 0/12 | -0.00349 | 0.09498 | 24.50740 | -0.01832 |
| O | 1/3 | 1/3 | -0.00048 | 0.02396 | 4.46106 | -0.00395 |

The nonfatal warning is exactly: selected epoch20 AND actual checkpoint AUROC20 > AUROC15 AND training loss15 > loss20. It means possible budget sensitivity, not proven undertraining; absence is not proof of convergence. Loss reductions are signed in original loss units and percent, with the actual epochs16–20 OLS slope. A zero loss15 has an explicitly undefined percentage. All 39 histories, all seeds (including imported late G7), layer/family denominators and complete curves are frozen in `../diagnostics/training_diagnostics.json` and `.csv`, whose three-file receipt is bound below. No warning authorizes new training, exclusion, replacement, changed selection or retuning. Architectural conclusions remain **under the fixed20 training budget**, never intrinsic inferiority.

| Fit | Selected | AUC15 | AUC20 | Loss15 | Loss20 | Loss drop | Drop % | Tail slope | Warning | Original status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| G/block2/seed7 | 13 | 0.85746 | 0.85175 | 0.37108 | 0.30778 | 0.06330 | 17.05768 | -0.01046 | False | complete_late_diagnostic |
| G/block5/seed7 | 7 | 0.85892 | 0.84952 | 0.36831 | 0.31208 | 0.05623 | 15.26677 | -0.01281 | False | complete_late_diagnostic |
| G/block8/seed7 | 13 | 0.85899 | 0.85453 | 0.37103 | 0.29158 | 0.07945 | 21.41405 | -0.01648 | False | complete_late_diagnostic |
| G/block11/seed7 | 14 | 0.85404 | 0.85552 | 0.39806 | 0.33610 | 0.06196 | 15.56648 | -0.01076 | False | complete_late_diagnostic |
| G/block2/seed17 | 14 | 0.86411 | 0.85109 | 0.38516 | 0.30300 | 0.08216 | 21.33234 | -0.01503 | False | complete |
| G/block5/seed17 | 18 | 0.86171 | 0.85193 | 0.37331 | 0.28066 | 0.09265 | 24.81887 | -0.01660 | False | complete |
| G/block8/seed17 | 9 | 0.86163 | 0.85065 | 0.36022 | 0.27922 | 0.08101 | 22.48758 | -0.01668 | False | complete |
| G/block11/seed17 | 10 | 0.86106 | 0.85198 | 0.39755 | 0.30483 | 0.09272 | 23.32241 | -0.01719 | False | complete |
| G/block2/seed27 | 8 | 0.84373 | 0.85060 | 0.38608 | 0.31241 | 0.07367 | 19.08211 | -0.01270 | False | complete |
| G/block5/seed27 | 8 | 0.85328 | 0.85120 | 0.37568 | 0.29302 | 0.08266 | 22.00207 | -0.01377 | False | complete |
| G/block8/seed27 | 8 | 0.85208 | 0.85313 | 0.37206 | 0.27685 | 0.09522 | 25.59134 | -0.01683 | False | complete |
| G/block11/seed27 | 8 | 0.86190 | 0.85935 | 0.40885 | 0.32224 | 0.08661 | 21.18380 | -0.01413 | False | complete |
| H/block2/seed7 | 1 | 0.68988 | 0.68301 | 0.22862 | 0.15062 | 0.07800 | 34.11654 | -0.01513 | False | complete |
| H/block5/seed7 | 3 | 0.70730 | 0.71243 | 0.11032 | 0.08271 | 0.02761 | 25.02739 | -0.00492 | False | complete |
| H/block8/seed7 | 4 | 0.76263 | 0.76559 | 0.12332 | 0.09217 | 0.03115 | 25.25785 | -0.00585 | False | complete |
| H/block11/seed7 | 18 | 0.85062 | 0.85425 | 0.43805 | 0.37555 | 0.06250 | 14.26740 | -0.01381 | False | complete |
| H/block2/seed17 | 1 | 0.68765 | 0.69173 | 0.22133 | 0.14109 | 0.08025 | 36.25623 | -0.01215 | False | complete |
| H/block5/seed17 | 3 | 0.69219 | 0.69394 | 0.12337 | 0.07114 | 0.05224 | 42.34026 | -0.00932 | False | complete |
| H/block8/seed17 | 3 | 0.76010 | 0.75881 | 0.11159 | 0.07727 | 0.03432 | 30.75830 | -0.00877 | False | complete |
| H/block11/seed17 | 18 | 0.84629 | 0.84391 | 0.44082 | 0.37265 | 0.06817 | 15.46366 | -0.01195 | False | complete |
| H/block2/seed27 | 5 | 0.65822 | 0.66527 | 0.22718 | 0.13487 | 0.09231 | 40.63318 | -0.01454 | False | complete |
| H/block5/seed27 | 3 | 0.71539 | 0.70633 | 0.10304 | 0.07598 | 0.02706 | 26.26258 | -0.00586 | False | complete |
| H/block8/seed27 | 2 | 0.75401 | 0.76793 | 0.11609 | 0.08737 | 0.02872 | 24.74341 | -0.00583 | False | complete |
| H/block11/seed27 | 13 | 0.85427 | 0.85563 | 0.44011 | 0.37220 | 0.06791 | 15.42987 | -0.00885 | False | complete |
| S/block2/seed7 | 15 | 0.86411 | 0.85359 | 0.40788 | 0.29290 | 0.11499 | 28.19087 | -0.01941 | False | complete |
| S/block5/seed7 | 7 | 0.85532 | 0.84371 | 0.36775 | 0.26748 | 0.10027 | 27.26539 | -0.01857 | False | complete |
| S/block8/seed7 | 13 | 0.84863 | 0.85002 | 0.38591 | 0.30434 | 0.08157 | 21.13774 | -0.01462 | False | complete |
| S/block11/seed7 | 10 | 0.85165 | 0.84897 | 0.39439 | 0.28207 | 0.11232 | 28.47844 | -0.01894 | False | complete |
| S/block2/seed17 | 8 | 0.85481 | 0.84079 | 0.37771 | 0.29749 | 0.08022 | 21.23895 | -0.01965 | False | complete |
| S/block5/seed17 | 8 | 0.85864 | 0.84645 | 0.36928 | 0.27487 | 0.09441 | 25.56657 | -0.01563 | False | complete |
| S/block8/seed17 | 10 | 0.84927 | 0.84409 | 0.37189 | 0.29590 | 0.07600 | 20.43489 | -0.01921 | False | complete |
| S/block11/seed17 | 10 | 0.85801 | 0.84638 | 0.39336 | 0.28899 | 0.10437 | 26.53264 | -0.01973 | False | complete |
| S/block2/seed27 | 8 | 0.84816 | 0.85516 | 0.38985 | 0.28056 | 0.10930 | 28.03549 | -0.02305 | False | complete |
| S/block5/seed27 | 8 | 0.85200 | 0.85591 | 0.39940 | 0.33237 | 0.06703 | 16.78194 | -0.01079 | False | complete |
| S/block8/seed27 | 13 | 0.85034 | 0.86348 | 0.39017 | 0.29516 | 0.09502 | 24.35212 | -0.01916 | False | complete |
| S/block11/seed27 | 8 | 0.85060 | 0.85103 | 0.40013 | 0.29580 | 0.10433 | 26.07380 | -0.02106 | False | complete |
| O/logits/seed7 | 13 | 0.86495 | 0.86188 | 0.53574 | 0.51183 | 0.02391 | 4.46380 | -0.00474 | False | complete |
| O/logits/seed17 | 19 | 0.86096 | 0.85457 | 0.53187 | 0.51223 | 0.01964 | 3.69259 | -0.00250 | False | complete |
| O/logits/seed27 | 20 | 0.85832 | 0.86636 | 0.54211 | 0.51378 | 0.02834 | 5.22678 | -0.00460 | True | complete |

## Cost and descriptive breakdowns

Source-derived parameter counts per detector are G=130,434, H=129,986, S=131,126, O=8,577; the allocated preflight is responsible for confirming them. Family/seed export durations and available memory measurements are preserved in `inference_cost.csv` and the bound prediction receipts. Full-role timings include loading and all four forwards for G/H/S; amortized record time is not isolated batch latency. Missing historical telemetry is labeled unavailable, never fabricated. Four-detector ensemble inference requires all four forwards plus averaging or a logistic head. The fixed nine-condition and severity tables in `conditions.csv` are descriptive, with no selected subgroup, new hypothesis family or evaluation-optimized threshold.

## Claim ceiling and provenance

- Development photographs and earlier EDA/results were already used; this is not an untouched independent test.
- Imported G seed7 remains complete_late_diagnostic under its original formally incomplete protocol.
- All three fixed neural seeds (7, 17, 27) are included, including seed7; sample SD uses ddof=1 and only three observations.
- The seed set was reduced from five to three on 2026-09-16, before any benchmark result, because only seeds 7/17/27 have compatible G fits.
- Photo-bootstrap uncertainty is conditional on fitted models/heads and these three seeds; it does not estimate full training-seed uncertainty or repair development-data reuse.
- Comparisons are under the fixed20 training budget; a boundary warning is nonfatal, and no warning is not proof of convergence.
- A possibly budget-limited H/S/G/O result does not establish intrinsic architectural inferiority.
- G versus S changes incidence/association and message passing together; a gap does not universally prove GNN necessity.
- G uses selected attention plus H12; H uses its corresponding hidden layer, so G versus H does not isolate message passing.
- Four-model ensembles cost four detector forwards; across-seed metric means are not new prediction ensembles.
- L is one deterministic base_train-only fit, not one fit per seed, and is not AdamW/fixed20 budget-matched.
- All six registered contrasts are reported without choosing a winner, exclusions, extensions, retuning, or a borrowed practical-effect threshold.
- No original-test arrays, new data, evaluation-fitted thresholds, classifier fine-tuning, or language-model generalization are used.

Scope: `final_comparison_20260916_fixed20`. Complete input identity: `b51ad29f90453bc6c3554df75b98ee7579eea33c14c0903589c2451d20a65fe6`. All prediction, head, linear, diagnostic, cache, campaign and executed-source identities are listed in `report.json`; `complete.json` hashes every emitted artifact. Historical source files, timestamps and scopes are not rewritten. A completed rerun verifies inputs/artifacts rather than overwriting them.
