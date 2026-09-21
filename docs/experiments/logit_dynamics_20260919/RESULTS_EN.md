# LogitDynamics comparison — factual report

Experiment completed: September 19, 2026. Report prepared: September 21, 2026.

**Why the experiment was run**

Yishai reported that the project-paper draft did not include a LogitDynamics comparison and requested that someone run it in addition to Omri's existing experiments. In that message, he described the existing baselines as "internal" and said this weakened the project's claim. Omri's completed September 17 comparison did not include a LogitDynamics arm. The September 19 experiment added a fixed adaptation of the published method and compared its predictions with the preserved results of that earlier experiment.

**What was run**

The task was to predict whether a frozen ViT-Base image classifier's CIFAR-100 prediction was incorrect. The ViT and the existing graph detectors were not retrained.

LogitDynamics used the CLS representations from all 12 transformer layers. A linear class-prediction head was trained for each layer. A separate linear error detector used the resulting class-score trajectories and seven dynamics features. Three pipelines were trained, with seeds **7, 17 and 27**. Each pipeline completed 16 auxiliary-head epochs and 100 error-detector epochs; the detector checkpoint was selected by validation average precision.

The source-image allocation was 1,200 for auxiliary-head training, 800 for error-detector training, 400 for checkpoint selection and 800 for development evaluation. All nine variants of an image stayed together. Evaluation contained **7,200 examples from 800 source images**, with **2,271 classifier errors (31.54%)**. The earlier detectors were evaluated using their existing predictions on those same examples.

**Results**

For trained methods, the table reports the mean of the three seed-specific metrics. SD is the sample standard deviation between seeds. MSP and entropy have one fixed score vector each.

| Method | Mean AUROC | AUROC seed SD | Mean average precision |
| --- | ---: | ---: | ---: |
| LogitDynamics | 0.898066 | 0.001574 | 0.799039 |
| GNN mean ensemble (G_mean) | 0.888895 | 0.004543 | 0.779509 |
| Set-model mean ensemble (S_mean) | 0.888559 | 0.003086 | 0.775108 |
| Final-logit detector (O) | 0.875708 | 0.000599 | 0.758339 |
| MSP | 0.861943 | — | 0.709357 |
| Entropy | 0.864851 | — | 0.715732 |

G_mean averages four graph detectors' error scores within each seed, using layers 3, 6, 9 and 12. The primary comparison was G_mean against LogitDynamics.

| Seed | LogitDynamics AUROC | G_mean AUROC |
| --- | ---: | ---: |
| 7 | 0.899177 | 0.891193 |
| 17 | 0.896264 | 0.891829 |
| 27 | 0.898756 | 0.883662 |

LogitDynamics had higher AUROC in all three seed pairs. The prespecified mean paired difference, **G_mean minus LogitDynamics**, was **−0.009171**, with a **95% interval of [−0.015874, −0.002361]**. The interval used 2,000 paired bootstrap draws grouped by source image, keeping its nine variants together and sharing draws across methods and seeds. Metrics were averaged across seeds; predictions were not combined across seeds. Average precision is scikit-learn AP, not trapezoidal precision–recall area.

**Recorded scope and verification limits**

The evaluation images had already been examined in earlier work. This was a fixed ViT-Base adaptation, not a replication of the original paper's ViT-Large experiments and hyperparameter search. The methods differed in their inputs, supervision allocation, parameter counts and training budgets. The interval is conditional on the fitted model pairs. Only the primary AUROC contrast received an interval.

On September 21, two additional Slurm audits loaded the published portable weights and replayed validation and development predictions for all three seeds from the saved CLS inputs. On GPU, all six score vectors exactly matched the original values, with maximum absolute difference zero. On CPU, all three development vectors and two validation vectors passed the unchanged absolute/relative tolerance of 0.0001. Seed-17 validation retained the previously recorded single failing example among 3,600, with a score difference of 0.000797. All six CPU checks were completed; the CPU tolerance failure remains recorded. CPU development metrics are not exactly identical in every case: seed-7 AUROC was 0.8991769878 versus the saved 0.8991770325.

The CPU audit also independently recalculated all 2,000 paired bootstrap draws using scikit-learn weighted AUROC: 12,000 AUROC calculations across the three seed pairs. The maximum discrepancy in a per-seed paired difference was 3.33 × 10⁻¹⁶; the interval matched at absolute tolerance 10⁻¹². Independent checks verified all 14 saved AUROC/AP vectors, metadata and historical predictions. The original GPU scores and reported results were retained unchanged.

These checks used the existing server environment and cached CLS representations. A clean dependency installation, fresh raw-image-to-output replay and full training replication were not performed.

All computation ran on Slurm. The original experiment used **3 hours, 59 minutes, 36 seconds** of allocated GPU time, including the unsuccessful prediction attempt and recovery. The September 21 GPU replay added **4 minutes, 43 seconds**, bringing the total to **4 hours, 4 minutes, 19 seconds**. The parallel CPU audit used 4 minutes, 39 seconds of allocated job time and no GPU.

**Sources**

- Rationale: Yishai's message supplied by Omri in this conversation; experiment scope in [PROTOCOL.md](PROTOCOL.md).
- Numerical results: the unchanged Slurm-produced [report.json](results/report.json) and [REPORT.md](results/REPORT.md).
- Verification and CPU replay: [AUDIT_REPRODUCIBILITY.md](AUDIT_REPRODUCIBILITY.md).
- Execution, timing and preserved artifacts: [SESSION.md](SESSION.md) and [MODEL_CARD.md](MODEL_CARD.md).
