# Scientific audit of the completed LogitDynamics comparison

September 19, 2026. Scope: the frozen scientific contract, completed numerical
reports, training/selection receipts and executed analysis source. This review
reads existing results; it performs no local training, inference, metric
calculation or bootstrap recomputation. Artifact/reload integrity and remote
backup verification are separately audited by the engineering and operations
roles. The methodology role contributed to the protocol and preflight tests;
this is not an external, blinded replication.

## Finding

**No blocking scientific discrepancy was found in the inspected evidence.**
The completed result supports an advantage for the fixed LogitDynamics ViT-B
adaptation over the prespecified historical GNN mean ensemble on this reused
development benchmark. It does not establish superiority of the published
method in general, graph irrelevance, or a new untouched-test result.

The saved primary estimate, **G mean minus LD AUROC**, is
**−0.009171241462913962**, with the ordinary 95% photograph-bootstrap interval
**[−0.015873773489983947, −0.0023608076406355953]**. The interval is entirely
below zero and all three paired seed estimates favor LD. These numbers are
transcribed from the completed [machine-readable report][report], not
recomputed for this audit.

| Training seed | G mean AUROC | LD AUROC | G mean − LD AUROC | Selected LD probe epoch |
| --- | ---: | ---: | ---: | ---: |
| 7 | 0.8911931193087148 | 0.899177032487478 | −0.007983913178763236 | 100 |
| 17 | 0.8918293667033568 | 0.8962643380119225 | −0.004434971308565738 | 95 |
| 27 | 0.8836615117406047 | 0.8987563516420176 | −0.015094839901412915 | 100 |

Mean AUROC is **0.8980659073804728 for LD** and
**0.8888946659175588 for G mean**. The saved sample seed SDs are respectively
0.0015743196104057561 and 0.004543195974930641. Those SDs describe three fitted
seeds; they are not confidence intervals or a reliable estimate of every
future-training outcome. The seed-specific numerical advantage is variable,
despite its common direction. Sources: [report][report], and probe completion
receipts for [seed 7][probe7], [seed 17][probe17] and [seed 27][probe27].

## Contract and statistical interpretation

The [protocol](PROTOCOL.md) fixed this one primary contrast, seeds 7/17/27,
and 2,000 paired bootstrap draws before LD fitting. The executed
[analysis source][analysis-source] constructs within-seed AUROC differences
and then averages those differences. It does not score predictions averaged
across seeds. Each draw samples 800 image_id groups with replacement and
shares each photograph's multiplicity across all nine views, both methods and
all three seeds. Thus the phrase “2,000 draws” means 2,000 bootstrap
resamples of the 800-source evaluation cohort, not 2,000 distinct test images.

The reported implementation uses RNG seed 20260919, half credit for AUROC
ties, linear percentile interpolation and one ordinary 95% interval. The
saved report records no undefined bootstrap draws. No multiplicity adjustment
was prespecified for this single primary contrast; secondary comparisons do
not acquire inferential claims from it. The source and the scientific
preflight tests cover the intended grouping and estimand. This review has not
independently regenerated the actual saved bootstrap array.

This interval is conditional on the three fitted model pairs and the chosen
configuration. It does not include full training-seed uncertainty, repair
previous development-set exposure, or account for the history of choosing
methods and G mean using earlier results. Therefore “the paired development
comparison favors LD” is supported; an unqualified confirmatory population
claim is stronger than the design supports. No equivalence margin, practical
deployment threshold or p-value was declared.

Reusing G also preserves its earlier provenance: seed 7 was a late diagnostic
under its original campaign, as recorded in the
[September 17 synthesis](../scientific_synthesis_20260917/HANDOFF.md). Its
presence here is not a new on-time or independent-data replication of that fit.

## Cohort and error-label checks

The executed report contains **7,200 records from 800 source photographs**,
with **2,271 erroneous and 4,929 correct frozen-classifier predictions**.
Reported error prevalence is **0.3154166666666667**, displayed as **31.54%**.
Every listed method uses those same record/outcome counts. The frozen
[role map][roles] records the following fitting and selection boundaries:

| LD role | Source photographs in the protocol | Recorded views | Recorded error labels |
| --- | ---: | ---: | ---: |
| Auxiliary class-head training | 1,200 | 10,800 | 3,365 |
| Error-probe training | 800 | 7,200 | 2,379 |
| Probe checkpoint selection | 400 | 3,600 | 1,272 |
| Development evaluation | 800 | 7,200 | 2,271 |

The auxiliary-head error count is cohort metadata; heads are trained against
the 100-class ground-truth label, not that error indicator. The deterministic
source split and all-view grouping remain fixed across training seeds.
The executed train/data source restricts auxiliary-head parameters to
head_train, normalization and probe parameters to probe_train, and checkpoint
selection to probe_val. Evaluation loading requires the common completed
three-seed freeze. The [full extraction manifest][extraction] reports all
28,800 records complete and maximum absolute differences of zero for the
reproduced original classifier logits and checked H12 CLS values. This
supports classifier/cohort compatibility; it does not turn the reused
development set into an independent test.

## Average precision and secondary outputs

The metric is error-positive **scikit-learn average precision**, with equal
score thresholds grouped. The executed [metric implementation][metric-source]
sets the legacy `auprc` field to that same value. Neither field denotes
trapezoidal PR area, and neither should be silently equated with an unspecified
AUCPR integration convention in the paper.

| Seed | G mean average precision | LD average precision |
| --- | ---: | ---: |
| 7 | 0.783218528511687 | 0.8000409413412997 |
| 17 | 0.7816841336308237 | 0.7937858858006084 |
| 27 | 0.7736238326854813 | 0.8032898867936457 |

Mean AP is **0.7795088316093306 for G mean** and
**0.7990389046451846 for LD**. This is descriptively consistent with the
AUROC result; no paired AP interval was prespecified or reported. AP depends
on error prevalence and the evaluation mixture. The reported numbers should
not be compared directly with the paper's different clean-image/backbone
experiment as evidence of successful numerical replication.

S mean, O, MSP and entropy have lower saved mean AUROC/AP point estimates than
LD. Their inclusion provides context, not four additional statistically
confirmed victories. AURC and fixed-coverage risk values are secondary
descriptions; they do not establish calibrated alert probabilities or an
operating threshold. Increasing raw LD logit means greater predicted error;
class-weighted binary training does not by itself calibrate probabilities.

## Optimization, recovery and limits on the claim

All three auxiliary-head completion receipts record the required final epoch
16. All three probes completed 100 epochs. Probe selection uses validation AP,
with strict improvement and earliest tie, as fixed in advance. Selected
epochs are 100, 95 and 100; the histories and completion receipts preserve
these choices. The endpoint selections for seeds 7 and 27 are a reason to
describe performance under the fixed budget, not proof of either convergence
or undertraining. They do not authorize more fitting after the result.

The [session record](SESSION.md) preserves the original prediction timeout,
completed seed-7 output, and prediction-only recovery. The eventual
[evaluation completion receipt][completion] binds all three seed outputs to
the same frozen gate and historical baseline. This is an operational recovery,
not another training replicate or a basis for selecting favorable seeds.

The result compares complete recipes with materially different resources:

- LD uses all 12 contiguous raw CLS states, class-supervised auxiliary heads,
  800 source photographs for error-probe fitting and AP-based selection. Its
  learned components contain **922,800 auxiliary-head parameters plus 86 probe
  parameters**, not merely 86 parameters.
- Historical G mean combines four separately fitted graph detectors at layers
  3/6/9/12, each using the graph recipe and final-layer token features. Those
  detectors used 1,600 source photographs for error fitting, a fixed 20-epoch
  budget and AUROC-based selection. G mean itself did not fit a meta head.

The supervised class labels, feature exposure, capacity, optimization and
selection objective therefore differ. The comparison neither isolates graph
topology nor estimates a method's fully tuned optimum. In particular, this
experiment does not isolate the contribution of the seven dynamics
statistics from the numeric class trajectories, and tests no cross-dataset
transfer. The paper's broad hyperparameter search was deliberately not run.

Recommended scientific wording: **On the previously examined 800-source,
nine-view ViT-B development benchmark, the fixed LogitDynamics adaptation
outperformed the preserved GNN mean ensemble in all three trained seeds;
the prespecified paired photograph-bootstrap AUROC interval favors LD.
This demonstrates a competitive alternative internal-signal recipe under
the stated training budgets, with development reuse and unequal supervision
and capacity limiting broader conclusions.**

## September 21 verification completion

The portable-checkpoint replay is now complete for validation and development
across all three seeds, on both CPU and CUDA. CUDA reproduced all original score
values exactly. CPU retained only the known seed-17 validation violation; all
development score vectors passed the unchanged tolerance, with small numerical
metric differences recorded separately. The independent CPU implementation also
recomputed every weighted AUROC in all 2,000 paired bootstrap draws, agreeing
with saved paired differences to 3.33e-16 and the interval within 1e-12.

This closes the previously unexecuted checks and strengthens verification of
the existing results; it is not another scientific experiment. It changes no
model, original score, confidence interval, primary finding or methodological
limitation. In particular, the reused development cohort and unequal method
inputs/supervision remain. The one CPU tolerance failure also remains recorded.
A clean installation and fresh raw-image-to-output pipeline were not tested.
See [the current reproducibility audit](AUDIT_REPRODUCIBILITY.md).

## September 19 disposition after the CPU audit (historical)

The completed comparison can close with its **original GPU predictions and
reported results unchanged**, and an explicit limitation on strict CPU replay.
The later Slurm CPU diagnostic, job **910651**, independently verified all
14 saved AUROC/AP score vectors, seed means/SDs, matching metadata and
historical scores. It also checked the saved bootstrap group/count structure,
within-draw seed aggregation and the reported percentile interval. It did
**not** recompute every weighted AUROC bootstrap draw. These checks strengthen
the evidence for the reported comparison without creating another experiment.
See the [diagnostic receipt][cpu-diagnostic] and the separate
[reproducibility audit](AUDIT_REPRODUCIBILITY.md).

The original strict CPU audit, job **910647**, remains failed: one of 3,600
seed-17 validation records exceeded the unchanged absolute/relative replay
tolerances of 1e-4. Record 15321 had a maximum absolute score difference of
0.0007970333099365234. Fresh published-file checks, native/portable parameter
comparisons, all three training histories/checkpoint selections and all three
training-only scaler reconstructions passed; that does not amount to a pass
of full numerical replay.

The targeted diagnostic found a CPU FP32/FP64 top-five membership change at
depth 4, with a boundary margin of 2.384185791015625e-7. Changes in dynamics
features dominated the resulting score difference. Computing the auxiliary
heads in FP64 on CPU matched the saved GPU score at the original tolerance
for the affected record. This supports sensitivity of discrete class-set
features to numerical precision. Historical GPU intermediate logits/features
were not saved, so the exact historical cause is not proved. The FP64 check
is a diagnosis, not a replacement implementation or a retrospectively chosen
fix.

Neither CPU job establishes complete validation/development prediction replay
for every seed. Accordingly, do not claim universal CPU/GPU numerical
equivalence or a clean end-to-end portable replay. The available evidence does
not require another GPU experiment to retain the scoped result for the
original execution. Any future cross-backend reproducibility claim would
need its own verification. No original score, model, method, tolerance,
checkpoint or statistical conclusion is changed by this disposition.

## Evidence identity

The completed evaluation is Slurm job **910574**. Its receipt identifies
`report.json` SHA-256
`d8daf3374f3fb2bb5279b6383a48a933ce26a1126d7efe66bddb2b9a2ff79112`,
campaign SHA-256
`f133671b92871129a0e859e47188c316e8ef535634a1fda33a9efa287dfa094f`,
and gate SHA-256
`8918ff80c58a74e53a747fb0c054fe18b93e0f3dd06d64ebb1ff1d42741c4ab0`.
The executed scientific source is the preserved
`release-400f1fbbc105a370`, not subsequent reporting edits. Compact JSON/Markdown
evidence is in the coordination workspace's `ops/results_20260919` directory;
heavy arrays and model files remain remote. The download manifest and the
operations audit, rather than this source-level review, establish transfer
and backup integrity.

### Portable evidence

The primary result and source links in this audit point to the private,
immutable Hugging Face revision
**`86605781c0528e286483e305072f7787393da860`** of
`omrifahn/polygraph-experiments`, under
`logit_dynamics_20260919_114500/snapshot_910575`. Permission to that repository
is required. These links retain the reviewed version even if `main` changes:

- [Pinned result report][report] and [evaluation completion receipt][completion].
- [Pinned executed analysis][analysis-source] and [metric implementation][metric-source].
- [Pinned scientific protocol][pinned-protocol] and [source manifest][source-manifest].
- [Pinned backup inventory][backup-manifest], which records the result/source
  paths and their checksums.
- [Pinned full extraction log][extraction-log] and
  [extraction completion receipt][extraction-receipt]. The extraction command
  emits its final manifest in that log; raw images and extracted CLS tensors
  are excluded from this result snapshot.

Absolute paths into the coordination workspace are **convenience mirrors**, not
the sole sources for the findings. The local `ops/results_20260919` copies were
the files read during review, and their identities can be checked against the
pinned receipt/inventory. The local extraction-manifest link remains useful
for inspecting that auxiliary cache record; the archived extraction log and
receipt provide the portable execution evidence. The original full CLS cache
and its manifest also remain at the Slurm experiment root recorded in
[SESSION.md](SESSION.md). Later reporting and reproducibility-audit documents
do not alter the pinned result revision.

[report]: https://huggingface.co/omrifahn/polygraph-experiments/blob/86605781c0528e286483e305072f7787393da860/logit_dynamics_20260919_114500/snapshot_910575/evaluation/report.json
[completion]: https://huggingface.co/omrifahn/polygraph-experiments/blob/86605781c0528e286483e305072f7787393da860/logit_dynamics_20260919_114500/snapshot_910575/evaluation/complete.json
[roles]: https://huggingface.co/omrifahn/polygraph-experiments/blob/86605781c0528e286483e305072f7787393da860/logit_dynamics_20260919_114500/snapshot_910575/role_map.json
[extraction]: run_records/cls_manifest.json
[probe7]: https://huggingface.co/omrifahn/polygraph-experiments/blob/86605781c0528e286483e305072f7787393da860/logit_dynamics_20260919_114500/snapshot_910575/runs/seed7/probe/complete.json
[probe17]: https://huggingface.co/omrifahn/polygraph-experiments/blob/86605781c0528e286483e305072f7787393da860/logit_dynamics_20260919_114500/snapshot_910575/runs/seed17/probe/complete.json
[probe27]: https://huggingface.co/omrifahn/polygraph-experiments/blob/86605781c0528e286483e305072f7787393da860/logit_dynamics_20260919_114500/snapshot_910575/runs/seed27/probe/complete.json
[analysis-source]: https://huggingface.co/omrifahn/polygraph-experiments/blob/86605781c0528e286483e305072f7787393da860/logit_dynamics_20260919_114500/snapshot_910575/source/pilots/logit_dynamics_20260919/evaluate.py
[metric-source]: https://huggingface.co/omrifahn/polygraph-experiments/blob/86605781c0528e286483e305072f7787393da860/logit_dynamics_20260919_114500/snapshot_910575/source/pilots/final_comparison_20260916/evaluate.py
[pinned-protocol]: https://huggingface.co/omrifahn/polygraph-experiments/blob/86605781c0528e286483e305072f7787393da860/logit_dynamics_20260919_114500/snapshot_910575/source/docs/experiments/logit_dynamics_20260919/PROTOCOL.md
[source-manifest]: https://huggingface.co/omrifahn/polygraph-experiments/blob/86605781c0528e286483e305072f7787393da860/logit_dynamics_20260919_114500/snapshot_910575/source/source_manifest.json
[backup-manifest]: https://huggingface.co/omrifahn/polygraph-experiments/blob/86605781c0528e286483e305072f7787393da860/logit_dynamics_20260919_114500/snapshot_910575/backup_manifest.json
[extraction-log]: https://huggingface.co/omrifahn/polygraph-experiments/blob/86605781c0528e286483e305072f7787393da860/logit_dynamics_20260919_114500/snapshot_910575/logs/extract_910058.out
[extraction-receipt]: https://huggingface.co/omrifahn/polygraph-experiments/blob/86605781c0528e286483e305072f7787393da860/logit_dynamics_20260919_114500/snapshot_910575/ops/stage_receipts/extract_910058.json
[cpu-diagnostic]: run_records/cpu_precision_diagnostic_v1.json
