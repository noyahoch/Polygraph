# Independent final scientific review

**Assessment: the completed analysis supports a small positive primary effect for the
tested full-feature MPNN against the matched set detector. It does not resolve whether
the benefit reaches the registered practical margin of 0.005 AUROC.** The secondary
feature interaction points opposite to the proposed larger graph benefit with attention
alone. No result establishes that original graph topology is necessary, that all non-GNN
models are inferior, or that the method detects language-model hallucinations.

This bounded review reads existing, unmodified Slurm outputs only. No new statistics,
model execution, resampling, test scoring or outcome-dependent changes were performed.
All 35 trained fits and scores and analysis job 879965 were reported complete. The final
HF backup is now complete at artifact revision
`a4a40d46b69e63450bc6ddcdfc6cfa2a4916ed6a`. A subsequent Operations receipt records
restoration job 880355 completed successfully at 22:16:50 Israel on September 11.
It verified all artifact/source/dependency bindings (435 downloaded artifact files and
63 source files) and numerical agreement for `full_graph/seed1` on 48 validation rows,
using freshly downloaded source/artifacts with the existing pinned Slurm environment.
The [verification record](RESULTS.md#reproduction-and-verification-scope) gives the exact
receipt and scope: no test rescoring, new training, fresh dependency installation or
numerical reevaluation of all 35 models. This administrative status update does not change
the scientific review below.

## Evidence and identity

The interpretation follows [PROTOCOL.md](PROTOCOL.md), the prior
[statistical implementation review](STATISTICAL_REVIEW.md), and the fixed pre-test
[rewiring decision](REWIRING_DECISION.md). The older startup state in `RESUME.md` is
superseded for scientific analysis by these later completed native outputs.

Native outputs were retrieved by Operations as exact bytes, with the hashes recorded in
the local `retrieval_receipt.json` audit. The portable pinned-artifact inventory is in
[RESULTS.md](RESULTS.md#pinned-artifacts).
All results below refer to freeze SHA-256
`f6795a359184d878fdf4b0c2204c88d3d21bcf5a0c7a20a453158ac31f364473`.

| Native source | SHA-256 |
|---|---|
| `summary.json` | `2f530a7ad7fc27c5091a4b9bf453e0da0d77187665ee65bfde27f7b9cbab2ab0` |
| `bootstrap.json` | `5c76653d30ad88632a7804c6ff11fea2b45cd668805ae0e810d4be022eb3648a` |
| `results.csv` | `209b77f921ca1aa61c43176322168cb1ba9e745bf1cfb842fc6962aeb8fb638d` |
| `evaluation_complete.json` | `16344ee7bf07f7d2f9456ecd2b4de5d1a1703fc49620f137255b17a322657790` |

## Primary comparison and seed variation

[summary.json](https://huggingface.co/omrifahn/polygraph-experiments/blob/a4a40d46b69e63450bc6ddcdfc6cfa2a4916ed6a/topology_20260910/results/summary.json)
reports the arithmetic mean of five paired seed-wise AUROC differences as
**0.006232532**, with paired-photo bootstrap 95% interval **[0.002618290, 0.009835779]**.
All 2,000 primary draws are defined. The bootstrap uses 800 original photographs,
retains each photograph's nine versions together, and shares each draw across models and
seeds. It does not treat 7,200 correlated versions as independent photographs. The
interval reflects source sampling conditional on these five fitted seeds, rather than
uncertainty over arbitrary future retraining.

Values below are copied from the native mixture rows and reported seed differences;
rounding is for presentation only.

| Seed | Full graph AUROC | Full set AUROC | Reported paired difference |
|---|---:|---:|---:|
| 1 | 0.896272 | 0.888666 | +0.007606 |
| 2 | 0.896360 | 0.889981 | +0.006379 |
| 7 | 0.889206 | 0.890371 | −0.001164 |
| 17 | 0.897435 | 0.889337 | +0.008097 |
| 27 | 0.898050 | 0.887806 | +0.010245 |

The reported standard deviation of the five differences is 0.004364503. Seed 7 must
remain visible. The primary interval excludes zero, supporting a positive difference
under this registered analysis, but spans 0.005. Neither “practically meaningful advantage
established” nor “no useful effect” is warranted. A nonzero AUROC difference is also not
a direct guarantee of operational benefit at a particular false-alarm cost.

## Secondary comparisons

These values are the already-produced `comparisons.mixture` entries in
[bootstrap.json](https://huggingface.co/omrifahn/polygraph-experiments/blob/a4a40d46b69e63450bc6ddcdfc6cfa2a4916ed6a/topology_20260910/results/bootstrap.json).
They are prespecified secondary/descriptive intervals and are not adjusted for multiple
comparisons. No adjusted analysis is claimed; these intervals must not be promoted to
confirmatory superiority findings.

| Comparison | Mean AUROC difference | 95% paired-photo interval |
|---|---:|---:|
| Full graph − full endpoint | +0.007366 | [0.003294, 0.011398] |
| Attention-only graph − set | −0.003686 | [−0.008349, 0.001045] |
| Full gap − attention-only gap | +0.009919 | [0.004259, 0.015219] |
| Full graph − rewired graph | +0.000494 | [−0.001945, 0.002906] |

The positive interaction means the graph-versus-set benefit is larger with the rich
feature bundle here. It does not support the suggested larger benefit with attention
alone; the attention-only graph-minus-set point differences are negative for all five
seeds, while its pooled interval includes zero. This does not disprove that intuition
for other architectures, feature definitions or tasks. Hidden states and class-conditioned
features were added together, so their individual contributions are not identified.

The endpoint contrast is consistent with a benefit of the selected MPNN over this
particular endpoint-set architecture. Endpoint records still contain relationships.
Differences in aggregation, optimization and inductive bias remain alternatives to an
interpretation exclusively about iterative message passing. Parameter matching does
not establish equal expressive capacity.

The output baselines also prevent an overbroad claim about unique internal information:
the five logit-only mixture AUROCs are 0.890919, 0.891250, 0.891497, 0.891162 and 0.890736
in seed order 1, 2, 7, 17, 27; MSP is 0.865021 and entropy is 0.867945. These are descriptive
context from native rows, not newly tested pairwise superiority claims. In particular,
full graph seed 7 is below its logit-only counterpart.

## Confidence, all nine conditions and fixed thresholds

The MSP ≥ 0.9 slice retains both outcomes: 5,015 records, comprising 900 errors and
4,115 correct predictions, from 795 photographs. There are 443 distinct photographs with
a confident error and 745 with a confident correct prediction; these sets may overlap.
Both exceed the fixed 200-photo support requirement. Nevertheless the full graph-minus-set
estimate is **0.005421007**, interval **[−0.001198050, 0.011848300]**. The subgroup supports
neither a resolved positive difference nor an established 0.005 benefit. The support floor
is not a power guarantee. No lower confidence cutoff or class search is justified.

Every one of the 35 fits, plus MSP and entropy, has native rows for the mixture, confidence
slice and all nine registered conditions in
[results.csv](https://huggingface.co/omrifahn/polygraph-experiments/blob/a4a40d46b69e63450bc6ddcdfc6cfa2a4916ed6a/topology_20260910/results/results.csv).
Each condition has 800 records from the same 800 photographs. The classifier error counts
and prevalence below are copied from the common condition metadata; they are not selected
according to detector performance.

| Condition | Classifier errors | Error prevalence |
|---|---:|---:|
| Clean | 75 | 0.09375 |
| Gaussian noise, severity 3 | 407 | 0.50875 |
| Gaussian noise, severity 5 | 494 | 0.61750 |
| Motion blur, severity 3 | 224 | 0.28000 |
| Motion blur, severity 5 | 262 | 0.32750 |
| Fog, severity 3 | 117 | 0.14625 |
| Fog, severity 5 | 303 | 0.37875 |
| JPEG compression, severity 3 | 257 | 0.32125 |
| JPEG compression, severity 5 | 320 | 0.40000 |

The mixture contains 2,459 errors among 7,200 records, reported prevalence 0.341527778.
Pooled discrimination can reflect differences in condition difficulty as well as
discrimination within conditions; a pooled positive effect must not be presented as a
benefit established separately for every condition. All corruption families also occur
in training, so these are not unseen-family generalization results.

The registered alarm remains strict `score > threshold`, calibrated on correct validation
records only. The reported validation false-alarm rate is 0.049908704. A fixed threshold
does not guarantee a 5% false-alarm rate in each test condition: for the designated first
fit, `full_graph/seed1`, the native clean rate is 0.009655172 and Gaussian-noise severity-5
rate is 0.120915033. Its mixture error recall is 0.494103294, coverage 0.799583333 and retained
error rate 0.216084766. These explicitly identified seed-1 examples illustrate threshold
transfer; they are not averages over seeds or chosen evidence of a uniform effect.

## Rewiring and limits of interpretation

The frozen full development evidence reports structural integrity passed, but mean
changed-edge fraction **0.503963667** over all 28,800 eligible training/validation graphs,
below the unchanged 0.80 requirement. The weak-graph fraction is 0.778020833. Mixing
quality remains **FAIL / non-diagnostic** despite explicit admission under the bound
pre-test decision. All graphs and all 35 fits were retained.

The small graph-minus-rewired difference describes sensitivity to this realized partial
perturbation. Its interval cannot be used to claim removal of topology, structural
necessity or non-necessity, or equivalence. Degrees, source-associated features and much
incidence remain informative; rewiring also changes destination coherence and paths.
The primary matched graph/set comparison remains interpretable, but this intended
mechanistic triangulation remains unavailable.

The defensible result is restricted to a frozen CIFAR-100 image classifier, this fixed
source-disjoint cohort from an already-used benchmark, the chosen corruption mixture,
these model families and their registered training recipe. Error labels are classifier
mistakes, not corruption indicators. Nothing here directly tests language-model
hallucinations, guarantees deployment reliability, or establishes universal GNN superiority.
No new experiment or changed conclusion threshold is proposed on the basis of these test
outcomes.
