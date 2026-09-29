# Polygraph scientific synthesis and handoff — September 17, 2026

Prepared for Omri from the preserved project reports, protocols, result files, source code and meeting/chat records. This is a retrospective scientific audit, not a new experiment or a literature-priority claim. Numerical results below were read from existing artifacts; no training, inference or statistical recomputation was performed for this synthesis.

## 1. The conclusion an incoming agent should preserve

The project has credible evidence that internal transformer features can improve the tested error detectors beyond simple output confidence, and sometimes beyond a learned full-logit baseline. It has substantially less evidence that graph incidence and repeated message passing are responsible for that improvement.

Omri contributed more than another attempted GNN win:

1. An independent July program established task dependence: ordinary classification-error detection, attack detection and OOD detection are different problems, and their outcomes differ.
2. September backdoor work exposed experimental-validity and threshold-transfer problems. A trained shortcut, visually plausible attention and a high ranking metric do not automatically produce a usable error detector.
3. The September-10th study, executed September 11, found a small positive rich-GNN versus matched-set difference over five seeds, but failed to establish a topology mechanism.
4. The September 14–16 study found a repeated modest benefit from combining separately trained detectors using attention from widely spaced layers.
5. The September 16–17 benchmark then tested whether that ensemble advantage needed message passing. A matched feature-set ensemble obtained almost the same AUROC as the GNN ensemble.

The fifth point is the most useful new refinement within the team’s work: **the promising multi-layer recipe does not currently have a demonstrated GNN-specific advantage.** This is a scientifically informative result. It is not a proof of equivalence, universal GNN irrelevance, or causal attribution of all useful signal to one feature type.

Do not erase the earlier positive GNN result to make a neat negative story. Do not erase Yishai’s earlier set/endpoint/rewiring controls to manufacture novelty for Omri. The contribution is a controlled extension and clarification of existing questions, with a materially different layer-combination study.

## 2. What the task is, and what each comparison can identify

In the main September experiments, a frozen ViT classifies CIFAR-100 images, including selected corruptions. A second model predicts whether the ViT’s classification is wrong. Its AUROC measures how well it ranks erroneous predictions above correct ones; it is not classification accuracy and does not by itself establish a usable alert threshold.

Use two broad scientific questions, with an explicit distinction inside the second:

- **Useful information:** Which output or internal measurements help predict errors? MSP/entropy, all logits, hidden tokens, attention values, and class-conditioned edge information are different inputs.
- **Useful graph modeling:** Does the tested graph architecture add value beyond alternative processing of comparable information? A stronger mechanistic subquestion asks whether the specific connectivity/message passing causes the gain.

Three comparisons have different meanings:

- A GNN beating MSP or a logit MLP establishes superiority of the entire tested recipe: information, capacity, pooling, optimization and architecture may all differ.
- GNN versus a capacity-matched set model with identical node and edge values narrows the explanation considerably. It still compares different encoders/readouts and optimization behavior; it does not isolate a single graph operation.
- Connectivity perturbations or endpoint controls address a more specific mechanism. A weak perturbation, inference-only distribution shift or unmatched feature-message association can prevent a causal interpretation.

Attention alone is also ambiguous. In the latest G/S experiment the inputs include final-layer hidden tokens, coordinates/CLS identity and attention diagonals, not just attention edge weights. Say exactly what was exposed.

The July attack and OOD studies and Noya’s POPE work use different labels. Keep them outside a single classifier-error leaderboard.

## 3. Source hierarchy, project structure and attribution

Prefer, in order: actual numerical artifacts and executed configuration; frozen protocol and recorded amendments; contemporaneous technical reports; later summaries; conversation recollection. Protocols explain intended design, while receipts and results determine what actually finished. A report’s filename is not necessarily its completion date.

Local roots:

- Outer course project: /Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project
- Shared team checkout: /Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph
- Omri’s separate checkout: /Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph-September-10th
- Earlier independent code: /Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/Early_Work_July_2026/Experiments
- Preserved context: /Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/Context
- Coordination workspace and downloaded remote receipts: /Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and

The nested Polygraph checkouts are separate Git repositories. Do not stage them through the outer course repository. At the start of this audit Omri’s checkout was on September-10th at d4f0185. Two pre-existing untracked artifact directories were present: docs/experiments/final_comparison_20260916/results/ and pilots/results_final_20260917/. This synthesis does not take ownership of their contents.

Use the [context source ledger][context-ledger] and [September 10–11 meeting synthesis][meetings] to reconcile people and dates. Yishai explicitly reported that his commits appeared under Noya’s name in the [WhatsApp export, line 627][chat]. Therefore Git author strings do not establish that Noya wrote everyone’s code. Noya’s [September 11 transcript][noya-sync] explains her work and limits what she personally checked about the backdoor branch. Eran’s documented contribution is literature/writing and research discussion; this audit has not verified a separate Eran experimental result. That is not a claim that he contributed nothing.

Compact chronology:

| Period | Work represented | Provenance boundary |
|---|---|---|
| July: initial work from July 11 | Omri independent small-ViT program | Phase-one closeout July 24; phase-two launches July 24–25; exact final completion dates not verified from job receipts |
| July / handoff updated August 16 | Yishai clean-image POC | Small exploratory sample, not the later corruption benchmark |
| August–early September | Noya scaled benchmark, generalization, combinations, VLM and backdoor branches | Different branches/tasks; not every planned detector comparison finished |
| September 4 | Yishai evidence-flow report completed | Filename says August 30; report commit eb1da9b |
| September 6 | Yishai topology/depth/last-four report completed | Filename says September 5; report commit 495ea05, shared September 6 |
| September 5–7 | Omri trigger diagnostics and clean-trained detector robustness | Separate classifier perturbation study |
| September 11 | Omri “September10” controlled study | Branch/namespace date differs from execution date |
| September 13–14 | Layer cache/preparation and superseded wider screen | Engineering/preflight is not a completed scientific comparison |
| September 14–15 | Four single-layer GNNs, seed 7, then late diagnostic evaluation | Preserve original deadline miss |
| September 16 | Seeds 17/27 layer-ensemble replication | Same development photographs, new fits |
| September 16–17 | Final G/H/S/output comparison | Imports earlier G fits; fresh H/S/O fits; prediction-only recovery completed |

## 4. Colleagues’ research strands

### 4.1 Yishai’s initial POC: is attention-graph error prediction feasible?

**Question.** Can a small detector learn error-related signal from a frozen ViT’s attention graph, and where should it read that graph?

**Design and result.** The [original handoff][poc] records 852 errors among 10,000 clean test images, detector training on 1,500 balanced examples and evaluation on 200 balanced examples, with no validation group and fixed 30 epochs. Last-layer mean-readout AUROC was 0.7476 in one layer sweep. A readout ablation improved from 0.7252 to 0.7898 using CLS plus gated pooling. Shared multi-layer variants were worse: last-four 0.7603 and last-two 0.7366 versus last-only 0.7898. MSP was 0.9101; adding the graph produced 0.9112 ± 0.0046 across three seeds.

**Shown.** The graph detector learns something; pooling/layer choices matter under this implementation. No clear incremental advantage over MSP was established.

**Not shown.** That the chosen layer/readout generalizes, that additional layers contain no useful information, or that graph topology is necessary. Repeated selection on 200 test examples, including only 100 errors, makes this feasibility exploration.

**Overlap/newness.** This was the initial operationalization of the project idea. Omri’s later layer experiments overlap the layer question but change the combination strategy and validation discipline substantially.

### 4.2 Noya’s scaled corruption benchmark: does the idea survive a serious error population?

**Question.** Does graph error prediction remain useful beyond a tiny balanced clean sample, and how does it compare with confidence and internal representations?

**Design and result.** [TEAM_REPORT][noya-team] and [CONSOLIDATED_STATE][noya-state] document a 1.01-million-verdict scan and a 75,000-record detector benchmark, with source-image-disjoint groups of 52,000/6,000/17,000 and balancing by correctness within corruption/severity. Main single-seed results include MSP 0.8695, attention GNN 0.8417, flat-attention MLP 0.7421, CLS MLP 0.8742, a 12-layer CLS GRU 0.8759 and graph-plus-token-hidden fusion 0.8758. A 12-layer CHARM-lite union scored 0.8233. A later DeepSets control scored 0.8127.

**Shown.** A substantial benchmark and baseline ladder were built. Rich representations modestly improved on confidence in that setting. The specific graph detector beat the specific flat/set controls, while increasing graph size/capacity or unioning layers was not automatically beneficial.

**Critical correction.** The original flat baseline did not receive all the GNN’s information: Yishai later identified missing coordinates, CLS identity, layer information and attention diagonals. Also, Noya’s “output MLP” used two summary features, not all 100 logits. Therefore neither a pure topology win nor an “output-information ceiling at MSP” follows. Equal edge counts are not equal information. See [Yishai’s audit][evidence-flow], especially its baseline critique.

**Overlap/newness.** Scaling and realistic benchmark construction are real contributions. Omri did not invent this baseline ladder; his later matched controls refine its interpretation.

### 4.3 Noya’s held-out corruption and complementarity studies

**Questions.** Do learned detectors generalize to unfamiliar corruption families? Can two imperfect rankings complement one another?

**Results.** In a weather-family holdout, unseen weather plus extras gave MSP 0.8885 versus graph-hidden fusion 0.8738, CLS MLP 0.8801, CLS sequence 0.8731 and pure graph 0.8380. A separate [unseen comparison][noya-unseen] found graph approximately 0.843 versus CLS probe 0.871, with graph-minus-probe interval [−0.038, −0.018].

The [complementarity report][noya-complementarity] and [G1 exploitation report][noya-g1] examined different captured/missed errors and combined scores. Five-fold OOF stacking on stored test scores gave MSP+CLS sequence 0.8829, plus graph 0.8845, and plus graph-hidden fusion 0.8895. Pure graph’s measured marginal increment was about 0.0016.

**Shown.** Familiar-corruption gains can disappear under family shift. Rankings can provide incremental predictive value even when a standalone model is weaker.

**Not shown.** “Structure generalizes, representations memorize” is not established. Catching some different errors is not enough: the graph can lose more errors than it recovers. OOF fitting on previously designated test scores is a diagnostic reuse of data, not a new untouched-test result; the report does not establish source-group folds/bootstraps for that G1 analysis. The measured 0.0016 is not a causal decomposition of topology’s contribution.

**Overlap/newness.** These are substantive generalization and conditional-value questions. Omri’s current nine-condition experiment, with all families represented in training, does not replace this unseen-family test.

### 4.4 Noya’s POPE/VLM branch: confident-error signal outside CIFAR

**Question.** Can internal representation detect confidently wrong answers in a vision-language model?

**Results and self-correction.** The [pilot][pope], [G1 report][noya-g1] and [category-holdout report][pope-category] cover LLaVA/POPE probes. An initially spectacular transfer number was discarded after discovering 76% duplicate questions. Deduplicated image-group OOF analysis gave internal probe 0.799 versus output 0.795 overall, but 0.790 versus 0.591 in the confident slice, containing 87 errors. Under category holdout the probe lost overall, 0.660 versus 0.775, while the confident slice favored it, 0.712 versus 0.522, with only 36 errors.

The distinction between absent-object hallucinations and present-object misses also mattered: the output detector was better for the former, while the internal probe was better for the latter in the reported breakdown.

**Shown.** Particular internal representations can help in a specific confident-error regime. The de-duplication audit and separation of error types materially improved validity.

**Not shown.** A GNN benefit: no completed POPE GNN comparison was found. Small slices, one backbone/recipe and adaptive exploration limit broad claims. This is not automatically a universal law about hallucinations.

**Overlap/newness.** A different task/domain and failure regime, not redundant with the CIFAR benchmark. Omri’s latest study does not replicate it.

### 4.5 Noya’s checkerboard testbed

**Question.** Can a learned trigger create confident errors for a detector study?

**Result.** The [testbed report][noya-backdoor] records a 16-pixel bottom-right trigger, 2% poisoning, target class 0 and two epochs: clean accuracy 0.9145, attack success 0.9995. The [routing protocol][noya-routing] recognized that almost no correct triggered cases remained and that fixed-target class identity could confound the detector task. A rotating-target/intermediate-ASR testbed was proposed.

**Shown.** The classifier learned a strong shortcut.

**Not shown.** That a graph detector detects those errors or that attention routing is the causal explanation. The detector comparison was unfinished. Noya’s transcript says she had not examined this delegated branch deeply.

**Overlap/newness.** Omri’s September work continued and audited this unfinished testbed rather than originating it.

### 4.6 Yishai’s evidence-flow study: stronger baselines and better features

**Question.** Is the improvement really graph structure, or are earlier baselines missing informative outputs/internal values?

**Results.** The [report completed September 4][evidence-flow] rebuilt the 75,000-record store, so its MSP 0.86510 should not be mixed with the historical 0.8695.

- Full-logit MLP: ordinary mean 0.87893; strict reference 0.87941.
- Matched attention-only controls: EdgeSet 0.84736, NodeEdgeSet 0.84556, EndpointSet 0.83754, versus raw graph M0 0.83477.
- Graph feature ladder: M0 0.83477; adding message magnitude 0.83685; class-conditioned direction 0.86602; compact node evidence 0.87725; full-hidden M5 ordinary 0.88574 and strict 0.88446.
- Strict output+M5 combination: 0.89128, using separate base/meta validation and grouped gate selection.

Class conditioning projects a transferred vector onto the difference between the predicted class and runner-up class directions. It describes whether that vector aligns with the eventual decision; it is a predictive proxy, not demonstrated causal attribution.

**Shown.** All logits are a stronger baseline than confidence summaries. Class-aligned/internal features matter substantially more than unsigned magnitude in this recipe. Matched non-graph controls undermine a simple topology interpretation. An internal-plus-output combined predictor can improve on full logits.

**Not shown.** Necessity of graph processing or causal “evidence flow.” The successful predictor bundles several ingredients.

**Overlap/newness.** This already tested raw-attention matched sets, rich information, endpoints and output competition. Do not credit those research questions as first introduced by Omri.

### 4.7 Yishai’s topology, operator, depth and last-four follow-up

**Question.** After enriching features, is a GNN still needed? Can a better operator or multi-layer graph improve the result?

**Results.** The [report completed September 6][topology-depth] records strict TransformerConv M5 0.88446, rich NodeEdgeSet 0.88339, EndpointSet 0.88132 and a single-seed HiddenTokenSet 0.88384. Capacities were not tightly matched. Perturbing edges/attributes reduced scores modestly; retraining on perturbed inputs recovered some performance.

Validation chose a two-layer edge-gated mean operator. Its final mean standalone score was 0.88832; output+edge-gated fusion reached 0.89391 versus full-output 0.87941. The reported standalone mean gain, 0.00891, fell below that study’s declared 0.01 target. The gate result was stronger and held up in the reported weather tests.

A union of the final four consecutive transformer layers, one-based 9–12, scored 0.88203 versus 0.88942 for the selected single-layer graph in that comparison. The union-set run was stopped early and some sequence variants were never launched.

**Shown.** Operator/representation design matters. A useful combined predictor was established under the report’s protocol. The tested last-four union did not improve on its single-layer comparator. Rich set controls were competitive.

**Not shown.** That every graph operator is unnecessary: the closest set comparison used TransformerConv M5, not an equally exhaustive matched-set comparison of every selected winning operator. Nor that all multi-layer approaches fail. Connectivity perturbations mix topology, message alignment and input shift.

**Overlap/newness.** Omri extends an already sophisticated critique, especially by testing the selected edge-gated family with close capacity matching and then crossing graph/set alternatives with independently trained spaced-layer ensembles.

## 5. Omri’s earlier research strands

### 5.1 July ordinary error detection, layers and token content

**Question.** On an independently fine-tuned small ViT, can attention graphs predict actual classification mistakes better than outputs, and do layers/token content help?

The [July final summary][july-summary], [raw metrics][july-metrics] and [backbone report][july-backbone] describe a ViT-S at 128 pixels, 65 tokens and 6 heads, clean accuracy about 89.93%. Source groups were 6,000/2,000/2,000, with eight seen and six unseen corruption families. Final error evaluation contained 86,000 views from 2,000 photographs. This is not the September ViT-B benchmark.

Results: MSP 0.873639, entropy 0.877745, logit-MLP approximately 0.8786 across three seeds, attention statistics 0.811440, last-layer GNN 0.808660, last-four 0.773862 and all-12 approximately 0.7731 across three seeds. On a restricted 30,000-row token-feature ablation, adding PCA48 token embeddings improved 0.753957 to 0.805119.

**Shown.** The tested graph recipe was inferior to strong output controls. Added token content helped that graph detector. Joint/shared integration of more layers did not help here.

**Not shown.** An information-theoretic limit of attention, absence of optimization issues, or uselessness of earlier layers. A handful of architecture trials cannot rule out all optimization/capacity explanations. A smaller seen-to-unseen drop does not establish better robustness when absolute performance remains much lower.

**Overlap/newness.** An independent precursor on another backbone, supporting task/representation caution. Its broad question overlaps colleagues’ work, but its numerical results are not reproductions of theirs.

### 5.2 July post-hoc combination and value-weighted edges

The reported rank blend of 75% logit detector and 25% GNN scored 0.8837 versus logit-only seed 0 at 0.8784. The weight was selected from five options on the test scores, explicitly disclosed in [the final summary][july-summary]. Treat this as a complementary-signal hint, not a clean held-out improvement.

The [phase-two raw CSV][july-phase2] corroborates value-weighted attention results: last-layer 0.811332 and all-12 0.791018. Last-layer improvement was small and below the declared 0.006 progression criterion. Compare one-seed value-weighted/all-12 against the corresponding raw seed when making a paired claim, not against a three-seed average.

**Shown.** A possible combination benefit worth testing with a dedicated meta group; value weighting helps some configurations but does not close the output-baseline gap.

**Not shown.** That message passing caused either gain. Edge weighting/sparsification changes also matter. Unsigned value magnitude is not the same idea as Yishai’s later class-conditioned signed projection.

**Newness.** Precursor evidence and ablations; later frozen meta-fitting improves directly on the test-chosen July blend.

### 5.3 July attack detection

**Question changed.** Clean versus successfully attacked input, not correct versus incorrect classification. Clean errors remain negative and unsuccessful attacks are excluded. Attacks target the classifier, not the detector.

The [raw phase-two metrics][july-phase2] give all-12 value-weighted GNN 0.997431 and raw GNN 0.993492; last-layer variants about 0.851/0.844; conventionally oriented MSP/entropy about 0.330.

**Shown.** Very strong separability of this dataset with these detectors.

**Not shown.** A GNN mechanism, adaptive attack resistance, or general error detection. No same-information set or strong learned-logit attack detector was compared. Below-chance MSP with a fixed conventional orientation is not absence of information: the ordering is inverted.

There is an additional source-visible confound. [Attack generation][july-attack-code] and [its report][july-attack-gen] use unclamped float bicubic resizing for clean negatives, while attacked positives are clipped and re-quantized to uint8 at 128 pixels. A preprocessing difference may contribute to separability. This does not disprove the numerical result, but prevents interpreting 0.997 as validated “routing fingerprint” evidence without controls.

Some intermediate phase reports remain marked IN PROGRESS; completed rows exist in the ignored phase-two CSV. Do not repeat an initial file-search miss as missing numerical evidence.

### 5.4 July OOD detection

**Question.** Can the detector separate in-distribution CIFAR images from a different dataset, including an unseen OOD family?

SVHN was seen during detector training; DTD was test-only. The [phase-two CSV][july-phase2] includes stronger all-12 results omitted from the final summary’s short table:

- Raw last-layer: SVHN 0.997405 / DTD 0.882504 / pooled 0.955971.
- Raw all-12: SVHN 0.999869 / DTD 0.912621 / pooled 0.968406.
- Value-weighted all-12: SVHN 0.999968 / DTD 0.905167 / pooled 0.965781.
- Mahalanobis CLS reference: SVHN 0.940076 / DTD 0.969203 / pooled 0.950579, in [the original metrics][july-metrics].

**Shown.** Excellent seen-OOD performance, but worse unseen-DTD discrimination than the representation-distance reference. A pooled GNN win hides that reversal.

**Not shown.** Universal OOD generalization, classifier-error prediction, or GNN necessity. This is a different label and sampling problem from corruption errors.

**Newness.** Useful empirical task dependence, not a direct answer to the course’s graph-mechanism question.

### 5.5 September 5–6 checkerboard diagnostics

Sources: [Omri’s running report][omri-sept5] and [fixed-target result][fixed-target].

**Clean negative control.** On 500 calibration photographs, clean correctness was 464/500 versus bottom-right checker 462/500. The error-rate difference was +0.4 percentage points with interval [−0.2, +1.2]. Attention rollout selected the patch in 16.8% of bottom-right cases even without poisoning. A salient checker in attention is therefore not by itself evidence of a learned backdoor mechanism.

**Rotating target.** One-epoch 2% poisoning toward (label+1) mod 100 yielded only 1.70% intended-target ASR. Most triggered errors were other wrong classes. This failed the intended intermediate-ASR testbed criterion, not a GNN evaluation.

**Tiny diagnostic.** Sixteen paired clean/triggered examples could be memorized with 100 updates. This shows the implementation can fit a tiny mapping, not that the low-poison full-data recipe should generalize.

**Training-source diagnostic.** The rotating model barely fitted the intended target even on its poison training subset. Pure train-to-test generalization failure was not established.

**Fixed target.** The all-CUDA one-epoch control achieved 99.85% ASR, overlapping Noya’s prior strong fixed-target testbed, but left only 23/2,000 correct triggered calibration examples, 20 in true class 0. It was too successful an attack to support the intended within-trigger error comparison. Backend/training history differed from the rotating run, so target mapping alone was not causally isolated.

**Newness.** These validity controls and calibration diagnostics added understanding. They did not produce a new graph detector result. A technically successful backdoor does not automatically provide an environment with sufficient correct and incorrect triggered examples for the intended comparison.

### 5.6 September 6–7: frozen clean-trained detector robustness

Best source: [final scientific review][robustness-review]; exact design in [frozen protocol][robustness-protocol].

A clean-base-trained full-logit MLP, seeds 7/17/27, and confidence baselines were evaluated on three classifiers (base, rotating C, fixed-target), each with clean/checker inputs. The same 3,000 final photographs were paired across all six cells. Alert thresholds were chosen once on clean validation at approximately 5% correct-case false alarms.

- Base clean: MSP AUROC 0.9151; MLP 0.8193/0.8300/0.8336. The MLP was already substantially worse before transfer.
- Rotating-C checker: MSP 0.9150; MLP 0.5185/0.3995/0.4414.
- At frozen thresholds in that cell, MSP/entropy rejected every image, including 360 correct examples; the MLPs accepted everything and missed all 2,640 errors.
- Fixed-target checker: 2,974 errors and only 26 correct examples. The predeclared support rule withheld AUROC. MSP caught 13 errors; MLPs caught 0/0/1. Class-conditioned correct support was extremely sparse.

**Shown.** Ranking performance and operating-threshold transfer are different properties. High AUROC can coexist with unusable all-reject behavior, while a learned threshold can accept catastrophic errors after model/input changes.

**Not shown.** That all learned error detectors fail or that colleagues’ corruption-trained MLPs are contradicted. The weak clean-trained baseline and changed classifier regimes limit that inference.

The paired graph-versus-bag branch failed its save/restore readiness check and never produced a valid matched comparison. Treat this as an incomplete branch, not evidence against GNNs. See [terminal graph review][robustness-graph].

**Newness.** A genuinely distinct operational-robustness question and a concrete warning against equating AUROC with deployed usefulness. Relevant to the project’s broader motivation, but not its strongest graph-specific evidence.

## 6. Omri’s controlled September studies

### 6.1 “September10,” executed September 11: same values, different processors

Sources: [protocol][sept10-protocol], [results][sept10-results], [final review][sept10-review] and [raw summary][sept10-raw].

**Design.** 4,000 official CIFAR-100 test-source photographs selected by SHA-256 ordering of polygraph-20260911:{base_index}; 2,400 train, 800 validation, 800 test. Nine versions per source: clean and four corruption types at two severities. No correctness balancing. Same source and its variants stay together. All families occur in all roles; this is not unseen-corruption generalization or a completely new benchmark.

Thirty-five fits: seven arms × seeds 1/2/7/17/27. Last transformer layer only. Rich features: 784 node values and 36 edge values, including hidden states and class-conditioned/magnitude information. Raw features: 16/12. Arms: rich graph, rich rewired graph, rich set, endpoint set, raw graph, raw set and 100-logit MLP. Internal detectors do not directly receive the full logit vector. Rich MPNN uses the edge-gated family, with close parameter matching to its controls. Maximum 60 epochs, patience 8, validation-selected checkpoint.

**Main question.** Does the rich graph architecture beat a non-message-passing processor of the same node/edge values?

**Primary result.** Mean rich graph minus rich set AUROC = +0.006233, paired photo-bootstrap 95% interval [+0.002618, +0.009836]. Four seeds favored graph, one did not. This is a genuine small positive architecture-level finding under that configuration. The interval crosses the declared practical margin 0.005, so a gain at least that large remains unresolved.

**Other questions/results.**

- Graph minus endpoint set: +0.007366, interval [+0.003294, +0.011398], secondary.
- Raw graph minus raw set: −0.003686, interval [−0.008349, +0.001045]. Raw point gaps were negative at all five seeds.
- Rich-minus-raw graph advantage interaction: +0.009919, interval [+0.004259, +0.015219], secondary and not a multiplicity-adjusted family claim.
- Rich graph minus rewired: +0.000494, interval [−0.001945, +0.002906], but rewiring changed only 50.396% of edges against a required 80%. It was classified non-diagnostic before evaluation; the null cannot establish connectivity irrelevance.
- High-confidence slice graph-minus-set: +0.005421, interval [−0.001198, +0.011848], unresolved.

**Interpretation.** This supports a modest benefit for this rich graph architecture, not causal necessity of topology. It does not support the simple hypothesis that graphs help most when deprived of rich features. The rich feature bundle changes multiple things together; the interaction does not identify which ingredient causes it.

**Overlap/newness.** Same broad question as Yishai, but a new closely matched control suite, five-seed assessment, endpoint comparison and explicit feature-regime interaction on another cohort. Rewiring’s intended mechanism test did not succeed. Thirty-five preserved models and a restoration audit improve reproducibility, not conceptual novelty by themselves.

All 35 fits were verified in private HF for this campaign, and one restoration check used 48 validation rows. Recorded allocation was 31.005 GPU-hours across the campaign; this is not measured productive GPU utilization. Read the campaign’s publication/restoration reviews before reproducing rather than assuming this backup covers later campaigns.

### 6.2 September 13 preparation is not another completed experiment

The [engineering status][sept13-status] and [layer-screen state][sept13-state] describe a wider initial screen, resource admission, extraction/cache and numerical/throughput audits. The wider union configurations were superseded by the smaller core.

These are meaningful engineering assets, but passing a preflight or preparing graphs does not establish a scientific result for a candidate. Do not count untrained union/all-layer candidates as negative findings or inflate the study count with technical retries.

### 6.3 September 14–15: independent layer detectors and score-level combination

Sources: [core protocol][sept14-protocol] and [core report][sept14-report].

**Question.** Rather than joining transformer layers into one graph, can separately fitted layer detectors provide useful scores when combined?

Four detectors use attention layers 3/6/9/12. Each still receives final-layer H12 token representations; this is not a sweep of matching-layer hidden states. Four roles contain 1,600 train / 400 checkpoint / 400 combiner / 800 development-evaluation photographs. All nine views remain grouped. Seed 7, exactly 20 epochs, best checkpoint from the checkpoint role. Fixed logistic stack fitted on the combiner role versus an identically specified final-layer-only logistic control; mean sigmoid scores are secondary.

Learned stack AUROC 0.891046 versus last-only 0.884395; difference +0.006651, interval [+0.003500, +0.010041]. Fixed mean scored 0.891193.

Four training jobs ran in parallel September 14 for roughly 2h24–2h36 each. Evaluation ultimately completed September 15 around 09:00, after the original deadline. Preserve the official incomplete-at-deadline and complete_late_diagnostic history; lateness alone does not invalidate a frozen numerical comparison.

**Shown.** A promising late-fusion recipe with cleanly separated meta fitting, unlike the July test-chosen weight.

**Not shown.** That GNNs are required, that a learned stack is better than an average, or that the gain is uniquely due to layer diversity. Four models are compared with one; there is no matched four-model final-layer ensemble.

**Newness.** Materially different from Yishai’s last-four consecutive union and Noya’s CLS trajectory: four widely spaced attention layers, separately trained detectors and score-level combination.

### 6.4 September 16 replication: does the same direction repeat?

Sources: [replication protocol][rep-protocol], [results][rep-results], [run record][rep-run].

Eight new fits cover seeds 17/27 at four layers. All finished 20 epochs; the recorded guardian completed at 10:03:09 Israel. Seed 17 stack minus last-only was +0.005323, interval [+0.000715, +0.010088]; seed 27 +0.004059, interval [+0.001692, +0.006513]. The new two-seed primary average was +0.004691, interval [+0.001936, +0.007565]. All three seeds combined are a separate descriptive result, +0.005344.

**Shown.** Repeated direction of the combination gain across additional trained seeds.

**Not shown.** Independent-data confirmation: the same development photographs were reused. Conditional photo-bootstrap uncertainty is not a full uncertainty distribution over future training seeds. Fixed averaging slightly exceeded learned stacking in every seed; one individual layer also exceeded the learned stack at seed 27. Retain the registered comparator rather than changing it after inspection.

**Newness.** Replication of the new integration recipe; importantly, it set up a discriminating follow-up: can the same layer advantage be obtained without graph processing?

## 7. September 16–17 final benchmark: the most informative extension

Sources: [final protocol][final-protocol], [executed report][final-report], [exact method scores][final-methods], [primary contrasts][final-contrasts], [training diagnostics][final-diagnostics].

### 7.1 Exactly what was matched

- G: edge-gated mean GNN on attention from layer 3, 6, 9 or 12, always with final H12 token vectors; two message-passing layers, width 64.
- S: the identical 784-dimensional node-value and 12-dimensional edge-value multisets as G, without endpoint assignment or repeated message passing; width 84.
- H: all 197 hidden token vectors from the corresponding layer plus coordinates/CLS flags, no attention; width 96.
- O: MLP on all 100 logits, 100→64→32→1.
- L: deterministic regularized logistic regression on the 100 logits, a secondary baseline.
- MSP and entropy: analytic output references.

G/H/S capacities were 130,434 / 129,986 / 131,126 parameters. O had 8,577. This is close capacity matching within the internal-model family, not a perfect fairness guarantee or a compute-matched comparison against O.

Code anchors: [feature materialization][layer-data], [extraction][layer-extract], [final model construction][final-models], [final data handling][final-data] and [set-model implementation][training-models]. G’s actual operator is edge_gated_mean, not an older proposed TransformerConv. The latest edge features are 12 raw per-head attention values, not the older rich 36-vector.

Important asymmetry: early G/S already see H12, while early H sees H3/H6/H9. Therefore G/H is not a clean attention-added-to-identical-hidden-state ablation.

### 7.2 Accounting and statistical scope

Same 1,600/400/400/800 source roles and nine views as the layer study. Three seeds 7/17/27. Exactly 20 epochs, selection by checkpoint-role AUROC. Fixed stack settings and scalers fitted only on the corresponding training roles.

There are 25 reported methods, not 25 independent studies. Thirty-nine neural fits comprise 12 imported G fits and 27 fresh H/S/O fits. Six G meta heads were imported and 12 fresh H/S heads were fitted; L adds one linear fit. The final benchmark is not another independent replication of the imported graph results.

The designated original test was not opened in this campaign; development evaluation had already been inspected in preceding work. Do not silently reinterpret “closed in this campaign” as a claim that every project artifact called test has never been evaluated: the September-10th study already has its own reported test results. The present benchmark is a controlled follow-up on reused development data, not a final independent-data confirmation.

Six registered primary contrasts use 10,000 shared paired bootstrap resamples grouped by source photo, averaging within-seed differences, with Bonferroni-adjusted intervals for that family of six. These intervals address photo-sampling variability conditional on fitted models, not all future-seed or adaptive-analysis uncertainty.

No equivalence margin was declared here. Do not import 0.005 from the September-10th protocol and retroactively call this a registered equivalence test.

### 7.3 Results and what each comparison means

| Method | Mean AUROC, seeds 7/17/27 |
|---|---:|
| GNN, equal-mean four-layer ensemble | 0.888895 |
| Same-value set, equal-mean ensemble | 0.888559 |
| GNN, learned stack | 0.888471 |
| Same-value set, learned stack | 0.887942 |
| GNN, last-only logistic control | 0.883127 |
| Hidden tokens, last-only logistic control | 0.880381 |
| Same-value set, last-only logistic control | 0.879744 |
| Full-logit MLP | 0.875708 |
| Hidden tokens, learned stack | 0.875702 |
| Entropy | 0.864851 |
| MSP | 0.861943 |
| Hidden tokens, equal-mean ensemble | 0.855308 |
| Full-logit linear model | 0.815501 |

Primary contrasts:

| Paired contrast | Mean ΔAUROC | Multiplicity-adjusted interval |
|---|---:|---|
| G stack − S stack | +0.000529 | [−0.003005, +0.004159] |
| G mean − S mean | +0.000336 | [−0.002800, +0.003409] |
| G stack − H stack | +0.012770 | [+0.007027, +0.018884] |
| G mean − H mean | +0.033587 | [+0.024244, +0.043216] |
| G stack − O | +0.012764 | [+0.006519, +0.019092] |
| G mean − O | +0.013187 | [+0.006879, +0.019715] |

**Same-value graph versus set is the key mechanism-oriented comparison.** The two ensemble recipes are numerically extremely close; G−S changes sign at seed 27. No extra GNN benefit is demonstrated. A competitive construction without incidence processing/repeated messages achieves nearly all the measured performance. This is stronger evidence than merely observing a nonsignificant result from a very weak comparator. It still does not prove exact equivalence or universal irrelevance.

**Graph versus outputs establishes a practical recipe gain.** The tested G ensemble beats the tested full-logit MLP under the specified regime. It also uses richer information and four larger models. The comparison cannot allocate the gain uniquely to attention, message passing, size or ensemble count. S’s similar score is crucial to interpretation.

**Graph versus hidden-state ensembles needs qualification.** H12 alone, 0.880381, beats both H ensemble recipes. Early H models are much weaker: approximately 0.7261/0.7548/0.8052 for H3/H6/H9. The large G-mean versus H-mean gap must not be described as superiority over the strongest hidden-only model or as a pure attention effect. Input exposure differs.

**Layer-combination gains are not unique to GNNs.** S mean 0.888559 versus S last-only 0.879744 is a descriptive improvement as well. This supports an effective collection/combination of layer-derived measurements, but there is still no same-layer four-model ensemble control to separate layer diversity from ensemble/training diversity completely.

**A learned combiner was not necessary for the observed point gain.** Equal mean numerically exceeded learned stacking for G and S. That is descriptive, not a registered primary mean-versus-stack claim.

### 7.4 Training, timing and completion caveats

All G/H/S fits selected checkpoints before epoch 20. Only O seed 27 met the predeclared budget-sensitivity warning by selecting epoch 20 with the specified improvement/loss behavior. Early H checkpoints often came from epochs 1–5, versus later H12 checkpoints. This is compatible with generalization/recipe difficulty; it is not a proof of its cause. No warning does not prove convergence, and a warning does not prove inadequate training is the whole explanation.

Do not claim clean G/S speedups from the cost table: G telemetry was imported, and H/S measurements include their own loading and four-forward-pass conventions.

The final campaign was authorized September 16 at 22:06:33 Israel. Six development-prediction groups initially exhausted a 90-minute cap after heavy shared-storage verification, reaching 6,744/7,200 records without a committed complete output. A September 17 12:15:22 resume reran those predictions and analysis with zero new fits/heads and unchanged frozen identities. Sources: [execution record][final-execution], [resume record][final-resume] and [completion receipt][final-complete].

Analysis job 904074 finished; results were downloaded and verified September 17 at 15:09. A later guard bookkeeping failure led to a CPU-only certification repair, not new scientific selection or training. Code history records a756f5f and d4f0185; certification jobs 904967–904969 checked completion while retaining the old failed history. The reconciled server terminal recorded complete around 16:39 Israel.

Authoritative remote namespace:

/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/final_comparison_20260916_215019

Certification receipt:

/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/final_comparison_20260916_215019_certify_control/manifests/terminal.json

The final campaign is complete. The imported seed-7 G result retains its historical late-diagnostic provenance; the entire final campaign should not inherit the old September-14 deadline. A completed bookkeeping gate is evidence of artifact completeness, not a substitute for the scientific limitations above.

## 8. Did Omri add something new, or only overlap?

| Omri strand | What overlaps | Distinct contribution within this project | Strength of conclusion |
|---|---|---|---|
| July ordinary errors | Core GNN-versus-output question | Independent backbone, broad baselines, feature/layer ablations | Negative for tested recipe; no universal mechanism |
| July attacks/OOD | Broad internal-signal motivation | Different target labels reveal task-dependent outcomes | Real measurements, important controls missing |
| Checkerboard diagnostics | Noya’s backdoor testbed | Clean patch control, calibration failures, eligibility problem | Improves validity; not a GNN win |
| Frozen detector robustness | Confident-error motivation | Paired six-cell threshold-transfer study | Strong example separating AUROC and deployed rejection |
| September-10th controls | Yishai already had sets/endpoints/rewiring | Closely matched edge-gated controls, five seeds, rich/raw interaction | Small positive architecture effect; rewiring non-diagnostic |
| Spaced-layer ensemble | Colleagues had layer sweeps/unions/trajectories | Separate detectors at 3/6/9/12 and meta-trained late fusion | Repeated gain vs final-only, with ensemble-count confound |
| Final comprehensive controls | Prior topology and hidden-feature comparisons | Tests whether the new ensemble advantage requires message passing | No demonstrated G-specific advantage over strong matched sets |

The studies collectively make a stronger project than a sequence of attempted positive results. They reveal a progression from “a graph score works” to “what was the useful input?” to “what was the useful computation?” and then to “does the gain survive a serious alternative explanation?”

However, the final result does not overturn Yishai: it strengthens and extends his skepticism. His report already found rich sets competitive. Omri’s distinct addition is showing that the new multi-layer ensemble benefit can also be obtained by the set approach, in a matched contemporary benchmark using the chosen graph family.

The September-10th positive result and the latest near-zero gap are not a controlled contradiction. They differ in edge features (36 versus 12), training budget (up to 60 versus exactly 20 epochs), training population (2,400 versus 1,600 photographs), single-layer versus ensemble contrasts and evaluation cohort/history. They suggest context sensitivity of architecture gains; the cause of the difference has not been isolated.

Likewise, a union model losing and a score ensemble winning can both be true. Unioning consecutive last layers, sharing weights across graphs, pooling hidden-state trajectories and combining separately learned spaced-layer detectors are different operations.

No result here establishes worldwide originality, a new state of the art or paper-level novelty. That would require a separate literature-positioning exercise. There is enough source-backed incremental scientific contribution to support a substantive course-project narrative without those claims.

## 9. Claims to use, claims to avoid

Use:

- “Internal measurements and their combination improve the tested error-prediction recipe.”
- “A small rich-graph advantage appeared in the five-seed September-10th study.”
- “Spaced-layer score combination improved on the final-layer control across the tested seeds.”
- “A matched feature-set ensemble attained nearly the same performance as the graph ensemble; an added GNN advantage was not demonstrated.”
- “The evidence points toward the importance of feature content and aggregation, while leaving some architecture-specific benefits unresolved.”
- “The controlled comparisons concern one family of implementations and data regimes.”

Avoid:

- “We proved GNNs are unnecessary” or “the models are equivalent.”
- “Omri first asked whether topology matters.”
- “Attention alone beats hidden states,” when G/S receive H12.
- “The hidden ensemble loss proves attention causally adds 0.034 AUROC.”
- “The GNN beat a logit MLP, therefore message passing was the reason.”
- “The rewired model matched the graph, therefore connectivity does not matter,” given failed perturbation admission.
- “More layers are useless” or “more layers always help.”
- “The July 0.997 attack result proves hallucination detection/routing.”
- “The test was untouched throughout the project.”
- “Every failed job is a failed scientific hypothesis.”
- “All 39 final fits were new,” or “25 methods are 25 independent experiments.”
- Comparing 0.8889 here with Yishai’s 0.8939 as a shared-benchmark ranking.

## 10. Current state and continuation boundaries

The final comprehensive benchmark has results, diagnostics, frozen-input receipts and a completed certification. The last reconciled queue snapshot was empty; this document is not a live monitor. A future agent must inspect current Slurm state before asserting that nothing is running or submitting anything.

This request is scientific synthesis. It does not authorize further experiments. Remaining work is principally interpretation, presentation and reproducibility packaging, not a required unfinished training stage:

1. Use the Hebrew executive report as the project discussion entry point and this document for provenance.
2. Build the final course report around the strongest controlled questions, retaining the negative and small-positive evidence.
3. Preserve the original reports as historical records; correct overclaims in the new synthesis rather than silently rewriting history.
4. Confirm current remote/HF backup coverage before promising that the latest final models can be restored from HF. The September-10th 35-model backup is verified in its own records; this audit did not establish equivalent HF coverage for the final campaign.
5. If an independent final evaluation or extra mechanism controls are wanted, agree on that next question and protocol first. Do not automatically open a held-back set or start a new search.
6. Keep local, small, single-purpose commits; no push. Preserve unrelated untracked/staged changes.

For future compute work, read [Omri’s Slurm guide][slurm-guide]. Heavy data, extraction, training, inference and statistical work belong on Slurm; the MacBook is for light inspection, writing and coordination. Reconcile job IDs and durable server receipts before resuming to avoid duplicate jobs. Never put HF/SSH secrets in chat, Git or handoff text.

Use bounded specialist agents with concrete ownership: one scientific interpretation reviewer, one implementation/provenance reviewer, and one Slurm/artifact operator when operations are actually needed. A reviewer who has no active subtask should not run expensive polling or invent a new experiment. User approval and the current protocol outrank historical plans embedded in source documents.

The old [September-16 Claude handoff][old-handoff] is useful historical context but predates the completed final benchmark. Its unfinished-state statements are superseded by the final campaign’s receipts, not by an optimistic conversation summary.

## 11. Evidence index

These links point to the local preserved sources. They intentionally distinguish code-repository documents from local result downloads and outer-project context. Heavy models/caches remain remote. On another machine, reconstruct the root layout or obtain the referenced artifacts rather than treating this synthesis as a substitute for them.

[context-ledger]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/Context/Project Summaries/SOURCES.md>
[meetings]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/Context/Project Summaries/2026-09-10_2026-09-11-meetings-and-handoff.md>
[chat]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/Context/WhatsApp/Exported/export sep 10th - WhatsApp/_chat.txt>
[noya-sync]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/Context/Transcriptions/Transcription - Omri and Noya sync - September 11th.txt>
[poc]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/Context/WhatsApp/Exported/export sep 10th - WhatsApp/00000024-HANDOFF.md>
[noya-team]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph/docs/TEAM_REPORT.md>
[noya-state]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph/docs/CONSOLIDATED_STATE.md>
[noya-unseen]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph/docs/results/step3_unseen.md>
[noya-complementarity]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph/docs/results/complementarity.md>
[noya-g1]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph/docs/results/exploitation_g1.md>
[pope]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph/docs/results/pilot_vlm_pope.md>
[pope-category]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph/docs/results/step4_category_holdout.md>
[noya-backdoor]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph/docs/results/backdoor_testbed.md>
[noya-routing]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph/docs/results/routing_backdoor.md>
[evidence-flow]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph-September-10th/docs/results/EVIDENCE_FLOW_RESEARCH_REPORT_2026-08-30.md>
[topology-depth]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph-September-10th/docs/results/POLYGRAPH_TOPOLOGY_DEPTH_LAST4_REPORT_2026-09-05.md>
[july-summary]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/Early_Work_July_2026/Experiments/reports/final_summary.md>
[july-metrics]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/Early_Work_July_2026/Experiments/results/metrics.csv>
[july-phase2]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/Early_Work_July_2026/Experiments/results/metrics_cluster_phase2.csv>
[july-backbone]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/Early_Work_July_2026/Experiments/reports/phase2_backbone.md>
[july-attack-code]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/Early_Work_July_2026/Experiments/src/gen_adv.py>
[july-attack-gen]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/Early_Work_July_2026/Experiments/reports/phase9_adv_gen.md>
[omri-sept5]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph/docs/results/omri_20260905.md>
[fixed-target]: </Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/outputs/september6-fixed-target-control-result-he.md>
[robustness-review]: </Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/work/science-night-20260906/final-robustness-scientific-review.md>
[robustness-protocol]: </Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/outputs/september6-night-scientific-protocol.md>
[robustness-graph]: </Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/work/science-night-20260906/graph-terminal-scientific-review.md>
[sept10-protocol]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph-September-10th/docs/experiments/september10/PROTOCOL.md>
[sept10-results]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph-September-10th/docs/experiments/september10/RESULTS.md>
[sept10-review]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph-September-10th/docs/experiments/september10/FINAL_SCIENTIFIC_REVIEW.md>
[sept10-raw]: </Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/work/september10/results_native/summary.json>
[sept13-status]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph-September-10th/docs/experiments/september13/ENGINEERING_STATUS_20260914.md>
[sept13-state]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph-September-10th/pilots/layer_screen_20260913/STATE.md>
[sept14-protocol]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph-September-10th/docs/experiments/september14/PROTOCOL.md>
[sept14-report]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph-September-10th/docs/experiments/september14/REPORT.md>
[rep-protocol]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph-September-10th/docs/experiments/layer_ensemble_replication/PROTOCOL.md>
[rep-results]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph-September-10th/docs/experiments/layer_ensemble_replication/RESULTS_20260916.md>
[rep-run]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph-September-10th/docs/experiments/layer_ensemble_replication/RUN_20260916.md>
[final-protocol]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph-September-10th/docs/experiments/final_comparison_20260916/PROTOCOL.md>
[final-report]: </Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/work/results-final-20260917/raw/evaluation/REPORT.md>
[final-methods]: </Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/work/results-final-20260917/raw/evaluation/method_summary.csv>
[final-contrasts]: </Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/work/results-final-20260917/raw/evaluation/primary_contrasts.csv>
[final-diagnostics]: </Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/work/results-final-20260917/raw/diagnostics/training_diagnostics.csv>
[layer-data]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph-September-10th/pilots/layer_screen_20260913/data.py>
[layer-extract]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph-September-10th/pilots/layer_screen_20260913/extract.py>
[final-models]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph-September-10th/pilots/final_comparison_20260916/models.py>
[final-data]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph-September-10th/pilots/final_comparison_20260916/data.py>
[training-models]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph-September-10th/polygraph/training/models.py>
[final-execution]: </Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/work/results-final-20260917/raw/execution.json>
[final-resume]: </Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/work/results-final-20260917/raw/resume_20260917_121522.json>
[final-complete]: </Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/work/results-final-20260917/raw/evaluation/complete.json>
[slurm-guide]: </Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/Context/OMRI_SLURM_GUIDE.md>
[old-handoff]: </Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/work/claude-code-handoff-20260916/CLAUDE_CODE_HANDOFF.md>
