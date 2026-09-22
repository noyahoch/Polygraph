# Polygraph: project synthesis for the team

**Scope:** work through September 22, 2026.

**Purpose:** explain what we asked, what we ran, what we learned, and what the evidence supports. This is a synthesis of preserved results, not a new experiment or reanalysis.

## Executive overview

Our project asks whether a detector can use the internal computation of a frozen image classifier to predict when that classifier is wrong. Attention graphs provide one representation of that computation. The scientific challenge is to distinguish the usefulness of the information from the usefulness of processing it with a GNN.

The work developed in three stages:

1. **Yishai and Noya established the benchmark and useful internal representations.** Attention alone was generally weaker than confidence or hidden-state baselines. Class-conditioned messages and hidden representations substantially improved detection. Their strongest graph/output fusion beat the tested output baselines, but set, endpoint and rewiring controls did not establish that graph topology was necessary.
2. **Omri tested narrower comparisons and layer combinations.** A five-seed rich-feature experiment found a small graph-over-set advantage. A later, different experiment found nearly identical graph and set ensemble performance. Combining detectors from several layers improved on the final-layer control, but there was no matched ensemble of several final-layer detectors to isolate the benefit of layer diversity.
3. **The September 19–21 follow-ups added LogitDynamics and tested complementarity.** The fixed LogitDynamics adaptation beat the existing graph ensemble alone. Combining its score with the graph score improved it, while combining it with the set score achieved almost the same result. A separate decomposition found additional predictive value in intermediate-layer projections; the added dynamics block did not improve this particular readout.

**The strongest supported project story is useful internal information and complementary detectors, with conditional evidence about graph architectures—not a general demonstration that topology is indispensable.** These are meaningful results for a graph-learning project, including where carefully chosen controls weaken the stronger claim.

The current manuscript is available as [PDF](../../tex/overleaf_20260921/acl_latex.pdf) and [TeX](../../tex/overleaf_20260921/acl_latex.tex). Its checked build has **five main-text pages plus one references page**. This report provides the longer explanation behind it.

## 1. What the task and scores mean

For most September experiments, a frozen ViT-Base classifies CIFAR-100 images. A separate detector predicts whether the classifier's selected class is wrong. Training a detector does not improve or fine-tune that ViT.

An attention graph has image/CLS tokens as nodes and selected attention connections as directed edges. A GNN processes this graph; a set model processes collections of node/edge values without the same message-passing structure. Hidden-state and output-logit models provide other controls.

CLS is the classifier's summary token. Logits are class scores before softmax; MSP is the maximum softmax probability, used in the error-detection direction as one minus confidence.

We distinguish two questions:

- **Predictive usefulness:** does this detector or combination improve error detection over the specified baseline?
- **Graph-specific usefulness:** does graph processing improve on controls with comparable information, and does the experiment isolate why?

The second question is harder. Matching parameter counts alone does not match optimization, pooling, features or computational cost.

**AUROC** measures how well scores rank errors above correct predictions; it is not classification accuracy or a deployment threshold guarantee. **Average precision (AP)** summarizes precision–recall performance and depends on error prevalence. The recent reports use scikit-learn AP, not trapezoidal precision–recall area.

For multi-seed studies, mean AUROC means the average of seed-specific metrics, not the AUROC of predictions averaged across seeds. Averaging four layer detectors within one seed is a separate operation.

“Hallucination detection” is the broader motivation. The principal evidence here concerns **image-classification errors**, not verified transfer to language-model hallucinations.

## 2. Chronology and which numbers can be compared

Dates below identify the work or completion, not necessarily the date in a filename. The September 4 evidence-flow report is named August 30; the September 6 topology report is named September 5.

| Period | Main work | Evaluation setting/status |
|---|---|---|
| July–August | Initial graph proof of concept; Omri's earlier vision pilots | Small or different backbones/tasks; background evidence |
| August–early September | Noya's larger benchmark, generalization and diagnostics | Balanced corruption benchmark; separate weather/VLM/backdoor studies |
| September 4–6 | Yishai's rich messages, stronger output baselines, topology/depth controls | Original team benchmark; strict validation/meta-validation split |
| September 5–7 | Omri's checkerboard and threshold-transfer diagnostics | Separate backdoored classifiers; not the later ensemble benchmark |
| September 11 | “September 10” rich/raw comparison | 4,000 photos: 2,400 train / 800 validation / 800 evaluation; five seeds |
| September 14–16 | Four layers and learned score combination | 1,600 train / 400 checkpoint / 400 combiner / 800 evaluation; seed 7 late diagnostic, then seeds 17/27 |
| September 16–17 | Matched graph/set/hidden/output comparison | Same four-way roles; 800-photo development evaluation; three seeds |
| September 19; replay September 21 | LogitDynamics comparison | Same 800 evaluation photos, different supervision allocation |
| September 21 | Fusion and LD decomposition | **Separate** 400-photo fusion assessment and 800-photo decomposition assessment |
| September 21–22 | Manuscript integration, shortening and preservation | Documentation/build work; no new scientific result |

The recent nine-view studies use clean images plus Gaussian noise, motion blur, fog and JPEG at severities 3 and 5. All nine views of a photograph stay in one partition. These studies do not balance correct and incorrect predictions and do not test corruption families absent from training.

The older benchmark has a different sampling/balancing scheme and includes held-out corruption evaluations. **Do not combine these studies into one leaderboard.** Nor should the September 21 fusion numbers be compared directly with numbers from the full 800-photo assessment.

## 3. Original proof of concept and Noya's benchmark

**Question:** can attention graphs predict errors, and do they beat simple confidence or representation baselines?

Yishai's initial proof of concept used 1,500 balanced training examples and a small 200-example evaluation set, without a separate validation set. The CLS-plus-gated graph readout achieved about **0.7898 AUROC**, versus **0.9101 for MSP**. A confidence/graph combination reached **0.9112 ± 0.0046** across three seeds. Repeated selection on that small evaluation set makes this feasibility evidence rather than a confirmed fusion improvement.

Noya scaled the task to a 75,000-record, photo-grouped benchmark: 52,000 training, 6,000 validation and 17,000 test records. Correctness was balanced within corruption/severity cells.

Representative historical seed-7 results were:

| Method | AUROC on that historical benchmark |
|---|---:|
| MSP confidence | 0.8695 |
| Attention GNN | 0.8417 |
| Flattened attention MLP | 0.7421 |
| CLS hidden-state MLP | 0.8742 |
| CLS-sequence GRU | 0.8759 |
| Graph plus hidden information | 0.8758 |
| Tested 12-layer CHARM-lite union | 0.8233 |

**What this established:** attention-only detection was not enough to beat confidence; hidden representations were strong alternatives. Adding layers in one graph was not automatically beneficial.

**What it did not establish:** the flattened-attention comparison was not information-matched—it omitted node information available to the GNN. Also, the early “output MLP” used two summary features, not all 100 logits. Beating it would not establish superiority over a strong full-output baseline. The historical `logit_dyn` recurrent baseline was not the later supervised-head LogitDynamics adaptation.

Noya also examined unseen weather/extra corruptions and score complementarity. On a separate unseen slice, MSP reached **0.8885**, compared with **0.8738** for graph-plus-hidden and **0.8801** for the CLS MLP. A diagnostic stacking analysis of stored test scores suggested complementarity, but reusing test predictions for fitting limits its evidential strength. Different error sets alone do not demonstrate useful fusion.

Sources: [Noya's team report](../TEAM_REPORT.md), [original handoff](../HANDOFF.md), and the [historical scientific synthesis](../experiments/scientific_synthesis_20260917/HANDOFF.md).

## 4. Yishai's rich messages and topology controls

**Question:** which information makes a graph detector competitive, and does the graph structure explain the gain?

The evidence-flow work added information about the **direction of class evidence**, rather than attention weights or message magnitude alone. One feature projects a message onto the classifier direction separating the predicted class from its runner-up. This is a predictive proxy for class support, not a causal attribution guarantee.

On the rebuilt store, the attention set model scored **0.84736**, compared with **0.83477** for the raw graph model. Class-direction features raised the graph result to **0.86602**. The richer hidden-message model reached **0.88446**, versus **0.87941** for a strict full-100-logit MLP; their conditional fusion reached **0.89128**. The rebuilt-store MSP was **0.86510**, distinct from the historical **0.8695**; that difference is not detector progress on an unchanged evaluation pool.

The subsequent architecture/topology study produced this original-benchmark comparison:

| Method | AUROC |
|---|---:|
| MSP | 0.86510 |
| Full-logit MLP | 0.87941 |
| Selected edge-gated GNN | 0.88832 |
| Output + graph conditional fusion | 0.89391 |

The strongest fusion also improved on the reported output baseline on an unseen weather/extra slice: **0.89119 versus 0.88162**.

However, rich node/edge sets, endpoint-aware sets and hidden-token sets were close to the earlier rich graph model. Their capacities were not perfectly matched. Rewiring and feature perturbations caused limited degradation, with recovery after retraining in some controls. A last-four-layer union scored **0.88203**, below a same-seed final-layer result of **0.88942**.

**Scientific contribution:** this work already established the importance of class-conditioned/internal information, stronger output baselines, fusion and non-graph structural controls. Omri's later work extends those questions; it does not originate them.

**Limit:** the evidence supports the tested representation and detector recipes, but not a general necessity of topology or repeated message passing.

Sources: [evidence-flow report](../results/EVIDENCE_FLOW_RESEARCH_REPORT_2026-08-30.md) and [topology/depth report](../results/POLYGRAPH_TOPOLOGY_DEPTH_LAST4_REPORT_2026-09-05.md).

## 5. Exploratory branches: July pilots, VLM and checkerboards

These studies explain the project's decisions but should not be mixed into the central September comparison.

**Omri's July pilots** used a different, smaller ViT and investigated error, attack and out-of-distribution detection. For error detection, the output MLP was about **0.8786 AUROC**, compared with **0.8087** for a last-layer GNN. A graph/output blend improved descriptively, but its weight was selected using evaluation results. Strong attack-detection scores addressed a different label and had preprocessing confounds. Neither those scores nor OOD results demonstrate improved ordinary classification-error detection on the later backbone.

**Noya's VLM/POPE branch** exposed an important evaluation problem: an initially impressive result was discarded after finding extensive duplicate questions. Deduplicated analyses suggested internal-signal value for some high-confidence-error slices, but category transfer was weaker and error counts were small. No completed POPE GNN comparison establishes a language/vision-language graph benefit.

**The checkerboard studies** tried to create confidently wrong predictions. Noya established the original checkerboard testbed; Omri continued and audited that direction. A trained fixed-target backdoor should make a patched image predict one chosen class; merely placing a checkerboard on a classifier not trained with the trigger does not guarantee this behavior. Omri's rotating-target attempt achieved only **1.7% target attack success**, so it did not create the intended testbed. A fixed-target control achieved roughly **99.85%**, leaving very few correctly classified patched examples and a target-class confound.

The September 6–7 transfer diagnostics froze detectors and thresholds learned on clean data, then evaluated clean/patched inputs across classifier variants. One setting retained high MSP ranking AUROC while its fixed threshold rejected everything; the learned logit detector could instead accept everything and miss the errors. This distinguishes ranking from threshold transfer. The graph/set branch failed its save/restore gate and did not yield a valid graph-versus-set conclusion.

These are useful feasibility and validity lessons, including failed approaches—not evidence against all graph detectors. Their primary archives are distributed across older project folders; the tracked [historical synthesis](../experiments/scientific_synthesis_20260917/HANDOFF.md) documents the sources and limitations. This report does not claim to republish every underlying pilot artifact.

## 6. September 11: controlled rich-feature versus attention comparisons

**Question:** when graph and non-graph detectors receive comparable values, does graph processing help, and does the answer depend on feature richness?

This completed experiment trained seven arms across five seeds: rich graph, rewired rich graph, rich set, endpoint-aware set, attention graph, attention set and a full-logit MLP. It used the last layer and a fixed 4,000-photo sample with nine views each.

The rich representation combined 784-dimensional node features with 36 edge features: attention, message-magnitude and class-support groups, plus hidden-state information in the node representation. The attention regime used 16 node and 12 edge features. The internal-feature detectors did not directly concatenate the entire output-logit vector.

| Contrast | Mean AUROC difference | 95% paired-photo interval |
|---|---:|---|
| Rich graph − rich set, primary | +0.006233 | [0.002618, 0.009836] |
| Rich graph − endpoint set, secondary | +0.007366 | [0.003294, 0.011398] |
| Attention graph − attention set, secondary | −0.003686 | [−0.008349, 0.001045] |

The rich graph advantage was positive in four of five seeds. Its interval crossed the registered practical margin of **0.005**, leaving the practical magnitude unresolved. The attention-only contrast did not establish a graph advantage.

The rewiring control changed only **50.396%** of edges on average, below the required **80%**. It was retained as **non-diagnostic**; its small score change cannot prove that topology is unnecessary.

This is a modest positive architectural result under one rich-feature recipe. Pooling and other implementation differences limit causal interpretation.

Source: [results, limits and preservation](../experiments/september10/RESULTS.md).

## 7. September 14–16: combine layer detectors rather than merge their graphs

**Question:** do error scores from attention graphs at different depths usefully complement the final-layer score?

Four separate GNNs processed layers **3, 6, 9 and 12**, each for exactly 20 epochs. All received final-layer hidden token states. After freezing them, a logistic combiner learned from a separate 400-photo group. Its control was the same kind of learner using only the final-layer score. Fixed averaging was also reported.

The original seed-7 run finished after its deadline and remains a **late diagnostic**: stack AUROC **0.891046** versus final-layer control **0.884395**, difference **+0.006651 [0.003500, 0.010041]**.

A fixed replication with fresh seeds **17 and 27** completed September 16. Its primary mean difference was **+0.004691 [0.001936, 0.007565]**, positive in both seeds. The three-seed summary is descriptive; it does not retroactively turn seed 7 into an on-time primary result.

**What this adds:** combining separately trained layer detectors can help even when earlier joint multi-layer graphs did not. These are different constructions.

**What remains unresolved:** no matched ensemble of four final-layer detectors was tested. The improvement therefore does not isolate layer diversity from generic ensembling. Fixed averaging slightly exceeded the learned stack in each seed, so the work also does not show that learning the combination was necessary.

Sources: [original run record](../experiments/september14/REPORT.md) and [replication results](../experiments/layer_ensemble_replication/RESULTS_20260916.md).

## 8. September 16–17: broader, matched ensemble controls

**Question:** does the layer-ensemble result remain specifically a graph advantage when sets, hidden states and output logits receive stronger controls?

The comparison covered three seeds and four layer positions, with 20-epoch neural training budgets. Its **39 neural fits comprise 12 imported graph fits and 27 new hidden/set/output fits**. The 25 reported methods include aggregations and controls, not 25 independent discoveries.

Notation:

- **G:** graph detectors; **S:** node/edge set detectors.
- **H:** hidden-state detectors; **O:** full-logit MLP.
- **Mean:** average the four detector probabilities within a seed.
- **Stack:** learn a combination of their raw error scores on the reserved combiner group.

G and S had closely matched parameter counts and shared feature-value collections. This recipe used **12 attention edge channels**, zero-filling each subthreshold head value on retained edges, rather than the earlier rich36 edge representation. S removed endpoint associations, while pooling also differed. G/H comparisons did not match hidden-layer exposure: G used final-layer hidden states even at earlier attention layers, whereas H used the corresponding layer.

Selected results on the **800-photo development evaluation**:

| Method | Mean AUROC |
|---|---:|
| G mean | 0.888895 |
| S mean | 0.888559 |
| G stack | 0.888471 |
| S stack | 0.887942 |
| Full-logit MLP O | 0.875708 |
| H stack | 0.875702 |
| MSP | 0.861943 |

The six primary contrasts used 10,000 paired-photo draws and Bonferroni-adjusted intervals. For **G mean − S mean**, the difference was **+0.000336 [−0.002800, 0.003409]**; for **G stack − S stack**, **+0.000529 [−0.003005, 0.004159]**. No additional graph advantage was established. No equivalence margin was registered, so this is not an equivalence result.

Graph ensembles exceeded the tested hidden and output controls, but those comparisons concern the whole input/model/ensemble recipe. The fixed 20-epoch budget does not guarantee equal convergence.

This does not erase September 11's positive rich-feature result: feature construction, sample allocation and training design changed. The pair of studies shows that the graph advantage is **not consistent across these recipes**; it does not isolate which change caused that difference.

Sources: [frozen protocol](../experiments/final_comparison_20260916/PROTOCOL.md), [preserved native report](evidence/final_comparison_20260917/REPORT.md), [all method metrics](evidence/final_comparison_20260917/method_summary.csv), and [all primary contrasts](evidence/final_comparison_20260917/primary_contrasts.csv).

## 9. September 19: add the missing LogitDynamics comparator

**Question:** how does our existing graph ensemble compare with a literature-based method that reads class-score evolution across layers?

Yishai requested this missing comparison for the paper. “External baseline” here means a method from outside our own detector designs; LogitDynamics still uses the classifier's internal representations.

We trained 12 auxiliary class-prediction heads on the frozen ViT's layerwise CLS representations. Their projections, the original classifier and seven dynamics features formed an 85-feature input to a linear error detector. Three seeds used 16 auxiliary-head epochs and 100 readout epochs. The allocation was 1,200 photos for heads, 800 for the error readout, 400 for selection and the same 800-photo development evaluation used by G/S.

| Method | Mean AUROC on those 800 photos |
|---|---:|
| LogitDynamics D | 0.898066 |
| G mean, preserved | 0.888895 |
| S mean, preserved | 0.888559 |
| Full-logit MLP O, preserved | 0.875708 |
| MSP | 0.861943 |

The primary **G − D** difference was **−0.009171**, with ordinary 95% paired-photo interval **[−0.015874, −0.002361]**. D scored higher in all three seed pairs.

This strengthens the baseline comparison and weakens any claim that our graph ensemble is the strongest standalone detector tested. It is a fixed ViT-Base adaptation, not a full replication of the original paper. Inputs, supervision allocation, capacity and budgets differ; this is not a controlled mechanism comparison.

Source: [factual results and replay limits](../experiments/logit_dynamics_20260919/RESULTS_EN.md).

## 10. September 21: does the graph score add to LogitDynamics?

**Question:** even if G is weaker alone, does it add useful information to D—and more than S does?

The existing 800 development photos were split deterministically into **400 fusion-training and 400 assessment photos**. No base detector was retrained. Identical standardized logistic combiners fit D+G, D+S and D+D′ for each of three seeds.

D entered fusion as its **pre-sigmoid error logit**. G/S entered as their existing four-layer mean **probabilities**. Scaling was fitted on the fusion-training group only. The D′ pairings were fixed cyclically across seeds; these overlapping pairs are not independent replications or a compute-matched control.

Results below belong only to the **400-photo fusion assessment**:

| Method | Mean AUROC |
|---|---:|
| D alone | 0.900126 |
| G alone | 0.895609 |
| S alone | 0.894954 |
| D+G | 0.907329 |
| D+S | 0.907286 |
| D+D′ | 0.902116 |

- **D+G − D:** **+0.007203 [0.003415, 0.011330]**, primary adjusted interval.
- **D+G − D+S:** **+0.000043 [−0.001311, 0.001508]**, primary adjusted interval.
- **D+G − D+D′:** **+0.005213 [0.001942, 0.008673]**, secondary descriptive 95% interval.

**Conclusion:** the graph score added useful information to this LD detector. The tested set score produced almost the same fusion result, so the experiment did not establish a graph-specific advantage. In the secondary comparison, graph–LD fusion also outperformed the tested two-seed LD combination; this does not establish superiority over generic ensembling generally.

These comparisons extend the group's earlier fusion work by using the stronger, independently motivated LD baseline and a same-information set alternative.

## 11. September 21: which parts of LogitDynamics help?

**Question:** what do intermediate-layer projections and the dynamics block add under the fixed LD readout recipe?

Using the same saved heads, we trained only A/B/C and reused the original D. This study retained the **full 800-photo evaluation**, separately from fusion.

| Readout | Input | Mean AUROC |
|---|---|---:|
| A | Six original-classifier features | 0.863598 |
| B | A plus six auxiliary final-layer-head features | 0.894831 |
| C | All 78 numerical features across layer heads and original classifier | 0.898966 |
| D | C plus seven dynamics features | 0.898066 |

A is **not** the existing MLP on all 100 logits. The numerical blocks retain the original predicted class's score and five competing scores. The dynamics block also carries class-identity information; it is not merely a summary of values already preserved in C.

- **C − B:** **+0.004135 [0.000428, 0.007911]**, primary adjusted interval.
- **B − A:** **+0.031233 [0.025273, 0.037483]**, secondary 95% interval.
- **D − C:** **−0.000900 [−0.001750, −0.000048]**, secondary 95% interval.

The largest step among these nested readouts came from the auxiliary final-layer projection. Intermediate-layer projections added a smaller benefit. The dynamics block did not improve this recipe. This does not show that dynamics are generally harmful, isolate the effect of extra supervision, or justify replacing D in the already completed fusion experiment after seeing results.

For both September 21 studies, the primary family contained **DG−D, DG−DS and C−B**. The fixed 2,000 paired-photo bootstrap draws used adjusted percentile endpoints 0.0083333333 and 0.9916666667. Fusion resampled training and assessment photos separately and refitted the combiners/scalers; decomposition held fitted readouts fixed. Point estimates are the original fits, not bootstrap averages. About 17 draws fall in each primary tail, so the small positive C−B lower endpoint should not be portrayed as numerically decisive.

Sources for sections 10–11: [full results, all seeds and diagnostics](../experiments/complementarity_20260921/results/report/REPORT.md), [frozen protocol](../experiments/complementarity_20260921/PROTOCOL.md), and [machine-readable statistics](../experiments/complementarity_20260921/results/statistics/results.json).

## 12. What Omri's work contributes beyond the earlier group work

There is deliberate overlap: replication and stronger controls are valuable even when they do not invent a new method.

| Contribution | Relationship to previous work |
|---|---|
| Five-seed rich/raw graph-versus-set study | Tightens an existing question; adds uncertainty and a feature-regime comparison, while exposing an insufficient rewiring control |
| Separate layer-detector ensembles and replication | Tests a different use of depth from the earlier joint-layer graphs |
| Comprehensive matched ensemble controls | Shows the ensemble result is not demonstrated to be uniquely graph-based in that recipe |
| LogitDynamics comparison | Adds a missing literature-based comparator requested for the paper |
| LD fusion versus set fusion and two-seed LD fusion | Tests complementarity beyond the stronger comparator and narrows the graph-specific claim |
| Nested LD decomposition | Identifies which projection blocks add predictive value under this adaptation |
| Preservation and validation | Makes completed results traceable and usable, including failed attempts and unresolved checks |

**Yes, Omri's work adds meaningful evidence.** Its contribution is a more discriminating account of when graph-based detectors help and which stronger explanations survive the controls—not a universal GNN win or a claim of field-wide novelty.

Eran contributed to project discussions and proposed further threshold/topology directions. The records reviewed here do not support assigning him a separate completed quantitative experiment; this report does not infer experiment authorship from Git commit names alone.

## 13. Reliability, artifacts and what remains open

The September 11–21 comparison campaigns were executed on Slurm. Reports, protocols and compact evidence are in Git; heavy caches remain on the server, and model/code packages have private Hugging Face preservation records. **Private HF links require access**; publication of this Git branch does not grant it.

Key evidence packages:

- [September 10 model catalog and restoration scope](../models/SEPTEMBER10_CATALOG.md).
- [September 16–17 compact evidence and checksum provenance](evidence/final_comparison_20260917/README.md).
- [LD reuse guide](../experiments/logit_dynamics_20260919/REUSE.md) and [reproducibility audit](../experiments/logit_dynamics_20260919/AUDIT_REPRODUCIBILITY.md).
- [September 21 preservation receipt](../experiments/complementarity_20260921/PRESERVATION.md), including immutable HF revision and downloaded archive/member verification.
- [Long historical handoff](../experiments/scientific_synthesis_20260917/HANDOFF.md) for details intentionally omitted here.

The final LD replay was **not left unfinished**: all CPU checks ran. All six GPU validation/development vectors matched exactly; all three CPU development vectors and two validation vectors passed the existing tolerance. One seed-17 CPU validation example still failed it. The bootstrap calculations were independently reproduced. These audits used saved CLS inputs and the existing environment; they are not a fresh raw-image-to-output or full training reproduction.

The September 21 campaign completed all nine new linear readouts, nine fusion fits and all 2,000 fixed bootstrap draws. Semantic-cache/new-cache-D replay and weighted-versus-literal-duplication gates passed. Its backup was verified at an immutable revision. The short runtime came from reusing base detectors, heads and cached inputs—not from training the whole project in minutes.

Across the recent studies, uncertainty is conditional on the fitted models and specified partitions. Photo-group resampling preserves dependent views, but it does not remove previous development-data exposure or cover all new training randomness. Near-zero nonsignificant contrasts do not prove equivalence.

The remaining scientific gaps are **not incomplete authorized runs**:

- a genuinely fresh evaluation cohort for confirmation;
- a matched same-layer ensemble to isolate layer diversity;
- an effective, information-controlled intervention to isolate topology;
- a controlled explanation of why the earlier rich36 graph/set result differs from the later attention-edge ensemble result.

These are possible follow-ups, not experiments already performed or automatically scheduled. The immediate paper task is to choose a concise narrative, retain the original benchmark and the distinct follow-up cohorts, and state both the positive complementarity finding and the lack of demonstrated graph-specific fusion benefit.

No new experiments, numerical analyses or manuscript edits were performed to prepare this report.
