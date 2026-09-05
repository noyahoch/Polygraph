# Polygraph: Graph Necessity, Detector Architecture, and Last-Four-Layer Study

> Status: experiments in progress. This file is updated as each durable experiment
> completes. Test evaluation for new configurations remains withheld until the
> base-validation selection manifest is frozen.

## Executive answer

The follow-up is not yet complete. The established strict reference is a raw-logit MLP
at 0.87941 mean AUROC, M5 (full final-token hidden states plus evidence-flow edges) at
0.88446, and their strict mixture gate at 0.89128 on the rebuilt main test. The present
study is testing whether M5's explicit message passing is necessary, whether a better
edge-aware detector family or depth helps, and whether correctly aligned observations
from ViT blocks 8–11 add information. No new test result will be reported until the
predeclared configurations have been selected using strict `base_val` only.

## Protocol and current dataset

- Immutable graph store: 75,000 records; no graph extraction is repeated.
- Detector training: original train split.
- Detector and architecture selection: strict `base_val` only.
- Gate fitting: disjoint strict `meta_val`, grouped by base photograph.
- Test: evaluated only after the selection manifest is frozen.
- Existing test outcomes were seen in the preceding study, so this is a bounded
  follow-up on an existing benchmark, not an independent fresh-test confirmation.
- Primary baseline: the strongest full-output detector, not MSP alone.

The strict split audit found zero record and zero base-photograph overlap between
`base_val` and `meta_val`. The main strict detector plan contains 52,000 train, 2,991
base-validation, and 17,000 test records; meta-validation contains 3,009 records. The
weather strict detector plan contains 36,238 train, 2,496 base-validation, and 14,232
test records; meta-validation contains 2,362 records.

## Planned comparisons

The mandatory controls use exactly M5's 784-dimensional node tensor (coordinates, CLS
identity/layer coordinate, 12 attention diagonals, and the 768-dimensional final token
state) and its signed 36-dimensional evidence-flow edge tensor. S0 removes edges. S1
pools the full node and edge multisets without endpoint association. S2 uses factored
endpoint records plus an independent complete node branch, but performs no iterative
neighborhood aggregation. S1 and S2 each receive a base-validation-selected capacity
check and three-seed confirmation.

The architecture screen holds M5 information fixed while comparing a residual
TransformerConv scaffold, GINE, an edge-gated mean MPNN, and edge-conditioned GATv2.
Detector depth is screened separately from ViT depth. The required multilayer minimum
is L0 token trajectories without a graph, L1 a four-block attributed union graph, and
L1-SET its matched endpoint-record control.

## Durable experiment table

<!-- AUTO_RESULTS_START -->
| experiment | seed | selected base-val AUROC | epochs | status |
|---|---:|---:|---:|---|
| S0 full-hidden token set | 7 | 0.87210 | 13 | completed; best epoch 5 |

S0 used 75,138 parameters and removed all edge and endpoint information. This is a
base-validation result only; its test metrics remain sealed until configuration
selection is frozen.
<!-- AUTO_RESULTS_END -->

## Environment

The existing repository-local environment is retained: NVIDIA RTX A5000 (24 GiB),
driver 535.161.08, CUDA-enabled PyTorch 2.5.1+cu121 (runtime CUDA 12.1), PyG 2.7.0,
Transformers 5.16.1, and Python 3.11. No driver, system CUDA, or package replacement was
performed. The CUDA toolkit compiler at `/usr/local/cuda-11.8/bin/nvcc` is present but
is not required by the installed binary wheels.

## Remaining sections

The final report will add matched gate comparisons, rewiring diagnostics, architecture
and detector-depth ablations, correctly aligned last-four-block results, all-degradation
and per-severity metrics, inference/runtime profiles, weather confirmation,
base-photograph paired bootstrap intervals, negative and skipped results, reproduction
commands, an artifact map, and final git state.
