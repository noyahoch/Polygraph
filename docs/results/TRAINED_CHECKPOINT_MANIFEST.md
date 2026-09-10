# Polygraph Trained Checkpoint Manifest

This manifest describes the 55 final `model_seed*.pt` checkpoints committed with the
Polygraph evidence-flow and topology/depth studies. Their total size is approximately
30 MB. Each file contains the learned model state and its training configuration.

Only completed model checkpoints are included. Four `state_seed*.pt` files are
excluded because they are interrupted/resumable training state, not selected final
models. Generated scores, logs, the 198 GB graph store, and feature sidecars remain
outside Git.

## Reproduction dependency

The checkpoints are useful for preservation and evaluation, but most internal models
still require the aligned local graph store and the sidecars documented in the final
research reports. In particular, hidden/evidence-flow models require `hidden12` and
`message_stats_l11`; last-four models require `hidden_last4_missing`; output models
require the logits sidecar. These large inputs are intentionally not committed.

## Evidence-flow study checkpoints

Paths are relative to the repository root.

| Directory | Checkpoint filenames | Experiment and short description |
|---|---|---|
| `runs/research_20260830/combiners/strict/main/output/` | `model_seed1.pt`, `model_seed2.pt`, `model_seed7.pt` | Strict main raw-logit MLP; strongest output-only detector trained from all 100 frozen-ViT logits. |
| `runs/research_20260830/combiners/strict/main/M5/` | `model_seed1.pt`, `model_seed2.pt`, `model_seed7.pt` | Strict main M5; two-layer TransformerConv using final token hidden states and 36-D evidence-flow edges. |
| `runs/research_20260830/combiners/strict/weather/output/` | `model_seed1.pt`, `model_seed2.pt`, `model_seed7.pt` | Weather-plan strict raw-logit MLP confirmation. |
| `runs/research_20260830/combiners/strict/weather/M5/` | `model_seed1.pt`, `model_seed2.pt`, `model_seed7.pt` | Weather-plan strict M5 confirmation. |
| `runs/research_20260830/controls/C1_edge_set/` | `model_seed7.pt` | C1 EdgeSet: attention-edge multiset only; no nodes, endpoints, or connectivity. |
| `runs/research_20260830/controls/C2_node_edge_set/` | `model_seed7.pt` | C2 NodeEdgeSet: independent base-node and attention-edge multisets; no incidence. |
| `runs/research_20260830/controls/C3_endpoint_set/` | `model_seed7.pt` | C3 EndpointSet: source/target/edge records with invariant pooling; no message passing. |
| `runs/research_20260830/message_flow/M0_attention/` | `model_seed7.pt` | M0 raw-attention TransformerConv baseline with base node features. |
| `runs/research_20260830/message_flow/M1_attention_message/` | `model_seed7.pt` | M1 adds transformed-value message magnitude to raw attention. |
| `runs/research_20260830/message_flow/M2_attention_decision/` | `model_seed7.pt` | M2 adds the predicted-versus-runner-up class-direction flow proxy. |
| `runs/research_20260830/message_flow/M3_evidence_flow/` | `model_seed7.pt` | M3 combines attention, message magnitude, and class-direction flow. |
| `runs/research_20260830/message_flow/M4_evidence_compact/` | `model_seed7.pt` | M4 adds compact class-conditioned token evidence to M3. |
| `runs/research_20260830/message_flow/M5_evidence_hidden/` | `model_seed1.pt`, `model_seed2.pt`, `model_seed7.pt` | Ordinary-plan M5 confirmation: full final-token hidden states plus evidence-flow edges. |
| `runs/research_20260830/message_flow/M5_tcp_multitask/` | `model_seed7.pt` | M5 TCP multitask ablation with error and TCP heads. |
| `runs/research_20260830/node_evidence/D1_compact_attention/` | `model_seed7.pt` | D1 TransformerConv with compact class evidence and raw-attention edges. |
| `runs/research_20260830/node_evidence/D2_compact_node_edge_set/` | `model_seed7.pt` | D2 matched compact-evidence NodeEdgeSet control. |
| `runs/research_20260830/node_evidence/D3_compact_endpoint_set/` | `model_seed7.pt` | D3 matched compact-evidence EndpointSet control. |
| `runs/research_20260830/node_evidence/H0_hidden_attention/` | `model_seed7.pt` | H0 full hidden-token nodes with raw-attention edges; isolates hidden semantics before evidence-flow edges. |

## Topology, architecture, and last-four-layer checkpoints

| Directory | Checkpoint filenames | Experiment and short description |
|---|---|---|
| `runs/topology_depth_last4_20260905/controls/S0_h64/` | `model_seed7.pt` | S0 HiddenTokenSet: full hidden-token multiset without graph edges. |
| `runs/topology_depth_last4_20260905/controls/S1_h96/` | `model_seed7.pt` | S1 capacity candidate: h96 full-information NodeEdgeSet. |
| `runs/topology_depth_last4_20260905/controls/S1_h128/` | `model_seed1.pt`, `model_seed2.pt`, `model_seed7.pt` | Selected S1 h128 confirmation: independent hidden-node and evidence-edge multisets. |
| `runs/topology_depth_last4_20260905/controls/S2_h64/` | `model_seed7.pt` | S2 capacity candidate: h64 full-information EndpointSet. |
| `runs/topology_depth_last4_20260905/controls/S2_h96/` | `model_seed1.pt`, `model_seed2.pt`, `model_seed7.pt` | Selected S2 h96 confirmation: endpoint-local records without iterative message passing. |
| `runs/topology_depth_last4_20260905/controls/weather_S2_h96/` | `model_seed1.pt`, `model_seed2.pt`, `model_seed7.pt` | Weather-plan confirmation of selected S2. |
| `runs/topology_depth_last4_20260905/controls/R1_target_permuted_M5/` | `model_seed7.pt` | R1 M5 trained with deterministic target-endpoint permutation. |
| `runs/topology_depth_last4_20260905/controls/R2_attribute_shuffled_M5/` | `model_seed7.pt` | R2 M5 trained with edge attributes shuffled against fixed topology. |
| `runs/topology_depth_last4_20260905/architectures/A_transformerconv_residual/` | `model_seed7.pt` | Residual TransformerConv architecture screen with hidden/evidence-flow inputs. |
| `runs/topology_depth_last4_20260905/architectures/A_gine/` | `model_seed7.pt` | GINE architecture screen with the same information and training protocol. |
| `runs/topology_depth_last4_20260905/architectures/A_gatv2/` | `model_seed7.pt` | Edge-conditioned GATv2 architecture screen. |
| `runs/topology_depth_last4_20260905/architectures/A_edge_gated_mean/` | `model_seed1.pt`, `model_seed2.pt`, `model_seed7.pt` | Selected two-layer edge-gated mean evidence-flow MPNN; best standalone internal detector. |
| `runs/topology_depth_last4_20260905/architectures/weather_A_edge_gated_mean/` | `model_seed1.pt`, `model_seed2.pt`, `model_seed7.pt` | Weather-plan confirmation of the selected edge-gated detector. |
| `runs/topology_depth_last4_20260905/architectures/D_edge_gated_mean_d1/` | `model_seed7.pt` | One-message-passing-layer depth ablation. |
| `runs/topology_depth_last4_20260905/architectures/D_edge_gated_mean_d4/` | `model_seed7.pt` | Four-message-passing-layer depth ablation. |
| `runs/topology_depth_last4_20260905/multilayer/L0_token_trajectory/` | `model_seed7.pt` | L0 blocks 8–11 per-token trajectory set model without graph connectivity. |
| `runs/topology_depth_last4_20260905/multilayer/L1_union_graph/` | `model_seed7.pt` | L1 attributed union graph over ViT blocks 8–11. |

## Important limitation: conditional gates

The reported strict output+internal gates are five-fold score-level ensembles. The
original gate evaluator persisted aligned predictions and JSON summaries but did not
persist the five temporary fold models, so there are no gate `model_seed*.pt` files in
this checkpoint set. The underlying output and internal expert checkpoints are
included above. Re-running `scripts/evaluate_strict_gate.py` from their aligned
validation/test scores deterministically reconstructs the reported gate predictions.

## Related reports

- `docs/results/EVIDENCE_FLOW_RESEARCH_REPORT_2026-08-30.md`
- `docs/results/POLYGRAPH_TOPOLOGY_DEPTH_LAST4_REPORT_2026-09-05.md`
