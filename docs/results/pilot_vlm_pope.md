# VLM POPE pilot — initial headroom (LLaVA-1.5-7B, adversarial split)

n = 3000 questions · ViT... err, LLaVA accuracy 0.832 · hallucination rate (said-yes-when-absent) 0.061

Split: group-disjoint by image; train 732 (balanced) / test 900 (natural).

**Label-shuffle control (must be ~0.5): 0.4792**

## y_err — all test questions (n=3000, errors=137)

| detector | AUROC |
|---|---:|
| out_msp | 0.7633 |
| out_margin | 0.7635 |
| probe_lr | 0.7562 |
| probe_mlp | 0.8092 |

**headroom (best internal − best output) = +0.0457**

## y_err — confident answers (p(chosen) >= 0.9) (n=1587, errors=29)

| detector | AUROC |
|---|---:|
| out_msp | 0.6284 |
| out_margin | 0.6293 |
| probe_lr | 0.7805 |
| probe_mlp | 0.8117 |

**headroom (best internal − best output) = +0.1824**

## y_hall slice — questions where the object is absent (n=1500, errors=51)

| detector | AUROC |
|---|---:|
| out_msp | 0.8776 |
| out_margin | 0.8768 |
| probe_lr | 0.6949 |
| probe_mlp | 0.7497 |

**headroom (best internal − best output) = -0.1280**

## Decision (§1.6), corrected with bootstrap CIs and the honest linear probe

The +0.0457 above uses probe_mlp, which overfits (128-wide MLP on 732 rows). The linear
probe (probe_lr) is the honest denominator. Powered picture (n=3000, 137 errors), probe_lr
vs out_msp, 95% bootstrap CIs:

| slice | errors | probe_lr | out_msp | headroom (95% CI) |
|---|---:|---:|---:|---|
| all | 137 | 0.756 | 0.763 | **-0.007 [-0.053, +0.041]** — flat |
| confident (p>=0.9) | 29 | 0.781 | 0.628 | **+0.152 [+0.027, +0.279]** — CI excludes 0 |
| unconfident | 108 | 0.637 | 0.680 | -0.043 [-0.120, +0.036] — flat |

**Verdict: no average headroom (POPE's yes/no label couples to the output, like the
corruption task), but a real, significant headroom in the confident-error regime** — where
output confidence collapses to near-chance (0.628) and the internal probe holds (0.781).

### Cross-domain: the confident-error effect is consistent and scales with decoupling

Same confident-slice test on the corruption benchmark (fusion vs MSP), bootstrap CIs:

| domain | all-slice headroom | confident-slice headroom |
|---|---|---|
| corruption (fusion vs msp) | +0.006 [+0.002, +0.010] | +0.019 [+0.012, +0.026] |
| VLM POPE (probe vs output) | -0.007 [-0.053, +0.041] | +0.152 [+0.027, +0.279] |

Internal signals beat output confidence specifically on confident errors, in both domains,
with magnitude scaling by how decoupled confidence is from correctness (small under
corruption, ~8x larger under VLM hallucination).

### Caveat that gates the next step

This is a REPRESENTATION (probe / hidden-state) result, not a STRUCTURE (graph) result.
On the corruption data the pure attention-graph does NOT beat MSP even on the confident
slice (-0.024 [-0.031, -0.016]); the confident-error win is carried by hidden states, not
topology. So a POPE attention-graph is unlikely to beat this probe (yes/no is not
routing-shaped). Pursuing a GRAPH win needs a task whose failure signature is inherently
relational (spurious-correlation / shortcut reliance), not POPE.

