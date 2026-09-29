# Literature baselines — main test split (n=17,000, seed 7)

Logit-lens sanity: corr 1.000000, max|diff| 6.57e-05 vs stored confidence.

| slice | msp | energy | mahalanobis | logit_dyn |
|---|---|---|---|---|
| all (n=17000) | **0.8695** | 0.8653 | 0.6623 | 0.8632 |
| clean (n=176) | **0.9042** | 0.8902 | 0.7603 | 0.8212 |
| seen_corruptions (n=13286) | **0.8679** | 0.8636 | 0.6572 | 0.8671 |
| unseen_extra_family (n=3538) | **0.8745** | 0.8713 | 0.6774 | 0.8512 |
| severity_1 (n=3348) | **0.8998** | 0.8933 | 0.7010 | 0.8716 |
| severity_3 (n=3364) | **0.8698** | 0.8669 | 0.6798 | 0.8587 |
| severity_5 (n=3378) | 0.8382 | 0.8390 | 0.6085 | **0.8628** |
