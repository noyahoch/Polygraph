# Canonical Polygraph versus LogitDynamics

17,000 identical canonical views, 1,998 source photographs; seeds 1/2/7. Original scan error labels are preserved.

| Method | Mean AUROC | Seed SD | Mean AP | Mean AURC |
|---|---:|---:|---:|---:|
| Polygraph | 0.89390638 | 0.00045790 | 0.87483908 | 0.20710602 |
| LogitDynamics | 0.89029565 | 0.00053918 | 0.87330330 | 0.20995887 |

Mean within-seed Polygraph − LD AUROC: 0.00361072; conditional 95% paired-photo interval: [-0.0021257525845460624, 0.009329021286420967].

- Canonical benchmark was previously used for development; not an untouched-test claim.
- Canonical records were selected using classifier correctness with constructed 50% error prevalence; AP/AURC describe this benchmark, not natural deployment prevalence.
- Fresh LD auxiliary heads and probes use a different supervision allocation from Polygraph.
- Fixed ViT-B adaptation, not full reproduction of the original paper search.
- Photo bootstrap conditions on these fitted models; it does not include configuration-selection or future-seed variability.
- An advantage between these complete recipes does not establish graph-topology necessity.
