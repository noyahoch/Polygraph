# LogitDynamics frozen development comparison

800 source photographs, all nine views each, seeds 7/17/27. Original test remains closed.

Frozen classifier error prevalence: 2271/7200 (31.54%).

| Method | Mean AUROC | Seed SD | Mean average precision | Mean AURC |
|---|---:|---:|---:|---:|
| G_mean | 0.888895 | 0.004543195974930641 | 0.779509 | 0.097136 |
| LogitDynamics | 0.898066 | 0.0015743196104057561 | 0.799039 | 0.092500 |
| S_mean | 0.888559 | 0.003086111581356541 | 0.775108 | 0.097596 |
| O | 0.875708 | 0.0005987859314172579 | 0.758339 | 0.102748 |
| MSP | 0.861943 | N/A | 0.709357 | 0.104663 |
| entropy | 0.864851 | N/A | 0.715732 | 0.103819 |

Prespecified mean paired G_mean − LogitDynamics ΔAUROC: -0.009171; ordinary 95% photograph-cluster bootstrap interval: [-0.015873773489983947, -0.0023608076406355953]. All 2,000 draws use shared photograph multiplicities across methods and seeds. Average precision is sklearn AP, not trapezoidal PR area. Seed means average metrics, never predictions.

- Development data were previously exposed; this is not an untouched-test confirmation.
- This is a fixed-budget ViT-Base adaptation, not a reproduction of the ViT-Large paper search.
- The confidence interval conditions on these fitted models and does not measure full training variability.
- G_mean and LogitDynamics have different capacity and training allocation; this does not isolate topology.
- Only the prespecified G_mean-minus-LogitDynamics AUROC contrast has an inferential interval.
