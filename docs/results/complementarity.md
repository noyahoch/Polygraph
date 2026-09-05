# Complementarity analysis — main test split (n=17,000, seed 7)

Sanity (must match the report): msp 0.8695, margin 0.8681, cls_mlp 0.8744, cls_seq 0.8764, graph 0.8417, fusion 0.8758

## 1. Spearman correlation of detector rankings

| | msp | margin | cls_mlp | cls_seq | graph | fusion |
|---|---|---|---|---|---|---|
| **msp** | 1.000 | 0.997 | 0.820 | 0.812 | 0.794 | 0.827 |
| **margin** | 0.997 | 1.000 | 0.811 | 0.801 | 0.776 | 0.815 |
| **cls_mlp** | 0.820 | 0.811 | 1.000 | 0.923 | 0.805 | 0.905 |
| **cls_seq** | 0.812 | 0.801 | 0.923 | 1.000 | 0.808 | 0.882 |
| **graph** | 0.794 | 0.776 | 0.805 | 0.808 | 1.000 | 0.841 |
| **fusion** | 0.827 | 0.815 | 0.905 | 0.882 | 0.841 | 1.000 |

## 2. Errors caught at a 50 percent flag budget (total errors: 8500)

| detector | caught | caught & MSP missed | MSP caught & detector missed | overlap with MSP |
|---|---:|---:|---:|---:|
| msp | 6725 | 0 | 0 | 6725 |
| margin | 6713 | 59 | 71 | 6654 |
| cls_mlp | 6781 | 761 | 705 | 6020 |
| cls_seq | 6785 | 813 | 753 | 5972 |
| graph | 6456 | 818 | 1087 | 5638 |
| fusion | 6748 | 847 | 824 | 5901 |

## 3. MSP-blind errors (n=1775): fraction each detector flags

| detector | share of MSP-blind errors it flags (own 50 percent budget) |
|---|---:|
| margin | 3.3% |
| cls_mlp | 42.9% |
| cls_seq | 45.8% |
| graph | 46.1% |
| fusion | 47.7% |

## 4. AUROC restricted to confident predictions (conf >= 0.9; n=10851, errors=3375)

| detector | AUROC |
|---|---:|
| msp | 0.8423 |
| margin | 0.8410 |
| cls_mlp | 0.8535 |
| cls_seq | 0.8598 |
| graph | 0.8186 |
| fusion | 0.8614 |

## 5. Per-source AUROC (winner bolded)

| source | msp | margin | cls_mlp | cls_seq | graph | fusion |
|---|---|---|---|---|---|---|
| brightness (n=880) | **0.9302** | 0.9291 | 0.8960 | 0.8982 | 0.8842 | 0.9053 |
| clean_test (n=176) | 0.9042 | **0.9046** | 0.8514 | 0.8559 | 0.8200 | 0.8409 |
| contrast (n=884) | 0.8735 | 0.8725 | 0.8855 | 0.8753 | 0.8460 | **0.8869** |
| defocus_blur (n=882) | 0.9147 | **0.9149** | 0.8997 | 0.9056 | 0.8814 | 0.9009 |
| elastic_transform (n=884) | **0.9012** | 0.9008 | 0.8889 | 0.8973 | 0.8710 | 0.8983 |
| fog (n=884) | 0.9005 | 0.8982 | **0.9025** | 0.8919 | 0.8795 | 0.8930 |
| frost (n=886) | 0.8848 | 0.8822 | 0.8765 | 0.8880 | 0.8467 | **0.8910** |
| gaussian_blur (n=884) | **0.8882** | 0.8882 | 0.8694 | 0.8690 | 0.8352 | 0.8731 |
| gaussian_noise (n=890) | 0.8499 | 0.8441 | 0.8798 | **0.8880** | 0.8439 | 0.8733 |
| glass_blur (n=890) | 0.8058 | 0.8008 | 0.8623 | **0.8741** | 0.8046 | 0.8435 |
| impulse_noise (n=888) | 0.8271 | 0.8238 | 0.8596 | **0.8812** | 0.8084 | 0.8611 |
| jpeg_compression (n=888) | 0.8431 | 0.8438 | 0.8420 | **0.8464** | 0.8155 | 0.8461 |
| motion_blur (n=888) | 0.8726 | 0.8718 | **0.8852** | 0.8824 | 0.8460 | 0.8769 |
| pixelate (n=884) | 0.8602 | 0.8561 | 0.8602 | **0.8637** | 0.8443 | 0.8568 |
| saturate (n=884) | **0.8847** | 0.8833 | 0.8657 | 0.8610 | 0.8467 | 0.8649 |
| shot_noise (n=888) | 0.8424 | 0.8397 | 0.8587 | **0.8629** | 0.8332 | 0.8578 |
| snow (n=886) | **0.8883** | 0.8851 | 0.8730 | 0.8751 | 0.8481 | 0.8824 |
| spatter (n=882) | 0.9124 | 0.9101 | 0.9046 | 0.9037 | 0.8759 | **0.9128** |
| speckle_noise (n=888) | 0.8183 | 0.8128 | 0.8465 | **0.8620** | 0.8253 | 0.8553 |
| zoom_blur (n=884) | 0.8894 | 0.8885 | 0.8892 | 0.8912 | 0.8552 | **0.8928** |
