# Frozen ViT classifier accuracy (the proposal's required benchmark table)

Model: `edumunozsala/vit_base-224-in21k-ft-cifar100`. Clean test accuracy: **0.9148** (10,000 images). Each cell below is accuracy over the same 10,000 images under one corruption/severity.

| corruption | s1 | s2 | s3 | s4 | s5 |
|---|---:|---:|---:|---:|---:|
| brightness | 0.9157 | 0.9128 | 0.9056 | 0.8950 | 0.8592 |
| contrast | 0.9107 | 0.8870 | 0.8641 | 0.8140 | 0.5583 |
| defocus_blur | 0.9114 | 0.9005 | 0.8777 | 0.8381 | 0.7508 |
| elastic_transform | 0.8577 | 0.8598 | 0.8277 | 0.6893 | 0.5052 |
| fog | 0.9084 | 0.8838 | 0.8553 | 0.7917 | 0.6209 |
| frost | 0.8862 | 0.8506 | 0.7848 | 0.7738 | 0.7065 |
| gaussian_blur | 0.9111 | 0.8787 | 0.8475 | 0.8126 | 0.7206 |
| gaussian_noise | 0.7937 | 0.6713 | 0.5235 | 0.4687 | 0.4042 |
| glass_blur | 0.5294 | 0.5236 | 0.5076 | 0.3328 | 0.3118 |
| impulse_noise | 0.8292 | 0.7487 | 0.6660 | 0.4971 | 0.3641 |
| jpeg_compression | 0.8148 | 0.7374 | 0.7079 | 0.6700 | 0.6172 |
| motion_blur | 0.8740 | 0.8233 | 0.7566 | 0.7565 | 0.6835 |
| pixelate | 0.8870 | 0.8314 | 0.8152 | 0.6910 | 0.4740 |
| saturate | 0.8592 | 0.7828 | 0.9094 | 0.8677 | 0.8153 |
| shot_noise | 0.8418 | 0.7800 | 0.6053 | 0.5361 | 0.4215 |
| snow | 0.8867 | 0.8319 | 0.8194 | 0.7938 | 0.7432 |
| spatter | 0.8939 | 0.8638 | 0.8109 | 0.8692 | 0.8280 |
| speckle_noise | 0.8390 | 0.7317 | 0.6685 | 0.5331 | 0.4150 |
| zoom_blur | 0.8555 | 0.8510 | 0.8296 | 0.8022 | 0.7504 |
