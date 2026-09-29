# Polygraph

## Layout

```
polygraph/
├── config.py, records.py     shared vocabulary: source taxonomy, keys, scan records
├── data/                     dataset creation (run once)      python3 -m polygraph.data
│   ├── sources.py            CIFAR-100 + CIFAR-100-C parquet pools, auto-download
│   ├── pipeline.py           frozen ViT: scan + extract
│   ├── graphs.py             attention -> sparse threshold graphs
│   ├── splits.py             group-disjoint stratified plans
│   └── storage.py            the key-indexed graph store
└── training/                 the detector (run many times)    python3 -m polygraph.training
    ├── models.py             GNN architectures (pinned to the POC, commit 8036fd9)
    ├── train.py              per-seed checkpoints, early stopping on val AUROC
    ├── evaluate.py           slices, metrics, combiner
    └── baselines.py          trained non-graph baselines
legacy/                       Yishai's original POC (frozen) + readers for the old store
docs/HANDOFF.md               the running results log (historical record)
tests/test_polygraph.py       35 tests; python3 tests/test_polygraph.py
```

## Pipeline

```bash
python3 -m polygraph.data scan       # ViT verdicts, full grid; resumable
python3 -m polygraph.data split --train-cap 26000 --val-cap 3000 --test-cap 8500
python3 -m polygraph.data extract    # attention graphs into the store; resumable
python3 -m polygraph.training train      # one checkpoint per seed (default 7 1 2)
python3 -m polygraph.training evaluate   # slices + all baselines from checkpoints
```

Only flags someone actually decides per run exist; settled constants (model, tau, paths)
live in `config.py`.
