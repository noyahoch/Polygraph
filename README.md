# Polygraph

Can a GNN reading a frozen ViT's attention graphs predict the ViT's classification
errors — especially under unseen corruptions — better than output-, representation-, and
non-graph baselines? (Project proposal: Lavi, Hochwald, Fahn, Kramf.)

The classifier under study is `edumunozsala/vit_base-224-in21k-ft-cifar100`, frozen
throughout. Each image induces one directed graph per ViT layer over its 197 tokens; only
a small GNN detector is trained, to predict `y_err = 1[argmax f(x) != y]`.

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

## The data

**Scan** — 1,010,000 records: clean test (10k) + clean train (50k) + 19 corruptions x 5
severities from CIFAR-100-C (950k). The ViT makes **239,921 errors (23.8%)**; only 852 of
them are on clean test images, which is why the POC could never scale — the positive class
was exhausted, not the images. `clean_train` is scanned but excluded from every plan
(measured: 99.45% accuracy, confidence inflated by fine-tuning memorization).

**Split plan** — 75,000 records: train 52k / val 6k / test 17k, all 50/50 balanced.
Three properties are enforced structurally and tested:

- *Group-disjoint*: every corruption and severity of one photograph shares a split
  (corruption shards keep the clean test row order — verified — so `fog(img 41)` and
  `snow(img 41)` are the same picture and can never straddle train/test).
- *Stratified*: equal wrong/correct within every (source, severity) cell. Error rate
  climbs from 8.5% clean to ~60% at severity 5, so without this a detector could score
  by reading corruption strength instead of impending failure.
- *Held-out corruptions*: the four extra CIFAR-C corruptions (`speckle_noise`,
  `gaussian_blur`, `spatter`, `saturate` — the benchmark's own designated validation set)
  appear only in test: train/val cover 76 cells, test 96.

**Store** — graphs extracted once at `tau = 0.02` (`max_h A[h,i,j] > tau`, edge j->i,
per-head attention as 12-dim edge features; ~7,600 edges/layer). Edges are stored sorted
by descending strength, so any stricter tau and any top-K view are free prefix slices at
load time — one extraction serves the whole threshold-sensitivity ablation and the old
top-100 comparisons. Per-layer CLS embeddings are stored alongside for the representation
baselines. ~2.6 MB/record, ~190 GB total. The store is key-indexed, never organised by
split: changing the plan costs nothing, and a plan needing new records extends the store
incrementally. Extraction is resumable (at most one shard lost) and self-checking (every
record's prediction is compared to the scan; systematic drift aborts).

Node features are minimal (patch coordinates, CLS flag, layer position, per-head
attention diagonals — 16 dims); ViT hidden states are deliberately not node features in
the primary condition.

## The comparison ladder

Evaluation reports every detector on identical records, per slice (all / clean / seen
corruptions / unseen sources / held-out extra family / severity 1-5), with AUROC, AUPRC,
and selective prediction (risk-coverage, AURC, risk@{0.5,0.8,0.9,1.0}):

| detector | trained? | reads |
|---|---|---|
| `msp` | no | max softmax probability |
| `margin` | no | top-1 minus top-2 probability |
| `output_lr` | yes | logistic on [logit(msp), margin] |
| `cls_mlp` | yes | MLP on the final CLS embedding (what the softmax head reads) |
| `graph` | yes | the GNN on attention graphs |
| `msp_plus_graph` | yes | standardized logistic combiner on [graph logit, logit(msp)] |

Training the *softmax itself* is deliberately absent twice over: monotone recalibration
cannot change a ranking metric, and fine-tuning the ViT would change whose failures
`y_err` describes. The trained baselines are what makes the comparison fair: the graph's
advantage must come from attention, not from being the only trained model in the room.

Reference points: MSP scores 0.9092 on all 10k clean test images (the POC measured 0.9101
on its 200-image subset), and margin is 0.978-correlated with MSP.

## Why the numbers can be trusted

- The edge rule is **bit-identical** to the POC reference (tested against Yishai's frozen
  `legacy/poc_gnn_vit_cifar100.py`), and the model architectures match his originals to
  1e-6 across all readouts — so results stay comparable with every number in
  [docs/HANDOFF.md](docs/HANDOFF.md).
- The current scan code was verified against the previous pipeline's ground truth on real
  images: 600/600 identical predictions, confidence deltas <= 7e-7.
- The 35-test suite was **mutation-audited**: five deliberately planted bugs (reversed
  edge direction, dropped edges, holdout leak, the Stage-5 combiner scaling bug, changed
  readout activation) are each caught. The audit also exposed and fixed one tautological
  test.

## Legacy

`legacy/` holds the original POC verbatim (git commit `8036fd9`) and readers for the
earlier top-100-edges store (`data/graph_dataset/graphs/`, 6.4 GB) via
`legacy/legacy_dataset.py`. All historical results and their caveats are in
[docs/HANDOFF.md](docs/HANDOFF.md).
