# Completed evidence and preservation

The original September 19 campaign remains unchanged. The September 21
follow-up completed without scientific source changes, retries, replacement
draws or new model arms. All 13 jobs completed on September 21, 2026; the final
operator queue snapshot at 14:23:30 Israel was empty.

## Immutable private snapshot

- Repository: `omrifahn/polygraph-experiments` (private).
- Revision: `769e8b48c9e33edaebac81e97405945197054c11`.
- Prefix: `complementarity_20260921_135802/snapshot_v1`.
- [Pinned snapshot](https://huggingface.co/omrifahn/polygraph-experiments/tree/769e8b48c9e33edaebac81e97405945197054c11/complementarity_20260921_135802/snapshot_v1).
- `scientific_state.tar.gz`: 48,354,170 bytes; SHA-256
  `b0022d597ffda2c336e7791526635d7ade601fe184460cecb762b1b64238448c`.
- `FILE_MANIFEST.json`: SHA-256
  `5a62b0d97a95e35639fa318c5067ac05eb71c6eab7f324605cb27b8cfa12b290`.
- `BACKUP_MANIFEST.json`: SHA-256
  `00c416cf4d90996e77a9b256f629184cfeb85eadaeeafe6b247af3e6db030f2e`.

The publisher downloaded every uploaded object at the pinned revision and
verified both archive hashes and every one of its 2,429 file members. All 1,391
previous Hub files retained their identities. The archive contains the exact
executable release, protocol, environment records, configuration, split and
draw manifests, new feature cache, models/scalers/resume states, histories,
predictions, all fixed draw states, reports and operational evidence, plus small
bound parent inputs and models. Original G/S score files are included; the
snapshot does not duplicate all original G/S detector checkpoints. Recomputing
those scores requires the preserved parent assets and workflow. Heavy original
images/CLS caches remain on Slurm; their identities and regeneration sources
are recorded.

The publisher's own in-progress receipt/log was captured before its completion;
the separately retained [publication receipt](results/publication/receipt.json)
records the completed verification. Reviews and this local completion note were
written after the snapshot and are additional local Git documentation. They
are not claimed to be included in that earlier immutable revision.

## Entry points and scope of validation

- [Human summary in Hebrew](SUMMARY_HE.md).
- [Full server-generated report](results/report/REPORT.md).
- [Structured complete result](results/statistics/results.json).
- [Statistical completion hashes](results/statistics/complete.json).
- [Weighted/duplicated evidence](results/statistics/weighted_duplication.json).
- [Operational closure](results/ops/CLOSING_RECEIPT.json) and
  [resource ledger](results/ops/resource_ledger_final.json).
- [Parent preservation](results/ops/parent_integrity_after_fits.json).
- [Downloaded report hash comparison](results/ops/retrieved_report_checksums.json).
- [Executed cache/fit/freeze gate review](reviews/gate_review.md).
- [Detailed scientific results audit](reviews/scientific_results_review.md).
- [Independent fusion/statistics cross-review](reviews/independent_scientific_cross_review.md).
- [Preservation metadata audit](reviews/preservation_review.md).

The compact `results/` tree preserves retrieved reports and receipts unchanged.
Receipt paths resolve against the recorded remote experiment root; many heavy
files they identify are intentionally absent from this compact local tree.
All calculation was performed in Slurm allocations. Local follow-up reviews
inspect receipts, sources and already-computed values; they are not independent
numerical recomputation, fresh-install recreation or raw-image reproduction.
The detailed results audit was written by the fusion/statistics implementer;
the separate scientific cross-review was performed by the other engineer,
who did not implement those modules. That engineer did implement the
cache/ablation modules. This is team cross-review, not external replication.

The exact runnable release in the snapshot is based on local Git checkpoint
`45b0c4f`, with all 133 parent files preserved. The new source-manifest SHA-256
is `4f2debe7aaa861c44538240ce624d4ff1ad4e4160dbe538f1f1298e24db08fed`;
the executed configuration SHA-256 is
`19c04dc936a958fa9864a008cf01171d4446a9b55734681294b38b1f6cbe31bd`.
Absolute paths inside frozen manifests identify the original server layout.
Restore that layout or explicitly document a relocation before reusing commands.

No further experiment or training is pending. Do not resubmit the completed
DAG, reopen old training matrices, push/merge this branch or edit Overleaf as
part of this completed scope.
