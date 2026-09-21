# Independent replay-completion review — September 21, 2026

Reviewer: `english_report_factcheck`. This review inspected the audit source,
final CPU/CUDA receipts, operator preservation evidence, and the current changes
to `RESULTS_EN.md`, `RESULTS_HE.md`, `AUDIT_REPRODUCIBILITY.md`,
`AUDIT_SCIENCE.md`, `REUSE.md`, `MODEL_CARD.md`, `SESSION.md`, and
`REPLAY_COMPLETION_20260921.md`. No numerical computation, remote operation,
model change or scientific-output edit was performed by the reviewer.

The [CPU receipt](final_receipts/audit/replay_completion_20260921_cpu_v1.json)
and [CUDA receipt](final_receipts/audit/replay_completion_20260921_cuda_v1.json)
support the following conclusions:

- Both jobs executed all six seed/role cases: seeds 7, 17 and 27, each with
  3,600 validation and 7,200 development records. Production inference routines
  and batch size 512 were used with published portable weights and scalers.
- CUDA matched every saved score value exactly, with zero maximum absolute
  difference in every case. The equality check is `np.array_equal`; no separate
  raw-byte comparison was performed.
- CPU execution completed, but aggregate tolerance success is false. The sole
  violation remains seed-17 validation record 15321, image 659, with absolute
  score difference 0.0007970333099365234 at unchanged `atol=rtol=1e-4`.
  Every development case and the other two validation cases passed tolerance.
  Small CPU metric differences, including seed-7 development AUROC/AP, are
  retained separately from the original scientific results.
- The CPU audit independently recomputed all 2,000 shared photograph-bootstrap
  draws for three seed pairs using 12,000 scikit-learn weighted AUROC
  calculations. No draw was undefined. Maximum paired-difference discrepancy
  was 3.3306690738754696e-16; the interval agreed at absolute tolerance 1e-12.
- Final accounting records 279 allocated CPU-job seconds and 283 allocated
  GPU-job seconds. The current total GPU allocation is 14,659 seconds (4:04:19).
  The CUDA runtime reported an NVIDIA GeForce RTX 2080 Ti.

The [preservation receipt](final_receipts_verification_20260921.json) records
unchanged original scores, report, bootstrap, completion record, campaign,
role map, freeze gate, six portable model files, 133 source files, and earlier
failed-audit/diagnostic records against their pinned references. The new audit
does not replace the original predictions or change the scientific findings.

The reviewed documents distinguish September 19 history from September 21
completion and preserve the CPU limitation. Three wording issues concerning
historical GPU accounting, diagnostic chronology and numerical metric
differences were corrected and rechecked. **No unresolved documentation or
numerical-evidence blocker was found for this bounded verification scope.**

The supported verification claim is portable checkpoint-to-score replay from
cached CLS inputs in the existing server environment. Universal CPU tolerance
success, a clean dependency installation, fresh raw-image-to-output inference,
full training replication and replication of the paper's broader study are
not established. Private publication of the new addendum is a separate delivery
step and was not verified by this review.
