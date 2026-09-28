# Canonical LD: one Slurm operator

`submit.py` accepts an exact reviewed configuration hash and a named phase. It
checks all stage reservations against the entire 12 GPU-hour / 2 CPU-job-hour
ceiling (including consumed prior attempts), enforces synchronized groups of at
most three GPU stages, and writes an intent before calling Slurm. Existing
receipts prevent duplicate submission; an intent without a receipt must be
reconciled manually. Completed predecessors use checksummed completion receipts
instead of depending on scheduler IDs that may have expired. Preserve job user,
account, name, submission time and source identities when reconciling receipts.

`stage_runner.py` verifies the immutable release and config inside the allocation,
prepares the preserved node-local import bundle and dependency overlay, and
retains success/failure receipts and command timings. The model Hub cache must
point to the existing pinned model snapshots for inference; the download stage
can use a new dedicated dataset cache and enables online access explicitly.
No GPU should be allocated just to wait for dataset downloading.

`backup.py` archives original supplied benchmark metadata/scores, derived roles,
all executable source releases, environment/configuration records, model states,
predictions, evaluation/bootstrap outputs and logs to a new private Hub prefix.
Heavy raw parquet and full CLS shards stay on Slurm with their checksummed
manifests and reproduction code. An immutable revision and downloaded archive
member hashes must verify before the backup reports success. A failed upload
does not authorize retraining; preserve its intent/receipt and reconcile the
upload separately. No credential values are logged or saved in Git.

These helpers do not decide scientific settings or approve resource expansion.
CPU-only dataset preparation can precede final campaign-source freezing. GPU
preflight requires reviewed scientific code/protocol; full extraction requires
its successful parity/throughput results and an approved full reservation.
