# Sole-operator execution helpers

Scientific commands and numerical tests run only in Slurm allocations. The
Mac may assemble source/configuration files and coordinate submission. The
immutable parent release is copied byte for byte into a new source snapshot;
the parent experiment and original outputs are never edited.

`build_release.py` and `build_config.py` produce proposals, not submissions.
The coordinator reviews their exact source/configuration hashes. Before any
submission, read the newest operator HANDOFF and reconcile user, account,
names, submission timestamps, intents and job IDs against the live scheduler.
An existing intent without an unambiguous receipt must never be blindly retried.

The proposed initial envelope is 105 GPU-minutes: cache plus CUDA checks 20,
three independent seed fits 25 each, and one post-freeze prediction job 10.
The new campaign ceiling is 120 GPU-minutes including failures; this envelope
leaves 15 minutes unreserved. CPU allocations total 230 job-wall-minutes including
preparation, analysis and backup, leaving 10 minutes under the 240 minute ceiling.
These are allocation ceilings rather than estimates of elapsed calendar time.

`stage_runner.py` verifies all source files, preserves the exact parent source,
stages the pinned dependency bundle, fixes numerical-library thread counts and
sanitizes credentials. `submit.py` preserves one-shot intents and enforces full
reservation ceilings plus previous-attempt accounting. Automatic requeue is off.
Failure requires reconciliation and a separately reviewed mechanical recovery.

The first 50 bootstrap draws are retained. `advance_after_timing.py` sees only
their completion/timing metadata, and submits the remaining fixed1950draws only
when the measured estimate with a 1.25 safety factor fits the already approved
150 minute remainder allocation. No scientific effect is used for this decision.
Jobs after the timing controller continue independently of the laptop.

`backup.py` uses the existing protected server token only during the private
upload. It snapshots all new scientific and operational state plus small bound
parent inputs/models, with a 512 MiB uncompressed scope guard. Original image/CLS
caches are not duplicated. A compressed archive retains every per-draw file;
the publisher verifies every remotely downloaded object at an immutable revision,
then checks each downloaded archive member against its per-file manifest.
The source release must be physically inside the new experiment root.

## Upload recovery

An upload failure never causes training, fitting, prediction or statistics to
run again. Keep publication/bundle, upload_intent.json and every failure receipt.
If receipt.json contains a revision but verification failed, the next task is
verification-only against that exact immutable revision and unchanged local
bundle/manifest. Check repository privacy, every pre-existing blob recorded at
the parent revision, exact new prefix inventory, downloaded object hashes and
all archive members. Write a new verification receipt; preserve the failed one.

If upload was interrupted before a revision receipt was written, first reconcile
the intent's parent revision and new prefix with Hub commit history and compare
the remote manifest. Do not assume failure and do not upload again. If no commit
exists, preserve the failed operational attempt and request a reviewed upload-only
recovery. Neither recovery modifies scientific settings or original artifacts.
