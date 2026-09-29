# Independent statistical implementation review — 2026-09-11

Scope: source-only review of `evaluate.py`, statistical regression tests, the imported
validation-threshold helper, and freeze checks against registered protocol sections 4–5.
This reviewer did not author the evaluator or statistical tests. No numerical imports,
tests, model execution, bootstrap, SSH, or test-data inspection ran on the Mac.

**Final static assessment: PASS.** The registered statistical calculations pass source
review, and the one material interrupted-score recovery defect is corrected in the
independently reviewed patch recorded below. No remaining correctness blocker was found
within this bounded review.
This document does not authorize production evaluation or substitute for Slurm evidence.

## Registered calculations

| Requirement | Source evidence and assessment |
| --- | --- |
| Primary mean of five paired seed differences | `CONTRASTS`, `_contrast_values`, and `paired_bootstrap` calculate AUROC separately for each arm/seed, subtract within seed, then average the five differences. No score averaging or ensemble AUROC enters the primary estimate. `test_seed_contrast_is_mean_of_auc_differences` deliberately constructs different answers for those two estimands. |
| Paired photograph bootstrap | Defaults are 2,000 draws and RNG seed 20260911. `image_id`, not corruption `source_id`, defines the 800 source groups; each draw samples 800 indices with replacement. One multiplicity vector expands to all nine rows per photograph and is shared across arms, seeds and supported slices. Cohort verification requires 800 photographs with all nine registered conditions before evaluation. |
| AUROC ties and intervals | `WeightedAUC` groups exactly equal native scores and assigns tied positive/negative pairs half credit. Its weighted pair-count formula equals expansion by photograph multiplicity. The interval uses the registered 2.5th/97.5th percentiles with explicit NumPy linear interpolation. Undefined draws remain in the recorded sequence, with no redraw, and any undefined draw suppresses the interval. |
| Practical margin | `practical_conclusion` compares both interval endpoints with 0.005 using the registered strict inequalities. Bounds touching or spanning the margin remain unresolved. The report explicitly avoids interpreting exclusion of a 0.005 benefit as general equivalence. |
| Frozen threshold and zero denominators | `train.validation_threshold` uses correct records from the complete validation mixture, one-based rank `ceil(.95*n)`, and strict `score > threshold`. Freeze recomputes the threshold from the selected validation artifact. The evaluator reuses that one threshold for mixture, each condition and confidence slice. Ties remain accepted. Undefined AUROC/recall/risk denominators become `None`/CSV `NA`; zero retained records gives zero coverage and undefined retained risk. No ordering-based coverage curve is computed, so no additional tie-breaking rule is invoked. |
| Confidence slice and support | The mask is MSP >= 0.9, without filtering by correctness. Distinct `image_id` counts are computed separately for confident errors and confident correct predictions, permitting overlap. Both must reach 200 before subgroup intervals are generated. Descriptive subgroup rows remain available when support fails; cutoff and outcome requirements are fixed. |
| Nine conditions and interaction | `VIEWS` fixes clean plus four corruption families at severities 3 and 5. Every model/reference receives every condition row. Interaction coefficients are full graph minus full set minus raw graph plus raw set; a negative value means a larger graph benefit with attention-only features. Secondary intervals remain explicitly nonconfirmatory. |
| Freeze and completed results | Freeze requires completed registered fits, correct validation alignment/calibration, and passing cache readiness. Before test scoring, checks bind the protocol/code, statistical definitions, cache/cohort, original/rewired tensor provenance, run artifacts and calibration references. Scoring receipts and final evaluation manifests reject changed hashes. Completed evaluation returns verified existing artifacts and rejects replacement model scores. The pre-completion orphan case below was the exception. |

The statistical tests cover tied/weighted AUROC, photograph multiplicities, deterministic
draws, undefined draws, the estimand, interaction sign, support floors, missing classes,
zero acceptance, practical-margin boundaries, the complete matrix and rewired identity.
They have been inspected rather than executed by this reviewer. These focused tests are
not a full end-to-end production freeze/scoring demonstration.

## Material finding: orphan score replacement

**P2, correctness of recovery:** the original `score_one` implementation at lines 443–451
detected an existing prediction NPZ without its completion receipt, recomputed predictions,
and overwrote that NPZ before marking recovery. It did not compare the pre-existing score
content or preserve its byte identity. A crash between atomic NPZ creation and receipt
creation can reach this branch. Protocol section 5 permits byte-identical interrupted
artifact recovery, not replacement by a numerically different later test pass.

Root authorized the evaluator owner to correct only this path: bind new score artifacts
to their frozen provenance, verify the orphan's metadata/schema and frozen model identity,
require exact array names/dtypes/shapes/content on recomputation, and retain the original
NPZ bytes while attaching the missing receipt. Differing or unverifiable content must
fail closed. ZIP serialization metadata must not create a false inequality: compare
array content and retain the original archive/hash. This strict artifact-recovery rule
is separate from the 1e-5 numerical tolerance allowed for validation restore audits.

## Independent correction re-review

**PASS, pending Slurm execution.** The evaluator owner implemented the correction; this
reviewer inspected the stable patch without modifying it. New score files embed a scalar
provenance record containing the freeze, run, selected config/checkpoint, implementation
and device identity. An orphan must match that provenance, align with the canonical
metadata/cohort, have finite score arrays, and match frozen recomputation in every array
name, dtype, shape and value. Missing legacy provenance or any difference fails closed.

The accepted recovery branch does not call the NPZ writer. It rechecks the original file's
identity and SHA, revalidates the freeze, and attaches a receipt to the preserved archive.
The receipt states the exact recovery rule. Existing completed receipt/results behavior
remains unchanged. The added scalar is compatible with the evaluator's metadata checks
and file-based publisher checks.

New regression tests assert unchanged NPZ bytes, SHA and modification time on successful
recovery, and preserved bytes/no receipt on a score change smaller than the restore
tolerance or on wrong/missing provenance. They simulate the crash gap directly. Tests
were read, not executed here; Operations must run the updated suite inside Slurm.

Reviewed correction identities:

- `evaluate.py`: `f5ef7405407d5478bef24c9e47b4fc3cd26cdc9ab1f040ff6876c590385358e3`
- `tests/test_topology_evaluation.py`: `a2fea224b161e14b126ac6f7e1e117580152183b6ab2a66b7c12b2f0aa9fa069`

Operations subsequently reported **PASS** for Slurm CPU job **877580**: all 18 updated
evaluation-suite and existing threshold checks passed on immutable release
`dfc58324c469f4a3`, using the repaired source hashes above. This is execution evidence
reported by Operations; the reviewer did not execute tests on the Mac. The full frozen
scoring/analysis workflow remains a later gate. No production scores were opened.
This review does not need repeating unless source or runtime evidence changes.

## Reviewed baseline identities

- `evaluate.py`: `b18a9223cd769110ffd133eba8bc6589aa1c2afdb1e8baf1cbc9e7092b718e02`
- `tests/test_topology_evaluation.py`: `b732ae00438618df91b60261ad49b265fb4894eb03aa776fa7f4b5652c0bd248`
- `tests/test_topology_training.py`: `74af71f4f3d77cf352b9a4f0f6f604b113a4503a5eba59d6ee4dfe28370dd1d7`
- `PROTOCOL.md`: `fcdbc850550b3d31ff98f556b1d3584cec6f5150901830df75a5b8e00cc1b5db`
