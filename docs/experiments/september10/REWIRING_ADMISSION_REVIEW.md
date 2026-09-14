# Independent review of explicit rewiring admission — 2026-09-11

**Static result: PASS.** The implementation faithfully applies the root's recorded
pre-test decision without changing the construction or relabeling its quality result.
No remaining correctness blocker was found in the reviewed admission, freeze, publication
or operational-wrapper integration. Slurm execution remains a separate requirement.

The reviewer authored [REWIRING_DECISION.md](REWIRING_DECISION.md), but did not author
the admission module or its integration patch. Earlier versions of data/validation and
publisher code were reviewed only for the new changes made by their current owner.
No numerical imports, tests, SSH, feature generation or detector scoring ran on the Mac.

## Decision and preserved evidence

The decision is explicit, restricted to the expected path, and pinned to SHA-256
`0453bb052c668642e42ec1fbd0b6d3359df33f2a5288fb4a6c51de0c47381099`.
It retains all 35 fixed fits, leaves 0.80/2E/20E unchanged, admits only insufficient
mixing, and limits the rewired contrast to descriptive interpretation. Unknown or
changed decision documents fail admission.

The preserved Operations receipt `diagnostic_readiness_877598.json`, SHA-256
`ad20e334662ca8bbc3219c4c57da764f6fefb29493ad0c8b1c204f5a6ccce2bc`, independently
corroborates 180 eligible development graphs, mean changed fraction
0.5441644076028652, failed quality, and completed structural construction. The reviewer
read the saved values without recomputing them. The original failed construction
manifest remains unchanged.

## Checked behavior

- Admission writes a separate `rewire/admission.json`. Construction completeness,
  structural integrity, actual mixing pass/fail, original `complete` value and explicit
  approval remain distinct fields. Failed quality never becomes `passed=true` merely
  because admission is allowed.
- Admission verifies source/cohort/index and per-shard hashes, source-code provenance,
  complete retained record coverage, sidecar correspondence, offsets and endpoint
  structure. It recomputes graph validity rather than trusting a sidecar's PASS claim.
  Unchanged source tensors and source-associated edge construction remain bound.
- Every graph receives structural verification. Only training/validation records enter
  changed-fraction recomputation and quality summaries; test-side receipts are restricted
  to opaque identity/split/integrity fields. Missing graphs, wrong provenance, modified
  sidecars/tensors and hidden test diagnostics fail closed.
- Readiness and freeze bind the admission, original null manifest, decision, diagnostics
  and implementation. Final freeze requires the cache's `diagnostic_only=false`,
  all 36,000 retained records, and all 28,800 training/validation records verified.
  The small 270-record diagnostic cannot substitute for that full-cohort evidence.
- Rewired training configuration and restored test access bind the admission and decision.
  The original constructor remains byte-identical. No model, feature, seed, optimizer,
  threshold, primary estimand or resampling rule changes.
- Evaluator output records the actual quality result and explicitly marks every
  graph-versus-rewired contrast descriptive. The Hebrew report forbids topology-removal,
  structural-necessity and equivalence claims, even if full-cohort mixing later passes.
  Publication copies/binds the admission and exact decision and carries the same limits
  into the collection manifest and model cards.
- Focused tests cover explicit approval, original manifest byte preservation, idempotent
  admission, diagnostic/full-cohort separation, unknown/changed decisions, missing graphs,
  fake structural PASS, tensor/sidecar tampering and forbidden test diagnostics. Publisher
  tests preserve failed quality in the uploaded metadata. These tests were inspected,
  not executed in this source review.

## Operational integration and repaired retry issue

The original constructor still exits nonzero when mixing fails. The Slurm wrapper
accepts that exit only when a completed construction explicitly records failed quality,
then calls the full admission verifier. Validation waits for the successful wrapper and
requires the explicit admission receipt; the generic stage reader checks admitted and
structural fields plus the workflow's decision hash, leaving quality untouched.

The reviewer identified a retry defect in the first wrapper: invoking the unchanged
constructor again on a completed, failed-quality manifest rewrote its elapsed-time field
and invalidated immutable admission provenance. The corrected wrapper skips construction
when `construction_complete=true` and performs full admission verification on the existing
bytes. Absent/incomplete construction still follows the ordinary construction path.
This preserves the original failed receipt during an interrupted-wrapper retry.

## Reviewed source identities

| File | SHA-256 |
| --- | --- |
| `rewiring_decision.py` | `ac155c5b97b82b23eb6ecff2314f478ac73650288ee58a8b80f786628827f6c8` |
| `data.py` | `f23f408b0a185f71094eb6c2cf4c56118d81c3c58a3b5d942730093754e8bfaa` |
| `train.py` | `a623be4a37573a9de5e79c1dfeb845e8023511f2af73bc0dec53ed0551a20d59` |
| `validate.py` | `4ea4bf8c7e85ae868cc0463ea1bb0608f2b2eb45d02e4682b2edb3c2a8e31610` |
| `evaluate.py` | `c032605e2a68be977637e35d84863e5bc819b90788701cfb5571a836f5b6a7bd` |
| `publish.py` | `db79c3b7b32b99b5428db99532809553fe88f61a2c3a8913816c93ffd8f38979` |
| `rewire.py` (unchanged) | `7b4b7eb767b2b3fcbc454af2fee31acd4b474d6b14f27493d6e3d633f7c89b26` |
| `slurm/rewire_stage.py` | `3408b0a95c663071bae74508a4229cb12e8bb2309feedcfa4ce5550d7c5e8498` |
| `slurm/stage.py` | `a611d12290bd34cc03129306da10a5e4553f64fa331fd771787f7f6d6a0f862a` |
| `tests/test_topology_rewiring_admission.py` | `bf5bdd393f530ee4cbb4132c22282f62c12639de04cf59551bf753ed85aee595` |
| `test_publish.py` | `235d40f734fc70ccf64b01ce2a42bd7c205d99f4757a941472e5f240631652c8` |
| `tests/test_topology_evaluation.py` | `7ce04a6992685bf91f4eee06ac54b6fdfd5c5c6f84a1341e5576b03f6d4be75a` |

Unqualified filenames above are under `pilots/topology_20260910/`; `tests/` names are
repository-relative. Required runtime evidence: focused admission tests, successful
admission of the preserved diagnostic without changing its original hashes, and the
remaining readiness stages. Production admission still requires full-cohort evidence
and the actual reviewed workflow; this static result is not a claim that training ran.
