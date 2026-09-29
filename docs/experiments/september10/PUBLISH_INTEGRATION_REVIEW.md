# Independent publisher integration re-review — 2026-09-11

**Static result: PASS for the focused integration correction.** No remaining concrete
blocker was found in the final-artifact inventory, frozen provenance, or manifest-backed
Slurm source-bundle paths. This is conditional source approval, not evidence of a passing
Slurm test, authenticated publication, successful numerical restore, or production GO.

The reviewer authored the original publisher, but did not author the integration repair
under review. The repair's author documented its motivation and implementation separately
in [PUBLISH_OPS_REVIEW.md](PUBLISH_OPS_REVIEW.md). This review inspected the repaired
implementation and tests against the actual evaluator and Operations interfaces. No
imports, numerical code, tests, SSH commands, or Hub operations ran on the Mac.

Reviewed source identities:

- `pilots/topology_20260910/publish.py`:
  `39adcc045358e473521ee54ee3c16b45cb4fb3a2f7b516aa0b42afe66966ec84`
- `pilots/topology_20260910/test_publish.py`:
  `4edd651bed8e9a8dd9e21b305096047cb1479f6ea9981af2ccf4b381834e2653`

## Checks and conclusions

| Check | Static conclusion |
| --- | --- |
| Evaluator output coverage | `copy_final_results` requires exactly the evaluator's six summary/state artifacts, every frozen model prediction and receipt, and both baseline prediction files. It verifies `evaluation_complete.json`, every artifact hash, and the freeze identity. Missing or incomplete supplied results fail publication. |
| Calibration and frozen provenance | `stage_snapshot` includes both sibling analytic-validation references with their frozen hashes, protocol/cohort/cache identities, readiness receipt, rewiring manifest, and declared processor configuration. The run mapping must contain all 35 fits, or the explicitly approved pre-test 25-fit reduction. |
| Gitless Slurm release | `bundle_source` accepts the actual `source_manifest.json` contract, requires the full registered base commit, verifies declared hashes, copies only allowed source, and checks copied bytes against the manifest. It records release provenance and correctly omits the unavailable Git audit patch. |
| Exact source restoration | Every run's implementation map and the frozen evaluator implementation must match the source overlay. Both the imported `polygraph` package and experiment package are allowed. The restore instruction uses the finished overlay; an optional Git audit patch is not applied twice. |
| Rewired restore identity | The smoke check now passes the configured arm into `_cache_identity`, retaining the rewiring-manifest binding for `full_rewired`. Numerical equivalence still requires the Slurm restore audit. |
| Focused regression coverage | New source tests exercise the real Gitless bundler with Git explicitly forbidden, excluded private-context paths, changed source rejection, a complete 35-fit final inventory, missing result rejection, and missing calibration/model rejection. Existing tests cover private-repository rejection, checksum/revision enforcement, retries, locking and immutable run files. Tests were inspected, not executed in this review. |

The reuse guide now describes the exact source overlay, release-manifest provenance,
optional Git audit patch, and full frozen/final artifact scope consistently with the code.

## Operations interface

Operations confirmed the proposed central invocation uses the experiment venv Python,
`--repo-id omrifahn/polygraph-experiments`, separate run and publication directories,
the immutable release as `--code-root`, the production cache as `--cache-metadata-root`,
and `--once`. Final publication adds the original freeze and results root, and must run
after `evaluation_complete.json` exists. Operations also confirmed its source manifest
includes the full registered `base_commit` field.

`stage.py` logs command arguments, while `submit.py` exports only the workflow hash.
Operations confirmed use of the same Unix user's persisted Hugging Face credential store,
with no `HF_HOME` override and no token in command arguments or logged environment values.
Its `HF_HUB_CACHE` override affects the public-model cache, not the token location. This
is a compatible credential-path design; successful authentication has not been claimed.

## Outstanding execution gates

- Run the updated publisher tests inside Slurm, including the real manifest-backed path.
- Verify the installed Hub API and actual private-repository access, then perform one
  real upload and a download pinned to its immutable revision with all checksums verified.
- Run validation-only fresh-environment restores on the required representative completed
  architectures, including the rewired cache identity and logit preprocessing.
- Review the generated immutable workflow's actual commands and dependencies. Publication
  failure must preserve scientific work and cannot imply successful training/evaluation.

Operations reported that the unified diagnostic GPU preflight was running at review time;
this review does not pre-empt its result. No concrete production workflow or authenticated
publisher job was available for execution-level certification.
