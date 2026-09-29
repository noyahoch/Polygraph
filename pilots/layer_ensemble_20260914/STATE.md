# September14 core — restart entry point

The active approved scope is PLAN.md: four single-layer seed7 fits,20 epochs, four disjoint photo roles, learned stack and matched last-only head. This supersedes the earlier six/18-fit plans for the overnight run. Both layer unions and extra seeds are deferred. Local commits only; no push or local ML/tests.

Core protocol/data/train/predict/smoke source is prepared for independent review and Slurm readiness execution. No new scientific fits, smoke pass or results are claimed by this source handoff. The September13 feature capture and old releases remain unchanged. Ops reported canceling old pending admission/dispatch/guardian jobs892205/892206/892207 at13:15 while retaining capture892193; inspect fresh Ops state before any new submission.

Canonical remote experiment suffix: experiments/layer_ensemble_20260914. Inputs at its root are role_map.json and execution.json. Outputs: runs/{block2,block5,block8,block11}/seed7; base_freeze.json; predictions/meta.npz andmeta.json; heads/{stack,last_only}.json plusheads_freeze.json; predictions/dev_eval.npz anddev_eval.json; evaluation/report.json andREPORT.md. The shared cache remains in the September13 experiment and is referenced by immutable hash.

Execution order: prepare role map/plan → one bounded readiness job → all four base fits eligible in parallel → all-four base freeze → meta GPU predictions → two CPU heads/freeze → dev_eval GPU predictions → CPU report/private backup. Ops owns allocation/deadline policy and independent remote partial-status reporting. Root must review actual readiness evidence; code existing on disk is not a passing test or completed model.

Base completion is valid only after20 complete epochs plus selected-checkpoint restoration/export, before23:00 Israel. Training stores latest model/optimizer/RNG/sampler every completed epoch and preserves previous atomic checkpoints on interruption. Existing fits from the old split cannot be resumed into this scope. Deadline or required-stage failure prevents subset/partial scientific comparison. No automatic extra seeds.
