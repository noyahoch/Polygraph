"""Dispatch the explicitly authorized four-fit subset after full capture.

The original scientific protocol/cache is preserved. The versioned admission
selects exactly block11/union4 with seeds7/17; no other model is submitted.
"""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import time

from .ramp import FOUR_SCOPE, FOUR_MATRIX, prior_job, read
from .runtime import atomic_json, file_sha256, require_slurm, submit_stage


def ensure_ramp(args, dispatch_path):
    # The dispatch is immutable once complete; the ramp binds its exact bytes.
    # A restart after publication must still reconcile/ensure the ramp stage.
    stage="ramp_four_20260914"
    job=prior_job(args.root,stage,360,False,2)
    if job is None:
        job=submit_stage(args.root,args.old_root,args.release,stage,360,False,2)
    atomic_json(args.root/"manifests/dispatch_four_ramp_receipt.json",
                {"job_id":str(job),"dispatch_sha256":file_sha256(dispatch_path),"scope_id":FOUR_SCOPE})


def main():
    require_slurm()
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("root","old-root","release","admission"):
        p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    if os.environ.get("SLURM_JOB_GPUS"): raise RuntimeError("Dispatcher requires a CPU allocation")
    out=args.root/"manifests/dispatch_four_20260914.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        admission=read(args.admission)
        if admission.get("admitted") is not True or admission.get("scope_id")!=FOUR_SCOPE:
            raise RuntimeError("The authorized versioned four-fit admission is required")
        if admission.get("budget_gpu_minutes")!=4800 or set(admission["fits"])!=FOUR_MATRIX:
            raise RuntimeError("The authorized80-GPU-hour/four-fit matrix changed")
        total=admission["diagnostic_cap_minutes"]+admission["extraction_limit_minutes"]
        total+=sum(fit["limit_minutes"] for fit in admission["fits"].values())
        if total!=admission["reserved_gpu_minutes"] or total>4800:
            raise RuntimeError("Declared reservation exceeds or misstates the budget")
        cache=args.root/"feature_cache"
        manifest=read(cache/"manifest.json")
        if manifest.get("complete") is not True or manifest.get("diagnostic_only") is not False or manifest.get("records")!=28800:
            raise RuntimeError("Complete28800-record development cache required before real fits")
        if file_sha256(cache/"manifest.json")!=admission["full_cache_manifest_sha256"]:
            raise RuntimeError("Full cache differs from the admitted manifest")
        entries=[]
        for arm,seed in (("block11",7),("union4",7),("block11",17),("union4",17)):
            fit=admission["fits"][f"{arm}/seed{seed}"]
            if fit["cpus"]!=6 or fit["workers"] not in (0,2,4):
                raise RuntimeError("Unexpected admitted loader allocation")
            entries.append({"stage":f"fit_four_{arm}_s{seed}","arm":arm,"seed":seed,
                            "cap_minutes":fit["limit_minutes"],
                            "run_dir":str(args.root/"runs_four_20260914"/arm/f"seed{seed}")})
        bindings={"admission_sha256":file_sha256(args.admission),"source_sha256":file_sha256(__file__),
                  "cache_manifest_sha256":file_sha256(cache/"manifest.json"),"release":str(args.release)}
        state=read(out) if out.exists() else {"scope_id":FOUR_SCOPE,"bindings":bindings,
            "initial_fits":[],"remaining_fits":entries[2:],"complete":False,
            "finalize":{"stage":"finalize_four_20260914","cap_minutes":90,"cpus":2}}
        if state["bindings"]!=bindings: raise RuntimeError("Dispatch restart input/source changed")
        if state.get("complete"):
            ensure_ramp(args,out)
            return
        if "error" in state: state.setdefault("previous_errors",[]).append(state.pop("error"))
        state["job_id"]=os.environ["SLURM_JOB_ID"]
        atomic_json(out,state)
        try:
            for entry in entries[:2]:
                job=prior_job(args.root,entry["stage"],entry["cap_minutes"],True,6)
                if job is None:
                    job=submit_stage(args.root,args.old_root,args.release,entry["stage"],entry["cap_minutes"],True,6)
                if not any(row["stage"]==entry["stage"] for row in state["initial_fits"]):
                    state["initial_fits"].append({**entry,"id":str(job)})
                atomic_json(out,state)
            # Publish the final dispatcher bytes BEFORE submitting the ramp,
            # whose restart identity binds this file. Its receipt is separate.
            state["complete"]=True
            state["completed_unix"]=time.time()
            atomic_json(out,state)
            ensure_ramp(args,out)
        except BaseException as error:
            # Do not mutate dispatcher bytes after a ramp might have accepted
            # them. Preserve errors separately for the independent guardian.
            failure={"error":repr(error),"scope_id":FOUR_SCOPE,"initial_fits":state["initial_fits"],
                     "failed_unix":time.time(),"dispatcher_job_id":os.environ["SLURM_JOB_ID"]}
            ramp_intent=prior_job(args.root,"ramp_four_20260914",360,False,2)
            if ramp_intent is None and not state.get("complete") and state["initial_fits"]:
                try:
                    final=state["finalize"]
                    dep="afterany:"+":".join(row["id"] for row in state["initial_fits"])
                    job=submit_stage(args.root,args.old_root,args.release,final["stage"],final["cap_minutes"],False,final["cpus"],dep)
                    failure["finalize_job_id"]=str(job)
                except BaseException as final_error: failure["finalizer_error"]=repr(final_error)
            atomic_json(args.root/"manifests/dispatch_four_failure.json",failure)
            raise


if __name__=="__main__": main()
