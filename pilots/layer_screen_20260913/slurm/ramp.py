"""Resource-only release gate for the two predeclared real detector fits.

Run in an allocated CPU job. This reads publication metadata, never tensors or
validation scores, and submits only workflow-declared stages via runtime.py.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from .runtime import atomic_json, file_sha256, require_slurm, submit_stage

INITIAL = {"block11/seed7", "union12/seed7"}
MATRIX = {f"{arm}/seed{seed}" for arm in ("block2","block5","block8","block11","union4","union12")
          for seed in (7,17)} | {"block11/seed1","block11/seed27"}
BAD_STATES = {"FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL","BOOT_FAIL","PREEMPTED"}


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def key(entry):
    return f"{entry['arm']}/seed{int(entry['seed'])}"


def accounting(job):
    result = subprocess.run(["sacct","-X","-nP","-j",str(job),
        "--format=JobIDRaw,State,ExitCode,Start,ElapsedRaw"],capture_output=True,text=True,timeout=15)
    if result.returncode: return {"available":False,"error":result.stderr[-1000:]}
    for line in result.stdout.splitlines():
        parts=line.split("|")
        if len(parts)>=5 and parts[0]==str(job):
            try: elapsed=int(parts[4])
            except ValueError: elapsed=None
            return {"available":elapsed is not None,"state":parts[1].split()[0],
                    "exit_code":parts[2],"start_raw":parts[3],"elapsed_seconds":elapsed,
                    "observed_unix":time.time()}
    return {"available":False,"reason":"Accounting record unavailable"}


def validate_config(run, entry, fit, release):
    config=read(run/"config.json")
    if config.get("arm")!=entry["arm"] or config.get("seed")!=entry["seed"]:
        raise RuntimeError("Initial run identity changed")
    if config.get("loader",{}).get("num_workers")!=fit["workers"]:
        raise RuntimeError("Initial run worker count differs from admission")
    if config.get("training",{}).get("epochs")!=60 or config["training"].get("minimum_epochs")!=20:
        raise RuntimeError("Frozen training horizon changed")
    if config.get("implementation_sha256")!=digest(config["implementation"]):
        raise RuntimeError("Run implementation digest mismatch")
    for name,wanted in config["implementation"].items():
        target=(release/name).resolve()
        if not target.is_relative_to(release.resolve()) or file_sha256(target)!=wanted:
            raise RuntimeError("Initial run uses another source: "+name)
    numerical=config.get("numerical_runtime",{})
    if numerical.get("deterministic_algorithms") is not False or numerical.get("cublas_workspace_config")!=":4096:8":
        raise RuntimeError("Initial run numerical runtime differs from measured production")
    return file_sha256(run/"config.json")


def observe(root, release, entry, fit, observations):
    run=Path(entry["run_dir"])
    if not run.resolve().is_relative_to(root.resolve()): raise RuntimeError("Run path escapes experiment")
    status=accounting(entry["id"])
    if status.get("state") in BAD_STATES: raise RuntimeError(f"Initial run {key(entry)} is {status['state']}")
    receipt_path=root/"manifests/execution"/f"{entry['stage']}_{entry['id']}.json"
    if not receipt_path.exists(): return {"ready":False,"reason":"Waiting for launcher receipt","slurm":status}
    receipt=read(receipt_path)
    if receipt.get("status")=="failed" or receipt.get("exit_code") not in (None,0):
        raise RuntimeError("Initial launcher reported failure: "+key(entry))
    for name in ("config.json","history.json","latest.pt","attempts.json"):
        if not (run/name).exists(): return {"ready":False,"reason":"Waiting for "+name,"slurm":status}
    attempts=[row for row in read(run/"attempts.json") if str(row.get("slurm_job_id"))==str(entry["id"])]
    if len(attempts)!=1 or attempts[0].get("resume") is not False:
        raise RuntimeError("Ramp requires a verified fresh same-job attempt; resumed history cannot prove new epoch wall")
    if attempts[0].get("status")=="failed": raise RuntimeError("Initial trainer attempt failed: "+key(entry))
    config_sha=validate_config(run,entry,fit,release)
    path=run/"history.json"
    before=path.stat()
    history_bytes=path.read_bytes()
    history=json.loads(history_bytes)
    after=path.stat()
    latest=(run/"latest.pt").stat()
    if (before.st_ino,before.st_mtime_ns,before.st_size)!=(after.st_ino,after.st_mtime_ns,after.st_size):
        return {"ready":False,"reason":"History publication in progress","slurm":status}
    if latest.st_mtime_ns>after.st_mtime_ns or latest.st_size<=0:
        return {"ready":False,"reason":"Waiting for checkpoint/history publication boundary","slurm":status}
    n=len(history)
    if n>60 or [row["epoch"] for row in history]!=list(range(1,n+1)):
        raise RuntimeError("Published history has invalid epoch sequence")
    if not n: return {"ready":False,"reason":"No completed epoch","slurm":status}
    launch=dt.datetime.fromisoformat(receipt["started_utc"].replace("Z","+00:00"))
    if launch.tzinfo is None: raise RuntimeError("Launcher timestamp must have explicit timezone")
    launcher_seconds=after.st_mtime-launch.timestamp()
    if launcher_seconds<=0: raise RuntimeError("Invalid complete-epoch publication timestamp")
    point={"epoch":n,"history_mtime_unix":after.st_mtime,"history_sha256":hashlib.sha256(history_bytes).hexdigest(),
           "latest_mtime_unix":latest.st_mtime,"latest_bytes":latest.st_size,
           "launcher_to_publication_seconds":launcher_seconds,"observed_unix":time.time()}
    if not observations or observations[-1]["epoch"]!=n:
        if observations and observations[-1]["epoch"]>n: raise RuntimeError("Completed epoch cursor regressed")
        observations.append(point)
    if n<2 or not status.get("available") or status.get("state") not in ("RUNNING","COMPLETING","COMPLETED"):
        return {"ready":False,"reason":"Need two complete epochs and live accounting","epoch":n,"slurm":status}
    cycles=[]
    for a,b in zip(observations,observations[1:]):
        count=b["epoch"]-a["epoch"]
        seconds=b["history_mtime_unix"]-a["history_mtime_unix"]
        if count<=0 or seconds<=0: raise RuntimeError("Invalid publication interval")
        cycles.append({"epoch_delta":count,"seconds":seconds,"seconds_per_epoch":seconds/count})
    # Charge the actual prefix once. Current Slurm elapsed may include part of
    # the next epoch/poll interval; keeping it makes the estimate conservative.
    prefix=max(launcher_seconds,status["elapsed_seconds"])
    if cycles:
        epoch_seconds=max(row["seconds_per_epoch"] for row in cycles)
        method="maximum observed publication interval per epoch"
    else:
        epoch_seconds=prefix/n
        method="startup-inclusive cumulative average fallback; no consecutive boundary observed"
    limited=not cycles or any(row["epoch_delta"]!=1 for row in cycles)
    reserve=float(fit["startup_final_reserve_seconds"])
    if reserve<0: raise RuntimeError("Negative terminal reserve")
    projected=1.5*(prefix+(60-n)*epoch_seconds+reserve)
    cap=int(fit["limit_minutes"])*60
    return {"ready":True,"passes_reserved_cap":projected<=cap,"completed_epochs":n,
            "projected_full_60_epoch_seconds":projected,"reserved_cap_seconds":cap,
            "actual_prefix_seconds":prefix,"future_epoch_seconds":epoch_seconds,
            "terminal_reserve_seconds":reserve,"headroom_multiplier":1.5,
            "cycle_observations":cycles,"method":method,"limited_boundary_sample":limited,
            "config_sha256":config_sha,"slurm":status,
            "memory_observation":"32GB allocation and no recorded OOM through observed epochs; peak RSS/GPU memory unavailable, not zero",
            "scope":"resource-only; no loss/AUROC decisions; existing cap is never reduced or increased"}


def prior_job(root, stage, minutes, gpu, cpus):
    path=root/"manifests/submissions.tsv"
    found=set()
    if path.exists():
        for line in path.read_text().splitlines():
            row=line.split("\t")
            if len(row)==6 and row[2]==stage:
                if row[3:]!=[f"{minutes}min",f"{int(gpu)}gpu",f"{cpus}cpu"]:
                    raise RuntimeError("Previously submitted stage has different resources: "+stage)
                found.add(row[1])
    if len(found)>1: raise RuntimeError("Ambiguous prior submissions require operator review: "+stage)
    return next(iter(found),None)


def main():
    require_slurm()
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("root","old-root","release","admission","dispatch"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--max-wait-seconds",type=int,default=21600)
    args=p.parse_args()
    if os.environ.get("SLURM_JOB_GPUS"): raise RuntimeError("Ramp must run in a CPU allocation")
    out=args.root/"manifests/ramp_20260914.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    with (out.parent/"ramp_20260914.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        admission,dispatch=read(args.admission),read(args.dispatch)
        if admission.get("admitted") is not True: raise RuntimeError("Full-budget admission has not passed")
        initial,remaining=dispatch["initial_fits"],dispatch["remaining_fits"]
        if len(initial)!=2 or {key(e) for e in initial}!=INITIAL:
            raise RuntimeError("The two fixed ramp runs changed")
        if len(remaining)!=12 or {key(e) for e in initial+remaining}!=MATRIX:
            raise RuntimeError("The fixed fourteen-fit matrix changed")
        if len({e["stage"] for e in initial+remaining})!=14: raise RuntimeError("Duplicate fit stage")
        for entry in initial+remaining:
            fit=admission["fits"][key(entry)]
            if entry["cap_minutes"]!=fit["limit_minutes"] or fit["cpus"]!=6:
                raise RuntimeError("Dispatch resources differ from admitted reservation")
        bindings={"admission_sha256":file_sha256(args.admission),"dispatch_sha256":file_sha256(args.dispatch),
                  "source_sha256":file_sha256(__file__)}
        state=read(out) if out.exists() else {"bindings":bindings,"observations":{},"submitted_remaining":[],"released":False}
        if state["bindings"]!=bindings: raise RuntimeError("Ramp restart input/source changed")
        if state.get("complete"): return
        if "error" in state:
            state.setdefault("previous_errors",[]).append(state.pop("error"))
        state["job_id"]=os.environ["SLURM_JOB_ID"]
        began=time.monotonic()
        try:
            while True:
                evaluations={}
                for entry in initial:
                    run_key=key(entry)
                    evaluations[run_key]=observe(args.root,args.release,entry,admission["fits"][run_key],
                                                state["observations"].setdefault(run_key,[]))
                state["evaluations"]=evaluations
                state["updated_unix"]=time.time()
                atomic_json(out,state)
                if all(value.get("ready") for value in evaluations.values()):
                    if not all(value["passes_reserved_cap"] for value in evaluations.values()):
                        raise RuntimeError("Observed complete-epoch wall exceeds fixed reserved cap; no remaining fits released")
                    break
                if time.monotonic()-began>args.max_wait_seconds:
                    raise TimeoutError("Two-run ramp evidence did not arrive within the bounded wait")
                time.sleep(15)
            if any(file_sha256(path)!=bindings[name] for path,name in
                   ((args.admission,"admission_sha256"),(args.dispatch,"dispatch_sha256"))):
                raise RuntimeError("Admission/dispatch changed while observing initial fits")
            state["released"]=True
            state["release_basis"]="Both fixed fits published >=2complete epochs and fit their original resource caps; no scientific-score selection"
            atomic_json(out,state)
            # Eight lanes: two initial jobs plus six immediately available slots.
            # Later independent fits wait for a lane to free, even on failure;
            # finalization still audits all fourteen outputs.
            lanes=[str(entry["id"]) for entry in initial]+[None]*6
            for index,entry in enumerate(remaining):
                lane=index+2 if index<6 else (index-6)%8
                dependency="afterany:"+lanes[lane] if lanes[lane] else None
                job=prior_job(args.root,entry["stage"],entry["cap_minutes"],True,6)
                if job is None:
                    job=submit_stage(args.root,args.old_root,args.release,entry["stage"],entry["cap_minutes"],True,6,dependency)
                lanes[lane]=str(job)
                item={**entry,"id":str(job),"dependency":dependency}
                if not any(row["stage"]==entry["stage"] for row in state["submitted_remaining"]):
                    state["submitted_remaining"].append(item)
                atomic_json(out,state)
            final=dispatch["finalize"]
            ids=[str(e["id"]) for e in initial]+[row["id"] for row in state["submitted_remaining"]]
            job=prior_job(args.root,final["stage"],final["cap_minutes"],False,final["cpus"])
            if job is None:
                job=submit_stage(args.root,args.old_root,args.release,final["stage"],final["cap_minutes"],False,
                                 final["cpus"],"afterany:"+":".join(ids))
            state["finalize_job_id"]=str(job)
            state["complete"]=True
            atomic_json(out,state)
        except BaseException as error:
            state["error"]=repr(error); state["updated_unix"]=time.time()
            atomic_json(out,state)
            raise


if __name__=="__main__": main()
