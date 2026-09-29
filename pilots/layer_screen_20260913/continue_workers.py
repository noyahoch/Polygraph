"""Selection/deadline-only continuation of frozen worker profiling helpers.

Ops freezes the parent result and missing-case plan after the parent job ends.
Completed case files can be reused on restart; their measurements are not rerun.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import tempfile
import time
import traceback

import torch

from .profile_workers import (ObservedDataset, backend_settings, check_parity,
    compare_steps, deterministic_parity, settings, timing)
from .protocol import atomic_json, digest, file_sha256, implementation_identity, protocol, require_slurm
from .train import _seed

ARMS = ("block11", "union12", "block2", "block5", "block8", "union4")
WORKERS = (0, 2, 4)


def complete(case):
    return bool(case.get("parity", {}).get("passed") and "timing" in case
                and "timing_error" not in case)


def validate(plan, parent, args):
    if plan.get("schema_version") != 1: raise RuntimeError("Unknown continuation plan")
    if plan["parent_terminal_state"].split()[0] not in ("COMPLETED","FAILED","TIMEOUT","CANCELLED"):
        raise RuntimeError("Ops must freeze the missing list only after the parent job is terminal")
    if plan["parent_result_sha256"] != file_sha256(args.parent_result):
        raise RuntimeError("Parent evidence changed after missing-case freeze")
    if str(parent["job_id"]) != str(plan["parent_job_id"]):
        raise RuntimeError("Parent job mismatch")
    root = Path(__file__).resolve().parents[2]
    helper = "pilots/layer_screen_20260913/profile_workers.py"
    core = {**implementation_identity(), helper: file_sha256(root/helper)}
    if core != plan["core_files_sha256"] or parent["source_sha256"] != core[helper]:
        raise RuntimeError("Frozen measurement/audit/training helpers differ")
    universe = {f"{arm}/workers{workers}" for arm in ARMS for workers in WORKERS}
    completed = {key for key,case in parent["cases"].items() if complete(case)}
    if completed != set(plan["completed_cases"]): raise RuntimeError("Plan misstates complete parent cases")
    missing = universe - completed
    if missing != set(plan["missing_cases"]) or len(plan["missing_cases"]) != len(missing):
        raise RuntimeError("Plan missing cases must be exactly the unfinished original matrix")
    manifest = json.loads((args.cache/"manifest.json").read_text())
    if not manifest.get("complete") or not manifest.get("diagnostic_only") or manifest.get("records") != 36:
        raise RuntimeError("Only the existing complete36-record diagnostic cache is permitted")
    binding = {"core_files_sha256":core,"protocol_sha256":digest(protocol()),
        "cache_manifest_sha256":file_sha256(args.cache/"manifest.json"),
        "production_numerical_runtime":backend_settings(),"torch":torch.__version__,
        "gpu":torch.cuda.get_device_name(),"cpus_per_task":int(os.environ["SLURM_CPUS_PER_TASK"])}
    for key,value in binding.items():
        if value != plan[key]: raise RuntimeError(f"Continuation compatibility mismatch: {key}")
    for key in ("protocol_sha256","production_numerical_runtime","torch","gpu"):
        if parent[key] != binding[key]: raise RuntimeError(f"Parent compatibility mismatch: {key}")
    for key in completed:
        if parent["cases"][key]["timing"]["numerical_runtime"] != binding["production_numerical_runtime"]:
            raise RuntimeError(f"Parent timing used another numerical runtime: {key}")
    if binding["cpus_per_task"] != 6: raise RuntimeError("Continuation requires the same6-CPU allocation")
    selected = [f"{arm}/workers{workers}" for arm in ARMS if arm in args.arms
                for workers in WORKERS if workers in args.workers and f"{arm}/workers{workers}" in missing]
    if not selected: raise RuntimeError("Selection contains no missing cases")
    return binding,selected


def main():
    require_slurm()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--parent-result",type=Path,required=True)
    p.add_argument("--arms",choices=ARMS,nargs="+",default=list(ARMS))
    p.add_argument("--workers",choices=WORKERS,type=int,nargs="+",default=list(WORKERS))
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--max-seconds",type=int,default=840)
    args=p.parse_args()
    if not 1 <= args.max_seconds <= 900: p.error("Internal cap must be1..900 seconds; default840")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8": p.error("Export CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python")
    if torch.are_deterministic_algorithms_enabled(): p.error("Production baseline must use ordinary CUDA")
    _seed(7)
    plan=json.loads(args.plan.read_text())
    parent=json.loads(args.parent_result.read_text())
    binding,selected=validate(plan,parent,args)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    case_dir=args.out.parent/(args.out.stem+"_cases")
    case_dir.mkdir(exist_ok=True)
    trace_dir=args.out.parent/(args.out.stem+"_io_"+os.environ["SLURM_JOB_ID"])
    trace_dir.mkdir(exist_ok=False)
    started=time.monotonic()
    lineage={"job_id":os.environ["SLURM_JOB_ID"],"wrapper_sha256":file_sha256(__file__),
        "plan_sha256":file_sha256(args.plan),"parent_result_sha256":plan["parent_result_sha256"],
        "parent_job_id":plan["parent_job_id"],"parent_source_release":plan["parent_source_release"],
        "compatibility":binding,"compatibility_sha256":digest(binding)}
    report={**lineage,"technical_only":True,"scientific_fits_created":0,"test_evaluated":False,
        "production_admitted":False,"selected_cases":selected,"cases":{},"support_references":{},
        "selected_cases_passed":False,"limitations":[
            "The original18-case matrix is unchanged; this wrapper only selects missing cases and bounds time.",
            "Regenerated workers0 parity is necessary when completed parent JSON lacks gradient reference tensors; its timing is never replaced.",
            "Same36-record one-shard diagnostic; no measurement of production multishard NFS contention.",
            "Per-arm continuation repeats process startup and may encounter different OS/NFS caches; support/setup time remains charged.",
            "Selected coverage is not full-matrix validation or resource admission."]}
    def save():
        report["elapsed_seconds"]=time.monotonic()-started
        atomic_json(args.out,report)
    def record(key,case):
        payload={**lineage,"key":key,"case":case,"elapsed_seconds":time.monotonic()-started}
        atomic_json(case_dir/(key.replace("/","_")+".json"),payload)
        report["cases"][key]=payload
        save()
    def expired(signum,frame): raise TimeoutError("Missing-case continuation internal time cap")
    signal.signal(signal.SIGALRM,expired); signal.alarm(args.max_seconds)
    save()
    try:
        # Reuse only fully completed, compatible case files from this wrapper.
        # Failed/partial case evidence is retained before another attempt.
        pending=[]
        for key in selected:
            previous=case_dir/(key.replace("/","_")+".json")
            if previous.exists():
                payload=json.loads(previous.read_text())
                if payload["compatibility"] != binding or payload["wrapper_sha256"] != lineage["wrapper_sha256"]:
                    raise RuntimeError(f"Existing continuation case is incompatible: {key}")
                if complete(payload["case"]):
                    report["cases"][key]=payload
                    continue
                archive=previous.with_name(previous.stem+"_attempt_"+str(payload["job_id"])+".json")
                if not archive.exists(): atomic_json(archive,payload)
            pending.append(key)
        save()
        with tempfile.TemporaryDirectory(prefix="continuation_disposable_",dir=args.out.parent) as temporary:
            for arm in ARMS:
                chosen=[w for w in WORKERS if f"{arm}/workers{w}" in pending]
                if not chosen: continue
                reference=None
                if 0 not in chosen:
                    began=time.monotonic()
                    ds=ObservedDataset(args.cache,"train",arm,diagnostic=True,trace_dir=trace_dir,phase=arm+"/reference_only_workers0")
                    support={"workers":0,"timing_rerun":False,"purpose":"reconstruct absent per-step gradient/input reference"}
                    try:
                        with deterministic_parity():
                            reference,parity=check_parity(ds,arm,0,Path(temporary))
                            support["parity"]=parity
                    except TimeoutError: raise
                    except Exception as error: support["error"]=repr(error); support["traceback"]=traceback.format_exc()
                    finally:
                        support["seconds"]=time.monotonic()-began
                        report["support_references"][arm]=support
                        save()
                    del ds
                for workers in chosen:
                    key=f"{arm}/workers{workers}"
                    ds=ObservedDataset(args.cache,"train",arm,diagnostic=True,trace_dir=trace_dir,phase=key)
                    case={"loader":settings(workers),"parity":{"passed":False}}
                    began=time.monotonic()
                    try:
                        with deterministic_parity() as audit_backend:
                            case["parity_numerical_runtime"]=audit_backend
                            steps,parity=check_parity(ds,arm,workers,Path(temporary))
                            if workers == 0: reference=steps
                            if reference is None: raise RuntimeError("No passing workers0 gradient/input reference")
                            parity["gradient_max_abs_difference_vs_workers0"]=compare_steps(reference,steps)
                            case["parity"]=parity
                    except TimeoutError: raise
                    except Exception as error:
                        case["parity"]={"passed":False,"error":repr(error),"traceback":traceback.format_exc()}
                    finally:
                        case["parity_seconds"]=time.monotonic()-began
                        case["restored_numerical_runtime"]=backend_settings()
                        record(key,case)
                    if backend_settings() != binding["production_numerical_runtime"]:
                        raise RuntimeError("Audit did not restore production numerical runtime")
                    try: case["timing"]=timing(ds,arm,workers)
                    except TimeoutError: raise
                    except Exception as error:
                        case["timing_error"]={"error":repr(error),"traceback":traceback.format_exc()}
                    finally: record(key,case)
                    print(json.dumps({"event":"continuation_case_complete","key":key,
                        "passed":complete(case),"elapsed_seconds":time.monotonic()-started}),flush=True)
                    del ds
        report["selected_cases_passed"]=all(key in report["cases"] and complete(report["cases"][key]["case"]) for key in selected)
    except BaseException as error:
        report["error"]=repr(error); report["traceback"]=traceback.format_exc()
        raise
    finally:
        signal.alarm(0)
        report["ended_unix"]=time.time()
        save()


if __name__ == "__main__": main()
