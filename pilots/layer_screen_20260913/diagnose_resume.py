"""Bounded Slurm-only localization of the serial resume discrepancy in job891833.

This creates disposable diagnostic states, never a scientific fit. All original
1e-5 comparisons remain visible; no unknown parameter is exempted from a gate.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch

from .data import CachedDataset, atomic_torch
from .loader_runtime import settings
from .models import build_model
from .profile_workers import cpu_tree, start_iterator, tensor_hash
from .protocol import atomic_json, digest, file_sha256, implementation_identity, protocol, require_slurm
from .train import (BlockShuffleSampler, _collect, _cpu_state, _finite, _forward,
                    _loader, _restore_rng, _rng_state, _seed)

ATOL = RTOL = 1e-5
CONFIG = {"arm": "block11", "batch_size": 24, "loader": settings(0)}


def differences(reference, actual, path="state"):
    """Visit every leaf, with full names; retain even within-tolerance drift."""
    rows = []
    def visit(a, b, name):
        if isinstance(a, np.ndarray): a = torch.from_numpy(a.copy())
        if isinstance(b, np.ndarray): b = torch.from_numpy(b.copy())
        if torch.is_tensor(a) and torch.is_tensor(b):
            a, b = a.detach().cpu(), b.detach().cpu()
            if a.shape != b.shape or a.dtype != b.dtype:
                rows.append({"path": name, "exact": False, "close": False,
                             "structure": [str((a.shape,a.dtype)), str((b.shape,b.dtype))]})
                return
            exact = torch.equal(a, b)
            numeric = a.is_floating_point() or a.is_complex()
            close = bool(torch.allclose(a, b, atol=ATOL, rtol=RTOL)) if numeric else exact
            row = {"path": name, "shape": list(a.shape), "dtype": str(a.dtype),
                   "exact": exact, "close": close}
            if numeric and a.numel():
                delta = (a.double() - b.double()).abs()
                row["max_abs"] = float(delta.max())
                row["outside_tolerance"] = int((delta > ATOL + RTOL * b.double().abs()).sum())
                row["max_abs_reference"] = float(a.abs().max())
                row["max_abs_actual"] = float(b.abs().max())
                if a.numel() <= 8:
                    row["reference_values"], row["actual_values"] = a.tolist(), b.tolist()
            rows.append(row)
        elif isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                rows.append({"path": name+".keys", "exact": False, "close": False,
                             "reference": list(map(str,a)), "actual": list(map(str,b))})
            for key in a.keys() & b.keys(): visit(a[key], b[key], f"{name}.{key}")
        elif isinstance(a, (list,tuple)) and isinstance(b, (list,tuple)):
            if len(a) != len(b):
                rows.append({"path": name+".length", "exact": False, "close": False})
            for i,(aa,bb) in enumerate(zip(a,b)): visit(aa,bb,f"{name}[{i}]")
        else:
            equal = type(a) is type(b) and a == b
            rows.append({"path": name, "exact": bool(equal), "close": bool(equal),
                         **({"reference": repr(a), "actual": repr(b)} if not equal else {})})
    visit(reference,actual,path)
    return {"exact": all(r["exact"] for r in rows), "close": all(r["close"] for r in rows),
            "leaves_compared": len(rows), "differences": [r for r in rows if not r["exact"]]}


def named_optimizer(model, optimizer):
    names = {id(p): name for name,p in model.named_parameters()}
    return {"state": {names[id(p)]: cpu_tree(v) for p,v in optimizer.state.items()},
            "param_groups": [{**{k: cpu_tree(v) for k,v in group.items() if k != "params"},
                              "params": [names[id(p)] for p in group["params"]]}
                             for group in optimizer.param_groups]}


def snapshot(model, optimizer, sampler, generator):
    return {"model": _cpu_state(model), "optimizer": cpu_tree(optimizer.state_dict()),
            "optimizer_named": named_optimizer(model,optimizer),
            "sampler": sampler.state_dict(), "loader_rng": generator.get_state().clone(),
            "rng": cpu_tree(_rng_state())}


def selected_state(value):
    return {k:value[k] for k in ("model","optimizer_named","sampler","loader_rng","rng")}


def make_pair(device):
    model = build_model("block11").to(device)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    return model,optimizer


def synchronize(device):
    if device.type == "cuda": torch.cuda.synchronize()


def epoch(ds, model, optimizer, sampler, generator, device, directory, name, progress):
    loader = _loader(ds,CONFIG,sampler=sampler,generator=generator)
    model.train()
    observed, ids = [], []
    for i,batch in enumerate(start_iterator(loader,generator)):
        input_hash = tensor_hash(batch)
        record_ids = batch.record_id.detach().cpu().clone()
        ids.extend(record_ids.tolist())
        before_rng = cpu_tree(_rng_state())
        optimizer.zero_grad(set_to_none=True)
        scores,labels = _forward(model,batch,"block11",device)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scores,labels,
                    pos_weight=torch.tensor(1.,device=device))
        _finite(loss,"resume diagnostic loss")
        loss.backward()
        gradients = {}
        for key,p in model.named_parameters():
            if p.grad is not None:
                _finite(p.grad,f"resume diagnostic gradient {key}")
                gradients[key] = p.grad.detach().cpu().clone()
        optimizer.step()
        synchronize(device)
        row = {"input_sha256":input_hash,"record_ids":record_ids,
               "loss":loss.detach().cpu().clone(),"scores":scores.detach().cpu().clone(),
               "labels":labels.detach().cpu().clone(),"gradients":gradients,
               "rng_before":before_rng,"after":snapshot(model,optimizer,sampler,generator)}
        path = directory / f"{name}_step{i+1}.pt"
        atomic_torch(path,row)
        observed.append(row)
        progress({"event":"step_saved","branch":name,"batch":i+1,"file":path.name})
        del batch, scores, labels, loss
    assert sorted(ids) == sorted(r["record_id"] for r in ds.entries)
    assert len(observed) == 2, "Only the original36-row diagnostic epoch is allowed"
    return observed, snapshot(model,optimizer,sampler,generator)


def compare_runs(reference, actual, reference_final, actual_final, eval_reference, eval_actual):
    original_steps = []
    for index,(a,b) in enumerate(zip(reference,actual)):
        original_steps.append({"batch":index+1,
            "input_exact": a["input_sha256"] == b["input_sha256"],
            "original_loss_gate": abs(float(a["loss"])-float(b["loss"])) <= ATOL + RTOL*abs(float(a["loss"])),
            "loss":differences(a["loss"],b["loss"],"loss"),
            "gradients":differences(a["gradients"],b["gradients"],"gradients"),
            "training_logits":differences(a["scores"],b["scores"],"scores"),
            "rng_before":differences(a["rng_before"],b["rng_before"],"rng_before"),
            "model_after":differences(a["after"]["model"],b["after"]["model"],"model"),
            "optimizer_after":differences(a["after"]["optimizer_named"],b["after"]["optimizer_named"],"optimizer")})
    final = differences(selected_state(reference_final),selected_state(actual_final))
    model_close = differences(reference_final["model"],actual_final["model"],"model")["close"]
    return {"steps":original_steps,"final_state":final,
            "fixed_eval_logits":differences(eval_reference,eval_actual,"fixed_eval_logits"),
            "original_input_loss_gradient_gate": len(reference)==len(actual) and all(
                r["input_exact"] and r["original_loss_gate"] and r["gradients"]["close"] for r in original_steps),
            "original_model_parameter_gate":model_close,
            "all_final_state_close":final["close"]}


def phase(ds,device,deterministic,directory,progress):
    directory.mkdir()
    torch.use_deterministic_algorithms(deterministic)
    _seed(7)
    model,optimizer = make_pair(device)
    sampler = BlockShuffleSampler(ds,7)
    generator = torch.Generator().manual_seed(1000010)
    _, checkpoint = epoch(ds,model,optimizer,sampler,generator,device,directory,"epoch1",progress)
    atomic_torch(directory/"epoch1_checkpoint.pt",checkpoint)
    # Copy the pair together: optimizer Parameter references must follow the
    # cloned model. This branch performs no state_dict load or serialization.
    live_model,live_optimizer = copy.deepcopy((model,optimizer))
    synchronize(device)
    model_ids = {id(p) for p in live_model.parameters()}
    assert {id(p) for group in live_optimizer.param_groups for p in group["params"]} == model_ids
    assert not model_ids & {id(p) for p in model.parameters()}
    clone_start = snapshot(live_model,live_optimizer,sampler,generator)
    clone_equality = differences(selected_state(checkpoint),selected_state(clone_start))
    atomic_json(directory/"clone_prebranch_equality.json",clone_equality)

    # Restore the captured RNG after copy/instrumentation, before original
    # continuation as well, so all three branches start from the same cursor.
    _restore_rng(checkpoint["rng"])
    reference,reference_final = epoch(ds,model,optimizer,sampler,generator,device,directory,"uninterrupted_epoch2",progress)
    eval_reference = torch.from_numpy(_collect(model,ds,CONFIG,device).copy())
    atomic_torch(directory/"uninterrupted_final.pt",{"state":reference_final,"eval_logits":eval_reference})

    branches = {}
    for kind in ("in_memory_replay","disk_resume_reused","disk_resume_fresh"):
        if kind == "in_memory_replay":
            target,target_optimizer = live_model,live_optimizer
            saved = checkpoint
        else:
            saved = torch.load(directory/"epoch1_checkpoint.pt",map_location="cpu",weights_only=False)
            # Job891833 reloaded into the already-used pair. Also compare a
            # fresh pair to reveal any optimizer-object state outside its saved
            # state_dict, instead of silently changing the failing fixture.
            target,target_optimizer = (model,optimizer) if kind == "disk_resume_reused" else make_pair(device)
            target.load_state_dict(saved["model"],strict=True)
            target_optimizer.load_state_dict(saved["optimizer"])
        next_sampler = BlockShuffleSampler(ds,7)
        next_sampler.load_state_dict(saved["sampler"])
        next_generator = torch.Generator().set_state(saved["loader_rng"])
        _restore_rng(saved["rng"])
        initial = snapshot(target,target_optimizer,next_sampler,next_generator)
        restoration = differences(selected_state(checkpoint),selected_state(initial))
        atomic_torch(directory/f"{kind}_before.pt",initial)
        atomic_json(directory/f"{kind}_before_comparison.json",restoration)
        actual,actual_final = epoch(ds,target,target_optimizer,next_sampler,next_generator,
                                   device,directory,kind,progress)
        eval_actual = torch.from_numpy(_collect(target,ds,CONFIG,device).copy())
        atomic_torch(directory/f"{kind}_final.pt",{"state":actual_final,"eval_logits":eval_actual})
        branches[kind] = {"immediate_state_restore":restoration,
                          **compare_runs(reference,actual,reference_final,actual_final,eval_reference,eval_actual)}
        atomic_json(directory/"comparisons.json",branches)
        progress({"event":"branch_compared","branch":kind,
                  "immediate_state_exact":restoration["exact"],
                  "original_model_parameter_gate":branches[kind]["original_model_parameter_gate"]})
    return {"device":str(device),"deterministic_algorithms":deterministic,
            "clone_prebranch_equality":clone_equality,"branches":branches,
            "all_original_gates_pass": all(b["original_input_loss_gradient_gate"] and
                 b["original_model_parameter_gate"] and b["all_final_state_close"] for b in branches.values())}


def main():
    require_slurm()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--max-seconds",type=int,default=600)
    args=parser.parse_args()
    if not 1 <= args.max_seconds <= 600: parser.error("Internal cap must be1..600 seconds")
    args.out_dir.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.cache/"manifest.json").read_text())
    if not manifest.get("complete") or not manifest.get("diagnostic_only") or manifest.get("records") != 36:
        raise RuntimeError("Requires the original complete36-record diagnostic cache")
    ds=CachedDataset(args.cache,"train","block11",diagnostic=True)
    started=time.monotonic()
    report={"technical_only":True,"scientific_fits_created":0,"test_evaluated":False,
            "job_id":os.environ["SLURM_JOB_ID"],"torch":torch.__version__,"gpu":torch.cuda.get_device_name(),
            "source_sha256":file_sha256(__file__),"implementation_sha256":implementation_identity(),
            "profile_workers_helpers_sha256":file_sha256(Path(__file__).with_name("profile_workers.py")),
            "protocol_sha256":digest(protocol()),"cache_manifest_sha256":file_sha256(args.cache/"manifest.json"),
            "original_atol":ATOL,"original_rtol":RTOL,"arm":"block11","seed":7,"workers":0,
            "cublas_workspace_config":os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "phases":{},"events":[],"diagnosis_complete":False,
            "limitations":["One36-record shard and one seed; diagnostic model states are never scientific checkpoints.",
                "A single same-checkpoint in-memory replay control does not estimate a universal CUDA noise bound.",
                "Instrumentation and CUBLAS workspace settings may affect execution; original891833 is preserved.",
                "No worker throughput or production admission is established by this diagnostic."]}
    def progress(event):
        event={**event,"elapsed_seconds":time.monotonic()-started}
        report["events"].append(event)
        atomic_json(args.out_dir/"summary.json",report)
        print(json.dumps(event),flush=True)
    def expired(signum,frame): raise TimeoutError("Resume localization internal time cap")
    signal.signal(signal.SIGALRM,expired); signal.alarm(args.max_seconds)
    try:
        for name,device,deterministic in (("cuda_original",torch.device("cuda"),False),
                                         ("cuda_deterministic",torch.device("cuda"),True)):
            try:
                report["phases"][name]=phase(ds,device,deterministic,args.out_dir/name,progress)
            except RuntimeError as error:
                report["phases"][name]={"execution_error":repr(error),"traceback":traceback.format_exc()}
                if name == "cuda_original": raise
            finally:
                gc.collect(); torch.cuda.empty_cache()
                progress({"event":"phase_finished","phase":name})
        if "execution_error" in report["phases"]["cuda_deterministic"]:
            # Diagnostic CPU reference only; never a production device switch.
            report["phases"]["cpu_deterministic"]=phase(ds,torch.device("cpu"),True,
                                                       args.out_dir/"cpu_deterministic",progress)
        report["diagnosis_complete"]=True
    except BaseException as error:
        report["execution_error"]=repr(error)
        report["traceback"]=traceback.format_exc()
        raise
    finally:
        signal.alarm(0)
        torch.use_deterministic_algorithms(False)
        report["elapsed_seconds"]=time.monotonic()-started
        report["ended_unix"]=time.time()
        atomic_json(args.out_dir/"summary.json",report)


if __name__ == "__main__": main()
