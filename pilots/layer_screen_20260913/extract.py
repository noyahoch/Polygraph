"""Slurm-only, restartable all-layer sparse attention extraction, development only."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time
import torch
from transformers import AutoImageProcessor, ViTForImageClassification
from pilots.topology_20260910.extract import OfficialImages
from pilots.topology_20260910.data import atomic_torch
from .protocol import (MODEL_ID, MODEL_REVISION, PROCESSOR_ID, PROCESSOR_REVISION,
                       atomic_json, cohort, digest, file_sha256, protocol, require_slurm, write_frozen)

class Capture:
    def __init__(self):
        if not torch.cuda.is_available(): raise RuntimeError("CUDA allocation required")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.processor = AutoImageProcessor.from_pretrained(PROCESSOR_ID, revision=PROCESSOR_REVISION, use_fast=True)
        self.model = ViTForImageClassification.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
            use_safetensors=True, attn_implementation="eager").eval().to("cuda")
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def batch(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to("cuda")
        result = self.model(**inputs, output_attentions=True, output_hidden_states=True)
        if len(result.attentions) != 12 or result.hidden_states[12].shape[1:] != (197,768):
            raise RuntimeError("Unexpected frozen ViT dimensions")
        logits = result.logits.float().cpu()
        if not torch.isfinite(logits).all(): raise FloatingPointError("Nonfinite logits")
        hidden = result.hidden_states[12].to("cpu", torch.float16)
        diagonals = torch.stack([a.diagonal(dim1=-2, dim2=-1).transpose(1,2) for a in result.attentions], 1).to("cpu", torch.float16)
        sparse = [[] for _ in images]
        sparse_values = [[] for _ in images]
        mask_diagonal = torch.eye(197, device="cuda", dtype=torch.bool)[None]
        for layer, attention in enumerate(result.attentions):
            for index in range(len(images)):
                a = attention[index].float()
                if not torch.isfinite(a).all(): raise FloatingPointError("Nonfinite attention")
                positions = ((a > 0.02) & ~mask_diagonal).reshape(-1).nonzero().flatten()
                sparse[index].append((positions + layer * 12 * 197 * 197).to("cpu", torch.int32))
                sparse_values[index].append(a.reshape(-1)[positions].to("cpu", torch.float16))
        return {"logits": logits, "hidden": hidden, "diagonals": diagonals}, [
            (torch.cat(pos), torch.cat(val)) for pos,val in zip(sparse,sparse_values)]

def capture(args):
    require_slurm()
    start = time.monotonic()
    cache = args.cache
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "shards").mkdir(exist_ok=True)
    frozen, group = protocol(), cohort()
    records = group["records"]
    diagnostic = args.max_records is not None
    if diagnostic:
        if not 24 <= args.max_records <= 288: raise ValueError("Preflight sample must be 24..288 records")
        records = records[:args.max_records]
    for name,value in (("protocol",frozen),("cohort",group)): write_frozen(cache / f"{name}.json", value)
    implementation = {name: file_sha256(Path(__file__).with_name(name)) for name in ("extract.py", "protocol.py")}
    identity = {"schema_version": 1, "protocol_sha256": digest(frozen), "cohort_sha256": digest(group),
                "record_ids_sha256": digest([r["record_id"] for r in records]), "records": len(records),
                "diagnostic_only": diagnostic, "shard_size": args.shard_size, "extraction_implementation": implementation}
    manifest_path = cache / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if any(previous.get(k) != v for k,v in identity.items()): raise RuntimeError("Resume extraction identity changed")
        if previous.get("complete"):
            for shard in previous["shards"]:
                if file_sha256(cache / shard["path"]) != shard["sha256"]: raise RuntimeError("Completed shard changed")
            print(json.dumps({"event":"already_complete", "cache":str(cache)}), flush=True)
            return
        if not args.resume: raise RuntimeError("Partial cache requires --resume")
    images = OfficialImages(args.data_root)
    # Hash once for provenance before any classifier outcome is inspected.
    provenance = images.provenance()
    write_frozen(cache / "data_provenance.json", provenance)
    model = Capture()
    write_frozen(cache / "processor.json", model.processor.to_dict())
    versions = {n: importlib.metadata.version(n) for n in ("torch", "transformers", "numpy", "torchvision", "Pillow")}
    atomic_json(cache / "attempt.json", {"started_unix":time.time(), "job_id":os.environ["SLURM_JOB_ID"],
                "versions":versions, "gpu":torch.cuda.get_device_name(), "cuda":torch.version.cuda,
                "batch_size":args.batch_size, "tf32":False})
    extraction_start = time.monotonic()
    shards, rows = [], []
    for shard_no, first in enumerate(range(0,len(records),args.shard_size)):
        subset = records[first:first+args.shard_size]
        name = f"shard_{shard_no:05d}.pt"
        path, sidecar = cache / "shards" / name, cache / "shards" / name.replace(".pt",".json")
        if path.exists() and sidecar.exists() and args.resume:
            done = json.loads(sidecar.read_text())
            if done["record_ids"] != [r["record_id"] for r in subset] or file_sha256(path) != done["sha256"]:
                raise RuntimeError("Resume shard hash or IDs changed")
            rows.extend(done["rows"])
            shards.append({"path":f"shards/{name}","sha256":done["sha256"],"records":len(subset)})
            continue
        buffers, positions, values, offsets, shard_rows = {}, [], [], [0], []
        for at in range(0,len(subset),args.batch_size):
            batch_records = subset[at:at+args.batch_size]
            payload, sparse = model.batch([images.image(r) for r in batch_records])
            for key, value in payload.items(): buffers.setdefault(key,[]).append(value)
            probabilities = payload["logits"].softmax(-1)
            top = probabilities.topk(2,dim=-1).values
            for i,(record,(pos,val)) in enumerate(zip(batch_records,sparse)):
                positions.append(pos); values.append(val); offsets.append(offsets[-1]+len(pos))
                label = int(images.labels[record["image_id"]])
                pred = int(payload["logits"][i].argmax())
                shard_rows.append({**record,"shard":name,"offset":len(shard_rows),"label":label,"pred":pred,
                    "y":int(pred!=label),"confidence":float(top[i,0]),"margin":float(top[i,0]-top[i,1]),
                    "sparse_values":len(pos)})
        output = {k:torch.cat(v) for k,v in buffers.items()}
        output.update(sparse_positions=torch.cat(positions),sparse_values=torch.cat(values),
                      sparse_offsets=torch.tensor(offsets,dtype=torch.int64),
                      record_id=torch.tensor([r["record_id"] for r in subset],dtype=torch.int64))
        if not torch.isfinite(output["hidden"]).all() or not torch.isfinite(output["diagonals"]).all():
            raise FloatingPointError("Nonfinite stored node features")
        atomic_torch(path,output)
        sha = file_sha256(path)
        atomic_json(sidecar,{"record_ids":[r["record_id"] for r in subset],"sha256":sha,"rows":shard_rows})
        rows.extend(shard_rows)
        shards.append({"path":f"shards/{name}","sha256":sha,"records":len(subset)})
        atomic_json(manifest_path,{**identity,"complete":False,"shards":shards,"completed_records":len(rows)})
        print(json.dumps({"event":"capture_shard","records":len(rows),"total":len(records),
                          "elapsed_seconds":time.monotonic()-start,"bytes":path.stat().st_size}),flush=True)
    atomic_json(cache / "index.json",rows)
    total_bytes = sum((cache / s["path"]).stat().st_size for s in shards)
    elapsed = time.monotonic()-extraction_start
    summary = {"records":len(rows),"total_bytes":total_bytes,"extraction_seconds":elapsed,
               "startup_seconds":extraction_start-start,"projected_28800_bytes":total_bytes*28800/len(rows),
               "projected_28800_seconds":elapsed*28800/len(rows), "free_bytes":shutil.disk_usage(cache).free,
               "projection_caveat":"Small deterministic development sample; excludes unknown contention and full-data graph variation"}
    atomic_json(cache / "capture_summary.json",summary)
    atomic_json(manifest_path,{**identity,"complete":True,"shards":shards,"completed_records":len(rows),
        "index_sha256":file_sha256(cache / "index.json"),"data_provenance_sha256":file_sha256(cache / "data_provenance.json"),
        "processor_sha256":file_sha256(cache / "processor.json")})
    print(json.dumps({"event":"capture_complete",**summary}),flush=True)

def main():
    require_slurm()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--shard-size",type=int,default=64)
    p.add_argument("--max-records",type=int)
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    if args.batch_size<1 or args.shard_size<1: p.error("Positive batch and shard sizes required")
    capture(args)

if __name__=="__main__": main()
