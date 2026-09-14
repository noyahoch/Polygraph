"""Validation-only summary after all 14 fits, including fixed probability ensembles."""
import argparse
import json
from pathlib import Path
import numpy as np
from .protocol import MATRIX, atomic_json, require_slurm, file_sha256
from .train import _verify_complete, auroc

def main():
    require_slurm()
    p=argparse.ArgumentParser()
    p.add_argument("--run-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    rows, arrays={},{}
    for arm,seed in MATRIX:
        directory=args.run_root/arm/f"seed{seed}"
        complete=_verify_complete(directory)
        with np.load(directory/"validation.npz",allow_pickle=False) as z:
            data={k:z[k].copy() for k in z.files}
        key=f"{arm}/seed{seed}"
        arrays[key]=data
        rows[key]={"validation_auroc":auroc(data["y"],data["score"]),"best_epoch":complete["best_epoch"],
                   "completed_epochs":complete["completed_epochs"],"validation_sha256":file_sha256(directory/"validation.npz")}
    ensembles={}
    specifications={f"distinct_layers_seed{s}":[f"block{l}/seed{s}" for l in (2,5,8,11)] for s in (7,17)}
    specifications["same_layer_four_seeds_descriptive"]=[f"block11/seed{s}" for s in (1,7,17,27)]
    for name,keys in specifications.items():
        reference=arrays[keys[0]]
        for key in keys[1:]:
            assert all(np.array_equal(reference[k],arrays[key][k]) for k in ("record_id","image_id","source_id","severity","y","label","pred"))
        probabilities=np.mean([1/(1+np.exp(-np.clip(arrays[k]["score"].astype(np.float64),-700,700))) for k in keys],axis=0)
        ensembles[name]={"members":keys,"rule":"fixed arithmetic mean of sigmoid failure logits",
                         "validation_auroc":auroc(reference["y"],probabilities)}
        np.savez_compressed(args.run_root/f"{name}_validation.npz",record_id=reference["record_id"],
                            image_id=reference["image_id"],y=reference["y"],probability=probabilities)
    atomic_json(args.out,{"development_only":True,"test_evaluated":False,"fits":rows,"ensembles":ensembles,
                "caveat":"Validation was used for checkpoint selection. Two distinct-layer seed replicates; the one same-layer ensemble is descriptive, not independent replication."})

if __name__=="__main__": main()
