"""Headroom readout for the VLM POPE pilot (§1.6).

headroom := AUROC(best internal-signal detector) - AUROC(best output-derived score),
on a group-disjoint (by image) held-out split, balanced train, Polygraph protocol.

Output-derived (untrained, the denominator): msp-analog = 1 - p(chosen answer), and
-margin. Internal (trained): act_probe as a linear probe (logistic regression) and a small
MLP on the answer-position hidden state. A label-shuffle control must collapse to ~0.5
before any number is trusted. Reports y_err headline, the y_hall slice, and the
confident-error slice (p(chosen) >= 0.9) where outputs are blind by construction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from polygraph.training.baselines import _fit_tabular, _standardize
from .run import OUT


def load_shards(run_dir: Path = OUT) -> dict:
    shards = sorted(run_dir.glob("shard_*.npz"))
    assert shards, f"no shards in {run_dir}"
    parts = [np.load(s) for s in shards]
    keys = ("question_id", "image_id", "y_true", "said_yes", "p_yes", "p_no",
            "margin", "y_err", "y_hall")
    data = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    data["hidden"] = np.concatenate([p["hidden"] for p in parts]).astype(np.float32)
    return data


def group_split(image_id: np.ndarray, seed: int = 7, val_frac: float = 0.3):
    """Group-disjoint by image: every question about one image lands in one split."""
    rng = np.random.default_rng(seed)
    images = np.unique(image_id)
    rng.shuffle(images)
    cut = int(len(images) * (1 - val_frac))
    train_imgs = set(images[:cut].tolist())
    is_train = np.array([i in train_imgs for i in image_id])
    return is_train, ~is_train


def balance(mask: np.ndarray, y: np.ndarray, seed: int = 7) -> np.ndarray:
    """Subsample the majority class within `mask` to 50/50."""
    rng = np.random.default_rng(seed)
    idx = np.where(mask)[0]
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    k = min(len(pos), len(neg))
    keep = np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)])
    out = np.zeros(len(y), bool)
    out[keep] = True
    return out


def probe(hidden, y, tr, va, device, shuffle=False, kind="mlp", seed=7):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    y_tr = y[tr].copy()
    if shuffle:
        y_tr = np.random.default_rng(999).permutation(y_tr)
    if kind == "lr":
        scaler = StandardScaler().fit(hidden[tr])
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(scaler.transform(hidden[tr]), y_tr)
        return clf.predict_proba(scaler.transform(hidden[va]))[:, 1]
    x_tr, x_va = _standardize(hidden[tr], hidden[va])
    from torch import nn
    build = lambda: nn.Sequential(nn.Linear(hidden.shape[1], 128), nn.ReLU(),
                                  nn.Dropout(0.15), nn.Linear(128, 1))
    return _fit_tabular(build, x_tr, y_tr, x_va, y[va], x_va, seed, device)


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def main():
    device = torch.device("cpu")
    d = load_shards()
    n = len(d["y_err"])
    p_chosen = np.maximum(d["p_yes"], d["p_no"]) / (d["p_yes"] + d["p_no"] + 1e-9)
    out_msp = 1 - p_chosen
    out_margin = -d["margin"]

    is_tr, is_va = group_split(d["image_id"])
    tr = balance(is_tr, d["y_err"])
    va = is_va  # evaluate on the natural distribution

    lines = ["# VLM POPE pilot — initial headroom (LLaVA-1.5-7B, adversarial split)", "",
             f"n = {n} questions · ViT... err, LLaVA accuracy {1 - d['y_err'].mean():.3f} "
             f"· hallucination rate (said-yes-when-absent) {d['y_hall'].mean():.3f}", ""]
    lines += [f"Split: group-disjoint by image; train {int(tr.sum())} (balanced) / "
              f"test {int(va.sum())} (natural).", ""]

    # label-shuffle control
    ctrl = auroc(d["y_err"][va], probe(d["hidden"], d["y_err"], tr, va, device,
                                       shuffle=True, kind="lr"))
    lines += [f"**Label-shuffle control (must be ~0.5): {ctrl:.4f}**", ""]

    scores = {
        "out_msp": out_msp, "out_margin": out_margin,
        "probe_lr": None, "probe_mlp": None,
    }
    scores["probe_lr"] = probe(d["hidden"], d["y_err"], tr, va, device, kind="lr")
    scores["probe_mlp"] = probe(d["hidden"], d["y_err"], tr, va, device, kind="mlp")

    def table(mask, title):
        rows = [f"## {title} (n={int(mask.sum())}, errors={int(d['y_err'][mask & is_va].sum())})",
                "", "| detector | AUROC |", "|---|---:|"]
        m = mask & is_va
        vals = {}
        for name, s in scores.items():
            if name.startswith("out_"):
                vals[name] = auroc(d["y_err"][m], s[m])
            else:
                # probe scores are defined on the test rows (va); index within va
                sub = s[d_index_in_va(is_va, m)]
                vals[name] = auroc(d["y_err"][m], sub)
        for name, v in vals.items():
            rows.append(f"| {name} | {v:.4f} |")
        best_out = max(vals["out_msp"], vals["out_margin"])
        best_in = max(vals["probe_lr"], vals["probe_mlp"])
        rows += ["", f"**headroom (best internal − best output) = {best_in - best_out:+.4f}**", ""]
        return rows, best_in - best_out

    def d_index_in_va(is_va, m):
        # scores for probes are aligned to va-order; map absolute mask m (subset of va) into that order
        va_positions = np.where(is_va)[0]
        wanted = np.where(m)[0]
        return np.searchsorted(va_positions, wanted)

    all_rows, hr_all = table(np.ones(n, bool), "y_err — all test questions")
    lines += all_rows
    conf = p_chosen >= 0.9
    conf_rows, _ = table(conf, "y_err — confident answers (p(chosen) >= 0.9)")
    lines += conf_rows
    hall_rows, _ = table(d["y_true"] == 0, "y_hall slice — questions where the object is absent")
    lines += hall_rows

    verdict = ("GRADUATE (headroom >= 0.05)" if hr_all >= 0.05 else
               "STOP (headroom < 0.03)" if hr_all < 0.03 else "MARGINAL (0.03-0.05)")
    lines += [f"## Decision (§1.6): headroom {hr_all:+.4f} -> **{verdict}**", ""]

    report = Path("docs/results/pilot_vlm_pope.md")
    report.write_text("\n".join(lines) + "\n")
    np.savez("runs/vlm_pope_scores.npz", y_err=d["y_err"], y_hall=d["y_hall"],
             is_va=is_va, out_msp=out_msp, out_margin=out_margin,
             probe_lr=scores["probe_lr"], probe_mlp=scores["probe_mlp"])
    print("\n".join(lines), flush=True)
    print(f"WRITTEN: {report}", flush=True)


if __name__ == "__main__":
    main()
