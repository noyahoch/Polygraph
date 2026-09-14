"""Private, allowlisted Hub snapshots. Run on one Slurm CPU allocation only.

Run files are never modified. A durable staging directory and external receipts
make retries safe; only a checksummed immutable Hub revision counts as a backup.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import time

from .protocol import ARMS, SEEDS, SOURCE_COMMIT, atomic_json, digest, file_sha256, require_slurm
from .rewiring_decision import DECISION_PATH, LIMITATION, approved_decision, read_admission

DEFAULT_REPO = "omrifahn/polygraph-experiments"
PREFIX = "topology_20260910"
RUN_FILES = ("config.json", "history.json", "best.safetensors", "latest.pt",
             "validation.npz", "validation.json", "complete.json",
             "scaler.json", "scaler.safetensors", "threshold.json",
             "attempts.json", "resume_audit.json",
             "restore_audit.json", "restore_reference.npz", "final_metrics.json",
             "test_metrics.json", "test_predictions.npz")
BOUND_FILES = {"config_sha256": "config.json", "best_sha256": "best.safetensors",
               "validation_npz_sha256": "validation.npz", "validation_json_sha256": "validation.json",
               "complete_sha256": "complete.json"}
FINAL_FILES = ("summary.json", "results.csv", "bootstrap.json", "bootstrap_draws.npz",
               "report.he.md", "evaluation_state.json", "evaluation_complete.json")
SOURCE_PREFIXES = ("polygraph/", "pilots/topology_20260910/", "docs/experiments/september10/",
                   "docs/models/", "scripts/", "tests/")
SOURCE_SUFFIXES = {".py", ".md", ".toml", ".sh", ".sbatch"}
REVISION = re.compile(r"[0-9a-f]{40}\Z")


def safe_relative(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"Unsafe relative path: {value!r}")
    return path


def safe_file(root, relative):
    relative = safe_relative(relative)
    root = Path(root).resolve()
    path = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"Symlinks are not publishable: {relative}")
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise RuntimeError(f"Missing or unsafe file: {relative}")
    return path


def stamp(path):
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def copy_stable(source, destination):
    before = stamp(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with source.open("rb") as incoming, destination.open("wb") as outgoing:
        for chunk in iter(lambda: incoming.read(8 * 1024 * 1024), b""):
            outgoing.write(chunk)
            h.update(chunk)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if before != stamp(source):
        raise RuntimeError(f"Snapshot source changed during copy: {source.name}")
    return {"sha256": h.hexdigest(), "bytes": before[2]}


def write_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


@contextlib.contextmanager
def publisher_lock(receipts):
    receipts.mkdir(parents=True, exist_ok=True)
    with (receipts / "publisher.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another centralized publisher owns this receipt directory") from error
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def source_allowed(relative):
    path = safe_relative(relative)
    return (relative == "requirements.txt" or
            (relative.startswith(SOURCE_PREFIXES) and path.suffix in SOURCE_SUFFIXES
             and not any(part.startswith(".") or part.lower() in {"context", "secrets", "checkpoints"}
                         for part in path.parts)))


def git(code_root, *args):
    return subprocess.check_output(["git", "-C", str(code_root), *args])


def bundle_source(code_root, destination):
    """Complete allowlisted source overlay + base diff, without historical weights.

    This includes the imported polygraph package, so restoration does not require
    a checkout containing historical runs, or access to the original laptop.
    """
    code_root = Path(code_root)
    release_path = code_root / "source_manifest.json"
    release, tracked = None, []
    if release_path.exists():
        release = json.loads(safe_file(code_root, "source_manifest.json").read_text())
        if release.get("base_commit") != SOURCE_COMMIT or not isinstance(release.get("files"), dict):
            raise RuntimeError("Immutable source release must bind the registered base commit and every source file")
        for name, expected in release["files"].items():
            if not isinstance(expected, str) or file_sha256(safe_file(code_root, name)) != expected:
                raise RuntimeError(f"Immutable source release checksum mismatch: {name}")
        names = sorted(name for name in release["files"] if source_allowed(name))
        provenance = {"kind": "immutable_slurm_release", "base_commit": SOURCE_COMMIT,
                      "release_manifest_sha256": file_sha256(release_path),
                      "allowlisted_release_files": {name: release["files"][name] for name in names},
                      "audit_patch": "unavailable: this immutable source release contains no Git metadata"}
    else:
        tracked = git(code_root, "ls-files", "-z").decode().split("\0")
        untracked = git(code_root, "ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
        names = sorted(name for name in set(tracked + untracked) if name and source_allowed(name))
        provenance = {"kind": "git_working_tree", "base_commit": SOURCE_COMMIT,
                      "working_head": git(code_root, "rev-parse", "HEAD").decode().strip()}
    source_files = {}
    for name in names:
        if not (code_root / name).exists():
            continue  # Tracked deletions are also represented in working-tree.patch.
        source_files[name] = copy_stable(safe_file(code_root, name), destination / "source" / name)
        if release is not None and source_files[name]["sha256"] != release["files"][name]:
            raise RuntimeError(f"Immutable release changed while copying: {name}")
    if "pilots/topology_20260910/train.py" not in source_files:
        raise RuntimeError("Source bundle is not ready: train.py is required for restoration")
    patch_sha256 = None
    if release is None:
        patch = git(code_root, "diff", "--binary", SOURCE_COMMIT, "--",
                    *[name for name in tracked if name and source_allowed(name)])
        (destination / "working-tree.patch").write_bytes(patch)
        patch_sha256 = file_sha256(destination / "working-tree.patch")
    # A Slurm overlay can expose duplicate distributions from parent environments.
    # Record the effective import-metadata version, not the lexically last version.
    package_names = sorted({dist.metadata["Name"] for dist in importlib.metadata.distributions() if dist.metadata.get("Name")})
    packages = [(name, importlib.metadata.version(name)) for name in package_names]
    environment = {"python": sys.version.split()[0], "packages": dict(packages)}
    atomic_json(destination / "environment.json", environment)
    write_text(destination / "requirements.lock.txt", "".join(f"{name}=={version}\n" for name, version in packages))
    bundle = {"schema_version": 1, "base_commit": SOURCE_COMMIT,
              "upstream_base_url": f"https://github.com/noyahoch/Polygraph/tree/{SOURCE_COMMIT}",
              "provenance": provenance,
              "source_files": source_files, "source_sha256": digest(source_files),
              "patch_sha256": patch_sha256,
              "environment_sha256": file_sha256(destination / "environment.json"),
              "restore": "Use this exact source/ overlay as the Python project root. An audit patch is included only for Git-backed snapshots; never apply it to the finished overlay.",
              "scope": "Allowlisted Python source, protocol/docs and dependency versions; no historical weights, data, credentials or environment variables."}
    atomic_json(destination / "bundle.json", bundle)
    return bundle


def file_manifest(root):
    return {path.relative_to(root).as_posix(): {"sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in sorted(root.rglob("*")) if path.is_file() and not path.is_symlink()}


def run_bindings(directory):
    return {key: file_sha256(directory / name) for key, name in BOUND_FILES.items()}


def check_frozen_runs(freeze, runs):
    if freeze.get("frozen") is not True or not isinstance(freeze.get("runs"), dict):
        raise RuntimeError("Freeze document must be frozen and bind completed runs")
    for name, expected in freeze["runs"].items():
        safe_relative(name)
        actual = runs.get(name, {}).get("bindings")
        if not actual or any(actual.get(key) != expected.get(key) for key in BOUND_FILES):
            raise RuntimeError(f"Frozen run mismatch: {name}")


def check_final_freeze(freeze):
    """Partial training backups are allowed; a final freeze cannot omit fits."""
    if freeze.get("schema_version") != 1 or freeze.get("frozen") is not True:
        raise RuntimeError("A final publication requires the evaluator's schema-1 freeze")
    matrix = freeze.get("matrix")
    arms = tuple(ARMS) if matrix == "full" else ("full_graph", "full_rewired", "full_set", "full_endpoint", "logit") if matrix == "primary-only" else ()
    if not arms or set(freeze.get("runs", {})) != {f"{arm}/seed{seed}" for arm in arms for seed in SEEDS}:
        raise RuntimeError("Final freeze must bind the complete approved arm/seed matrix")
    if matrix == "primary-only":
        approval = freeze.get("reduction_approval", {}).get("document", {})
        if not (approval.get("approved") is True and approval.get("approved_by") == "root"
                and approval.get("test_results_visible") is False and approval.get("matrix") == matrix
                and approval.get("date") and approval.get("reason")):
            raise RuntimeError("Reduced final publication requires recorded pre-test approval")


def copy_final_results(args, root, freeze, freeze_hash, watched):
    if not args.results_root.is_dir():
        raise RuntimeError("Requested final results directory does not exist")
    completion_path = safe_file(args.results_root, "evaluation_complete.json")
    completion = json.loads(completion_path.read_text())
    if completion.get("status") != "complete" or completion.get("freeze_sha256") != freeze_hash:
        raise RuntimeError("Final evaluation is incomplete or belongs to another freeze")
    required = set(FINAL_FILES) - {"evaluation_complete.json"}
    for name in freeze["runs"]:
        stem = "predictions/" + name.replace("/", "__")
        required.update({stem + ".npz", stem + ".receipt.json"})
    required.update({"predictions/msp.npz", "predictions/entropy.npz"})
    artifacts = completion.get("artifacts", {})
    if set(artifacts) != required:
        raise RuntimeError("Final evaluation manifest must bind every expected result, model score, and receipt; unknown artifacts are excluded")
    for name in sorted(required):
        path = safe_file(args.results_root, name)
        if file_sha256(path) != artifacts[name]:
            raise RuntimeError(f"Final evaluation checksum mismatch: {name}")
        watched.append((path, stamp(path)))
        copy_stable(path, root / "results" / name)
    watched.append((completion_path, stamp(completion_path)))
    copy_stable(completion_path, root / "results/evaluation_complete.json")
    for name in ("summary.json", "evaluation_state.json"):
        if json.loads((root / "results" / name).read_text()).get("freeze_sha256") != freeze_hash:
            raise RuntimeError("Final result metadata does not match the published freeze")


def stage_snapshot(args, previous=None):
    previous = previous or {}
    stage_parent = args.receipt_dir / "staging"
    stage_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="building-", dir=stage_parent))
    root = temporary / PREFIX
    root.mkdir()
    watched = []
    runs = {}
    try:
        for arm in ARMS:
            for seed in SEEDS:
                name = f"{arm}/seed{seed}"
                source = args.run_root / name
                result = {"arm": arm, "seed": seed, "status": "not_started", "bindings": None}
                runs[name] = result
                if not (source / "config.json").exists():
                    continue
                result["status"] = "in_progress"
                complete_before = (source / "complete.json").exists()
                for filename in RUN_FILES:
                    if (source / filename).exists():
                        if filename in {"final_metrics.json", "test_metrics.json", "test_predictions.npz"} and not args.freeze:
                            raise RuntimeError("Final run artifact publication requires --freeze")
                        path = safe_file(args.run_root, f"{name}/{filename}")
                        watched.append((path, stamp(path)))
                        copy_stable(path, root / "runs" / name / filename)
                saved = root / "runs" / name
                config = json.loads((saved / "config.json").read_text())
                if config.get("arm") != arm or config.get("seed") != seed:
                    raise RuntimeError(f"Run/config identity mismatch: {name}")
                if complete_before != (source / "complete.json").exists():
                    raise RuntimeError(f"Run completed while being snapshotted: {name}")
                if (saved / "complete.json").exists():
                    result["bindings"] = run_bindings(saved)
                    completion = json.loads((saved / "complete.json").read_text())
                    if completion.get("complete") is not True or completion.get("selection_split") != "val":
                        raise RuntimeError(f"Invalid completion marker: {name}")
                    for key, filename in BOUND_FILES.items():
                        if key != "complete_sha256" and completion.get(key) != result["bindings"][key]:
                            raise RuntimeError(f"Completion hash mismatch: {name}/{filename}")
                    for key, filename in (("latest_sha256", "latest.pt"), ("history_sha256", "history.json")):
                        if completion.get(key) != file_sha256(saved / filename):
                            raise RuntimeError(f"Completion hash mismatch: {name}/{filename}")
                    validation = json.loads((saved / "validation.json").read_text())
                    if validation.get("preprocessing") != config.get("preprocessing"):
                        raise RuntimeError(f"Validation preprocessing mismatch: {name}")
                    if any(validation.get(key) != result["bindings"][key] for key in ("config_sha256", "best_sha256")):
                        raise RuntimeError(f"Validation/config/checkpoint mismatch: {name}")
                    result["status"] = "training_complete"
                    old = previous.get("completed_runs", {}).get(name)
                    if old and old != result["bindings"]:
                        raise RuntimeError(f"Previously completed run changed: {name}")
                result["final_metrics_available"] = any((saved / f).exists() for f in ("final_metrics.json", "test_metrics.json"))
        for name, bindings in previous.get("completed_runs", {}).items():
            if runs.get(name, {}).get("bindings") != bindings:
                raise RuntimeError(f"Previously completed run disappeared or changed: {name}")
        if previous.get("freeze_sha256") and not args.freeze:
            raise RuntimeError("A previously published freeze cannot be omitted on a later snapshot")
        rewiring_status = None
        if args.cache_metadata_root:
            for name in ("manifest.json", "cohort.json", "protocol.json"):
                path = safe_file(args.cache_metadata_root, name)
                watched.append((path, stamp(path)))
                copy_stable(path, root / "cache_metadata" / name)
            cache_manifest = json.loads((root / "cache_metadata/manifest.json").read_text())
            if cache_manifest.get("processor_sha256"):
                path = safe_file(args.cache_metadata_root, "processor.json")
                if file_sha256(path) != cache_manifest["processor_sha256"]:
                    raise RuntimeError("Pinned image-processor metadata changed")
                watched.append((path, stamp(path)))
                copy_stable(path, root / "cache_metadata/processor.json")
            if (args.cache_metadata_root / "rewire/admission.json").exists():
                rewiring_status = read_admission(args.cache_metadata_root, required=True)
                for name in ("rewire/manifest.json", "rewire/admission.json"):
                    path = safe_file(args.cache_metadata_root, name)
                    watched.append((path, stamp(path)))
                    copy_stable(path, root / "cache_metadata" / name)
        if any(run["arm"] == "full_rewired" and run["status"] != "not_started" for run in runs.values()):
            if rewiring_status is None:
                raise RuntimeError("Publishing a rewired fit requires its bound cache admission and decision")
            for name, run in runs.items():
                if run["arm"] == "full_rewired" and run["status"] != "not_started":
                    config = json.loads((root / "runs" / name / "config.json").read_text())
                    if (config.get("rewire_admission_sha256") != file_sha256(root / "cache_metadata/rewire/admission.json")
                            or config.get("rewire_decision_sha256") != approved_decision()["sha256"]):
                        raise RuntimeError("Rewired run lost its admission/decision provenance")
        freeze_hash, freeze = None, None
        if args.freeze:
            freeze_path = safe_file(args.freeze.parent, args.freeze.name)
            watched.append((freeze_path, stamp(freeze_path)))
            copy_stable(freeze_path, root / "freeze.json")
            freeze = json.loads((root / "freeze.json").read_text())
            check_final_freeze(freeze)
            check_frozen_runs(freeze, runs)
            if freeze.get("rewiring_status") != rewiring_status or rewiring_status is None:
                raise RuntimeError("Final publication lacks the frozen rewiring qualification")
            if freeze.get("rewire_decision_sha256") != approved_decision()["sha256"]:
                raise RuntimeError("Frozen rewiring decision mismatch")
            freeze_hash = file_sha256(root / "freeze.json")
            if previous.get("freeze_sha256") not in (None, freeze_hash):
                raise RuntimeError("A previously published freeze document changed")
            references = freeze.get("references", {})
            for extension in ("npz", "json"):
                name = references.get(extension + "_path", "")
                relative = safe_relative(name)
                if len(relative.parts) != 1 or not name.endswith(".validation_references." + extension):
                    raise RuntimeError("Freeze reference must be a named validation-references NPZ/JSON beside the freeze")
                path = safe_file(args.freeze.parent, name)
                if file_sha256(path) != references.get(extension + "_sha256"):
                    raise RuntimeError("Frozen analytic validation calibration artifact changed")
                watched.append((path, stamp(path)))
                copy_stable(path, root / name)
            if not args.cache_metadata_root:
                raise RuntimeError("Final freeze publication requires --cache-metadata-root for bound provenance")
            for name, field, canonical in (("protocol.json", "protocol_sha256", True),
                                           ("cohort.json", "cohort_sha256", True),
                                           ("manifest.json", "cache_manifest_sha256", False),
                                           ("validation.json", "cache_validation_sha256", False),
                                           ("rewire/manifest.json", "rewire_manifest_sha256", False),
                                           ("rewire/admission.json", "rewire_admission_sha256", False)):
                path = safe_file(args.cache_metadata_root, name)
                identity = digest(json.loads(path.read_text())) if canonical else file_sha256(path)
                if identity != freeze.get(field):
                    raise RuntimeError(f"Frozen cache provenance mismatch: {name}")
                watched.append((path, stamp(path)))
                copy_stable(path, root / "cache_metadata" / name)
        if args.results_root:
            if not args.freeze:
                raise RuntimeError("Final result publication requires --freeze")
            copy_final_results(args, root, freeze, freeze_hash, watched)
            for name in freeze["runs"]:
                runs[name]["final_metrics_available"] = True
        bundle = bundle_source(args.code_root, root / "code")
        if rewiring_status is not None and bundle["source_files"].get(DECISION_PATH, {}).get("sha256") != approved_decision()["sha256"]:
            raise RuntimeError("Exact approved rewiring decision is missing from the source bundle")
        if freeze is not None:
            implementation = freeze.get("implementation")
            if not isinstance(implementation, dict) or freeze.get("implementation_sha256") != digest(implementation):
                raise RuntimeError("Frozen evaluation lacks a valid implementation binding")
            for path, expected in implementation.items():
                if bundle["source_files"].get(path, {}).get("sha256") != expected:
                    raise RuntimeError(f"Bundled code does not match frozen evaluation implementation: {path}")
        for name, run in runs.items():
            if run["status"] == "not_started":
                continue
            config = json.loads((root / "runs" / name / "config.json").read_text())
            implementation = config.get("implementation")
            if not isinstance(implementation, dict) or config.get("implementation_sha256") != digest(implementation):
                raise RuntimeError(f"Run lacks a valid implementation binding: {name}")
            for path, expected in implementation.items():
                if bundle["source_files"].get(path, {}).get("sha256") != expected:
                    raise RuntimeError(f"Bundled code does not match trained implementation: {name}/{path}")
        if any(stamp(path) != original for path, original in watched):
            raise RuntimeError("An artifact changed while constructing the snapshot; retry next cycle")
        files = file_manifest(temporary)
        payload = {"schema_version": 1, "experiment": PREFIX, "repo_type": "model", "runs": runs,
                   "files": files, "source_sha256": bundle["source_sha256"], "freeze_sha256": freeze_hash,
                   "rewiring_status": rewiring_status, "rewiring_limitation": LIMITATION,
                   "total_bytes": sum(item["bytes"] for item in files.values()),
                   "test_results_status": "complete_evaluation_available" if (root / "results").exists()
                   else "partial_run_artifacts_available" if any(run.get("final_metrics_available") for run in runs.values())
                   else "not_published",
                   "scope": "Every planned arm and seed is listed; availability never selects a winning seed."}
        payload["snapshot_id"] = digest(payload)
        atomic_json(root / "manifest.json", payload)
        destination = stage_parent / payload["snapshot_id"]
        if destination.exists():
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, destination)
        return destination, payload
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def ensure_private(api, repo_id, revision=None):
    info = api.repo_info(repo_id=repo_id, repo_type="model", revision=revision)
    if getattr(info, "private", None) is not True:
        raise RuntimeError("Destination must already be a private model repository; visibility will not be changed")
    return info


def quota_preflight(api, repo_id, payload, available_bytes=None):
    report = {"estimated_snapshot_bytes": payload["total_bytes"], "remaining_quota_bytes": available_bytes,
              "remaining_quota_status": "operator_provided" if available_bytes is not None else "unknown",
              "repo_used_storage_bytes": None,
              "policy": "No purchases or automatic paid storage; backend quota failures leave backup incomplete."}
    try:
        info = api.model_info(repo_id=repo_id, expand=["usedStorage"])
        value = getattr(info, "usedStorage", getattr(info, "used_storage", None))
        if isinstance(value, int):
            report["repo_used_storage_bytes"] = value
    except Exception as error:
        report["storage_lookup_error_type"] = type(error).__name__
    if available_bytes is not None and payload["total_bytes"] > available_bytes:
        raise RuntimeError("Estimated upload exceeds the operator-provided available storage ceiling")
    return report


def model_cards(stage, payload, repo_id, artifact_revision, destination):
    base = f"https://huggingface.co/{repo_id}/tree/{artifact_revision}/{PREFIX}"
    status = payload.get("rewiring_status")
    rewire_note = ("Development mixing quality: **" + ("PASS" if status["mixing_quality_passed"] else "FAIL / non-diagnostic")
                   + "**. Mean changed-edge fraction: " + str(status["development_diagnostics"]["mean_changed_fraction"])
                   + "; registered requirement: 0.80. " if status else "Development mixing quality: pending. ")
    rewire_note += LIMITATION + f" [Pre-test decision]({base}/code/source/{DECISION_PATH})."
    table = ["| Arm / seed | State | Configuration |", "|---|---|---|"]
    for name, run in payload["runs"].items():
        config_link = "pending" if run["status"] == "not_started" else f"[config]({base}/runs/{name}/config.json)"
        table.append(f"| {name} | {run['status']} | {config_link} |")
        source = stage / PREFIX / "runs" / name
        if not source.exists():
            continue
        config = json.loads((source / "config.json").read_text())
        spec = ARMS[run["arm"]]
        feature_text = ("All 100 frozen classifier logits." if run["arm"] == "logit" else
                        "784 node / 36 edge features: attention, full hidden states and class-conditioned edge proxies."
                        if spec["full"] else "16 node / 12 edge features: attention and token/CLS metadata only.")
        card = f"""# {PREFIX}: {name}

Status: **{run['status']}**. Final metrics available: **{run['final_metrics_available']}**.
A training completion marker is not a scientific result or proof that restore checks passed.

This detector predicts whether the frozen CIFAR-100 classifier is wrong (error = 1).
Architecture: `{spec['architecture']}`; width `{spec['width']}`. {feature_text}
Rewired incidence: `{spec['rewired']}`. Full endpoint sets preserve endpoint relationships;
they remove explicit iterative message passing, not all relational information.

{rewire_note}

Data: 4,000 source photographs, source-disjoint 2,400/800/800 train/validation/test;
each contributes clean and severity 3/5 Gaussian noise, motion blur, fog and JPEG.
All arms train on the same nine-condition mixture. This does not test unseen corruption families.
AdamW, lr 0.002, weight decay 0.0001, dropout 0.15; at most 60 epochs,
validation-AUROC checkpoint selection, patience 8. Batch 24 (logit 256).
Training-error class weighting and any fitted preprocessing are saved with the run.
The validation-only threshold uses the 95th percentile of correct-example scores,
with strict `score > threshold`; it is not retuned on test conditions.

Exact saved [configuration]({base}/runs/{name}/config.json),
[history]({base}/runs/{name}/history.json), and
[validation artifacts]({base}/runs/{name}) are authoritative for this fit.
The [collection manifest]({base}/manifest.json) lists every planned arm and seed,
including missing fits; no best-seed selection is implied.

Immutable artifact revision: `{artifact_revision}`.
Source bundle identity: `{payload['source_sha256']}`.
Use the [exact bundled code]({base}/code/source) and
[dependency fingerprint]({base}/code/environment.json).
The [upstream base](https://github.com/noyahoch/Polygraph/tree/{SOURCE_COMMIT}) is
`{SOURCE_COMMIT}`; current additions are bundled as exact source. Bundle metadata
states whether a Git audit patch is available. No pushed Git branch is implied.
See the relative [restoration guide](../../../code/source/docs/models/README.md)
and [protocol](../../../code/source/docs/experiments/september10/PROTOCOL.md).

Limitations: a fixed established benchmark and selected corruption mixture;
finite five-seed uncertainty; parameter matching does not equate expressiveness;
rewiring changes destination/attribute coherence and paths. Corruption is not an error label.
No deployment guarantee, hallucination-detection claim, or graph superiority follows from availability.

Saved run configuration:

```json
{json.dumps(config, indent=2, sort_keys=True)}
```
"""
        write_text(destination / PREFIX / "runs" / name / "README.md", card)
    write_text(destination / "README.md", f"""---
library_name: pytorch
tags:
  - graph-neural-networks
  - error-detection
  - cifar100
---
# Polygraph experiment collection

Private research checkpoints for `{PREFIX}`. This repository records every fixed arm and
seed (1, 2, 7, 17, 27), not one selected best model. Availability is not a claim of success.
Snapshot `{payload['snapshot_id']}`; [immutable artifacts]({base});
artifact commit `{artifact_revision}`. Test results: **{payload['test_results_status']}**.

{chr(10).join(table)}

Read the [protocol]({base}/code/source/docs/experiments/september10/PROTOCOL.md),
[restoration guide]({base}/code/source/docs/models/README.md),
and [checksummed manifest]({base}/manifest.json).
Base code: [noyahoch/Polygraph at {SOURCE_COMMIT[:7]}](https://github.com/noyahoch/Polygraph/tree/{SOURCE_COMMIT}).
The exact source overlay and provenance are included; an audit diff is available only
for Git-backed snapshots. No unpublished GitHub branch link is asserted.

The primary comparison is full graph minus full set across all five paired seeds.
Endpoint, rewired, attention-only and output baselines provide prespecified context.
{rewire_note}
Missing or failed fits remain visible and limit interpretation. A null or inconclusive result is valid.
No original image archives, private conversations, tokens or historical checkpoint trees are included.
""")


def verify_local(directory):
    directory = Path(directory)
    manifest = json.loads(safe_file(directory, f"{PREFIX}/manifest.json").read_text())
    snapshot_id = manifest.pop("snapshot_id")
    if digest(manifest) != snapshot_id:
        raise RuntimeError("Manifest identity mismatch")
    manifest["snapshot_id"] = snapshot_id
    for name, expected in manifest["files"].items():
        path = safe_file(directory, name)
        if path.stat().st_size != expected["bytes"] or file_sha256(path) != expected["sha256"]:
            raise RuntimeError(f"Artifact checksum mismatch: {name}")
    if manifest.get("freeze_sha256"):
        freeze = safe_file(directory, f"{PREFIX}/freeze.json")
        if file_sha256(freeze) != manifest["freeze_sha256"]:
            raise RuntimeError("Freeze checksum mismatch")
        check_frozen_runs(json.loads(freeze.read_text()), manifest["runs"])
    for name, run in manifest["runs"].items():
        if run.get("bindings") and run_bindings(directory / PREFIX / "runs" / name) != run["bindings"]:
            raise RuntimeError(f"Run binding mismatch: {name}")
    publication_path = directory / "publication.json"
    if publication_path.exists():
        publication = json.loads(publication_path.read_text())
        if publication["snapshot_id"] != snapshot_id:
            raise RuntimeError("Publication belongs to another snapshot")
        for name, expected in publication["documentation_files"].items():
            if file_sha256(safe_file(directory, name)) != expected["sha256"]:
                raise RuntimeError(f"Documentation checksum mismatch: {name}")
    return manifest


def download_verified(repo_id, revision, destination, *, downloader=None, api=None):
    if not REVISION.fullmatch(revision or ""):
        raise ValueError("Download requires a full immutable 40-character Hub commit revision")
    if downloader is None or api is None:
        from huggingface_hub import HfApi, hf_hub_download
        downloader = downloader or hf_hub_download
        api = api or HfApi()
    info = ensure_private(api, repo_id, revision)
    if getattr(info, "sha", None) != revision:
        raise RuntimeError("Hub resolved a different revision")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    def fetch(name):
        safe_relative(name)
        return Path(downloader(repo_id=repo_id, repo_type="model", revision=revision,
                               filename=name, local_dir=str(destination)))
    publication = json.loads(fetch("publication.json").read_text())
    if not REVISION.fullmatch(publication.get("artifact_revision", "")):
        raise RuntimeError("Publication lacks an immutable artifact revision")
    manifest = json.loads(fetch(f"{PREFIX}/manifest.json").read_text())
    for name in sorted(set(manifest["files"]) | set(publication["documentation_files"])):
        fetch(name)
    verified = verify_local(destination)
    if verified["snapshot_id"] != publication["snapshot_id"]:
        raise RuntimeError("Downloaded snapshot mismatch")
    return verified


def upload_once(args, api=None, downloader=None):
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    ensure_private(api, args.repo_id)
    previous_path = args.receipt_dir / "latest.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    stage, payload = stage_snapshot(args, previous)
    receipt_path = args.receipt_dir / f"{payload['snapshot_id']}.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {
        "snapshot_id": payload["snapshot_id"], "repo_id": args.repo_id, "backup_complete": False,
        "freeze_sha256": payload["freeze_sha256"],
        "completed_runs": {name: run["bindings"] for name, run in payload["runs"].items() if run["bindings"]}}
    if receipt.get("backup_complete"):
        ensure_private(api, args.repo_id, receipt["revision"])
        atomic_json(previous_path, receipt)
        return receipt
    try:
        receipt["quota"] = quota_preflight(api, args.repo_id, payload, args.quota_available_bytes)
        atomic_json(receipt_path, receipt)
        if not receipt.get("artifact_revision"):
            ensure_private(api, args.repo_id)
            commit = api.upload_folder(repo_id=args.repo_id, repo_type="model", folder_path=str(stage),
                                       allow_patterns=list(payload["files"]) + [f"{PREFIX}/manifest.json"],
                                       commit_message=f"Snapshot {PREFIX} {payload['snapshot_id'][:12]}")
            receipt["artifact_revision"] = commit.oid
            if not REVISION.fullmatch(commit.oid):
                raise RuntimeError("Upload did not return an immutable commit")
            atomic_json(receipt_path, receipt)
        cards = args.receipt_dir / "cards" / payload["snapshot_id"]
        model_cards(stage, payload, args.repo_id, receipt["artifact_revision"], cards)
        documentation = {name: item for name, item in file_manifest(cards).items() if name != "publication.json"}
        publication = {"schema_version": 1, "snapshot_id": payload["snapshot_id"],
                       "artifact_revision": receipt["artifact_revision"], "documentation_files": documentation}
        atomic_json(cards / "publication.json", publication)
        if not receipt.get("revision"):
            ensure_private(api, args.repo_id)
            commit = api.upload_folder(repo_id=args.repo_id, repo_type="model", folder_path=str(cards),
                                       allow_patterns=list(documentation) + ["publication.json"],
                                       commit_message=f"Document immutable snapshot {payload['snapshot_id'][:12]}")
            receipt["revision"] = commit.oid
            atomic_json(receipt_path, receipt)
        verification = args.receipt_dir / "verified" / payload["snapshot_id"]
        download_verified(args.repo_id, receipt["revision"], verification, api=api, downloader=downloader)
        receipt.update(backup_complete=True, verified_sha256=payload["snapshot_id"])
        atomic_json(receipt_path, receipt)
        atomic_json(previous_path, receipt)
        return receipt
    except Exception as error:
        # Never serialize request objects, tokens, environment variables or raw HTTP bodies.
        receipt.update(backup_complete=False, error_type=type(error).__name__)
        atomic_json(receipt_path, receipt)
        raise


def smoke_restoration(directory, run_name, cache, device, output):
    """Verify one completed run on a bounded, validation-only batch in Slurm.

    Invoke from the downloaded code/source root in a fresh environment. The
    trainer itself rejects a different implementation, scaler, or cache identity.
    """
    require_slurm()
    import numpy as np
    import torch
    from . import train
    from .data import CachedDataset
    payload = verify_local(directory)
    if run_name not in payload["runs"] or payload["runs"][run_name]["status"] != "training_complete":
        raise RuntimeError("Smoke restoration requires a completed, manifest-bound arm/seed")
    run = Path(directory) / PREFIX / "runs" / run_name
    model, config = train.load_run(run, device)
    for key, value in train._cache_identity(cache, config["arm"]).items():
        if config[key] != value:
            raise RuntimeError(f"Restore cache identity mismatch: {key}")
    if config["device_type"] != torch.device(device).type:
        raise RuntimeError("Restore audit requires the recorded device type")
    dataset = CachedDataset(cache, "val", config["arm"])
    count = min(48, len(dataset))
    indices = list(range(min(len(dataset), max(count, config["batch_size"]))))
    restored = train._collect(model, dataset, config, torch.device(device), indices=indices)[:count]
    reference = np.load(run / "validation.npz", allow_pickle=False)
    metadata = dataset.metadata()
    if not np.array_equal(reference["record_id"][:count], metadata["record_id"][:count]):
        raise RuntimeError("Restore reference record identities do not match")
    threshold = json.loads((run / "validation.json").read_text())["threshold"]
    report = train.compare_scores(reference["score"][:count], restored, threshold)
    alternative = train._collect(model, dataset, config, torch.device(device), indices=indices,
                                  batch_size=max(1, config["batch_size"] // 2))[:count]
    report["alternative_partition"] = train.compare_scores(restored, alternative, threshold)
    report.update(status="passed", run=run_name, snapshot_id=payload["snapshot_id"],
                  device=str(device), split="val", implementation_sha256=config["implementation_sha256"])
    output = Path(output).resolve()
    if output.is_relative_to(Path(directory).resolve()) or output.is_relative_to(Path(cache).resolve()):
        raise RuntimeError("Restore audit receipt must be outside immutable download and cache")
    atomic_json(output, report)
    return report


def completed_run_markers(run_root):
    """Observe only the 35 registered atomic completion markers, without globbing."""
    return tuple(f"{arm}/seed{seed}" for arm in ARMS for seed in SEEDS
                 if (run_root / arm / f"seed{seed}" / "complete.json").is_file())


def wait_for_terminal_or_interval(interval, terminal_marker, run_root=None, completion_baseline=None):
    """Wake the central uploader on fit completion, termination, or the interval.

    The caller retains the publisher lock and performs one final upload before
    exiting. No extra uploader or parallel snapshot is started here.
    """
    deadline = time.monotonic() + interval
    while True:
        if run_root is not None and completed_run_markers(run_root) != completion_baseline:
            return "completion"
        if terminal_marker is not None and terminal_marker.exists():
            return "terminal"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return "interval"
        time.sleep(min(30.0, remaining))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--receipt-dir", type=Path)
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--cache-metadata-root", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=3600)
    parser.add_argument("--max-cycles", type=int, default=48)
    parser.add_argument("--terminal-marker", type=Path)
    parser.add_argument("--quota-available-bytes", type=int)
    parser.add_argument("--download", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--smoke-run", help="Completed ARM/seedN; requires --verify or --download")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-output", type=Path)
    args = parser.parse_args()
    require_slurm()
    if args.download or args.verify:
        payload = (download_verified(args.repo_id, args.revision, args.download) if args.download
                   else verify_local(args.verify))
        print(json.dumps({"status": "verified", "snapshot_id": payload["snapshot_id"]}))
        if args.smoke_run:
            if not args.cache or not args.smoke_output:
                parser.error("--smoke-run requires --cache and --smoke-output")
            report = smoke_restoration(args.download or args.verify, args.smoke_run, args.cache,
                                       args.device, args.smoke_output)
            print(json.dumps(report))
        return
    if args.smoke_run:
        parser.error("--smoke-run requires --verify or --download")
    for name in ("run_root", "receipt_dir", "code_root"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required for publication")
        setattr(args, name, getattr(args, name).resolve())
    if args.receipt_dir.is_relative_to(args.run_root) or args.run_root.is_relative_to(args.receipt_dir):
        parser.error("Receipt/staging directory and run root must be disjoint")
    if args.interval < 60 or args.max_cycles < 1:
        parser.error("Use interval >= 60 seconds and max-cycles >= 1")
    if args.quota_available_bytes is not None and args.quota_available_bytes < 0:
        parser.error("Storage ceiling must be nonnegative")
    with publisher_lock(args.receipt_dir):
        failures = 0
        # Completion events do not consume the configured hourly-progress budget.
        # At most 35 new fit completions and one terminal catch-up add cycles.
        completion_budget = len(ARMS) * len(SEEDS)
        cycle_limit = 1 if args.once else args.max_cycles + completion_budget + 1
        counts = {"progress": 0, "completion": 0, "terminal": 0}
        wake_reason = "initial"
        for cycle in range(cycle_limit):
            kind = "progress" if wake_reason in {"initial", "interval"} else wake_reason
            counts[kind] += 1
            if counts["completion"] > completion_budget or counts["terminal"] > 1:
                raise RuntimeError("Publisher completion/terminal cycle budget exceeded")
            # Capture before snapshot/upload. A completion during upload then
            # causes another cycle, even if the terminal marker arrives too.
            completion_baseline = completed_run_markers(args.run_root)
            try:
                receipt = upload_once(args)
                failures = 0
                after_upload = completed_run_markers(args.run_root)
                atomic_json(args.receipt_dir / "last_cycle.json", {
                    "cycle": cycle + 1, "wake_reason": wake_reason, "cycle_counts": dict(counts),
                    "completion_baseline": completion_baseline, "completion_after_upload": after_upload,
                    "revision": receipt["revision"], "snapshot_id": receipt["snapshot_id"],
                    "backup_complete": True})
                print(json.dumps({"status": "verified", "revision": receipt["revision"],
                                  "snapshot_id": receipt["snapshot_id"], "wake_reason": wake_reason,
                                  "cycle_counts": counts}), flush=True)
                if args.once:
                    return
                terminal_seen = args.terminal_marker and args.terminal_marker.exists()
                # Read after observing terminal: a final fit can finish between
                # the post-upload audit and that terminal-marker observation.
                pending_completion = completed_run_markers(args.run_root) != completion_baseline
                if terminal_seen and not pending_completion:
                    return
                if counts["progress"] >= args.max_cycles and not pending_completion:
                    return
            except Exception as error:
                failures += 1
                atomic_json(args.receipt_dir / "last_failure.json", {"backup_complete": False,
                            "error_type": type(error).__name__, "cycle": cycle + 1,
                            "consecutive_failures": failures})
                print(json.dumps({"status": "backup_incomplete", "error_type": type(error).__name__,
                                  "consecutive_failures": failures}), flush=True)
                if args.once or failures >= 2 or counts["progress"] >= args.max_cycles or cycle + 1 == cycle_limit:
                    # A terse error avoids leaking raw HTTP bodies into logs.
                    raise RuntimeError("Publisher stopped with incomplete backup; inspect separate receipts and rerun after repair") from None
            if cycle + 1 < cycle_limit:
                wake_reason = wait_for_terminal_or_interval(args.interval, args.terminal_marker,
                                                            args.run_root, completion_baseline)


if __name__ == "__main__":
    main()
