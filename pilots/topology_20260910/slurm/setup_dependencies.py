"""Create a private dependency overlay without modifying any prior environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--old-root", type=Path, required=True)
    a = p.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        raise SystemExit("Allocated CPU required")
    os.umask(0o077)
    started = time.time()
    out = a.root / "manifests" / "environment"
    out.mkdir(parents=True, exist_ok=True)
    env = a.root / "env"
    if env.exists():
        raise SystemExit("New overlay already exists; inspect before any retry")
    old_python = str(a.old_root / "env" / "bin" / "python")
    subprocess.run([old_python, "-m", "venv", "--without-pip", str(env)], check=True)
    site = env / "lib" / "python3.12" / "site-packages"
    parents = [a.old_root / "env-overnight-graph/lib/python3.12/site-packages",
               a.old_root / "env/lib/python3.12/site-packages"]
    (site / "frozen_parents.pth").write_text("\n".join(map(str, parents)) + "\n")
    python = str(env / "bin" / "python")
    protected = {"torch": "2.14.0+cu126", "torchvision": "0.29.0+cu126", "numpy": "2.5.2",
                 "transformers": "5.16.1", "torch-geometric": "2.6.1", "safetensors": "0.8.0"}
    snapshot = "import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in " + repr(list(protected)) + "}))"
    before = json.loads(subprocess.check_output([python, "-c", snapshot], text=True))
    if before != protected:
        raise RuntimeError("Working parent environment differs from verified pins")
    (out / "protected_before.json").write_text(json.dumps(before, indent=2) + "\n")
    constraints = out / "constraints.txt"
    constraints.write_text("\n".join(k + "==" + v for k, v in protected.items()) + "\n")
    report = out / "resolver.json"
    pip = [old_python, "-m", "pip", "--python", python, "install", "--no-input",
           "--disable-pip-version-check", "--only-binary=:all:"]
    subprocess.run(pip + ["--dry-run", "--report", str(report), "--constraint", str(constraints),
                         "numba==0.67.0", "llvmlite==0.49.0", "pytest==9.1.1"], check=True)
    resolved = json.loads(report.read_text())["install"]
    lines = []
    for item in resolved:
        name = item["metadata"]["name"].lower().replace("_", "-")
        if name in protected:
            raise RuntimeError(f"Resolver attempted to alter protected dependency {name}")
        sha = item["download_info"]["archive_info"]["hashes"]["sha256"]
        lines.append(f"{name}=={item['metadata']['version']} --hash=sha256:{sha}")
    locked = out / "resolved_requirements.txt"
    locked.write_text("\n".join(lines) + "\n")
    subprocess.run(pip + ["--no-deps", "--require-hashes", "-r", str(locked)], check=True)
    after = json.loads(subprocess.check_output([python, "-c", snapshot], text=True))
    if after != before:
        raise RuntimeError("Protected numerical environment changed")
    subprocess.run([old_python, "-m", "pip", "--python", python, "check"], check=True)
    subprocess.run([python, "-c", "import numba,llvmlite,pytest,numpy; print('Installed:',numba.__version__,llvmlite.__version__,pytest.__version__,numpy.__version__)"], check=True)
    result = {"status": "complete", "job_id": os.environ["SLURM_JOB_ID"], "python": python,
              "protected_before": before, "protected_after": after,
              "new_dependencies": lines, "elapsed_seconds": time.time() - started,
              "resolver_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
              "old_environments_modified": False}
    temp = out / "complete.tmp"
    temp.write_text(json.dumps(result, indent=2) + "\n")
    temp.replace(out / "complete.json")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
