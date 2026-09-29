"""Construct the fixed null and explicitly admit only the approved mixing shortfall."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--decision", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    manifest_path = a.cache / "rewire/manifest.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    reused = existing is not None and existing.get("construction_complete") is True
    code = None
    if not reused:
        result = subprocess.run([sys.executable, "-u", "-m", "pilots.topology_20260910.rewire",
                                 "--cache", str(a.cache), "--workers", str(a.workers)])
        code = result.returncode
        if code not in (0, 1):
            raise SystemExit(code)
    # Exit1 is expected only when the construction succeeded and its unchanged
    # registered mixing criterion failed. Admission recomputes structural evidence.
    manifest = existing if reused else json.loads(manifest_path.read_text())
    if code == 1 and not (
            manifest.get("construction_complete") is True
            and manifest.get("complete") is False
            and manifest.get("development_diagnostics", {}).get("passed") is False):
        raise RuntimeError("Rewiring failed for a reason other than approved insufficient mixing")
    from pilots.topology_20260910.rewiring_decision import admit
    receipt = admit(a.cache, a.decision)
    print(json.dumps({"construction_exit_code": code, "reused_immutable_construction": reused,
                      "mixing_quality_passed": receipt["mixing_quality_passed"],
                      "admission": receipt}, indent=2), flush=True)


if __name__ == "__main__":
    main()
