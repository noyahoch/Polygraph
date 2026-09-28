"""Static packaging only: snapshot exact execution sources without running them."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    a = p.parse_args()
    repo, destination = a.repo.resolve(), a.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    files = list((repo / 'polygraph').rglob('*.py'))
    files += list((repo / 'pilots/canonical_logit_dynamics_20260924').rglob('*.py'))
    files += [repo / relative for relative in (
        'pilots/logit_dynamics_20260919/__init__.py',
        'pilots/logit_dynamics_20260919/features.py',
        'pilots/logit_dynamics_20260919/protocol.py',
        'pilots/logit_dynamics_20260919/test_science.py',
        'pilots/topology_20260910/__init__.py',
        'pilots/topology_20260910/protocol.py',
        'pilots/topology_20260910/slurm/imports.py',
        'docs/experiments/canonical_logit_dynamics_20260924/PROTOCOL.md',
        'docs/experiments/canonical_logit_dynamics_20260924/SCIENTIFIC_REVIEW.md')]
    manifest = {}
    for source in sorted(set(files)):
        if source.is_symlink() or not source.is_file():
            raise RuntimeError('Missing regular source file: ' + str(source))
        relative = source.relative_to(repo)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    value = dict(scope='canonical_logit_dynamics_20260924', files=manifest,
                 git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                 git_branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=repo, text=True).strip(),
                 source_authority='The complete copied bytes and SHA256 manifest, including uncommitted files; Git HEAD is provenance, not a substitute for the snapshot.')
    path = destination / 'source_manifest.json'
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(files=len(manifest), source_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          destination=str(destination))))


if __name__ == '__main__':
    main()
