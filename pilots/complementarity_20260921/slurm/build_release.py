"""Package reviewed new files beside byte-identical parent source; no tests run."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--parent-release', type=Path, required=True)
    parser.add_argument('--ops', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    parent_manifest = args.parent_release / 'source_manifest.json'
    if sha(parent_manifest) != 'd78438ba6a0c512ab66e5b59a16a772b0c4d972e39e9bf57ceb043b8936144b3':
        raise RuntimeError('Local parent snapshot is not the pinned immutable release')
    files = json.loads(parent_manifest.read_text())['files']
    args.out.mkdir(parents=True, exist_ok=False)
    for name, expected in files.items():
        source = args.parent_release / name
        if sha(source) != expected:
            raise RuntimeError('Pinned parent source mismatch: ' + name)
        target = args.out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    added = {}
    for subtree in ['pilots/complementarity_20260921', 'docs/experiments/complementarity_20260921']:
        for source in sorted((args.repo / subtree).rglob('*')):
            if not source.is_file() or '__pycache__' in source.parts or source.suffix == '.pyc':
                continue
            name = str(source.relative_to(args.repo))
            if name in files:
                raise RuntimeError('New source collides with immutable parent: ' + name)
            target = args.out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            added[name] = sha(target)
    mappings = {'stage_runner.py': 'complementarity_stage_runner.py',
                'submit.py': 'complementarity_submit.py',
                'advance_after_timing.py': 'complementarity_advance_after_timing.py',
                'environment_inventory.py': 'complementarity_environment_inventory.py',
                'backup.py': 'complementarity_backup.py'}
    for source_name, target_name in mappings.items():
        target = args.out / 'ops' / target_name
        if target.exists():
            raise RuntimeError('New operational source collides with parent')
        shutil.copyfile(args.ops / source_name, target)
        added['ops/' + target_name] = sha(target)
    files = {**files, **added}
    identity = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    manifest = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), files=files,
                    source_identity_sha256=identity, parent_source_manifest_sha256=sha(parent_manifest),
                    scope='Exact parent executable snapshot plus isolated September21 follow-up; no data or secrets')
    (args.out / 'source_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    archive = args.out.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(args.out, arcname=args.out.name)
    print(json.dumps(dict(release=str(args.out), archive=str(archive), source_identity_sha256=identity,
                          source_manifest_sha256=sha(args.out / 'source_manifest.json'),
                          archive_sha256=sha(archive), files=len(files), added_files=len(added)), indent=2))


if __name__ == '__main__':
    main()
