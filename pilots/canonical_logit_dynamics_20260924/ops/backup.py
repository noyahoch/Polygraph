"""Private immutable snapshot; archive all scientific state and verify remote bytes."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import urllib.parse
import urllib.request

REPO = 'omrifahn/polygraph-experiments'
TOKEN_PATH = '/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/topology_20260910/private/huggingface/token'
MAX_BYTES = 512 * 1024 * 1024


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


class ScopedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected and urllib.parse.urlparse(newurl).hostname != 'huggingface.co':
            redirected.remove_header('Authorization')
        return redirected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Backup runs inside its allocated CPU job')
    config_sha = sha(args.config)
    if config_sha != os.environ.get('CANONICAL_CONFIG_SHA256'):
        raise RuntimeError('Backup configuration changed')
    config = json.loads(args.config.read_text())
    root = args.root.resolve()
    if str(root) != config['root'] or not root.name.startswith('canonical_ld_20260924_'):
        raise RuntimeError('Backup scope mismatch')
    if not Path(config['release']).resolve().is_relative_to(root):
        raise RuntimeError('Executable release must be inside the archived experiment root')
    # Scientific completion is distinct from upload completion. A failed upload never reruns science.
    required = ['campaign.json', 'role_map.json', 'evaluation_gate.json', 'cls/manifest.json',
                'preflight/complete.json', 'evaluation/complete.json', 'data/dataset_manifest.json']
    for relative in required:
        if not (root / relative).is_file():
            raise RuntimeError('Incomplete scientific artifact: ' + relative)
    publication = root / 'publication'
    publication.mkdir(exist_ok=True)
    if (publication / 'upload_intent.json').exists() or (publication / 'receipt.json').exists():
        raise RuntimeError('Existing upload attempt requires reconciliation; no blind retry')
    bundle = publication / 'bundle'
    bundle.mkdir(exist_ok=False)
    source_files = {}
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if path.is_dir() or path.name.endswith('.lock') or '__pycache__' in relative.parts:
            continue
        if relative.parts[0] == 'publication':
            continue
        if relative.parts[0] == 'data' and str(relative) not in ('data/dataset_manifest.json', 'data/dataset_revisions.json'):
            continue
        if relative.parts[0] == 'cls' and str(relative) not in ('cls/manifest.json', 'cls/index.json'):
            continue
        if relative.parts[0] == 'dependencies':
            keep_manifest = str(relative) in ('dependencies/parquet_overlay_v1/complete.json',
                                             'dependencies/parquet_overlay_v1/runtime_verification.json')
            keep_wheel = len(relative.parts) == 4 and relative.parts[2] == 'wheels' and path.suffix == '.whl'
            if not (keep_manifest or keep_wheel):
                continue
        if path.is_symlink():
            raise RuntimeError('Unexpected symlink in experiment snapshot')
        source_files['experiment/' + str(relative)] = path
    # Original submitted plans/scan/scores are inside the root and archived; raw parquet
    # and full CLS shard caches stay on Slurm with checksummed reconstruction recipes.
    total_bytes = sum(path.stat().st_size for path in source_files.values())
    if total_bytes > MAX_BYTES:
        raise RuntimeError('Snapshot exceeds its reviewed 512MiB scope; request storage review')
    inventory = {name: dict(sha256=sha(path), bytes=path.stat().st_size) for name, path in source_files.items()}
    archive = bundle / 'scientific_state.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name, path in source_files.items():
            tar.add(path, arcname=name, recursive=False)
    # Detect mutations during packaging. The currently running backup receipt/log may change
    # only after this command exits; its in-progress form is labelled in the snapshot README.
    for name, path in source_files.items():
        if sha(path) != inventory[name]['sha256']:
            raise RuntimeError('Snapshot input changed during packaging: ' + name)
    save(bundle / 'FILE_MANIFEST.json', dict(created_utc=now(), files=inventory,
                                           uncompressed_bytes=total_bytes, configuration_sha256=config_sha))
    for relative in ['campaign.json', 'PROTOCOL.md']:
        shutil.copyfile(root / relative, bundle / Path(relative).name)
    (bundle / 'README.md').write_text("""# Canonical Polygraph versus LogitDynamics

Private preservation of the fixed September24 canonical-benchmark comparison.
Read PROTOCOL.md and the archived evaluation results for cohort allocation,
conditional uncertainty and exploratory-development-data limitations.

scientific_state.tar.gz preserves exact executable releases, environment and
operation receipts, original colleague plans/scan/reference scores, derived
splits, model states (selected and resumable), histories, predictions and fixed
bootstrap results. FILE_MANIFEST.json binds every archived file. Public raw
parquet files and the full CLS shards remain on Slurm; their manifests, source
revisions and reproduction code are included. No claim of fresh-install or
raw-image end-to-end replay follows merely from preservation. Recorded absolute
paths require restoration or an explicitly documented remapping. The backup
job's own log/receipt is archived in-progress; the separate publication receipt
provides the final immutable Hub revision and remote byte verification.
""")
    manifest = {str(path.relative_to(bundle)): dict(sha256=sha(path), bytes=path.stat().st_size)
                for path in sorted(bundle.iterdir()) if path.is_file()}
    prefix = root.name + '/snapshot_v1'
    save(bundle / 'BACKUP_MANIFEST.json', dict(files=manifest, repo_id=REPO, prefix=prefix,
                                            configuration_sha256=config_sha))
    manifest['BACKUP_MANIFEST.json'] = dict(sha256=sha(bundle / 'BACKUP_MANIFEST.json'), bytes=(bundle / 'BACKUP_MANIFEST.json').stat().st_size)
    os.environ['HF_TOKEN_PATH'] = TOKEN_PATH
    os.environ['HF_HOME'] = str(publication / 'hf_cache')
    os.environ['HF_HUB_CACHE'] = str(publication / 'hf_cache/hub')
    for key in ['HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE']:
        os.environ.pop(key, None)
    from huggingface_hub import HfApi
    api = HfApi()
    before = api.repo_info(REPO, repo_type='model', files_metadata=True)
    if before.private is not True:
        raise RuntimeError('Hub destination is not private')
    old_blobs = {item.rfilename: item.blob_id for item in before.siblings}
    if any(name.startswith(prefix + '/') for name in old_blobs):
        raise RuntimeError('Snapshot prefix already exists; reconcile instead of overwriting')
    save(publication / 'upload_intent.json', dict(created_utc=now(), repo_id=REPO, prefix=prefix,
                                               parent_revision=before.sha, configuration_sha256=config_sha,
                                               files=manifest, scope='New private snapshot only; preserve all existing files'))
    result = dict(repo_id=REPO, prefix=prefix, configuration_sha256=config_sha, private=True)
    try:
        commit = api.upload_folder(repo_id=REPO, repo_type='model', folder_path=str(bundle), path_in_repo=prefix,
                                   commit_message='Preserve canonical Polygraph versus LogitDynamics comparison', parent_commit=before.sha)
        result.update(status='uploaded_verification_pending', revision=commit.oid)
        info = api.repo_info(REPO, repo_type='model', revision=commit.oid, files_metadata=True)
        if info.private is not True or info.sha != commit.oid:
            raise RuntimeError('Immutable revision/private verification failed')
        after_blobs = {item.rfilename: item.blob_id for item in info.siblings}
        if any(after_blobs.get(name) != blob for name, blob in old_blobs.items()):
            raise RuntimeError('Existing Hub file changed')
        if set(after_blobs) - set(old_blobs) != {prefix + '/' + name for name in manifest}:
            raise RuntimeError('Unexpected remote snapshot inventory')
        credential = Path(TOKEN_PATH).read_text().strip()
        opener = urllib.request.build_opener(ScopedRedirect)
        checked = {}
        for name, spec in manifest.items():
            url = 'https://huggingface.co/' + REPO + '/resolve/' + commit.oid + '/' + urllib.parse.quote(prefix + '/' + name, safe='/')
            request = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + credential})
            digest, size = hashlib.sha256(), 0
            downloaded = publication / ('verified_remote_' + name)
            with opener.open(request, timeout=45) as response, downloaded.open('xb') as output:
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    size += len(block)
                    if size > MAX_BYTES:
                        raise RuntimeError('Remote verification response exceeds bound')
                    digest.update(block)
                    output.write(block)
            if digest.hexdigest() != spec['sha256'] or size != spec['bytes']:
                raise RuntimeError('Remote bytes differ: ' + name)
            checked[name] = spec
        downloaded_inventory = json.loads((publication / 'verified_remote_FILE_MANIFEST.json').read_text())['files']
        if downloaded_inventory != inventory:
            raise RuntimeError('Downloaded member manifest differs from the reviewed snapshot')
        with tarfile.open(publication / 'verified_remote_scientific_state.tar.gz', 'r:gz') as tar:
            members = tar.getmembers()
            if len(members) != len(inventory) or {member.name for member in members} != set(inventory):
                raise RuntimeError('Downloaded archive member inventory differs')
            for member in members:
                if not member.isfile() or member.size != inventory[member.name]['bytes']:
                    raise RuntimeError('Downloaded archive member metadata differs')
                digest = hashlib.sha256()
                with tar.extractfile(member) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(block)
                if digest.hexdigest() != inventory[member.name]['sha256']:
                    raise RuntimeError('Downloaded archive member checksum differs')
        result.update(status='verified', finished_utc=now(), files=checked,
                      all_existing_files_unchanged=True, previous_file_count=len(old_blobs),
                      archived_file_count=len(inventory), archive_members_verified=True,
                      uncompressed_bytes=total_bytes)
    except Exception as error:
        result.update(status='upload_or_verification_failed', error_type=type(error).__name__, finished_utc=now())
    save(publication / 'receipt.json', result)
    print(json.dumps({k: result.get(k) for k in ('status', 'repo_id', 'prefix', 'revision', 'archive_members_verified')}), flush=True)
    if result['status'] != 'verified':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
