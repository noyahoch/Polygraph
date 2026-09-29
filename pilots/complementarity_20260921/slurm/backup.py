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
    if config_sha != os.environ.get('COMPLEMENTARITY_CONFIG_SHA256'):
        raise RuntimeError('Backup configuration changed')
    config = json.loads(args.config.read_text())
    root = args.root.resolve()
    if str(root) != config['root'] or not root.name.startswith('complementarity_20260921_'):
        raise RuntimeError('Backup scope mismatch')
    if not Path(config['release']).resolve().is_relative_to(root):
        raise RuntimeError('Executable release must be inside the archived experiment root')
    # Scientific completion is distinct from upload completion. A failed upload never reruns science.
    required = ['evaluation_gate.json', 'cache/complete.json', 'ablation/freeze.json', 'fusion/freeze.json',
                'predictions/fusion_complete.json', 'predictions/ablation_complete.json',
                'statistics/draw_manifest.json', 'statistics/complete.json', 'report/REPORT.md', 'report/complete.json']
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
        if relative.parts[0] == 'publication' or path.is_dir() or path.name.endswith('.lock'):
            continue
        if path.is_symlink():
            raise RuntimeError('Unexpected symlink in experiment snapshot')
        source_files['experiment/' + str(relative)] = path
    # Preserve the bound small parent inputs/models; heavy CLS/image caches remain server-side.
    campaign = json.loads((root / 'campaign.json').read_text())
    parent = Path(config['parent_root'])
    for raw_path, expected in campaign['parent_hashes'].items():
        path = Path(raw_path)
        if sha(path) != expected:
            raise RuntimeError('A bound parent input changed')
        if path.is_relative_to(parent):
            relative = 'parent_inputs/logit_dynamics/' + str(path.relative_to(parent))
        elif path.is_relative_to(Path(campaign['parent_baseline_root'])):
            relative = 'parent_inputs/baseline/' + str(path.relative_to(Path(campaign['parent_baseline_root'])))
        elif path.is_relative_to(Path(config['parent_release'])):
            relative = 'parent_inputs/source/' + str(path.relative_to(Path(config['parent_release'])))
        else:
            raise RuntimeError('Unrecognized parent input location')
        source_files[relative] = path
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
    for relative in ['report/REPORT.md', 'report/REPORT_HE.md', 'campaign.json']:
        shutil.copyfile(root / relative, bundle / Path(relative).name)
    protocol = Path(config['release']) / 'docs/experiments/complementarity_20260921/PROTOCOL.md'
    shutil.copyfile(protocol, bundle / 'PROTOCOL.md')
    readme = '''# Complementarity and LogitDynamics readout decomposition

Private preservation of the fixed September21 follow-up. Read REPORT.md and
PROTOCOL.md for results, cohort reuse, uncertainty and explicit limitations.
The scientific_state.tar.gz archive contains the exact executable source,
environment receipts, frozen feature cache, selected and resumable model states,
training histories, predictions, all fixed bootstrap draw states, gates, failures
and operational receipts. FILE_MANIFEST.json binds every uncompressed file.
The original CLS/image caches are not duplicated; parent input/model identities
and small bound parent artifacts are included. This snapshot is not a claim of
a fresh raw-image or fresh-install end-to-end reproduction. Paths in frozen
manifests refer to the recorded server layout. Restore that layout or explicitly
record a path remapping before reusing executable commands. The running backup
job's own receipt/log can only be captured in its in-progress state; the separate
publication receipt supplies its completed immutable Hub revision.
'''
    (bundle / 'README.md').write_text(readme)
    manifest = {str(path.relative_to(bundle)): dict(sha256=sha(path), bytes=path.stat().st_size)
                for path in sorted(bundle.iterdir()) if path.is_file()}
    prefix = root.name + '/snapshot_v1'
    save(bundle / 'BACKUP_MANIFEST.json', dict(files=manifest, repo_id=REPO, prefix=prefix,
                                            configuration_sha256=config_sha, parent_snapshot_revision='86605781c0528e286483e305072f7787393da860'))
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
                                   commit_message='Preserve fixed complementarity and LD readout-decomposition follow-up', parent_commit=before.sha)
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
    print(json.dumps(result, indent=2), flush=True)
    if result['status'] != 'verified':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
