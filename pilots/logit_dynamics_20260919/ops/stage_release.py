"""Upload and checksum-verify a frozen source archive and configuration."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--ssh-wrapper',type=Path);a=p.parse_args()
meta=json.loads(a.receipt.read_text());config=json.loads(a.config.read_text());root=config['root'];release=config['release']
if release!=root+'/releases/'+meta['release_id']:raise RuntimeError('Release and configuration disagree')
ssh=a.ssh_wrapper or Path(__file__).resolve().parents[2]/'slurm-ops/ssh-c003.sh'
def remote(command,stdin=None):
    return subprocess.run(['bash',str(ssh),'bash -c '+shlex.quote(command)],input=stdin,check=True,capture_output=True)
remote('umask 077; mkdir -p '+shlex.quote(root+'/ops')+' '+shlex.quote(root+'/logs')+' '+shlex.quote(root+'/releases')+'; test ! -e '+shlex.quote(release))
archive=Path(meta['archive_path']);content=archive.read_bytes()
if hashlib.sha256(content).hexdigest()!=meta['archive_sha256']:raise RuntimeError('Local source archive changed')
remote_archive=root+'/ops/'+archive.name
remote('umask 077; test ! -e '+shlex.quote(remote_archive)+' && cat > '+shlex.quote(remote_archive),content)
result=remote('sha256sum '+shlex.quote(remote_archive)).stdout.decode().split()[0]
if result!=meta['archive_sha256']:raise RuntimeError('Uploaded source archive checksum failed')
remote('tar -xzf '+shlex.quote(remote_archive)+' -C '+shlex.quote(root+'/releases')+' && chmod -R a-w '+shlex.quote(release))
remote_config=root+'/ops/'+a.config.name
config_bytes=a.config.read_bytes()
remote('umask 077; test ! -e '+shlex.quote(remote_config)+' && cat > '+shlex.quote(remote_config),config_bytes)
config_hash=hashlib.sha256(config_bytes).hexdigest()
if remote('sha256sum '+shlex.quote(remote_config)).stdout.decode().split()[0]!=config_hash:raise RuntimeError('Uploaded configuration checksum failed')
remote('chmod 400 '+shlex.quote(remote_config))
print(json.dumps({'status':'staged_not_submitted','root':root,'release':release,'archive_sha256':result,'configuration':remote_config,'configuration_sha256':config_hash},indent=2))
