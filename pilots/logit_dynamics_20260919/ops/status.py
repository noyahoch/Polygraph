"""Read-only status tied to recorded user, submission time, and unique job names."""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args()
records={f.stem:json.loads(f.read_text()) for f in (a.root/'ops/submissions').glob('*.json') if not f.name.endswith('.intent.json')}
result={'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'root':str(a.root),'stages':{},'scheduler_warning':None}
if records:
    ids=','.join(r['job_id'] for r in records.values());start=min(r['created_utc'][:10] for r in records.values())
    query=subprocess.run(['sacct','-X','-P','-n','-u','omrifahn','-S',start,'-j',ids,'--format=JobIDRaw,User,JobName%80,State,ElapsedRaw,Start,End,AllocTRES%150,ExitCode'],text=True,capture_output=True,timeout=30)
    scheduler={}
    if query.returncode:result['scheduler_warning']='sacct unavailable'
    else:
        for line in query.stdout.splitlines():
            fields=line.split('|')
            if len(fields)>=9:scheduler[fields[0]]={'user':fields[1],'name':fields[2],'state':fields[3],'elapsed_seconds':fields[4],'start':fields[5],'end':fields[6],'allocation':fields[7],'exit_code':fields[8]}
    for stage,r in records.items():
        live=scheduler.get(r['job_id'])
        if live and (live['user']!='omrifahn' or live['name']!=r['job_name']):live={'state':'IDENTITY_MISMATCH'}
        receipt=a.root/'ops/stage_receipts'/(stage+'_'+r['job_id']+'.json')
        local=json.loads(receipt.read_text()) if receipt.exists() else None
        result['stages'][stage]={'job_id':r['job_id'],'scheduler':live,'receipt_status':local.get('status') if local else None,'elapsed_seconds':local.get('elapsed_seconds') if local else None}
print(json.dumps(result,indent=2,sort_keys=True))
