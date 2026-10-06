"""Read-only champion provenance and widening-design inspection."""
from pathlib import Path
import ast,datetime,json,sys
import psutil
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parents[2]
print('OBSERVED',datetime.datetime.now().astimezone().isoformat())
for rel in ('AGENTS.md','experiments/AGENTS.md','experiments/pure_dl/AGENTS.md','PROJECTS.md'):
 p=ROOT/rel;lines=p.read_text(encoding='utf-8-sig').splitlines();print('RULES',rel,len(lines))
 print('\n'.join(f'{i+1} | {lines[i]}' for i in range(max(0,len(lines)-12),len(lines))))
active=[]
for p in psutil.process_iter(['pid','name','cmdline']):
 try:
  d=p.info;argv=d.get('cmdline') or []
  if 'python' in (d.get('name') or '').lower() and any('pure_dl' in x and Path(x.replace('\\','/')).name.startswith(('train_','benchmark_')) for x in argv):active.append(d)
 except(psutil.NoSuchProcess,psutil.AccessDenied):pass
print('ACTIVE',active)
print('FREE_GIB',psutil.disk_usage(str(ROOT.anchor)).free/2**30)
print('RECENT',[p.name for p in sorted((ROOT/'outputs').iterdir(),key=lambda x:x.stat().st_mtime,reverse=True)[:5]])
def compact(x):
 if isinstance(x,dict):return {k:compact(v) for k,v in x.items() if k not in ('per_well','history','fit','validation','config','source_hashes','timing')}
 if isinstance(x,list):return [compact(v) for v in x]
 return x
for p in sorted((ROOT/'outputs').glob('pure_dl_dual8222*')):
 f=p/'fold_5_report.json'
 if f.exists():print('HEAD_RUN',p.name,json.dumps(compact(json.loads(f.read_text(encoding='utf-8-sig'))),ensure_ascii=False)[:4500])
p=ROOT/'outputs/pure_dl_paired_local_context_router80_sequence/fold_5_report.json'
print('SOURCE_BASE',json.dumps(compact(json.loads(p.read_text(encoding='utf-8-sig'))),ensure_ascii=False)[:4500])
for p in sorted((ROOT/'results/deep_learning').glob('dual8222*')): print('HEAD_STUDY',p.name)
for name in ('global_cosine_sequence.py','patch_hierarchy_sequence.py','multiscale_sequence.py','paired_calibrated_view_sequence.py','paired_local_context_router_sequence.py','dual8222_location_readout_v5.py'):
 p=ROOT/'experiments/pure_dl'/name
 print('MODEL_SOURCE',name)
 print(p.read_text(encoding='utf-8-sig'))
