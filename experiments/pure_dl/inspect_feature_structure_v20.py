"""Read-only scoped evidence for feature/backbone planning; no training."""
from pathlib import Path
import json, datetime, ast, sys
import psutil
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parents[2]
print('OBSERVED',datetime.datetime.now().astimezone().isoformat())
for rel in ('AGENTS.md','experiments/AGENTS.md','experiments/pure_dl/AGENTS.md','PROJECTS.md'):
    p=ROOT/rel; lines=p.read_text(encoding='utf-8-sig').splitlines()
    print('RULE_OR_STATUS',rel,'lines',len(lines))
    for i in range(max(0,len(lines)-16),len(lines)): print(i+1,lines[i])
print('LATEST_OUTPUTS',[p.name for p in sorted((ROOT/'outputs').iterdir(),key=lambda p:p.stat().st_mtime,reverse=True)[:8]])
p=ROOT/'results/deep_learning/por_pyramid_v19_20260928/assembled_report.json'
if p.exists():
    d=json.loads(p.read_text(encoding='utf-8'));print('LAST_SCORES',d['selected_scores'],[(r['mode'],r['scores']) for r in d['candidates']])
for p in psutil.process_iter(['pid','name','cmdline']):
    try:
        r=p.info; args=r.get('cmdline') or []
        if 'python' in (r.get('name') or '').lower() and any('pure_dl' in x and ('train_' in x or 'benchmark_' in x) for x in args):print('ACTIVE',r)
    except(psutil.NoSuchProcess,psutil.AccessDenied):pass
print('FREE_GIB',psutil.disk_usage(str(ROOT.anchor)).free/2**30)
manifest=json.loads((ROOT/'results/deep_learning/latent_fold5_v16_20260928/data_outer_manifest.json').read_text(encoding='utf-8'))
print('FEATURE_NAMES',list(enumerate(manifest['feature_names'])))
for rel in ('src/feature.py','experiments/pure_dl/paired_backbone_sequence.py','experiments/pure_dl/paired_well_view_sequence.py','experiments/pure_dl/independent_channels_v8.py'):
    p=ROOT/rel
    if not p.exists():continue
    s=p.read_text(encoding='utf-8-sig');print('SOURCE',rel,'lines',len(s.splitlines()))
    if len(s)<13000:print(s)
    else:
        for n in ast.walk(ast.parse(s)):
            if isinstance(n,(ast.FunctionDef,ast.ClassDef)): print('SYMBOL',n.name,n.lineno,n.end_lineno)
        for i,line in enumerate(s.splitlines()):
            if any(x in line for x in ('rolling','rank','quantile','log1p','interaction','gradient','well_norm','median')):print(i+1,line)
print('RELATED_PAST')
atlas=json.loads((ROOT/'docs/deep_learning/atlas/atlas-data.json').read_text(encoding='utf-8'))
for n in atlas['nodes']:
    text=' '.join(str(n.get(k,'')) for k in ('id','title','summary')).lower()
    if any(t in text for t in ('feature_embed','periodic','piecewise','raw_curve','curve_token','feature_token','revin','wavelet','localnorm','local_norm')):
        print(json.dumps({k:n.get(k) for k in ('id','title','primary','summary','conclusion')},ensure_ascii=False))
