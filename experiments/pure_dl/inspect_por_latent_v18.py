"""Read-only scoped source/status lookup before the POR architecture study."""
from pathlib import Path
import ast
import datetime
import json
import sys
import psutil

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parents[2]
print('OBSERVED', datetime.datetime.now().astimezone().isoformat())
for rel in ['AGENTS.md', 'experiments/AGENTS.md', 'experiments/pure_dl/AGENTS.md', 'PROJECTS.md']:
    a = (ROOT/rel).read_text(encoding='utf-8-sig').splitlines()
    print('DOC', rel, len(a))
    print('\n'.join(f'{i+1} | {a[i]}' for i in range(max(0,len(a)-14),len(a))))
print('RECENT', [p.name for p in sorted((ROOT/'outputs').iterdir(),key=lambda p:p.stat().st_mtime,reverse=True)[:6]])
active=[]
for p in psutil.process_iter(['pid','name','cmdline']):
    try:
        v=p.info; args=v.get('cmdline') or []
        if 'python' in (v.get('name') or '').lower() and '-c' not in args and any('pure_dl' in a and 'train_' in a for a in args): active.append(v)
    except (psutil.NoSuchProcess, psutil.AccessDenied): pass
print('ACTIVE',active)
print('V17_SCORES',json.loads((ROOT/'results/deep_learning/latent_gate_v17_20260928/assembled_report.json').read_text())['selected_scores'])
for name in ['train_invariant.py','por_t3_patch_loss.py','train_dual8222_heads80.py']:
    p=ROOT/'experiments/pure_dl'/name; s=p.read_text(encoding='utf-8-sig'); a=s.splitlines()
    print('SOURCE',name)
    if name=='por_t3_patch_loss.py':
        print(s)
        continue
    for node in ast.walk(ast.parse(s)):
        if isinstance(node,ast.FunctionDef) and node.name in ['kernel_matrix','prepare_data','distribution_probs','student_t3_cdf']:
            print('\n'.join(f'{i+1} | {a[i]}' for i in range(node.lineno-1,node.end_lineno)))
    for i,line in enumerate(a):
        if any(v in line for v in ['BOUNDS =','TRANSFORM =','INVERSE =','DEFAULTS =','linspace','StudentT','from ']): print(i+1,line)
