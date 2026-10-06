"""Paired terminal CPU result; no automatic adoption, GPU, or continuation."""
from pathlib import Path
import json
S=Path(__file__).resolve().parent;R=S.parents[2]
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
a=read(S/'DECORRELATION_TERMINAL.json');b=read(S/'IDENTITY_TERMINAL.json')
whole=a['report']['target_score']-b['report']['target_score']
cont=a['strata']['continuous']-b['strata']['continuous']
positive=whole>0 and cont>0
pairwall=max(v['timing']['started_epoch']+v['timing']['process_creation_to_terminal_seconds'] for v in (a,b))-min(v['timing']['started_epoch'] for v in (a,b))
f=dict(arms={'decorrelation':a,'identity':b},pair_whole_delta=whole,pair_CONT_delta=cont,pair_DEFAULT_delta=a['strata']['default']-b['strata']['default'],mechanism_positive=positive,absolute_whole_gain=a['whole_vs336']>0,absolute_CONT_gain=a['CONT_vs336']>0,adopted=False,serial=True,paired_wall_seconds_including_gaps=pairwall,individual_total_seconds=sum(v['report']['seconds'] for v in (a,b)),screening_fold=5,complete_five_folds=False,remaining_folds_started=False,independent_disk_reload=False,comparison_limits='Same architecture/parameters/init/budget policy, single seed and wholebest checkpoint. Covariance auxiliary not generalization proof; no coefficient/layer grid.')
(S/'PAIRED_FINAL.json').write_text(json.dumps(f,indent=2),encoding='utf-8')
p=read(S/'PLAN.json');p.update(paired_terminal=True,identity_terminal=True,decorrelation_terminal=True,mechanism_positive=positive,adopted=False,closed=not positive);(S/'PLAN.json').write_text(json.dumps(p,indent=2),encoding='utf-8')
t='\nV361 paired terminal: decorrelation minusidentity PERMwhole'+str(whole)+'/CONT'+str(cont)+'/DEFAULT'+str(f['pair_DEFAULT_delta'])+', finite mechanism_positive='+str(positive)+'. Absolute vsretained336 whole'+str(a['whole_vs336'])+'/CONT'+str(a['CONT_vs336'])+'. Pairwall'+str(pairwall)+'s includes serialgaps; individual total'+str(f['individual_total_seconds'])+'s. No automatic adoption/otherfolds/coefflayergrid/single-seedrobustnessclaim. Currentretained80.70612972323806 stays.\n'
for p in [S/'README.md',S/'PROJECTS.md',R/'PROJECTS.md',R.parent/'PROJECTS.md']:
 with p.open('a',encoding='utf-8') as h:h.write(t)
print(json.dumps(f))
