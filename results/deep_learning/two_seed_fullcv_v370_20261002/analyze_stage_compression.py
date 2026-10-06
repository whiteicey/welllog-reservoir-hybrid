"""Observed stage priorities; never claim an unrun pruning ablation."""
from pathlib import Path
import json,time,statistics

S=Path(__file__).resolve().parent;R=S.parents[2]
stages=['POR113','POR151','PERM154','SW104','POR341','PERM336','SW351']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
cases=[];grouped={s:[] for s in stages};pending_stage_evidence=[]
for p in sorted(S.glob('fold_*/*_PREFIX_SNAPSHOT.json')):
    case=read(p)
    if not case.get('complete8stage_single_fold'):
        for gain in case['stage_gains']:
            pending_stage_evidence.append(dict(identity=case['identity'],fold=case['fold'],stage=gain['stage'],
                gain=gain['actual_target_gain'],CONT=gain['strata']['CONT']['delta'],
                scope='Terminal stage within incomplete model; excluded from completed-model summary'))
        continue
    cases.append(dict(identity=case['identity'],fold=case['fold']))
    for stage in stages:
        terminal=read(p.parent/(case['identity']+'_'+stage+'_TERMINAL.json'))
        gain=terminal['stage_gains'][-1];assert gain['stage']==stage
        grouped[stage].append(dict(identity=case['identity'],fold=case['fold'],gain=gain['actual_target_gain'],TOTAL=gain['actual_TOTAL_contribution'],DEFAULT=gain['strata']['DEFAULT']['delta'],CONT=gain['strata']['CONT']['delta']))
assert cases and all(len(v)==len(cases) for v in grouped.values())
threshold=1e-4;summary={}
for stage,values in grouped.items():
    delta=[v['gain'] for v in values]
    summary[stage]=dict(observations=len(values),positive=sum(d>threshold for d in delta),near_zero=sum(abs(d)<=threshold for d in delta),negative=sum(d < -threshold for d in delta),minimum=min(delta),mean=statistics.mean(delta),maximum=max(delta),mean_TOTAL=statistics.mean(v['TOTAL'] for v in values),CONT_positive=sum(v['CONT']>threshold for v in values),CONT_negative=sum(v['CONT'] < -threshold for v in values),observations_by_case=values)
folds=sorted({c['fold'] for c in cases})
result=dict(observed_epoch=time.time(),cases=cases,summary=summary,near_zero_threshold=threshold,
    pending_stage_evidence=pending_stage_evidence,
    scope='Actual adjacent-stage changes on completed single-fold models; folds'+str(folds)+' and paired seeds are not independent replicates. Historical fifth has differing budgets and is not pooled here.',
    causal_pruning_ablation_completed=False,training_recipe_changed=False,new_GPU_job_started=False,
    conclusion='No verified lossless8-to5-stage reduction follows from these changes. Subsequent stages depend on earlier trained states, and probability-mixture decoding is nonlinear.',
    next_fixed_fifth_priority=['POR151 controlled omission with downstream POR341 retrained from POR113, same source/budget/seed; first7-stage candidate only. Positive completed or pending-stage gains forbid assuming universal redundancy.',
      'POR341 modification or omission: its observed benefit shifts DEFAULT/CONT; measure both and final native mixture, not solo score only.',
      'PERM154 omission needs neutral adapter and PERM336 retraining; PERM336 has meaningful fold2 gains, not universally redundant.',
      f"SW104 source-preserving additive alternative within the existing stage slot; observed SOURCE regressions {summary['SW104']['negative']}/{summary['SW104']['observations']}, positive cases {summary['SW104']['positive']}; no method-wide failure claim.",
      f"SW351 cannot be assumed redundant: {summary['SW351']['positive']} of {summary['SW351']['observations']} current observations positive. Any5-stage claim needs an actual fixed-fifth controlled recipe."],
    launch_policy='Complete already-authorized original+seed13 fullCV serially first. No new pruning/repair experiment launched.')
(S/'STAGE_COMPRESSION_ANALYSIS.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
lines=['# Stage compression evidence','',result['scope'],'',
    'Near-zero threshold0.0001 is a reporting convention, not a significance test. Actual raw values remain in STAGE_GAINS.csv.',
    'The current fixed8-stage recipe is unchanged. The table does not estimate full OOF or causal stage-removal effects.','',
    '| Stage | Positive / near-zero / negative | Target gain min / mean / max | Mean TOTAL increment | CONT positive / negative |',
    '|---|---|---|---:|---|']
for stage,x in summary.items():
    lines.append(f"| {stage} | {x['positive']} / {x['near_zero']} / {x['negative']} | {x['minimum']:+.6f} / {x['mean']:+.6f} / {x['maximum']:+.6f} | {x['mean_TOTAL']:+.6f} | {x['CONT_positive']} / {x['CONT_negative']} |")
lines+=['',result['conclusion'],'','Terminal stages in still-incomplete models (not pooled above):','']
for item in pending_stage_evidence:
    lines.append(f"- {item['identity']}/fold{item['fold']}/{item['stage']}: target {item['gain']:+.9f}, CONT {item['CONT']:+.9f}; {item['scope']}.")
lines+=['','Next isolated fixed-fifth comparisons, after fullCV:','']
lines+=['- '+idea for idea in result['next_fixed_fifth_priority']]
lines+=['',result['launch_policy'],'','Evidence: completed *_PREFIX_SNAPSHOT.json / *_TERMINAL.json and STAGE_GAINS.csv.','']
(S/'STAGE_COMPRESSION_ANALYSIS.md').write_text('\n'.join(lines),encoding='utf-8')
counts={stage:dict(positive=x['positive'],near_zero=x['near_zero'],negative=x['negative'],CONT_positive=x['CONT_positive'],CONT_negative=x['CONT_negative']) for stage,x in summary.items()}
note='\n2026-10-02 completed-stage compression analysis: '+str(len(cases))+' completed8-stage model/fold cases, actual adjacent changes only. Stage counts '+json.dumps(counts)+'. Historical fifth not pooled due differing budgets. No causal lossless8->5 proof or new training. Future controlled fifth priority documented in STAGE_COMPRESSION_ANALYSIS.md; current original+seed13 fullCV continues serially. Pending full grouped OOF and actual uniform pruning trials afterwards.\n'
for p in [S/'README.md',S/'PROJECTS.md',R/'PROJECTS.md',R.parent/'PROJECTS.md']:
    with p.open('a',encoding='utf-8') as stream:stream.write(note)
print(json.dumps(dict(cases=cases,summary={s:{k:v for k,v in x.items() if k!='observations_by_case'} for s,x in summary.items()},causal_pruning_ablation_completed=False)))
