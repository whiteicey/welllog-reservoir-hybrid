"""Record completed CPU-only paired results; no training or model selection."""
import json, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pure_dl'))
import train_invariant as base
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
data=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
truth=np.asarray(data['truth'][data['validation']]);default=base.dmask(truth)[:,2]
reports={};strata={}
for arm in ('independent','pooled'):
    out=ROOT/f'outputs/pure_dl_sw_rankone_continuous_v296_fold5/{arm}'
    reports[arm]=read(out/'report.json')
    assert reports[arm]['predictions_reproduced_exactly'] and reports[arm]['frozen_parameters_exact']
    assert (STUDY/f'{arm}_assembled.json').exists()
    pred=truth.copy();pred[:,2]=np.load(out/'channel_pred.npy')
    strata[arm]=dict(SW=base.metric(truth,pred)['SW'],default_SW=base.metric(truth[default],pred[default])['SW'],continuous_SW=base.metric(truth[~default],pred[~default])['SW'])
timing=read(STUDY/'BATCH_TIMING.json')
finished=max((ROOT/f'outputs/pure_dl_sw_rankone_continuous_v296_fold5/{arm}/report.json').stat().st_mtime for arm in reports)
timing.update(terminal=True,total_paired_wall_seconds=finished-min(timing['independent_start_epoch'],timing['pooled_start_epoch']),individual_seconds={a:r['seconds'] for a,r in reports.items()},peak_MiB={a:r['peak_MiB'] for a,r in reports.items()},measurement_scope='Data/initial/train/final replay included; staggered partial overlap and gap with other studies, not isolated speed benchmark; CPU record and codeprep excluded')
result=dict(reports=reports,strata=strata,independent_minus_pooled=reports['independent']['target_score']-reports['pooled']['target_score'],gain_pooled_vs289=reports['pooled']['target_score']-74.1537264289993,continuous_reference252=17.88486813991017,timing=timing,adopted=False,remaining_folds_started=False,complete_five_folds=False,conclusion='Independent member auxiliary CE does not beat pooled CE; tiny overall differences and lower continuous scores do not support continuing this setting. Historical TabM positives are not negated.')
for name,value in [('PAIRED_FINAL.json',result),('STRATA_DIAGNOSTIC.json',strata),('BATCH_TIMING.json',timing)]:
    (STUDY/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
note=f"\n\n## Paired completion\nIndependent SW{strata['independent']['SW']:.12f}, pooled{strata['pooled']['SW']:.12f}; independent-minus-pooled{result['independent_minus_pooled']:+.12f}. Pooled-minus289{result['gain_pooled_vs289']:+.12f}. Continuous scores independent{strata['independent']['continuous_SW']:.12f}, pooled{strata['pooled']['continuous_SW']:.12f}, below25217.88486813991017. No continuous breakthrough or adoption; close this independent-v-pooled auxiliaryCE setting, not historical TabM family. Original104 retained; allnew encoder/rankgains/readouts actuallytrained. Checkpoints retained temporarily as completed matched controls until current bank comparison closes; no automatic extra stages/folds. Paired wall{timing['total_paired_wall_seconds']:.3f}s, individual independent{reports['independent']['seconds']:.3f}s/pooled{reports['pooled']['seconds']:.3f}s, partial overlap/gap prevents speed ranking. Actual144665 assembly/own-best RAM replay, no independent disk reload.\n"
for path in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
    with path.open('a',encoding='utf-8') as f:f.write(note)
print(json.dumps({k:v for k,v in result.items() if k!='reports'}))
