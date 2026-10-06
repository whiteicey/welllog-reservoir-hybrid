"""CPU actual whole/stratum contrasts for completed flat/hierarchical runs."""
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3];STUDY=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pure_dl'))
import train_invariant as base
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
data=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
truth=np.asarray(data['truth'][data['validation']]);default=base.dmask(truth)[:,2]
reports={};strata={};configs={}
for arm in ('flat','hierarchical'):
    out=ROOT/f'outputs/pure_dl_sw_coarse_fine_v299_fold5/{arm}'
    assert (STUDY/f'{arm}_assembled.json').exists()
    reports[arm]=read(out/'report.json');configs[arm]=read(out/'config.json')
    assert reports[arm]['predictions_reproduced_exactly'] and reports[arm]['frozen_parameters_exact']
    pred=truth.copy();pred[:,2]=np.load(out/'channel_pred.npy')
    strata[arm]=dict(SW=base.metric(truth,pred)['SW'],default_SW=base.metric(truth[default],pred[default])['SW'],continuous_SW=base.metric(truth[~default],pred[~default])['SW'])
assert configs['flat']['initialization']['training_Q_weighted_continuous_bin_groups']==configs['hierarchical']['initialization']['training_Q_weighted_continuous_bin_groups']
assert configs['flat']['trainable_parameters']==configs['hierarchical']['trainable_parameters']==58777
timing=read(STUDY/'BATCH_TIMING.json')
finished=max((ROOT/f'outputs/pure_dl_sw_coarse_fine_v299_fold5/{a}/report.json').stat().st_mtime for a in reports)
timing.update(terminal=True,total_paired_wall_seconds=finished-min(timing['flat_start_epoch'],timing['hierarchical_start_epoch']),individual_seconds={a:r['seconds'] for a,r in reports.items()},peak_MiB={a:r['peak_MiB'] for a,r in reports.items()},scope='Partial overlap/stagger incldata/initial/train/finalrestore, codeprep/CPUassembly excluded; not speedbenchmark')
result=dict(reports=reports,strata=strata,hierarchical_minus_flat=reports['hierarchical']['target_score']-reports['flat']['target_score'],hierarchical_minus_flat_continuous=strata['hierarchical']['continuous_SW']-strata['flat']['continuous_SW'],hierarchical_minus297=reports['hierarchical']['target_score']-74.16003022301975,continuous_reference252=17.88486813991017,matched_group_boundaries=configs['flat']['initialization']['training_Q_weighted_continuous_bin_groups'],registered_and_active_capacity_same=True,initial_score_same=True,adopted=False,remaining_folds_started=False,complete_five_folds=False,timing=timing)
result['conclusion']='Small matched overall normalization benefit below retained297; absolute continuous stratum still must exceed252 to claim substantial breakthrough. Close count/quantile/FiLM grid; preserve compact paired evidence, do not generalize to all coarse-to-fine methods.'
for name,value in [('PAIRED_FINAL.json',result),('STRATA_DIAGNOSTIC.json',strata),('BATCH_TIMING.json',timing)]:
    (STUDY/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
note=f"\n\n## Final matched result\nFlat SW{strata['flat']['SW']:.12f}, hierarchical{strata['hierarchical']['SW']:.12f}; paired overall{result['hierarchical_minus_flat']:+.12f}, continuous flat{strata['flat']['continuous_SW']:.12f}/hierarchical{strata['hierarchical']['continuous_SW']:.12f}, pairedcontinuous{result['hierarchical_minus_flat_continuous']:+.12f}. Hierarchical-minus297{result['hierarchical_minus297']:+.12f}, not adopted. Compare absolutecontinuous25217.88486813991017; no continuous breakthrough from a local relative benefit alone. Same actual8boundaries/fit161423rows/58777params and sourceinitialfunction, allnewencoder/readout/state grads verified/144665composition/RAMbest replay, not independentdiskreload. Individual flat{reports['flat']['seconds']:.3f}s/hierarchical{reports['hierarchical']['seconds']:.3f}s, pairwall{timing['total_paired_wall_seconds']:.3f}s, partial overlap with otherjobs, no speedranking. Do not sweep groupcount/quantile/headwidth on thissmall overall effect. No additionalfolds, preserve current optional8/default5/oldfullOOFdelivery.\n"
for path in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
    with path.open('a',encoding='utf-8') as f:f.write(note)
print(json.dumps({k:v for k,v in result.items() if k not in ('reports','timing')}))
