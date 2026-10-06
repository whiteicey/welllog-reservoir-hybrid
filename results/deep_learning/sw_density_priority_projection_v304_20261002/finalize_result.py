"""CPU terminal evidence, including diagnostic decoded-action tradeoffs."""
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
out=ROOT/'outputs/pure_dl_sw_density_priority_projection_v304_fold5/density_priority'
report=read(out/'report.json');assert report['predictions_reproduced_exactly'] and report['frozen_parameters_exact']
assert (STUDY/'density_priority_assembled.json').exists()
paths={
    'SW252':ROOT/'outputs/pure_dl_sw_mass_cdf_v252_fold5/conditional/channel_pred.npy',
    'SW289':ROOT/'outputs/pure_dl_sw_state_bound_v289_fold5/balanced/channel_pred.npy',
    'SW297well':ROOT/'outputs/pure_dl_sw_raw_well_condition_v297_fold5/well/channel_pred.npy',
    'SW304':out/'channel_pred.npy',
}
strata={}
for name,path in paths.items():
    values=np.load(path);pred=truth.copy();pred[:,2]=values
    strata[name]=dict(SW=base.metric(truth,pred)['SW'],default_SW=base.metric(truth[default],pred[default])['SW'],continuous_SW=base.metric(truth[~default],pred[~default])['SW'],default_rows=int(default.sum()),continuous_rows=int((~default).sum()),atom_action=float(base.DEFAULTS['SW']),continuous_decoded_atom_fraction=float((values[~default]==base.DEFAULTS['SW']).mean()),default_decoded_atom_fraction=float((values[default]==base.DEFAULTS['SW']).mean()),continuous_near_default_action_fraction=float((values[~default]>=99.9).mean()))
timing=read(STUDY/'BATCH_TIMING.json')
timing.update(terminal=True,individual_seconds=report['seconds'],observed_launch_to_report_seconds=(out/'report.json').stat().st_mtime-timing['starts'][0]['start_epoch'],peak_MiB=report['peak_MiB'],concurrency='single GPU training task; historical289 had different contention, no speedranking')
history=read(out/'validation_history.json');gradient_evidence=dict(mean_conflict_fraction=float(np.mean([h['training_conflict_fraction'] for h in history])),mean_cosine=float(np.mean([h['training_state_density_cosine'] for h in history])),training_only=True,extra_task_backward_passes=2)
result=dict(gradient_evidence=gradient_evidence,report=report,strata=strata,whole_delta_vs289=strata['SW304']['SW']-strata['SW289']['SW'],continuous_delta_vs289=strata['SW304']['continuous_SW']-strata['SW289']['continuous_SW'],continuous_delta_vs252=strata['SW304']['continuous_SW']-strata['SW252']['continuous_SW'],adopted=False,remaining_folds_started=False,complete_five_folds=False,timing=timing,conclusion='Directed state-gradient projection compared with same-forward289 reference; decoded action fractions diagnose posterior decision tradeoffs, not direct state-classifier probabilities or an oracle. No priority/threshold/BCEweight grid.')
for name,value in [('FINAL.json',result),('STRATA_DECISION_DIAGNOSTIC.json',strata),('BATCH_TIMING.json',timing)]:
    (STUDY/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
note=f"\n\n## Final gradient-routing outcome\nSW{report['target_score']:.12f}/best{report['best_epoch']}/stop{report['stopped_epoch']}, versus289{result['whole_delta_vs289']:+.12f}; continuous{strata['SW304']['continuous_SW']:.12f}, versus289{result['continuous_delta_vs289']:+.12f}/252{result['continuous_delta_vs252']:+.12f}. Not adopted; no priority/threshold/BCE grid. Actualencoder/readout/state nonzerograd/frozen/RAMbest144665assembly; no independentdiskreload. Individual{report['seconds']:.3f}s/peak{report['peak_MiB']:.3f}MiB, single GPU, historical contention differs/no speedranking. Decoded atom actionfraction is a posterior decision diagnostic, not gateprobability/classifier accuracy; no roworacle or stratum-specific model selection. Currentrecipes unchanged; finishedcleanup remains blocked by automaticpolicy, no deletionperformed.\n"
for path in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
    with path.open('a',encoding='utf-8') as f:f.write(note)
print(json.dumps({k:v for k,v in result.items() if k not in ('report','timing')}))
