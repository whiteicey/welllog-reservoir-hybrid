"""Read-only held-out error audit; no fitting, thresholds or prediction replacement."""
import json
from pathlib import Path
import numpy as np
import torch
import train_invariant as base

torch.set_num_threads(2)
source=base.ROOT/'outputs/pure_dl_annealed_aux_sequence'
out=base.ROOT/'outputs/best_error_structure';out.mkdir(exist_ok=True)
proof=json.loads((source/'screening_verification.json').read_text());assert proof['passed']
with np.load(source/'oof.npz') as a:
    use=a['fold']==4;y=a['truth'][use];p=a['pred'][use];well=a['well'][use]
ck=torch.load(source/'fold_5.pt',map_location='cpu',weights_only=False)
raw=np.asarray(ck['validation_raw_rows']);assert len(raw)==len(y)==144665
with np.load(source/'fold_5_distributions.npz') as a:
    assert np.array_equal(a['indices'],np.flatnonzero(use))
    gate=a['gate'].copy()
d=base.dmask(y);pd=np.isclose(p,np.asarray([.1,.01,99.9]),rtol=0,atol=1e-9)
score=np.column_stack([np.maximum(0,1-np.abs(p[:,0]-y[:,0])/(.08*(abs(y[:,0])+1e-5))),
                       np.maximum(0,1-np.abs(np.log10(np.maximum(p[:,1]/y[:,1],1e-5)))),
                       np.maximum(0,1-np.abs(p[:,2]-y[:,2])/(.05*(abs(y[:,2])+1e-5)))])
weights=np.asarray([.3,.35,.35]);loss=1-score
assert abs(float(score.mean(0)@weights*100)-proof['screening_fold_score'])<1e-9
adjacent=(np.diff(raw)==1)&(np.diff(well)==0)
records={}
for j,name in enumerate(base.TS):
    partitions={}
    for truth_default in [False,True]:
        for prediction_default in [False,True]:
            m=(d[:,j]==truth_default)&(pd[:,j]==prediction_default)
            partitions[f'truth_default_{truth_default}_decoded_default_{prediction_default}']={
                'rows':int(m.sum()),'population_pct':float(m.mean()*100),
                'mean_target_score':float(score[m,j].mean()*100) if m.any() else None,
                'total_score_deficit_points':float(loss[m,j].sum()/len(y)*weights[j]*100)}
    state_boundary=np.zeros(len(y),dtype=bool)
    transitions=np.flatnonzero(adjacent&(d[1:,j]!=d[:-1,j]))
    state_boundary[transitions]=True;state_boundary[transitions+1]=True
    # Neighborhood by raw row distance, never cross a well or missing-row gap.
    band=state_boundary.copy()
    for _ in range(8):
        new=band.copy();new[1:]|=band[:-1]&adjacent;new[:-1]|=band[1:]&adjacent;band=new
    severe=loss[:,j]>=.8
    run_lengths=[];run_deficits=[]
    i=0
    while i<len(y):
        if not severe[i]:i+=1;continue
        end=i+1
        while end<len(y) and severe[end] and adjacent[end-1]:end+=1
        run_lengths.append(end-i);run_deficits.append(float(loss[i:end,j].sum()));i=end
    lengths=np.asarray(run_lengths);deficits=np.asarray(run_deficits)
    calibration=[]
    for lo,hi in zip([0,.1,.3,.7,.9],[.1,.3,.7,.9,1.000001]):
        m=(gate[:,j]>=lo)&(gate[:,j]<hi)
        calibration.append({'range':[lo,min(hi,1)],'rows':int(m.sum()),'mean_probability':float(gate[m,j].mean()) if m.any() else None,'default_frequency':float(d[m,j].mean()) if m.any() else None})
    records[name]={'partitions':partitions,'gate_brier':float(np.mean((gate[:,j]-d[:,j])**2)),
       'gate_calibration':calibration,'state_transition_edges':len(transitions),
       'within8_rows_of_transition_population_pct':float(band.mean()*100),
       'within8_rows_of_transition_deficit_pct':float(loss[band,j].sum()/loss[:,j].sum()*100),
       'severe_rows':int(severe.sum()),'severe_runs':len(lengths),'severe_runs_at_least16':int((lengths>=16).sum()),
       'deficit_in_severe_runs_at_least16_pct':float(deficits[lengths>=16].sum()/loss[:,j].sum()*100)}
well_rows=[]
for w in np.unique(well):
    m=well==w
    well_rows.append({'well_index':int(w),'rows':int(m.sum()),'score':float(score[m].mean(0)@weights*100),
        'total_deficit_contribution':float(loss[m].sum(0)@weights/len(y)*100)})
result={'source':str(source),'scope':'Held-out fifth-fold descriptive audit only. No optimization, correction, oracle score, test labels or training changes.',
        'rows':len(y),'score':float(score.mean(0)@weights*100),'total_deficit':float(loss.mean(0)@weights*100),
        'targets':records,'wells':sorted(well_rows,key=lambda r:r['total_deficit_contribution'],reverse=True)}
(out/'diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
for name,r in records.items():
    print(name,json.dumps({'partitions':r['partitions'],'boundary_population_pct':r['within8_rows_of_transition_population_pct'],'boundary_deficit_pct':r['within8_rows_of_transition_deficit_pct'],'long_severe_run_deficit_pct':r['deficit_in_severe_runs_at_least16_pct']}))
print('WORST_WELLS',json.dumps(result['wells'][:5]));print('PASS score identity',result['score'])
