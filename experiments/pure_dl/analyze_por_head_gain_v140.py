"""Requested descriptive gain analysis of frozen fifth-fold predictions; no fitting."""
import json
from pathlib import Path
import numpy as np
import torch
import train_invariant as base

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/por_joint_density_width_v140_20261001'
PATHS={
    'parent':'pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy',
    'location_local':'pure_dl_por_joint_context_head_v128_fold5/local_joint',
    'location_point':'pure_dl_por_joint_point_ablation_v129_fold5/point_joint',
    'location_weight64':'pure_dl_por_joint_density_head_v131_fold5/location_weight',
    'location_weight_gate':'pure_dl_por_joint_full_density_head_v133_fold5/location_weight_gate',
    'location_weight128':'pure_dl_por_joint_density_width_v134_fold5/wide128',
    'location_weight256':'pure_dl_por_joint_density_width_v140_fold5/wide256'}

def main():
    d=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
    ix=d['validation'];y=d['truth'][ix,0];wells=d['wells'][ix]
    default=base.dmask(d['truth'][ix])[:,0]
    scores={};preds={};evidence={}
    for name,path in PATHS.items():
        out=ROOT/'outputs'/path;p=np.load(out/'channel_pred.npy')
        assert p.shape==(144665,) and np.isfinite(p).all()
        point=np.maximum(0,1-abs(p-y)/(.08*(abs(y)+1e-5)))
        r=json.loads((out/'report.json').read_text(encoding='utf-8-sig'))
        assert abs(point.mean()*100-r['target_score'])<1e-9
        scores[name]=point;preds[name]=p
        cfg=json.loads((out/'config.json').read_text(encoding='utf-8-sig'))
        evidence[name]=dict(POR=r['target_score'],parameters=cfg['trainable_parameters'],best_epoch=r['best_epoch'],prediction_path=str((out/'channel_pred.npy').relative_to(ROOT)))
    pairs=[('local_vs_point','location_point','location_local'),('weight_added','location_point','location_weight64'),('width128_vs64','location_weight64','location_weight128'),('gate_added','location_weight64','location_weight_gate'),('overall_vs_parent','parent','location_weight128'),('width256_vs128','location_weight128','location_weight256'),('overall256_vs_parent','parent','location_weight256')]
    result=dict(screening_fold=5,rows=len(y),complete_five_folds=False,scope='Descriptive frozen prediction analysis only; no fit, threshold tuning, new model selection, or row-wise prediction choice',evidence=evidence,comparisons={})
    for key,a,b in pairs:
        gain=(scores[b]-scores[a])*100
        perwell=[dict(well=int(w),rows=int((wells==w).sum()),delta=float(gain[wells==w].mean())) for w in np.unique(wells)]
        intervals=[]
        for lo,hi in zip([0,5,10,15,20,30],[5,10,15,20,30,float('inf')]):
            mask=(~default)&(y>=lo)&(y<hi)
            if mask.any():intervals.append(dict(lower=lo,upper=None if np.isinf(hi) else hi,rows=int(mask.sum()),delta=float(gain[mask].mean())))
        result['comparisons'][key]=dict(control=a,candidate=b,delta=float(gain.mean()),per_well=perwell,positive_wells=sum(v['delta']>0 for v in perwell),negative_wells=sum(v['delta']<0 for v in perwell),nondefault_delta=float(gain[~default].mean()),default_delta=float(gain[default].mean()) if default.any() else None,default_rows=int(default.sum()),POR_ranges=intervals,improved_rows=int((gain>0).sum()),worsened_rows=int((gain<0).sum()),unchanged_rows=int((gain==0).sum()))
    (STUDY/'gain_analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# POR head gain analysis','', 'Original fixed fifth only; frozen predictions, no fitting or new selection. Single fold/seed and unequal active capacity limit causal attribution.','', '| Comparison | POR delta | Positive wells /16 | Nondefault delta | Default delta |','|---|---:|---:|---:|---:|']
    for key,c in result['comparisons'].items():
        dd='n/a' if c['default_delta'] is None else f"{c['default_delta']:+.6f}"
        lines.append(f"| {key} | {c['delta']:+.6f} | {c['positive_wells']} | {c['nondefault_delta']:+.6f} | {dd} |")
    lines+=['','Interpretation: point-local comparison provides no local-depth benefit. Adding conditional logweights alongside locations and then widening the nonlinear head each helped this exact POR recipe. Frozen original predictor/default gate/scales constrain the change to conditional mixture locations and masses. Extra output/capacity and different training trajectories prevent claiming isolated weight-only or width-only causal effects beyond this bounded comparison.','', 'Next: width256 from same frozen v113 initial function, replace width128 head phase; do not warm-start it and add a sixth phase. Preserve original proper mixed-family NLL/CDF and scorer. Only after useful POR result, test the same readout mechanism on PERM and SW with matched controls. PERM width128 previously +0.009542 but width256 negative, so transfer is not automatic; SW Student3 is a separate family experiment and does not test this head mechanism.','', 'Per-well/range details are in gain_analysis.json, for diagnosis only; no per-well/row model selection or label-derived input features.']
    (STUDY/'GAIN_ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:{f:v for f,v in c.items() if f in ['delta','positive_wells','nondefault_delta','default_delta']} for k,c in result['comparisons'].items()}))

if __name__=='__main__':main()
