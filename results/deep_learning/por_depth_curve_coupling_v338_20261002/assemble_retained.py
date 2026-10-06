"""Terminal CPU actual fixed-column combination; no GPU/folds and no overwrite."""
from pathlib import Path
import sys,json,hashlib,numpy as np,torch
S=Path(__file__).resolve().parent;R=S.parents[2];sys.path.insert(0,str(R/'experiments/pure_dl'));import train_invariant as b
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'));torch.set_num_threads(1)
result=read(S/'PAIRED_FINAL.json');a=result['arms']['depth'];assert a['report']['predictions_reproduced_exactly'] and a['report']['frozen_parameters_exact']
assert a['whole_vs335']>0 and a['CONT_vs335']>0,'No absolute whole+CONT adoption condition'
assert result['depth_minus_control_whole']>0 and result['depth_minus_control_CONT']>0,'Depth mechanism not positive vs completed control'
out=R/'outputs/pure_dl_por_depth_curve_coupling_v338_fold5/depth';perm=R/'outputs/pure_dl_perm_low_rank_cross_v336_fold5/cross';sw=R/'outputs/pure_dl_sw_latent_morphology_v326_fold5/depth'
p=np.column_stack([np.load(x/'channel_pred.npy') for x in (out,perm,sw)]);assert p.shape==(144665,3) and np.isfinite(p).all()
d=torch.load(R/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False);scores=b.metric(np.asarray(d['truth'][d['validation']]),p)
old=read(R/'results/deep_learning/perm_low_rank_cross_v336_20261002/CURRENT_EIGHT_STAGE_RECIPE.json')
assert scores['TOTAL']>old['scores']['TOTAL'];assert abs(scores['POR']-a['report']['target_score'])<1e-9
assert all(abs(scores[t]-old['scores'][t])<1e-9 for t in ('PERM','SW'))
pred=S/'eight_stage_pred.npy';recipefile=S/'CURRENT_EIGHT_STAGE_RECIPE.json';assert not pred.exists() and not recipefile.exists(),'New artifact only; no overwrites'
np.save(pred,p);recipe=dict(scores=scores,screening_fold=5,validation_rows=144665,strict_808_qualified=scores['TOTAL']>80.8,complete_five_folds=False,remaining_folds_started=False,fullOOF_verified=False,total_training_stages=8,stages=['SOURCE','POR113','POR151','PERM154','SW104','POR338depth','PERM336cross','SW326depth'],added_head_stages=dict(POR=1,PERM=1,SW=1),heads={t:dict(checkpoint=str(x/'model_best.pt'),sha256=hashlib.sha256((x/'model_best.pt').read_bytes()).hexdigest()) for t,x in zip(('POR','PERM','SW'),(out,perm,sw))},pred_sha256=hashlib.sha256(pred.read_bytes()).hexdigest(),verification='Actual144665fixedwhole-targetcolumns/terminalRAMbestreplay/frozenexact; notindependentdiskreload orfullOOF',independent_disk_reload=False,default_five_stage_unchanged=True,prior_optional8_artifacts_unchanged=True,gain_vs_prior336_recipe=scores['TOTAL']-old['scores']['TOTAL'],gain_vs_default_five=scores['TOTAL']-80.61247541799393,POR_continuous=a['strata']['continuous'],PERM_continuous=old['PERM_continuous'],SW_continuous=old['SW_continuous'],single_seed_small_gain_not_robustness=True,selection_reason='Depth jointlytrained encoder/readout whole+CONT >335 and completedcontrol; onefreshPORphase nottrainedrepairchain')
recipefile.write_text(json.dumps(recipe,indent=2),encoding='utf-8');print(json.dumps(recipe))
