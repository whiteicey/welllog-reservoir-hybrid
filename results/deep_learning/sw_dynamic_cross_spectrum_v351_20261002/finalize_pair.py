"""Finish fixed spectral pair and create a new optional8 recipe only on terminal positive evidence."""
from pathlib import Path
import json,hashlib,sys,numpy as np
S=Path(__file__).resolve().parent;R=S.parents[2];read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
a=read(S/'ALIGNED_TERMINAL.json');c=read(S/'PERMUTED_TERMINAL.json')
assert a['report']['stopped_epoch']>=10 and c['report']['stopped_epoch']>=10
f=dict(arms={'aligned':a,'permuted':c},aligned_minus_permuted_whole=a['report']['target_score']-c['report']['target_score'],aligned_minus_permuted_CONT=a['strata']['continuous']-c['strata']['continuous'],aligned_minus_permuted_DEFAULT=a['strata']['default']-c['strata']['default'],screening_fold=5,complete_five_folds=False,remaining_folds_started=False,independent_disk_reload=False,comparison_limits='92841samephysicalparams/init/loss/data, onlyfixed donorfrequency permutation; changes frequency correspondence and smoothness/functionclass, single seed not robustness. Historical326552lower capacity/nonconcurrent. Whole-target selected epoch, noCONToracle.')
f['mechanism_positive']=f['aligned_minus_permuted_whole']>0 and f['aligned_minus_permuted_CONT']>0
f['adopted']=f['mechanism_positive'] and a['whole_vs326']>0 and a['CONT_vs326']>0
reports=[R/'outputs/pure_dl_sw_dynamic_cross_spectrum_v351_fold5'/arm/'report.json' for arm in ('aligned','permuted')]
f['pair_wall_process_seconds']=max(p.stat().st_mtime for p in reports)-min(a['timing']['started_epoch'],c['timing']['started_epoch']);f['launch_stagger_seconds']=abs(c['timing']['started_epoch']-a['timing']['started_epoch']);f['pair_timing_limits']='Solo sequential pair includes explicit launch gaps; individual prep/warmup/train/restoration separately recorded, no speed claim from parametercount'
f['closed_setting']='Close current amplitude-donor filter if matchedwhole/CONT notpositive; no band-window-LRgrid' if not f['mechanism_positive'] else 'Positive finite mechanism; eligible for distinct isolated extension using positive code/init only, never trainedrepairchain'
(S/'PAIRED_FINAL.json').write_text(json.dumps(f,indent=2),encoding='utf-8')
if f['adopted']:
 old=read(R/'results/deep_learning/por_coupling_conditional_rps_v341_20261002/CURRENT_EIGHT_STAGE_RECIPE.json');q=dict(old)
 q.pop('gain_vs_prior338_recipe',None)
 q.update(scores=a['diagnostic_eight_stage_scores'],stages=['SOURCE','POR113','POR151','PERM154','SW104','POR341conditional','PERM336cross','SW351aligned'],gain_vs_prior341_recipe=a['TOTAL_vs_retained341'],gain_vs_default_five=a['diagnostic_eight_stage_scores']['TOTAL']-80.61247541799393,SW_continuous=a['strata']['continuous'],selection_reason='TerminalSWwhole+CONT > historical326 and samecapacity permuted control; fresh jointencoder/readout/spectrum code original104only, oneSWphase, not trainedrepairchain')
 q['heads']=dict(old['heads']);ck=R/'outputs/pure_dl_sw_dynamic_cross_spectrum_v351_fold5/aligned/model_best.pt';q['heads']['SW']=dict(checkpoint=str(ck),sha256=hashlib.sha256(ck.read_bytes()).hexdigest())
 p=S/'eight_stage_pred.npy';p.write_bytes((S/'aligned_diagnostic_eight_stage_pred.npy').read_bytes());q['pred_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
 q['strict_808_qualified']=q['scores']['TOTAL']>80.8;q['single_seed_small_gain_not_robustness']=True;q['verification']='Actual144665 fixed whole-target best columns/terminalRAMbestreplay/frozenexact, samecapacity pair; notindependentdiskreload/fullOOF'
 assert q['total_training_stages']==8 and all(v<=1 for v in q['added_head_stages'].values())
 (S/'CURRENT_EIGHT_STAGE_RECIPE.json').write_text(json.dumps(q,indent=2),encoding='utf-8')
text='\nV351 pairterminal aligned-minus-permuted whole'+str(f['aligned_minus_permuted_whole'])+'/CONT'+str(f['aligned_minus_permuted_CONT'])+', mechanismpositive='+str(f['mechanism_positive'])+', adopted='+str(f['adopted'])+'. Candidate actual341/336/351TOTAL'+str(a['diagnostic_eight_stage_scores']['TOTAL'])+'/delta'+str(a['TOTAL_vs_retained341'])+'; eightphases/oneadditionperhead/default5 andoldrecipes/deliveriesunchanged. Pairwall'+str(f['pair_wall_process_seconds'])+'s inclserialgap/stagger'+str(f['launch_stagger_seconds'])+'s,individual/peaks inarms.144665RAMreplay/frozenexact/notdiskreload/singleseed, strict808false/fullOOF82unmet/nootherfolds. '+f['closed_setting']+'\n'
for p in (S/'README.md',S/'PROJECTS.md',R/'PROJECTS.md',R.parent/'PROJECTS.md'):
 with p.open('a',encoding='utf-8') as h:h.write(text)
print(json.dumps({k:v for k,v in f.items() if k!='arms'}))
