"""Prepare exact champion data once and audit the retained gain chain."""
from pathlib import Path
from datetime import datetime
import hashlib,json,sys,time
import numpy as np
import torch
from train_dual8222_heads80 import SOURCE,prepare_data,cpu_copy,atomic_torch,atomic_json,SourceEvaluator
from champion_scaling_v21 import ROOT,STUDY,REFERENCE,PRIVATE,HEADS
from launch_latent_gate_v17 import active_jobs
import train_invariant as base

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def main():
 sys.stdout.reconfigure(encoding='utf-8')
 if active_jobs():raise RuntimeError('Active project trainer; inspect before preparing')
 if (STUDY/'prepared_data.pt').exists():raise RuntimeError('Prepared data immutable; explicit reuse only')
 torch.set_num_threads(3);torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
 start=time.perf_counter();master=torch.load(SOURCE,map_location='cpu',weights_only=False)
 d=prepare_data(master);reference=np.load(REFERENCE/'fold_5_pred.npy')
 truth=d['truth'][d['validation']];wells=d['wells'][d['validation']]
 scores=base.metric(truth,reference)
 assert reference.shape==(144665,3) and abs(scores['TOTAL']-80.27413533779057)<1e-9
 source_report=read(SOURCE.parent/'fold_5_report.json')
 source_pred=None;source_prediction_origin=None
 for name in ('fold_5_pred.npy','fold_5_predictions.npy'):
  p=SOURCE.parent/name
  if p.exists():
   v=np.load(p)
   if v.shape==reference.shape and all(abs(base.metric(truth,v)[k]-source_report['scores'][k])<1e-8 for k in base.TS):
    source_pred=v;source_prediction_origin=str(p.relative_to(ROOT));break
 if source_pred is None:
  from paired_local_context_router_sequence import Network as Original,retained_indices,calibration_indices
  names=master['selected_feature_names'];num,condition=calibration_indices(names)
  m=Original(retain_indices=retained_indices(names),numeric_indices=num,conditioning_indices=condition).cuda()
  m.load_state_dict(master['state_dict'],strict=True);m.eval()
  source_pred=SourceEvaluator(d,master)(m)
  if isinstance(source_pred,tuple):source_pred=source_pred[0]
  source_prediction_origin='fresh original SourceEvaluator replay'
  assert np.asarray(source_pred).shape==reference.shape
  del m
 old=base.metric(truth,source_pred)
 assert all(abs(old[k]-source_report['scores'][k])<1e-8 for k in (*base.TS,'TOTAL'))
 weights={'POR':.30,'PERM':.35,'SW':.35};tol={'POR':.08,'PERM':.30,'SW':.05}
 defaults=base.dmask(truth);contributions=[]
 for c,target in enumerate(base.TS):
  gain=weights[target]*(scores[target]-old[target])
  # PERM uses log-space scoring in the original scorer; exact gain allocation
  # is obtained from the actual full metric on each subset instead of a proxy.
  subsets={}
  for label,mask in (('default',defaults[:,c]),('nondefault',~defaults[:,c])):
   a=base.metric(truth[mask],source_pred[mask])[target];b=base.metric(truth[mask],reference[mask])[target]
   subsets[label]=dict(rows=int(mask.sum()),old_score=a,new_score=b,channel_delta=b-a,TOTAL_contribution=weights[target]*mask.mean()*(b-a))
  contributions.append(dict(target=target,old_score=old[target],new_score=scores[target],channel_delta=scores[target]-old[target],
    TOTAL_contribution=gain,fraction_of_total_gain=gain/(scores['TOTAL']-old['TOTAL']),subsets=subsets,
    changed_predictions=int(np.count_nonzero(source_pred[:,c]!=reference[:,c]))))
 chain_ids=['pure_dl_paired_local_context_router80_sequence','pure_dl_dual8222_v3_density_only80_fold5',
  'pure_dl_dual8222_selected_round2_fold5','pure_dl_dual8222_v4_por_mean_only80_fold5',
  'pure_dl_dual8222_v5_location_linear80_fold5','pure_dl_dual8222_v5_location_point3280_fold5']
 chain=[];previous=None
 for ident in chain_ids:
  report=read(ROOT/'outputs'/ident/'fold_5_report.json');score=report['scores']
  chain.append(dict(id=ident,scores=score,delta_previous=None if previous is None else score['TOTAL']-previous,
     evidence='outputs/'+ident+'/fold_5_report.json'))
  previous=score['TOTAL']
 reload_path=HEADS.parent/'checkpoint_reload_verification.json'
 verification=read(reload_path) if reload_path.exists() else None
 attribution=dict(observed_at=datetime.now().astimezone().isoformat(),source_scores=old,champion_scores=scores,
  total_gain=scores['TOTAL']-old['TOTAL'],contributions=contributions,retained_chain=chain,
  source_prediction_origin=source_prediction_origin,v5_residual_diagnostic=verification,
  caveats=['Chain entries are retained model replacements; not all historical trials are cumulative gains.',
    'v5 linear is a matched parallel control, not a literal parent of point32.',
    'Changing to three private model copies in v8 reproduced the same scores and did not itself add accuracy.',
    'There is no fully matched pretraining/Q/dual-view ablation in this gain decomposition.'])
 atomic_json(STUDY/'gain_attribution.json',attribution)
 per_well=[]
 for w in np.unique(wells):
  mask=wells==w;a=base.metric(truth[mask],source_pred[mask]);b=base.metric(truth[mask],reference[mask])
  per_well.append(dict(well=int(w),rows=int(mask.sum()),old=a,new=b,delta_TOTAL=b['TOTAL']-a['TOTAL']))
 atomic_json(STUDY/'gain_by_well.json',per_well)
 atomic_torch(STUDY/'prepared_data.pt',cpu_copy(d))
 atomic_json(STUDY/'data_manifest.json',dict(created_at=datetime.now().astimezone().isoformat(),
  source=str(SOURCE.relative_to(ROOT)),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
  champion_private=str(PRIVATE.relative_to(ROOT)),champion_sha256=hashlib.sha256(PRIVATE.read_bytes()).hexdigest(),
  heads=str(HEADS.relative_to(ROOT)),head_sha256=hashlib.sha256(HEADS.read_bytes()).hexdigest(),
  quality=d['qinfo'],fit_points=len(d['train_raw']),validation_points=len(d['validation']),features=d['x'].shape[1],
  original_row_order_verified=True,original_preprocessing_rebuilt_and_verified=True,validation_not_fitted=True,
  preparation_seconds=time.perf_counter()-start,cache_bytes=(STUDY/'prepared_data.pt').stat().st_size))
 lines=['# 80.274135收益来源','',f"原模型{old['TOTAL']:.12f}，保留模型{scores['TOTAL']:.12f}，增量{attribution['total_gain']:+.12f}。",'',
 '|通道|原分数|当前分数|对TOTAL贡献|占本次增量|','|---|---:|---:|---:|---:|']
 for c in contributions:lines.append(f"|{c['target']}|{c['old_score']:.9f}|{c['new_score']:.9f}|{c['TOTAL_contribution']:+.9f}|{c['fraction_of_total_gain']:.2%}|")
 lines+=['','保留路线：POR密度读出微调→仅位置微调→位置小残差；PERM仅门控微调；SW保持原头。v5点式残差相对同起点线性对照的新增TOTAL仅约0.002802，不能把整个0.052274都归因于新MLP。','',
 '原80.22包含已训练的双视图校准、多尺度骨干、局部与全局patch注意力、输入路由。这个预训练基础是当前方案的组成，但本次没有将其单因素因果贡献重新训练分离。','',
 'v8独立复制骨干未提高分数；v6/v7复杂注意力未提高当前总分。旧Q与新Q=1冷启动实验不同，不能把后者劣势全归因于架构。','',
 '按类别和井的精确分解见gain_attribution.json与gain_by_well.json；输出差值与训练阶段因果收益分开解释。']
 (STUDY/'GAIN_ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print('GAINS',json.dumps(attribution,ensure_ascii=False))
 print('DATA_READY',STUDY/'prepared_data.pt',time.perf_counter()-start)

if __name__=='__main__':main()
