"""Summarize completed scaling, score actual arrays, and retain only real gains.

Full disk inference is repeated only for a newly improved candidate. Each
trainer already restored/replayed its own selected checkpoint on every row.
"""
from pathlib import Path
from datetime import datetime
import hashlib,json,statistics,subprocess,sys,time
import numpy as np
import torch
from champion_scaling_v21 import ROOT,STUDY,REFERENCE,MODES,make_model,load_data
from launch_latent_gate_v17 import active_jobs,append_once
from train_dual8222_heads80 import atomic_json,atomic_array
import train_invariant as base

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def main():
 sys.stdout.reconfigure(encoding='utf-8')
 if active_jobs():raise RuntimeError('Only finalize after all scheduled training has completed')
 subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).with_name('report_scaling_v21.py')),'--record'],check=True,stdout=subprocess.DEVNULL)
 progress=read(STUDY/'progress.json');assert progress['all_complete']
 reference=np.load(REFERENCE/'fold_5_pred.npy')
 with np.load(REFERENCE/'oof.npz') as z:
  truth=z['truth'][z['fold']==4].copy();wells=z['well'][z['fold']==4].copy()
 assert truth.shape==reference.shape==(144665,3)
 ref=base.metric(truth,reference);candidates=[];counts={};launch_times=[];finish_times=[]
 for row in progress['rows']:
  out=ROOT/'outputs'/row['id'];cfg=read(out/'config.json');report=read(out/'fold_5_report.json')
  assert report['all144665target_predictions_restored_exactly'] and report['frozen_packed_rows_unchanged']
  for name,sha in cfg['source_hashes'].items():
   assert hashlib.sha256((ROOT/'experiments/pure_dl'/name).read_bytes()).hexdigest()==sha, name
  pred=np.load(out/'channel_pred.npy');assert pred.shape==(144665,) and np.isfinite(pred).all()
  joint=reference.copy();joint[:,0]=pred
  score=base.metric(truth,joint);assert abs(score['TOTAL']-report['reference_replacement_total'])<1e-9
  assert np.array_equal(joint[:,1:],reference[:,1:])
  if report['best_epoch']==0:assert np.array_equal(joint,reference)
  per_well=[]
  for w in np.unique(wells):
   mask=wells==w;a=base.metric(truth[mask],reference[mask]);b=base.metric(truth[mask],joint[mask])
   per_well.append(dict(well=int(w),rows=int(mask.sum()),delta_TOTAL=b['TOTAL']-a['TOTAL']))
  atomic_array(STUDY/('assembled_'+row['mode']+'_pred.npy'),joint)
  post=row['best_after_update'];launch=read(out/'launch.json')
  launch_times.append(datetime.fromisoformat(launch['started_at']));finish_times.append(datetime.fromisoformat(report['finished_at']))
  cp=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False)
  c=dict(mode=row['mode'],parameters=row['parameters'],added_parameters=row['added_parameters'],
   location_hidden_width=row['location_hidden_width'],scores=score,delta_champion_TOTAL=score['TOTAL']-ref['TOTAL'],
   best_epoch=report['best_epoch'],stopped_epoch=report['stopped_epoch'],
   best_after_update_epoch=post['epoch'],best_after_update_POR=post['target_score'],best_after_update_TOTAL=post['reference_replacement_total'],
   after_update_delta_champion=post['reference_replacement_total']-ref['TOTAL'],
   per_epoch_seconds_median=row['iteration_seconds_median'],run_seconds=report['seconds'],peak_allocated_MiB=report['peak_gpu_allocated_mib'],
   last_backbone_change_L2=report['last_backbone_parameter_change_L2'],selected_backbone_change_L2=report['selected_backbone_parameter_change_L2'],
   checkpoint_bytes=(out/'model_best.pt').stat().st_size,initial_predictions_exact=bool(row['initial']['different_target_elements_vs_shared_reference']==0),
   restored_predictions_exact=True,PERM_SW_arrays_exactly_unchanged=True,
   fresh_final_disk_inference=False,checkpoint_load_confirmed=True,
   checkpoint=f'outputs/{row["id"]}/model_best.pt',per_well=per_well)
  assert abs(cp['target_score']-score['POR'])<1e-8
  candidates.append(c);del cp
 best=max(candidates,key=lambda c:c['scores']['TOTAL'])
 adopted=best['scores']['TOTAL']>ref['TOTAL']+1e-9
 if adopted:
  from train_independent_channels_v8 import Evaluator
  torch.set_num_threads(3);torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
  data,master=load_data();model=make_model(best['mode'],master).cuda()
  cp=torch.load(ROOT/best['checkpoint'],map_location='cpu',weights_only=False);model.load_state_dict(cp['state_dict'],strict=True)
  replay,score,total=Evaluator(data,master,0,reference)(model)
  expected=np.load(ROOT/'outputs'/f'pure_dl_champion_scaling_v21_{best["mode"]}_fold5/channel_pred.npy')
  assert np.array_equal(replay,expected)
  best['fresh_final_disk_inference']=True
  del model,cp,data,master
 old_recipe=read(ROOT/'results/deep_learning/feature_structure_v20_20260928/CURRENT_RECIPE.json')
 choices=[dict(c) for c in old_recipe['choices']]
 if adopted:
  for c in choices:
   if c['target']=='POR':c.update(key='POR_v21_'+best['mode'],architecture=best['mode'],checkpoint=best['checkpoint'],
    model_module='champion_scaling_v21',builder='make_model',target_score=best['scores']['POR'],best_epoch=best['best_epoch'])
 selected=best['scores'] if adopted else ref
 contrasts=[];control=candidates[0]
 for c in candidates:
  contrasts.append(dict(mode=c['mode'],parameter_ratio=c['parameters']/control['parameters'],
   selected_TOTAL_delta_vs_control=c['scores']['TOTAL']-control['scores']['TOTAL'],
   after_update_TOTAL_delta_vs_control=c['best_after_update_TOTAL']-control['best_after_update_TOTAL']))
 timing=dict(execution='first two concurrent; third explicitly launched when slot available',maximum_jobs=2,
  first_launch_to_last_finish_seconds=(max(finish_times)-min(launch_times)).total_seconds(),
  per_run_seconds={c['mode']:c['run_seconds'] for c in candidates},
  epoch_seconds_median={c['mode']:c['per_epoch_seconds_median'] for c in candidates},
  preparation_seconds=read(STUDY/'data_manifest.json')['preparation_seconds'],
  caution='Epoch times include64updates,validation,checkpoint I/O; workloads overlap differently, not controlled throughput benchmarking. Wall includes explicit launch gaps, excludes code and prior data preparation.')
 assembled=dict(fold=5,rows=144665,source_scores=ref,candidates=candidates,comparisons=contrasts,selected_scores=selected,
  selected_new_POR=adopted,fixed_other_predictions_reused=True,all_trainers_replayed_selected_predictions=True,
  trained_sources_unchanged=True,no_other_folds=True,complete_five_folds=False,timing=timing,
  finished_at=datetime.now().astimezone().isoformat())
 atomic_json(STUDY/'assembled_report.json',assembled)
 recipe=dict(scores=selected,choices=choices,selected_new_POR=adopted,best_new_candidate=best['mode'],
  original_fifth_development_selection=True,remaining_folds_started=False,complete_five_folds=False,old_delivery_unchanged=True)
 atomic_json(STUDY/'CURRENT_RECIPE.json',recipe);atomic_json(STUDY/'timing.json',timing)
 gain=read(STUDY/'gain_attribution.json')
 lines=['# v21：80.274135收益分析与内部宽度Scaling','',f"完成时间：{assembled['finished_at']}",'',
 '## 收益来源','',f"原双专家{gain['source_scores']['TOTAL']:.12f} → 当前冠军{gain['champion_scores']['TOTAL']:.12f}，总增量{gain['total_gain']:+.12f}。",'',
 '|通道|对总分贡献|占增量|','|---|---:|---:|']
 for c in gain['contributions']:lines.append(f"|{c['target']}|{c['TOTAL_contribution']:+.9f}|{c['fraction_of_total_gain']:.2%}|")
 lines+=['','POR的收益来自密度/位置读出微调和小型残差，主要改善非默认数值段。v5非线性相对同起点线性对照只有约0.002802TOTAL，推理时仅移除已训练残差下降约0.000881；后者不是重训的结构效应。v8三个私有副本保留同分，复制模型本身无精度收益。详细阶段链和子集分解见gain_attribution.json。','',
 '## 本轮扩容范围','',
 '以实际保留POR私有双专家为基础，18个残差块的内部非线性宽度1/2/4倍，POR位置残差32/64/128维。外部阶段通道、骨干深度、注意力头数和视图/路由固定。所有POR参数可训练，固定PERM/SW。新增输出投影为零，三规模初始144665点全部精确匹配冠军。',
 '这是内部容量热启动扩容，不是全网所有通道翻倍/从零预训练/数据规模律。两个扩容子部位共同变化，不能分别断言其收益。方法参考函数保持网络增长：Chen et al., Net2Net, arXiv:1511.05641；具体实现采用零输出额外单元。',
 '原64/16井、原九曲线206特征、精确冠军预处理与固定Q50、原POR似然、80/18/max2、骨干1e-5/头3e-5、cosine到1e-6完全保留，优化器重置。', '',
 '## 实测结果（排除第0轮的训练后最高分）','',
 '|规模|POR模型参数|训练后最高POR|对应TOTAL|轮次|最终保留TOTAL|保留轮次|','|---|---:|---:|---:|---:|---:|---:|']
 for c in candidates:lines.append(f"|{c['mode']}|{c['parameters']}|{c['best_after_update_POR']:.9f}|{c['best_after_update_TOTAL']:.9f}|{c['best_after_update_epoch']}|{c['scores']['TOTAL']:.9f}|{c['best_epoch']}|")
 lines+=['',f"采纳新POR={adopted}；最终配方TOTAL={selected['TOTAL']:.12f}。保留第0轮代表回退原功能，不是扩容精度收益。",'',
 '## 开销','', '|规模|单轮中位秒数|单项报告秒数|峰值分配显存MiB|实际完成轮次|','|---|---:|---:|---:|---:|']
 for c in candidates:lines.append(f"|{c['mode']}|{c['per_epoch_seconds_median']:.3f}|{c['run_seconds']:.3f}|{c['peak_allocated_MiB']:.2f}|{c['stopped_epoch']}|")
 lines+=['',f"三项从首次显式启动至最后完成墙钟{timing['first_launch_to_last_finish_seconds']:.3f}秒，数据准备{timing['preparation_seconds']:.3f}秒另计。不同候选并发重叠不同，轮耗时不是严格同GPU占用的吞吐对照，不宣称加速比。",'',
 '## 验证与结论边界','',
 '各训练器均完成选中checkpoint的全量恢复验证；本次从磁盘加载所有选中checkpoint确认记录，再将实际保存POR预测与未变PERM/SW数组组合调用原metric。新冠军若存在才额外独立重建GPU全量推理，失败候选不重复全量推理。所有训练源指纹保持一致。',
 '第一批反向确认新增通路输出投影有梯度，最终训练骨干有实际变化。增加参数不保证验证收益；若连base1继续训练也退步，应优先研究热启动的泛化/更新范围，而非把负结果归为扩容初始化失败。',
 '本设置、学习率、单种子和固定开发折的结论不能推出任何更大模型都无效；整网宽度/深度、PERM/SW扩容、从零训练大模型和完整五折尚未测。本批不自动追加或补折，旧交付不改。',
 '下一步未执行：在冻结旧表示或更受约束的更新范围下，对新增容量单独训练作匹配对照；先解决继续训练造成的验证退步，再决定更大的模型预算。',
 '', '## 文件','', 'GAIN_ANALYSIS.md/gain_attribution.json为收益分析；METHOD.md为预定义实验；各outputs下config/history/report为实测；本目录assembled_report.json和CURRENT_RECIPE.json保存最终配方。']
 (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 marker='## v21冠军收益与扩容实验完成'
 note=marker+'\n'+datetime.now().astimezone().isoformat()+f"。1/2/4内部宽度三项训练/最佳恢复/实际预测组合均完成；采纳新POR={adopted}，保留TOTAL={selected['TOTAL']:.9f}。已完成收益来源/参数和开销记录；整网统一拓宽、其他目标扩容、其他折均未执行。证据champion_scaling_v21_20260928/FINAL.md。"
 for p in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):
  append_once(p,marker,note)
 for c in candidates:
  out=ROOT/'outputs'/f'pure_dl_champion_scaling_v21_{c["mode"]}_fold5'
  text=marker+f"\n更新后最高TOTAL={c['best_after_update_TOTAL']:.9f}@{c['best_after_update_epoch']}，最终保留={c['scores']['TOTAL']:.9f}@{c['best_epoch']}；完成{c['stopped_epoch']}轮。没有自动新实验或补折；解释和范围见研究FINAL.md。"
  for name in ('README.md','PROJECTS.md'):append_once(out/name,marker,text)
 graph_path=ROOT/'docs/deep_learning/atlas/curation.json';graph=read(graph_path)
 if not adopted:
  deadid='dead:champion_scaling_v21_unrestricted_warm'
  graph['deadends']=[r for r in graph.get('deadends',[]) if r.get('id')!=deadid]+[dict(id=deadid,title='v21冠军热启动内部扩容：本设置未增益',stage='architecture',
   parents=[r['id'] for r in progress['rows']],summary='1x/2x/4x从同80.274135起点，比较内部容量；保留原配方。',
   reason='只关闭当前全量微调/学习率/内部扩容组合；不否定其他更新范围或整网扩容。')]
  graph_path.write_text(json.dumps(graph,ensure_ascii=False,indent=2),encoding='utf-8')
 progress['assembly_complete']=True;(STUDY/'progress.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2),encoding='utf-8')
 print('FINAL_RESULT',json.dumps({k:v for k,v in assembled.items() if k!='candidates'},ensure_ascii=False))
 print('CANDIDATES',json.dumps([{k:v for k,v in c.items() if k!='per_well'} for c in candidates],ensure_ascii=False))
 print('ACTIVE',active_jobs());print('FINAL_REPORT',STUDY/'FINAL.md')

if __name__=='__main__':main()
