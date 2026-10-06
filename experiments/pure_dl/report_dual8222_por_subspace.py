"""CPU-only readout study snapshots, final score decomposition and atlas annotation.

No training launch, parameter fitting, target/row-specific cherry-picking, or old
report modification. Read completed reports before calling a result final.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/dual8222_por_subspace_20260928'
ATLAS=ROOT/'docs/deep_learning/atlas'
PARENT='pure_dl_dual8222_selected_round2_fold5'
SOURCE='pure_dl_paired_local_context_router80_sequence'
MODES=['mean_only','mean_scale','density_all']
PREFIX='pure_dl_dual8222_v4_por_'
TITLES={'mean_only':'POR·只更新分量位置','mean_scale':'POR·更新位置与尺度','density_all':'POR·全密度更新对照'}


def read(path,default=None):
    if not path.exists():return default
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path,value):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


def kernels(y,p):
    return np.c_[np.maximum(0,1-abs(p[:,0]-y[:,0])/(.08*(abs(y[:,0])+1e-5))),
                 np.maximum(0,1-abs(np.log10(np.maximum(p[:,1]/np.maximum(y[:,1],1e-5),1e-5)))),
                 np.maximum(0,1-abs(p[:,2]-y[:,2])/(.05*(abs(y[:,2])+1e-5)))]


def append_section(path,marker,text):
    current=path.read_text(encoding='utf-8-sig') if path.exists() else ''
    path.write_text(current.split(marker)[0].rstrip()+'\n\n'+marker+'\n'+text+'\n',encoding='utf-8')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--record',action='store_true');args=ap.parse_args()
    observed=datetime.now().astimezone().isoformat(timespec='seconds')
    parent=read(ROOT/'outputs'/PARENT/'fold_5_report.json')
    reference=np.load(ROOT/'outputs'/PARENT/'fold_5_pred.npy')
    with np.load(ROOT/'outputs'/SOURCE/'oof.npz') as z:
        mask=z['fold']==4;y=z['truth'][mask];wells=z['well'][mask]
    assert y.shape==reference.shape==(144665,3)
    reference_utility=kernels(y,reference);weights=np.array([.3,.35,.35])
    assert abs(reference_utility.mean(0)@weights*100-parent['scores']['TOTAL'])<1e-10
    records=[];predictions={};total_updates={};curated=read(ATLAS/'curation.json')
    for mode in MODES:
        run_id=PREFIX+mode+'80_fold5';folder=ROOT/'outputs'/run_id
        runtime=read(folder/'runtime_status.json',{'status':'planned'})
        final=read(folder/'fold_5_report.json');cfg=read(folder/'config.json',{})
        history=read(folder/'fold_5_validation_history.json',[]);last=history[-1] if history else {}
        initial=read(folder/'initial_verification.json',{})
        log=(folder/'train.log').read_text(encoding='utf-8',errors='replace') if (folder/'train.log').exists() else ''
        alive=False
        try:
            p=psutil.Process(runtime.get('pid',-1))
            alive=p.is_running() and 'train_dual8222_por_subspace80.py' in ' '.join(p.cmdline())
        except (ValueError,psutil.NoSuchProcess,psutil.AccessDenied):pass
        row={'mode':mode,'id':run_id,'runtime':runtime,'process_alive':alive,'complete':final is not None,
             'last_completed_epoch':last.get('epoch',0),'latest':last,'initial_verification':initial,
             'gradient_scope_verified':'GRADIENT_SCOPE_VERIFIED' in log,'trainable_parameters':cfg.get('trainable_parameters'),
             'final_report':final,'failure':read(folder/'failure.json'),'last_state_exists':(folder/'last_state.pt').exists()}
        if final:
            pred=np.load(folder/'fold_5_pred.npy');u=kernels(y,pred);change=(u-reference_utility)*100
            score=u.mean(0)*100
            assert abs(score@weights-final['scores']['TOTAL'])<1e-9
            assert np.array_equal(pred[:,1:],reference[:,1:])
            total=change@weights;predictions[mode]=u;total_updates[mode]=total
            state=y[:,0]<=.1
            groups={}
            for label,select in [('true_default',state),('true_nondefault',~state)]:
                groups[label]={'n':int(select.sum()),'baseline_POR_score':float(reference_utility[select,0].mean()*100),
                    'candidate_POR_score':float(u[select,0].mean()*100),
                    'contribution_to_TOTAL_delta':float(np.mean(change[:,0]*select)*.3)}
            perwell=[{'well_index':int(w),'n':int((wells==w).sum()),'TOTAL_delta':float(total[wells==w].mean()),
                      'TOTAL_contribution':float(total[wells==w].sum()/len(total))} for w in np.unique(wells)]
            row['diagnostic']={'recomputed_TOTAL':float(score@weights),'delta_initial':float(total.mean()),
               'POR_delta':float(change[:,0].mean()),'well_wins':sum(a['TOTAL_delta']>1e-12 for a in perwell),
               'well_losses':sum(a['TOTAL_delta']< -1e-12 for a in perwell),'unchanged_wells':sum(abs(a['TOTAL_delta'])<=1e-12 for a in perwell),
               'populations':groups,'per_well':perwell}
            score_text=f"完成{final['scores']['TOTAL']:.9f}，相对80.257639 {final['delta_initial']:+.9f}；最佳epoch{final['best_epoch']}，停止epoch{final['stopped_epoch']}"
        elif last:
            score_text=f"{runtime.get('status')}，epoch{last['epoch']}，观测最佳{last['best_score']:.9f}，未最终恢复"
        else:score_text=runtime.get('status','planned')+'；暂无训练评分'
        records.append(row)
        if args.record:
            text='观察时间：'+observed+'。'+score_text+'。\n'
            text+='初始差异元素数='+str(initial.get('different_elements'))+'；首批梯度范围通过='+str(row['gradient_scope_verified'])+'；完整训练状态已保存='+str(row['last_state_exists'])+'。\n'
            text+='目标仍是第五折>80.5且明确收益、完整OOF>82。未启动旧路线/补折，也未验证中断续跑等价性。\n'
            for name in ['README.md','PROJECTS.md']:append_section(folder/name,'## 本批实际结果',text)
            curated['nodes'][run_id]={'title':TITLES[mode],'summary':'由已验证80.257638头配方热启动，PERM/SW和两骨干固定，仅POR的'+mode+'参数更新。',
                'conclusion':score_text+'。同原单覆盖评分；开发集结果，不补折。','stage':'loss','highlight':True}
    complete=all(row['complete'] for row in records)
    pairwise={}
    if complete:
        for mode in ['mean_only','mean_scale']:
            difference=(predictions[mode]-predictions['density_all'])*100@weights
            wins=[float(difference[wells==w].mean()) for w in np.unique(wells)]
            pairwise[mode+'_minus_density_all']={'TOTAL_delta':float(difference.mean()),
                'well_wins':sum(v>1e-12 for v in wins),'well_losses':sum(v< -1e-12 for v in wins)}
    best={'mode':'previous_fixed_recipe','TOTAL':parent['scores']['TOTAL'],'delta':0.0}
    for row in records:
        if row['complete'] and row['final_report']['scores']['TOTAL']>best['TOTAL']+1e-10:
            best={'mode':row['mode'],'TOTAL':row['final_report']['scores']['TOTAL'],'delta':row['final_report']['delta_initial']}
    payload={'observed_at':observed,'parent':PARENT,'baseline_scores':parent['scores'],'runs':records,'all_complete':complete,
             'pairwise':pairwise,'best_completed':best,'remaining_folds_started':False,
             'limitations':'same repeatedly selected fifth fold; three predeclared warm-start interventions, not independent validation; no resume-equivalence test'}
    write_json(STUDY/'progress.json',payload)
    if args.record:
        edge_list=[{'source':PARENT,'target':PREFIX+mode+'80_fold5','type':'evolution',
                    'note':'固定80.257638读出热启动；保留PERM/SW，只更新POR预定义子空间。',
                    'evidence':'outputs/'+PREFIX+mode+'80_fold5/README.md'} for mode in MODES]
        for mode in ['mean_only','mean_scale']:
            edge_list.append({'source':PREFIX+'density_all80_fold5','target':PREFIX+mode+'80_fold5','type':'control',
                              'note':'同热启动/数据/损失/预算，限定可更新的密度参数；参数量差异来自更新范围。',
                              'evidence':'outputs/'+PREFIX+mode+'80_fold5/README.md'})
        pairs={(e['source'],e['target']) for e in edge_list}
        curated['edges']=[e for e in curated['edges'] if (e['source'],e['target']) not in pairs]+edge_list
        write_json(ATLAS/'curation.json',curated)
        lines=['# POR分布子空间优化结果','', '观察时间：'+observed,'',
               '三个对照同80.257638624609起点，只训练POR；所有其他目标/原评分保持不变。', '',
               '|方案|状态|可训练参数|TOTAL|相对起点|最佳新增轮次|','|---|---|---:|---:|---:|---:|']
        for row in records:
            report=row['final_report'];last=row['latest']
            score=report['scores']['TOTAL'] if report else last.get('best_score')
            delta=score-parent['scores']['TOTAL'] if score is not None else None
            values=[row['mode'],'完成并恢复' if report else row['runtime'].get('status','planned'),str(row['trainable_parameters']),
                    f'{score:.9f}' if score is not None else '尚无',f'{delta:+.9f}' if delta is not None else '—',
                    str(report['best_epoch'] if report else last.get('best_epoch','—'))]
            lines.append('|'+ '|'.join(values)+'|')
        lines+=['','当前已完成最佳：'+str(best),'',
                '每个完成结果均检查同一组144665点的实际预测与最佳参数恢复；PERM/SW逐元素不变，冻结参数没有受权重衰减影响。',
                '新优化器和学习率调度从头开始，属于匹配热启动新实验，不冒称延续上一轮的原轨迹。']
        if complete:
            lines+=['','## 匹配对照',json.dumps(pairwise,ensure_ascii=False,indent=2),'','## 误差分解']
            for row in records:
                d=row['diagnostic']
                lines.append(f"- {row['mode']}: {d['well_wins']}井改善、{d['well_losses']}井下降、{d['unchanged_wells']}井不变。")
                lines.append(json.dumps(d['populations'],ensure_ascii=False))
            if best['delta']<=1e-10:
                lines+=['','三组最终均未超过原固定配方；保留原80.257638，不新增模型交付。']
            lines+=['','## 后续边界',
                    '本批三个预先限定对照已经结束，不开追加宽度/LR扫描。按结果决定下一条机制；模型结果仍受反复使用第五折选模影响，完整OOF>82未完成。',
                    'epoch0若最佳，应明确为无新增训练收益。任何下一轮都应保持匹配控制，不能将增量预算或更换解码带来的差距归因于子空间。']
        else:lines+=['','尚未全部完成，不据早期结果决定胜负。']
        text='\n'.join(lines)+'\n'
        (STUDY/('FINAL.md' if complete else 'SNAPSHOT.md')).write_text(text,encoding='utf-8')
        note='\n\n## '+observed+' POR子空间结果核对\n'+str([(r['mode'],r['runtime'].get('status'),r['last_completed_epoch']) for r in records])+'\n已完成最佳='+str(best)+'；全部完成='+str(complete)+'。目标完整OOF>82尚未达成，补折暂停，旧交付不改。详见results/deep_learning/dual8222_por_subspace_20260928。\n'
        for path in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md',ATLAS/'PROJECTS.md']:
            with path.open('a',encoding='utf-8') as f:f.write(note)
    compact={**payload,'runs':[{k:v for k,v in row.items() if k not in ['diagnostic','initial_verification','final_report']} for row in records]}
    for short,row in zip(compact['runs'],records):
        short['final_summary']=({k:v for k,v in row['final_report'].items() if k not in ['initial_scores']} if row['final_report'] else None)
        if row.get('diagnostic'):short['diagnostic']={k:v for k,v in row['diagnostic'].items() if k!='per_well'}
    print(json.dumps(compact,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
