"""CPU evidence and atlas extension for the three predefined location readouts.
No fitting, training launch, or implicit selection by individual rows/wells.
"""
from datetime import datetime
from pathlib import Path
import argparse, json, sys
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/dual8222_location_v5_20260928'
ATLAS=ROOT/'docs/deep_learning/atlas'
PARENT='pure_dl_dual8222_v4_por_mean_only80_fold5'
SOURCE='pure_dl_paired_local_context_router80_sequence'
MODES=['linear','point32','local32']
PREFIX='pure_dl_dual8222_v5_location_'
NAMES={'linear':'线性位置继续训练','point32':'受约束逐点非线性位置头','local32':'受约束局部非线性位置头'}


def read(path,default=None):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else default


def atomic(path,obj):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


def section(path,body):
    marker='## 本批最新结果'
    old=path.read_text(encoding='utf-8-sig') if path.exists() else ''
    path.write_text(old.split(marker)[0].rstrip()+'\n\n'+marker+'\n'+body+'\n',encoding='utf-8')


def kernels(y,p):
    return np.c_[np.maximum(0,1-abs(p[:,0]-y[:,0])/(.08*(abs(y[:,0])+1e-5))),
                 np.maximum(0,1-abs(np.log10(np.maximum(p[:,1]/np.maximum(y[:,1],1e-5),1e-5)))),
                 np.maximum(0,1-abs(p[:,2]-y[:,2])/(.05*(abs(y[:,2])+1e-5)))]


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--record',action='store_true');args=ap.parse_args()
    now=datetime.now().astimezone().isoformat(timespec='seconds')
    baseline=read(ROOT/'outputs'/PARENT/'fold_5_report.json')
    reference=np.load(ROOT/'outputs'/PARENT/'fold_5_pred.npy')
    with np.load(ROOT/'outputs'/SOURCE/'oof.npz') as z:
        pick=z['fold']==4;y=z['truth'][pick];wells=z['well'][pick]
    assert y.shape==reference.shape==(144665,3)
    u0=kernels(y,reference);weight=np.array([.3,.35,.35])
    assert abs(u0.mean(0)@weight*100-baseline['scores']['TOTAL'])<1e-10
    data=[];utilities={};curated=read(ATLAS/'curation.json');timings=[]
    for mode in MODES:
        run_id=PREFIX+mode+'80_fold5';folder=ROOT/'outputs'/run_id
        status=read(folder/'runtime_status.json',{});final=read(folder/'fold_5_report.json')
        hist=read(folder/'fold_5_validation_history.json',[]);last=hist[-1] if hist else {}
        initial=read(folder/'initial_verification.json',{});cfg=read(folder/'config.json',{})
        alive=False
        try:
            p=psutil.Process(status.get('pid',-1))
            alive=p.is_running() and 'train_dual8222_location_v5.py' in ' '.join(p.cmdline())
        except (ValueError,psutil.NoSuchProcess,psutil.AccessDenied):pass
        log=(folder/'train.log').read_text(encoding='utf-8',errors='replace') if (folder/'train.log').exists() else ''
        row={'mode':mode,'id':run_id,'status':status,'process_alive':alive,'final_report':final,
             'latest':last,'initial_verification':initial,'failure':read(folder/'failure.json'),
             'gradient_scope_verified':'GRADIENT_SCOPE_VERIFIED' in log,
             'trainable_parameters':cfg.get('trainable_parameters'),
             'checkpoint_reload_verification':read(folder/'checkpoint_reload_verification.json')}
        value=final['scores']['TOTAL'] if final else last.get('best_score')
        row['TOTAL']=value
        if final:
            pred=np.load(folder/'fold_5_pred.npy');u=kernels(y,pred);utilities[mode]=u
            assert np.array_equal(pred[:,1:],reference[:,1:])
            assert abs(u.mean(0)@weight*100-value)<1e-10
            difference=(u-u0)*100;total=difference@weight
            by_well=[{'well_index':int(w),'n':int((wells==w).sum()),'TOTAL_delta':float(total[wells==w].mean())} for w in np.unique(wells)]
            population={}
            for label,m in [('true_default',y[:,0]<=.1),('true_nondefault',y[:,0]>.1)]:
                population[label]={'n':int(m.sum()),'baseline_POR':float(u0[m,0].mean()*100),
                                  'candidate_POR':float(u[m,0].mean()*100),'TOTAL_delta_contribution':float((difference[:,0]*m).mean()*.3)}
            row['diagnostic']={'delta_baseline':float(total.mean()),'per_well':by_well,'populations':population,
                               'well_wins':sum(w['TOTAL_delta']>1e-12 for w in by_well),
                               'well_losses':sum(w['TOTAL_delta']< -1e-12 for w in by_well)}
            launch=read(folder/'launch.json')
            timings.append((datetime.fromisoformat(launch['started_at']),datetime.fromisoformat(final['finished_at'])))
        state='完成并恢复' if final else status.get('status','planned')
        conclusion=state+'；TOTAL='+str(value)+'；最新epoch='+str(last.get('epoch',0))
        if value is not None:conclusion+='；相对80.268735 '+f"{value-baseline['scores']['TOTAL']:+.9f}"
        if not final:conclusion+='，不是最终成绩'
        if args.record:
            body=now+'\n'+conclusion+'。\n初始预测差异元素数='+str(initial.get('different_elements'))+'；实际首批梯度验证='+str(row['gradient_scope_verified'])+'。\n'
            body+='本批目标：验证相对线性对照的结构收益。未证明独立泛化，无补折或旧交付替换；续跑等价性未实测。\n'
            for name in ['README.md','PROJECTS.md']:section(folder/name,body)
            curated['nodes'][run_id]={'title':NAMES[mode],'summary':'同80.268735暖启动，POR线性位置可训练，其他分布因子固定；'+NAMES[mode],
                                    'conclusion':conclusion+'。重复开发第五折，未补折。','stage':'loss','highlight':True}
        data.append(row)
    all_complete=all(r['final_report'] is not None for r in data)
    pairs={}
    for a,b in [('point32','linear'),('local32','linear'),('local32','point32')]:
        if a in utilities and b in utilities:
            delta=(utilities[a]-utilities[b])*100@weight
            wd=np.array([delta[wells==w].mean() for w in np.unique(wells)])
            ns=np.array([(wells==w).sum() for w in np.unique(wells)])
            draws=np.random.default_rng(20260928).integers(0,len(ns),(10000,len(ns)))
            boots=(ns[draws]*wd[draws]).sum(1)/ns[draws].sum(1)
            pairs[a+'_minus_'+b]={'TOTAL_delta':float(delta.mean()),'well_wins':int((wd>1e-12).sum()),
                                 'well_losses':int((wd< -1e-12).sum()),'exploratory_95_interval':np.quantile(boots,[.025,.975]).tolist()}
    best={'mode':'previous','TOTAL':baseline['scores']['TOTAL'],'delta':0.}
    for r in data:
        if r['final_report'] and r['TOTAL']>best['TOTAL']+1e-10:
            best={'mode':r['mode'],'TOTAL':r['TOTAL'],'delta':r['TOTAL']-baseline['scores']['TOTAL']}
    timing=None
    if all_complete:
        events=sorted([(a,1) for a,b in timings]+[(b,-1) for a,b in timings],key=lambda t:(t[0],t[1]))
        count=max_count=0
        for _,change in events:count+=change;max_count=max(max_count,count)
        timing={'batch_span_seconds':(max(b for a,b in timings)-min(a for a,b in timings)).total_seconds(),
                'maximum_concurrent_training_processes':max_count,'scope':'includes loading, training, validation, replay and explicit launch gaps'}
    payload={'observed_at':now,'parent':PARENT,'baseline_scores':baseline['scores'],'runs':data,
             'all_complete':all_complete,'best_completed':best,'pairwise':pairs,'timing':timing,
             'limitations':'Fifth fold repeatedly used for selection; bootstrap is exploratory and selection-unadjusted; only3predeclared arms; no fullOOF claim.'}
    atomic(STUDY/'progress.json',payload)
    if args.record:
        edges=[(PARENT,PREFIX+'linear80_fold5','evolution'),(PREFIX+'linear80_fold5',PREFIX+'point3280_fold5','control'),
               (PREFIX+'point3280_fold5',PREFIX+'local3280_fold5','control')]
        eps={(a,b) for a,b,t in edges}
        curated['edges']=[e for e in curated['edges'] if (e['source'],e['target']) not in eps]
        curated['edges'] += [{'source':a,'target':b,'type':t,'note':'同起点、原评分及线性位置更新；按预定义新增支路作匹配比较。','evidence':'outputs/'+b+'/README.md'} for a,b,t in edges]
        atomic(ATLAS/'curation.json',curated)
        lines=['# 双专家位置读出v5结果','',now,'','|方案|状态|参数|TOTAL|相对80.268735|最佳新增轮次|','|---|---|---:|---:|---:|---:|']
        for r in data:
            value=r['TOTAL'];f=r['final_report'];st='完成并恢复' if f else r['status'].get('status','planned')
            lines.append('|'+r['mode']+'|'+st+'|'+str(r['trainable_parameters'])+'|'+(f'{value:.9f}' if value is not None else '尚无')+'|'+(f"{value-baseline['scores']['TOTAL']:+.9f}" if value is not None else '—')+'|'+str(f['best_epoch'] if f else r['latest'].get('best_epoch','—'))+'|')
        lines+=['','## 匹配结构收益',json.dumps(pairs,ensure_ascii=False,indent=2),'','## 当前判断',
                '已完成最佳：'+str(best),
                '三组均重新初始化优化器，从同一权重开始；只有新增支路受到0.25固定分量标准差的幅值约束。线性位置更新本身不限幅。',
                'point32/local32基础初始化一致，后者增加384个卷积参数；不是等参数比较。原评分/输入/Q/PERM/SW冻结。',
                '相对起点的改善需扣除linear继续训练效果，才讨论新结构增量。每个最终报告都来自实际恢复，而非日志最大值拼接。',
                '所有井使用相同已选配置。反复开发的第五折与探索性重采样不构成独立泛化证据，完整OOF>82未完成。',
                '未启动额外核长、宽度、学习率或阈值扫描；旧ABD、其他折与交付均不变。']
        if not all_complete:lines+=['','尚未全部完成，不将当前候选宣布为最终收益。']
        if timing:lines+=['','## 时间范围',json.dumps(timing,ensure_ascii=False)]
        (STUDY/('FINAL.md' if all_complete else 'SNAPSHOT.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
        reloaded=any(r.get('checkpoint_reload_verification') for r in data)
        pending=('剩余：独立泛化及中断续跑等价性尚未验证；本批不再增加候选。' if all_complete and reloaded else '待办：未完成实验的最终恢复和获益读出的保存重载验证。')
        body='已完成：代码、明确启动、实际状态记录；全部完成='+str(all_complete)+'，已验证保存重载='+str(reloaded)+'。\n当前最佳='+str(best)+'。\n'+pending+'\n目标：第五折>80.5且明确收益，完整OOF>82未完成。达成后固定方案做泛化确认，当前无补折授权。\n'
        section(STUDY/'PROJECTS.md',body)
        for p in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',ATLAS/'PROJECTS.md']:
            with p.open('a',encoding='utf-8') as f:f.write('\n\n## '+now+' 位置读出v5\n'+body)
    short={**payload,'runs':[{k:v for k,v in r.items() if k not in ['diagnostic','final_report']} for r in data]}
    for r,full in zip(short['runs'],data):
        r['final_scores']=full['final_report']['scores'] if full['final_report'] else None
        if full.get('diagnostic'):r['diagnostic']={k:v for k,v in full['diagnostic'].items() if k!='per_well'}
    print(json.dumps(short,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
