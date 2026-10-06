"""CPU snapshots and completed-prediction attribution; never launches training."""
from pathlib import Path
from datetime import datetime
import argparse,json,sys
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/dual8222_component_query_v7_20260928'
ATLAS=ROOT/'docs/deep_learning/atlas'
PARENT='pure_dl_dual8222_v5_location_point3280_fold5'
SOURCE='pure_dl_paired_local_context_router80_sequence'
PREFIX='pure_dl_dual8222_v7_query_'
MODES=['center','local']


def read(p,default=None):
    return json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else default


def atomic(p,data):
    temp=p.with_name(p.name+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(p)


def section(p,text):
    marker='## v7实际结果'
    old=p.read_text(encoding='utf-8-sig') if p.exists() else ''
    p.write_text(old.split(marker)[0].rstrip()+'\n\n'+marker+'\n'+text+'\n',encoding='utf-8')


def kernels(y,p):
    return np.c_[np.maximum(0,1-abs(p[:,0]-y[:,0])/(.08*(abs(y[:,0])+1e-5))),
                 np.maximum(0,1-abs(np.log10(np.maximum(p[:,1]/np.maximum(y[:,1],1e-5),1e-5)))),
                 np.maximum(0,1-abs(p[:,2]-y[:,2])/(.05*(abs(y[:,2])+1e-5)))]


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--record',action='store_true');args=ap.parse_args()
    now=datetime.now().astimezone().isoformat(timespec='seconds')
    parent=read(ROOT/'outputs'/PARENT/'fold_5_report.json')
    reference=np.load(ROOT/'outputs'/PARENT/'fold_5_pred.npy')
    with np.load(ROOT/'outputs'/SOURCE/'oof.npz') as z:
        m=z['fold']==4;y=z['truth'][m];wells=z['well'][m]
    assert y.shape==reference.shape==(144665,3)
    weights=np.array([.3,.35,.35]);u0=kernels(y,reference)
    assert abs(u0.mean(0)@weights*100-parent['scores']['TOTAL'])<1e-9
    cur=read(ATLAS/'curation.json');records=[];utilities={};intervals=[]
    for mode in MODES:
        folder=ROOT/'outputs'/(PREFIX+mode+'80_fold5')
        status=read(folder/'runtime_status.json',{})
        final=read(folder/'fold_5_report.json');history=read(folder/'fold_5_validation_history.json',[])
        cfg=read(folder/'config.json',{});initial=read(folder/'initial_verification.json',{})
        alive=False
        try:
            p=psutil.Process(status.get('pid',-1))
            alive=p.is_running() and 'train_dual8222_query_v7.py' in ' '.join(p.cmdline())
        except (psutil.NoSuchProcess,psutil.AccessDenied,ValueError):pass
        log=(folder/'train.log').read_text(encoding='utf-8',errors='replace') if (folder/'train.log').exists() else ''
        row={'mode':mode,'id':folder.name,'status':status,'process_alive':alive,'final_report':final,
             'latest':history[-1] if history else {},'initial_verification':initial,'failure':read(folder/'failure.json'),
             'trainable_parameters':cfg.get('trainable_parameters'),'gradient_scope_verified':'GRADIENT_SCOPE_VERIFIED' in log,
             'last_state_saved':(folder/'last_state.pt').exists()}
        best_trained=max(history,key=lambda h:h['official_total']) if history else None
        row['trained_peak']=({'epoch':best_trained['epoch'],'TOTAL':best_trained['official_total']} if best_trained else None)
        score=final['scores']['TOTAL'] if final else row['latest'].get('best_score')
        row['TOTAL']=score
        if final:
            p=np.load(folder/'fold_5_pred.npy');u=kernels(y,p);utilities[mode]=u
            assert np.array_equal(p[:,1:],reference[:,1:])
            assert abs(u.mean(0)@weights*100-score)<1e-9
            delta=(u-u0)*100;total=delta@weights
            perwell=[{'well_index':int(w),'n':int((wells==w).sum()),'TOTAL_delta':float(total[wells==w].mean())} for w in np.unique(wells)]
            pop={}
            for label,m in [('true_default',y[:,0]<=.1),('true_nondefault',y[:,0]>.1)]:
                pop[label]={'n':int(m.sum()),'source_POR':float(u0[m,0].mean()*100),'candidate_POR':float(u[m,0].mean()*100),
                            'TOTAL_contribution':float((delta[:,0]*m).mean()*.3)}
            row['diagnostic']={'delta_initial':float(total.mean()),'wins':sum(v['TOTAL_delta']>1e-12 for v in perwell),
                               'losses':sum(v['TOTAL_delta']< -1e-12 for v in perwell),'per_well':perwell,'populations':pop}
            launch=read(folder/'launch.json');intervals.append((datetime.fromisoformat(launch['started_at']),datetime.fromisoformat(final['finished_at'])))
        state='完成并恢复' if final else status.get('status','planned')
        text=now+'；'+state+'；最新完整epoch'+str(row['latest'].get('epoch',0))+'；TOTAL='+str(score)
        if score is not None:text+='；相对80.274135 '+f"{score-parent['scores']['TOTAL']:+.9f}"
        if final:text+='；选中epoch'+str(final['best_epoch'])+'；更新后最高='+str(row['trained_peak'])
        else:text+='，不是最终恢复成绩'
        if args.record:
            for filename in ['README.md','PROJECTS.md']:
                section(folder/filename,text+'。\n初始差异元素数='+str(initial.get('different_elements'))+'；梯度范围验证='+str(row['gradient_scope_verified'])+'。无其他折或旧交付变更。')
            cur['nodes'][folder.name]={'title':'分量query·'+('当前点记忆控制' if mode=='center' else '局部序列交叉注意力'),
                'summary':'完全冻结旧80.274135预测器，仅训练14081参数的16分量查询头；'+('重复当前两专家token' if mode=='center' else '读取前后8点的两专家token'),
                'conclusion':text+'。第五折开发结果，未补折。','stage':'loss','highlight':True}
        records.append(row)
    complete=all(r['final_report'] is not None for r in records)
    best={'mode':'previous','TOTAL':parent['scores']['TOTAL']};pairwise=None
    for r in records:
        if r['final_report'] and r['TOTAL']>best['TOTAL']+1e-10:best={'mode':r['mode'],'TOTAL':r['TOTAL']}
    if complete:
        change=(utilities['local']-utilities['center'])*100@weights
        wd=np.array([change[wells==w].mean() for w in np.unique(wells)])
        ns=np.array([(wells==w).sum() for w in np.unique(wells)])
        idx=np.random.default_rng(20260928).integers(0,16,(10000,16))
        draws=(ns[idx]*wd[idx]).sum(1)/ns[idx].sum(1)
        pairwise={'local_minus_center_TOTAL':float(change.mean()),'well_wins':int((wd>1e-12).sum()),
                  'well_losses':int((wd< -1e-12).sum()),'exploratory_selection_unadjusted_95':np.quantile(draws,[.025,.975]).tolist(),
                  'same_parameters':records[0]['trainable_parameters']==records[1]['trainable_parameters']}
    timing=({'batch_span_seconds':(max(b for a,b in intervals)-min(a for a,b in intervals)).total_seconds(),
             'scope':'first launch to last completion; includes loading/training/evaluation; excludes later diagnostics'} if complete else None)
    payload={'observed_at':now,'source_scores':parent['scores'],'runs':records,'all_complete':complete,'best_completed':best,
             'pairwise':pairwise,'timing':timing,'remaining_folds_started':False,
             'limitations':'Fixed fifth repeatedly used for development; no new fullOOF or independent-test evidence. Same architecture/initialization versus different local memory input.'}
    atomic(STUDY/'progress.json',payload)
    if args.record:
        edges=[(PARENT,PREFIX+'center80_fold5','evolution'),(PREFIX+'center80_fold5',PREFIX+'local80_fold5','control')]
        pairs={(a,b) for a,b,t in edges}
        cur['edges']=[e for e in cur['edges'] if (e['source'],e['target']) not in pairs]
        cur['edges'] += [{'source':a,'target':b,'type':t,'note':'旧预测器全冻结，16分量query，两组参数/初始化相同；只变记忆token是否来自邻近点。','evidence':'outputs/'+b+'/README.md'} for a,b,t in edges]
        atomic(ATLAS/'curation.json',cur)
        lines=['# 分量查询式预测头：结果','',now,'','|方案|状态|参数|保留TOTAL|更新后最高TOTAL|最佳/停止轮次|','|---|---|---:|---:|---:|---|']
        for r in records:
            f=r['final_report'];value=r['TOTAL'];peak=r['trained_peak']
            lines.append('|'+r['mode']+'|'+('完成' if f else r['status'].get('status','planned'))+'|'+str(r['trainable_parameters'])+'|'+str(value)+'|'+str(peak['TOTAL'] if peak else None)+'|'+(str(f['best_epoch'])+'/'+str(f['stopped_epoch']) if f else '—')+'|')
        lines+=['','当前保留：'+str(best),'','局部对照：'+str(pairwise),'',
                '两组均在同井两专家的隐藏表示上，为每专家8分量形成独立query。32维、2头、16查询×34记忆槽位；center重复当前token，local读真实邻近token。旧模型、线性位置、门控、权重、尺度、PERM/SW全部固定，仅新支路更新。',
                '相对v6同时改变了更新范围和query设计，不能直接据跨轮差值归因；本轮两组仅记忆内容不同，才是匹配的邻近特征比较。',
                '选择第0轮代表本次训练无新增收益，恢复后相同预测不意味着两种机制具有统计等价性。原第五折已反复开发，探索性区间不是独立测试证据。',
                '全部完成='+str(complete)+'；完整OOF>82及第五折>80.5仍待达成，本批无其他折或交付更换。']
        if complete:
            for r in records:lines+=['','### '+r['mode'],json.dumps(r.get('diagnostic',{}),ensure_ascii=False,indent=2)]
        (STUDY/('FINAL.md' if complete else 'SNAPSHOT.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
        section(STUDY/'PROJECTS.md','观察：'+now+'；全部完成='+str(complete)+'。\n已做：结构、启动与真实状态记录；当前保留='+str(best)+'。\n未做：胜出权重重载或实现诊断、独立泛化、续跑等价性。\n计划：完成本批两项并比较，不追加候选；目标达成后再固定方案验证泛化，当前不补折。')
        note='\n\n## '+now+' 分量query v7记录\n全部完成='+str(complete)+'；保留='+str(best)+'；对照='+str(pairwise)+'。记录results/deep_learning/dual8222_component_query_v7_20260928；无旧任务重启/补折/交付变更。\n'
        for p in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',ATLAS/'PROJECTS.md']:
            with p.open('a',encoding='utf-8') as f:f.write(note)
    summary={**payload,'runs':[{k:v for k,v in r.items() if k not in ['diagnostic','final_report']} for r in records]}
    for short,r in zip(summary['runs'],records):
        if r['final_report']:short['final_scores']=r['final_report']['scores']
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
