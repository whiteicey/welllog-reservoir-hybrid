"""Read bounded v6 experiments, score saved predictions, maintain independent records.
No training, rowwise selection, hyperparameter fitting, or old source modification.
"""
from __future__ import annotations
import argparse,json,sys
from datetime import datetime
from pathlib import Path
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/dual8222_attention_v6_20260928'
ATLAS=ROOT/'docs/deep_learning/atlas'
PARENT='pure_dl_dual8222_v5_location_point3280_fold5'
SOURCE='pure_dl_paired_local_context_router80_sequence'
MODES=['linear','deep_mlp','attention']
PREFIX='pure_dl_dual8222_v6_'
NAMES={'linear':'同起点线性继续训练','deep_mlp':'两层联合特征深MLP头','attention':'两层四头局部注意力头'}


def read(p,default=None):
    return json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else default


def save(p,value):
    temp=p.with_name(p.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(p)


def section(p,text):
    marker='## 本批实际核对'
    previous=p.read_text(encoding='utf-8-sig') if p.exists() else ''
    p.write_text(previous.split(marker)[0].rstrip()+'\n\n'+marker+'\n'+text+'\n',encoding='utf-8')


def kernels(y,p):
    return np.c_[np.maximum(0,1-abs(p[:,0]-y[:,0])/(.08*(abs(y[:,0])+1e-5))),
                 np.maximum(0,1-abs(np.log10(np.maximum(p[:,1]/np.maximum(y[:,1],1e-5),1e-5)))),
                 np.maximum(0,1-abs(p[:,2]-y[:,2])/(.05*(abs(y[:,2])+1e-5)))]


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--record',action='store_true');args=ap.parse_args()
    now=datetime.now().astimezone().isoformat(timespec='seconds')
    baseline=read(ROOT/'outputs'/PARENT/'fold_5_report.json')
    ref=np.load(ROOT/'outputs'/PARENT/'fold_5_pred.npy')
    with np.load(ROOT/'outputs'/SOURCE/'oof.npz') as z:
        pick=z['fold']==4;y=z['truth'][pick];wells=z['well'][pick]
    assert y.shape==ref.shape==(144665,3)
    weight=np.array([.3,.35,.35]);u0=kernels(y,ref)
    records=[];utilities={};intervals=[]
    cur=read(ATLAS/'curation.json')
    for mode in MODES:
        folder=ROOT/'outputs'/(PREFIX+mode+'80_fold5')
        status=read(folder/'runtime_status.json',{})
        final=read(folder/'fold_5_report.json');history=read(folder/'fold_5_validation_history.json',[])
        latest=history[-1] if history else {};cfg=read(folder/'config.json',{})
        alive=False
        try:
            proc=psutil.Process(status.get('pid',-1))
            alive=proc.is_running() and 'train_dual8222_attention_v6.py' in ' '.join(proc.cmdline())
        except (ValueError,psutil.NoSuchProcess,psutil.AccessDenied):pass
        log=(folder/'train.log').read_text(encoding='utf-8',errors='replace') if (folder/'train.log').exists() else ''
        row={'mode':mode,'id':folder.name,'status':status,'process_alive':alive,'final_report':final,
             'latest':latest,'initial_verification':read(folder/'initial_verification.json'),
             'failure':read(folder/'failure.json'),'gradient_scope_verified':'GRADIENT_SCOPE_VERIFIED' in log,
             'trainable_parameters':cfg.get('trainable_parameters'),
             'checkpoint_verification':read(folder/'checkpoint_reload_verification.json')}
        row['TOTAL']=final['scores']['TOTAL'] if final else latest.get('best_score')
        if final:
            p=np.load(folder/'fold_5_pred.npy');u=kernels(y,p);utilities[mode]=u
            assert np.array_equal(p[:,1:],ref[:,1:])
            assert abs(u.mean(0)@weight*100-final['scores']['TOTAL'])<1e-9
            change=(u-u0)*100;total=change@weight
            bywell=[{'well_index':int(w),'n':int((wells==w).sum()),'TOTAL_delta':float(total[wells==w].mean())} for w in np.unique(wells)]
            pop={}
            for label,m in [('true_default',y[:,0]<=.1),('true_nondefault',y[:,0]>.1)]:
                pop[label]={'n':int(m.sum()),'baseline_POR':float(u0[m,0].mean()*100),'candidate_POR':float(u[m,0].mean()*100),'contribution_to_TOTAL':float((change[:,0]*m).mean()*.3)}
            row['diagnostic']={'delta':float(total.mean()),'per_well':bywell,'populations':pop,
                               'wins':sum(w['TOTAL_delta']>1e-12 for w in bywell),'losses':sum(w['TOTAL_delta']< -1e-12 for w in bywell)}
            launch=read(folder/'launch.json');intervals.append((datetime.fromisoformat(launch['started_at']),datetime.fromisoformat(final['finished_at'])))
        records.append(row)
        conclusion=('完成并恢复' if final else status.get('status','planned'))+'；TOTAL='+str(row['TOTAL'])+'；最后epoch='+str(latest.get('epoch',0))
        if row['TOTAL'] is not None:conclusion+=f"；相对起点{row['TOTAL']-baseline['scores']['TOTAL']:+.9f}"
        if not final:conclusion+='；非最终恢复成绩'
        if args.record:
            body=now+'\n'+conclusion+'。\n初始验证='+str(row['initial_verification'])+'。\n首批梯度范围已验证='+str(row['gradient_scope_verified'])+'；失败记录='+str(row['failure'])+'。\n尚无跨折泛化或续跑等价性证明；不自动启动新路线或补折。'
            for name in ['README.md','PROJECTS.md']:section(folder/name,body)
            cur['nodes'][folder.name]={'title':NAMES[mode],'summary':'原80.274135完整模型热启动；POR线性位置可训练，旧非线性固定；'+NAMES[mode],
                                     'conclusion':conclusion+'。原单覆盖解码；重复开发第五折。','stage':'loss','highlight':True}
    pairs={}
    for a,b in [('deep_mlp','linear'),('attention','linear'),('attention','deep_mlp')]:
        if a not in utilities or b not in utilities:continue
        d=(utilities[a]-utilities[b])*100@weight
        wd=np.array([d[wells==w].mean() for w in np.unique(wells)])
        ns=np.array([(wells==w).sum() for w in np.unique(wells)])
        draws=np.random.default_rng(20260928).integers(0,len(ns),(10000,len(ns)))
        sampled=(ns[draws]*wd[draws]).sum(1)/ns[draws].sum(1)
        pairs[a+'_minus_'+b]={'TOTAL_delta':float(d.mean()),'well_wins':int((wd>1e-12).sum()),'well_losses':int((wd< -1e-12).sum()),
                            'exploratory_selection_unadjusted_95':np.quantile(sampled,[.025,.975]).tolist()}
    complete=all(r['final_report'] is not None for r in records)
    best={'mode':'previous','TOTAL':baseline['scores']['TOTAL']}
    for r in records:
        if r['final_report'] and r['TOTAL']>best['TOTAL']+1e-10:best={'mode':r['mode'],'TOTAL':r['TOTAL']}
    timing=None
    if complete:
        count=peak=0
        for _,v in sorted([(a,1) for a,b in intervals]+[(b,-1) for a,b in intervals],key=lambda x:(x[0],x[1])):
            count+=v;peak=max(peak,count)
        timing={'batch_span_seconds':(max(b for a,b in intervals)-min(a for a,b in intervals)).total_seconds(),'max_concurrent_trainers':peak,'scope':'first launch to last completion, including preparation and explicit launch gaps'}
    payload={'observed_at':now,'baseline_scores':baseline['scores'],'runs':records,'all_complete':complete,'best_completed':best,'pairwise':pairs,'timing':timing,
             'limitations':'all scores from repeatedly used fifth fold; intervals exploratory and selection-unadjusted; no independent fullOOF validation'}
    save(STUDY/'progress.json',payload)
    if args.record:
        edges=[(PARENT,PREFIX+'linear80_fold5','evolution'),(PREFIX+'linear80_fold5',PREFIX+'deep_mlp80_fold5','control'),(PREFIX+'deep_mlp80_fold5',PREFIX+'attention80_fold5','control')]
        unique={(a,b) for a,b,_ in edges}
        cur['edges']=[e for e in cur['edges'] if (e['source'],e['target']) not in unique]
        cur['edges'] += [{'source':a,'target':b,'type':t,'note':'同起点/评分；linear控制新增优化预算；attention与deepMLP近容量比较。','evidence':'outputs/'+b+'/README.md'} for a,b,t in edges]
        save(ATLAS/'curation.json',cur)
        lines=['# 复杂预测头与attention：本批结果','',now,'','|方案|状态|参数|TOTAL|相对起点|最佳/停止轮次|','|---|---|---:|---:|---:|---|']
        for r in records:
            f=r['final_report'];v=r['TOTAL']
            lines.append('|'+NAMES[r['mode']]+'|'+('完成并恢复' if f else r['status'].get('status','planned'))+'|'+str(r['trainable_parameters'])+'|'+(f'{v:.9f}' if v is not None else '尚无')+'|'+(f"{v-baseline['scores']['TOTAL']:+.9f}" if v is not None else '—')+'|'+(str(f['best_epoch'])+'/'+str(f['stopped_epoch']) if f else '—')+'|')
        lines+=['','## 匹配比较',json.dumps(pairs,ensure_ascii=False,indent=2),'','## 当前保留',str(best),'',
                '两层64维复杂头联合两专家特征。attention为每层4头、单头16维、局部33点及相对偏置；deepMLP为近容量点处理控制，参数相差392。',
                '只优化POR位置，共同线性lr3e-5、新增模块lr3e-4；门控/分量权重/尺度/PERM/SW与原v5修正冻结。增加结构与训练预算必须分开解释。',
                '三个实验同80.274135起点；不是原训练轨迹无缝续跑。原九曲线/206特征/分折/训练Q/单覆盖256格800动作不变。',
                '第0轮保留代表无新增收益。完成报告含全部144665点最佳恢复检查；本批不替换交付，不补其他折。完整OOF>82未实现。',
                '相关方法：Attention机制参见PyTorch官方scaled_dot_product_attention文档；本实现显式float32矩阵运算并处理全遮挡边界。']
        lines+=['','## 后续状态','全部完成='+str(complete)+'；重载验证='+str(any(r['checkpoint_verification'] for r in records))+'。不追加本批候选；独立泛化和中断续跑等价性未验证。']
        (STUDY/('FINAL.md' if complete else 'SNAPSHOT.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
        section(STUDY/'PROJECTS.md','观察时间：'+now+'；全部完成='+str(complete)+'；当前保留='+str(best)+'。\n已做：结构与正式状态记录。剩余：未完成训练/重载检查及独立泛化验证。目标>80.5和完整OOF>82尚未实现；无自动补折。')
        note='\n\n## '+now+' 复杂预测头v6核对\n全部完成='+str(complete)+'；当前保留='+str(best)+'；匹配差值='+str(pairs)+'。独立记录results/deep_learning/dual8222_attention_v6_20260928；无旧路线重启/补折或交付变更。\n'
        for p in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',ATLAS/'PROJECTS.md']:
            with p.open('a',encoding='utf-8') as f:f.write(note)
    summary={**payload,'runs':[{k:v for k,v in r.items() if k not in ['diagnostic','final_report']} for r in records]}
    for short,row in zip(summary['runs'],records):
        if row['final_report']:short['final_scores']=row['final_report']['scores']
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
