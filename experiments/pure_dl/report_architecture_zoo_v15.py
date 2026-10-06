"""Compact architecture comparison and non-destructive lineage overlay."""
from pathlib import Path
from datetime import datetime
import argparse,json
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/architecture_zoo_v15_20260928'
DATA=ROOT/'results/deep_learning/sw_group_generalization_v14_20260928'
CONTROL=ROOT/'outputs/pure_dl_sw_group_v14_inner_point80'
MODES=('adaptive','latent','state_space')
TITLES={'adaptive':'自适应深度采样','latent':'潜变量读写记忆','state_space':'双向门控状态空间'}

def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def score(pred,y):return 100*np.maximum(0,1-np.abs(pred-y)/(.05*(np.abs(y)+1e-5)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--record',action='store_true');args=ap.parse_args()
    control=load(CONTROL/'final_report.json');control_cfg=load(CONTROL/'config.json')
    with np.load(DATA/'data_inner.npz') as z:
        ids=z['monitor_raw'];y=z['truth'][ids];wells=z['wells'][ids];default=z['default_truth'][ids]
    basepred=np.load(CONTROL/'monitor_pred.npy');basepoints=score(basepred,y)
    assert len(y)==144621 and abs(basepoints.mean()-control['target_score'])<1e-9
    records=[];starts=[];ends=[]
    for mode in MODES:
        ident='pure_dl_architecture_v15_'+mode+'_inner';p=ROOT/'outputs'/ident
        status=load(p/'runtime_status.json') if (p/'runtime_status.json').exists() else dict(status='not_started')
        config=load(p/'config.json') if (p/'config.json').exists() else None
        history=load(p/'validation_history.json') if (p/'validation_history.json').exists() else []
        report=load(p/'final_report.json') if (p/'final_report.json').exists() else None
        r=dict(architecture=mode,id=ident,title=TITLES[mode],status=status,config=config,report=report,
               latest=history[-1] if history else None)
        if config:
            fields=['seed','epochs','patience','minimum_epochs','steps_per_epoch','batch_size','lr','eta_min','weight_decay','clip_norm','Q','fit_wells','monitor_wells']
            assert all(config[k]==control_cfg[k] for k in fields)
            r['matched_control_fields']=fields
        if report:
            pred=np.load(p/'monitor_pred.npy');assert pred.shape==basepred.shape
            delta=score(pred,y)-basepoints
            assert abs(score(pred,y).mean()-report['target_score'])<1e-9
            perwell=[dict(well=int(w),n=int((wells==w).sum()),SW_delta=float(delta[wells==w].mean())) for w in np.unique(wells)]
            counts=np.array([x['n'] for x in perwell]);sums=np.array([x['SW_delta']*x['n'] for x in perwell])
            rng=np.random.default_rng(20261010);draw=rng.integers(0,len(counts),(10000,len(counts)))
            boot=sums[draw].sum(1)/counts[draw].sum(1)
            winners=sum(x['SW_delta']>1e-10 for x in perwell)
            comparison=dict(SW_delta=float(delta.mean()),default_SW_delta=float(delta[default].mean()),numeric_SW_delta=float(delta[~default].mean()),
                default_contribution_to_SW=float(delta[default].sum()/len(delta)),numeric_contribution_to_SW=float(delta[~default].sum()/len(delta)),
                wins=winners,losses=sum(x['SW_delta']< -1e-10 for x in perwell),per_well=perwell,
                exploratory_selection_unadjusted_95=np.quantile(boot,[.025,.975]).tolist(),
                predeclared_signal=bool(delta.mean()>=.05 and winners>=9))
            r['comparison']=comparison
            starts.append(datetime.fromisoformat(load(p/'launch.json')['started_at']));ends.append(datetime.fromisoformat(report['finished_at']))
            verify=p/'implementation_verification.json'
            if verify.exists():r['implementation_verification']=load(verify)
        records.append(r)
    done=all(r['report'] is not None for r in records)
    configs=[r['config'] for r in records if r['config']]
    shared_init=len(set(c['common_stem_readout_initial_hash'] for c in configs))<=1
    assert shared_init
    result=dict(observed_at=datetime.now().astimezone().isoformat(),all_complete=done,control=control,control_reused_not_rerun=True,
        rows=records,common_input_output_initialization_equal=shared_init,score_scope='same inner48fit16monitor SW only',
        original_fifth_evaluated=False,complete_five_folds=False,model_parameter_counts_matched=False,
        old_three_model_TOTAL_unchanged=80.27413533779057,other_channels_trained_this_round=False)
    if done:
        best=max(records,key=lambda r:r['report']['target_score'])
        result['best_architecture']=best['architecture'];result['best_signal']=best['comparison']['predeclared_signal']
        result['timing']=dict(launch_to_last_finish_wall_seconds=(max(ends)-min(starts)).total_seconds(),maximum_concurrency=2,launches_explicit=True,
            per_run_seconds={r['architecture']:r['report']['seconds'] for r in records},peak_allocated_MiB={r['architecture']:r['report']['peak_allocated_MiB'] for r in records},
            scope='includes data loads and full fit/monitor eval,best restore,staggered gaps;excludes code work before first launch and post-run diagnostics')
    dump(STUDY/'progress.json',result)
    curpath=ROOT/'docs/deep_learning/atlas/curation.json';cur=load(curpath);nodes=cur.setdefault('nodes',{});edges=cur.setdefault('edges',[])
    for r in records:
        report=r['report'];status=r['status'];val=report['target_score'] if report else status.get('best_target_score')
        src='outputs/'+r['id']+('/final_report.json' if report else '/runtime_status.json')
        primary=None if val is None else dict(scope='内层48→16·SW'+('' if report else '·训练快照'),value=val,kind='diagnostic',scores={'SW':val},source=src)
        conclusion=(f"完成内层SW{val:.6f}，相对TCN{r['comparison']['SW_delta']:+.6f}，{r['comparison']['wins']}井升/{r['comparison']['losses']}井降。" if report else '状态='+status['status']+'，未形成最终结果。')
        conclusion+='完整可训练骨干筛查；参数量不同、单个内层开发划分，不作第五折或因果机制结论。'
        nodes[r['id']]=dict(title=TITLES[r['architecture']]+'·完整骨干v15',stage='backbone',status='recorded' if report else status['status'],
            primary=primary,results=[] if primary is None else [primary],summary=r['config']['description'] if r['config'] else '新完整架构，详见v15 METHOD.md',conclusion=conclusion,
            evidence=[src,'results/deep_learning/architecture_zoo_v15_20260928/METHOD.md'])
        parent='pure_dl_sw_group_v14_inner_point80'
        edges[:]=[e for e in edges if not(e['source']==parent and e['target']==r['id'])]
        edges.append(dict(source=parent,target=r['id'],type='evolution',note='相同48/16数据、损失、训练设置，完整架构替换；非参数匹配消融。',evidence='outputs/'+r['id']+'/config.json'))
    dump(curpath,cur)
    if done:
        lines=['# v15完整骨干结构探索：实测结果','',
            '本轮完成自适应采样、潜变量记忆、双向门控状态空间三种独立SW网络，从头训练骨干和输出头。原第五折未评价，原三模型不替换。',
            '', '|结构|参数量|内层SW|相对TCN|真实非默认段SW|最佳/停止轮次|', '|---|---:|---:|---:|---:|---|',
            f"|既有TCN对照|{control['trainable_parameters']}|{control['target_score']:.6f}|0|{control['monitor']['numeric_SW']:.6f}|{control['best_epoch']}/{control['stopped_epoch']}|"]
        for r in records:
            f=r['report'];c=r['comparison'];lines.append(f"|{r['title']}|{f['trainable_parameters']}|{f['target_score']:.6f}|{c['SW_delta']:+.6f}|{f['monitor']['numeric_SW']:.6f}|{f['best_epoch']}/{f['stopped_epoch']}|")
        lines+=['','## 结论边界',
            '相同训练数据、输入/输出、损失、窗口抽样、学习率与预算上限；输入与输出层初始值相同，隐藏架构及参数量不同。重复使用内层监控选模，不能把单设置胜出归因于唯一模块或称独立盲测。',
            '三个候选均未利用旧冻结预测，完整骨干真实更新。既有TCN控制未重复训练，采用相同缓存和原配置的完成结果。',
            '本轮只给SW做架构筛查，POR/PERM仍是原私有模型，不能声称三个目标均已完成新架构。原第五折TOTAL80.274135和完整OOF81.713233不变。']
        if result['best_signal']:
            lines+=['','## 下一步未执行',f"优先细化{TITLES[result['best_architecture']]}：先增加参数/计算预算匹配的对照，再检查该结构在POR/PERM独立模型上的效果；候选入选不代表已替换第五折方案。"]
        else:
            lines+=['','## 下一步未执行','本轮未达到预定义结构信号门槛。保留各类结构与局部诊断，不继续无界规模扫描；更换输入组织或层段归纳偏置需要单独方案。']
        lines+=['','## 证据与工程','详细逐井、默认/非默认贡献、训练/监控分数与时间见progress.json。模型源码architecture_zoo_v15.py；训练器train_architecture_zoo_v15.py。',
            '每模型保存model_best.pt、last_state.pt和预测；运行内最佳恢复已记录。单独磁盘重载与填充/SSM数值检查的完成范围以implementation_verification.json为准。',
            '图谱使用本报告的非破坏性curation.json扩展，再build_atlas.py；旧curate.py未改，不重置既有后续元数据。']
        (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        old=ROOT/'results/deep_learning/sw_group_generalization_v14_20260928/CURRENT_RECIPE.json'
        recipe=load(old);recipe.update(v15_selected_for_deployment=False,v15_inner_candidate=result['best_architecture'],v15_inner_signal=result['best_signal'],v15_original_fifth_evaluated=False)
        dump(STUDY/'CURRENT_RECIPE.json',recipe)
        if args.record:
            marker='## v15三种完整骨干筛查完成'
            note='\n\n'+marker+'\n'+result['observed_at']+'。'+str({r['architecture']:r['report']['target_score'] for r in records})+'；全部为48/16内层SW。是否结构信号='+str(result['best_signal'])+'。未训练原第五折，旧三私有模型80.274135保持。详细研究记录results/deep_learning/architecture_zoo_v15_20260928/FINAL.md。参数匹配消融/其他通道/完整OOF>82均未完成。\n'
            for p in [STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md']:
                if marker not in p.read_text(encoding='utf-8-sig'):
                    with p.open('a',encoding='utf-8') as f:f.write(note)
            for r in records:
                p=ROOT/'outputs'/r['id'];f=r['report']
                n='\n\n## 完成结论\n内层SW='+str(f['target_score'])+'；相对TCN='+str(r['comparison']['SW_delta'])+'；最佳/停止轮次='+str(f['best_epoch'])+'/'+str(f['stopped_epoch'])+'。只代表本配置架构比较，无原第五折新成绩。详见研究FINAL.md；后续参数匹配、其他通道和完整OOF验证未实施。\n'
                for name in ['README.md','PROJECTS.md']:
                    if '## 完成结论' not in (p/name).read_text(encoding='utf-8'):
                        with (p/name).open('a',encoding='utf-8') as f2:f2.write(n)
    active=[]
    for proc in psutil.process_iter(['pid','name','cmdline']):
        try:
            v=proc.info;argv=v.get('cmdline') or []
            if 'python' in (v.get('name') or '').lower() and any('pure_dl/' in x.replace('\\','/') and Path(x.replace('\\','/')).name.startswith(('train_','benchmark_')) for x in argv):active.append(v['pid'])
        except(psutil.NoSuchProcess,psutil.AccessDenied):pass
    brief=[dict(architecture=r['architecture'],status=r['status'],params=r['config']['trainable_parameters'] if r['config'] else None,
        SW=r['report']['target_score'] if r['report'] else r['status'].get('best_target_score'),comparison=r.get('comparison'),
        best_epoch=r['report']['best_epoch'] if r['report'] else None,stopped_epoch=r['report']['stopped_epoch'] if r['report'] else None) for r in records]
    print(json.dumps(dict(all_complete=done,control_SW=control['target_score'],rows=brief,best_architecture=result.get('best_architecture'),timing=result.get('timing'),active=active),ensure_ascii=False))

if __name__=='__main__':main()
