"""CPU diagnosis of the fixed inner split and optional explicit fifth-fold runs."""
from pathlib import Path
from datetime import datetime
import argparse,json
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/sw_group_generalization_v14_20260928'

def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def kernel(a,y):return 100*np.maximum(0,1-np.abs(a-y)/(.05*(np.abs(y)+1e-5)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['inner','outer'],default='inner');ap.add_argument('--record',action='store_true');a=ap.parse_args()
    manifest=load(STUDY/('data_'+a.phase+'_manifest.json'))
    with np.load(STUDY/('data_'+a.phase+'.npz')) as z:
        ids=z['monitor_raw'];truth=z['truth'][ids];wells=z['wells'][ids];default=z['default_truth'][ids]
    rows=[];configs=[];starts=[];ends=[]
    for mode in ['point','well']:
        ident='pure_dl_sw_group_v14_'+a.phase+'_'+mode+'80';p=ROOT/'outputs'/ident
        if not (p/'runtime_status.json').exists():continue
        status=load(p/'runtime_status.json');config=load(p/'config.json') if (p/'config.json').exists() else None
        hist=load(p/'validation_history.json') if (p/'validation_history.json').exists() else []
        report=load(p/'final_report.json') if (p/'final_report.json').exists() else None
        row=dict(id=ident,mode=mode,status=status,report=report,last=hist[-1] if hist else None)
        if config:configs.append(config)
        if report:
            pred=np.load(p/'monitor_pred.npy');score=kernel(pred,truth);assert abs(score.mean()-report['target_score'])<1e-9
            best_epoch=report['best_epoch'];best=next((h for h in hist if h['epoch']==best_epoch),None)
            rho=float(default.mean());f=report['fit'];v=report['monitor']
            standardized=rho*f['default_SW']+(1-rho)*f['numeric_SW']
            row['diagnostic']=dict(monitor_default_rate=rho,fit_default_rate=f['default_points']/(f['default_points']+f['numeric_points']),
                raw_fit_minus_monitor=f['SW']-v['SW'],default_prior_adjusted_fit_minus_monitor=standardized-v['SW'],
                numeric_fit_minus_monitor=f['numeric_SW']-v['numeric_SW'],default_fit_minus_monitor=f['default_SW']-v['default_SW'],
                monitor_minus_fit_conditional_CE=v['conditional_CE']-f['conditional_CE'],
                after_best=dict(epochs=report['stopped_epoch']-best_epoch,
                    fit_SW_change=hist[-1]['fit']['SW']-f['SW'],monitor_SW_change=hist[-1]['monitor']['SW']-v['SW'],
                    fit_CE_change=hist[-1]['fit']['conditional_CE']-f['conditional_CE'],
                    monitor_CE_change=hist[-1]['monitor']['conditional_CE']-v['conditional_CE']))
            starts.append(datetime.fromisoformat(load(p/'launch.json')['started_at']));ends.append(datetime.fromisoformat(report['finished_at']))
        rows.append(row)
    match=None
    if len(configs)==2:
        differences={k:[configs[0].get(k),configs[1].get(k)] for k in set(configs[0])|set(configs[1]) if configs[0].get(k)!=configs[1].get(k)}
        assert set(differences)=={'mode','weight'},differences
        match=dict(differences=differences,same_initial_state=True)
    done=len(rows)==2 and all(r['report'] for r in rows)
    result=dict(phase=a.phase,observed_at=datetime.now().astimezone().isoformat(),all_complete=done,rows=rows,matched_configuration=match,
        fit_wells=manifest['fit_wells'],monitor_wells=manifest['monitor_wells'],fit_points=manifest['fit_valid_points'],monitor_points=len(ids),
        original_fifth_evaluated=a.phase=='outer',complete_five_folds=False,complete_nestedCV=False,
        limitations='One inner development split, early-stop selected;architecture prechosen from earlier work. No untouched blind-test claim. Q=1 differs from historic Q recipes.')
    if done:
        p0=np.load(ROOT/'outputs'/rows[0]['id']/'monitor_pred.npy');p1=np.load(ROOT/'outputs'/rows[1]['id']/'monitor_pred.npy')
        delta=kernel(p1,truth)-kernel(p0,truth)
        perwell=[dict(well=int(w),n=int((wells==w).sum()),SW_delta=float(delta[wells==w].mean())) for w in np.unique(wells)]
        rng=np.random.default_rng(20261010);counts=np.array([p['n'] for p in perwell]);sums=np.array([p['SW_delta']*p['n'] for p in perwell]);draw=rng.integers(0,len(counts),(10000,len(counts)))
        boot=sums[draw].sum(1)/counts[draw].sum(1)
        comparison=dict(SW_delta=float(delta.mean()),well_macro_SW_delta=float(np.mean([p['SW_delta'] for p in perwell])),
            wins=sum(p['SW_delta']>1e-10 for p in perwell),losses=sum(p['SW_delta']< -1e-10 for p in perwell),per_well=perwell,
            default_delta=float(delta[default].mean()),numeric_delta=float(delta[~default].mean()),
            exploratory_selection_unadjusted_95=np.quantile(boot,[.025,.975]).tolist())
        comparison['predeclared_inner_screen_passed']=comparison['SW_delta']>=.05 and comparison['wins']>=9
        result['comparison']=comparison
        result['timing']=dict(concurrent_jobs=2,launch_to_last_finish_wall_seconds=(max(ends)-min(starts)).total_seconds(),
            per_run_seconds={r['mode']:r['report']['seconds'] for r in rows},
            peak_allocated_MiB={r['mode']:r['report']['peak_allocated_MiB'] for r in rows},
            preparation_seconds=manifest['preparation_seconds'],scope='training times include cached data load,all fit/monitor eval,best restore;exclude implementation and reports')
    write(STUDY/('progress_'+a.phase+'.json'),result)
    # Human-readable curation overlay. Never run the old curation generator that
    # would erase later independently maintained metadata.
    cpath=ROOT/'docs/deep_learning/atlas/curation.json';cur=load(cpath)
    for row in rows:
        report=row['report'];state=row['status'];val=report['target_score'] if report else state.get('best_target_score')
        scope='内层48井训练→16井监控·SW' if a.phase=='inner' else '原第五折·SW'
        source='outputs/'+row['id']+('/final_report.json' if report else '/runtime_status.json')
        primary=None if val is None else dict(scope=scope,value=val,scores={'SW':val},kind='diagnostic' if a.phase=='inner' else 'target',source=source)
        title=('按点平均' if row['mode']=='point' else '按井均衡')+'·'+('内层泛化诊断v14' if a.phase=='inner' else '原第五折v14')
        conclusion=('完成SW='+str(val)+'；最佳'+str(report['best_epoch'])+'轮，停止'+str(report['stopped_epoch'])+'轮。' if report else '训练中，尚无最终结果。')
        conclusion+='无旧模型/Q/统计复用；不与原第五折TOTAL混排。' if a.phase=='inner' else '新单通道结果，未完成全OOF。'
        cur.setdefault('nodes',{})[row['id']]=dict(title=title,stage='backbone',status='recorded' if report else 'running_snapshot',primary=primary,
            results=[] if primary is None else [primary],summary='相同从零TCN/初始化/窗口，只改变井大小加权；48训练/16内层监控，预处理严格重拟合。',
            conclusion=conclusion,evidence=[source,'results/deep_learning/sw_group_generalization_v14_20260928/data_'+a.phase+'_manifest.json'])
    if len(rows)==2:
        src,dst=rows[0]['id'],rows[1]['id'];edges=cur.setdefault('edges',[])
        edges[:]=[e for e in edges if not(e['source']==src and e['target']==dst)]
        edges.append(dict(source=src,target=dst,type='control',note='同参数/初始化/窗口/内层划分；仅训练井权重不同。',evidence='outputs/'+dst+'/config.json'))
    write(cpath,cur)
    if done:
        lines=['# v14：训练井内部泛化与井均衡对照','',
            '本阶段没有训练或评价原第五折，未替换旧三模型。仅原64训练井内部的48/16切分，监控集参与早停，不是完整嵌套交叉验证。','',
            '|方法|训练井SW|内层监控SW|训练非默认SW|监控非默认SW|最佳/停止轮次|',
            '|---|---:|---:|---:|---:|---|']
        for row in rows:
            r=row['report'];lines.append(f"|{row['mode']}|{r['fit']['SW']:.6f}|{r['monitor']['SW']:.6f}|{r['fit']['numeric_SW']:.6f}|{r['monitor']['numeric_SW']:.6f}|{r['best_epoch']}/{r['stopped_epoch']}|")
        lines+=['',f"井均衡相对按点SW差{comparison['SW_delta']:+.9f}；{comparison['wins']}井提高、{comparison['losses']}井下降。预设内部进一步检查门槛通过={comparison['predeclared_inner_screen_passed']}。",
            '井均衡只调节训练中不同井的贡献，不把困难井当坏标签；其权重是学习权重，Q恒为1。原始评估始终按点平均。',
            '进一步诊断见progress_inner.json：包含校正默认率占比后的训练-监控差，避免仅凭总体分差把人口组成差异当成过拟合。',
            '没有用本模型在内层监控井的成绩更新原第五折排行榜；旧80.274135及原交付保持。',
            '下一步依据训练/监控曲线决定：若均衡信号不足，不扩展其指数或学习率搜索；优先检查连续值表示是否包含足够的跨井共有信息。该机制未用新模型检验。']
        (STUDY/('FINAL_'+a.phase+'.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
        if a.record:
            note='\n\n## '+result['observed_at']+' v14内层诊断完成\n两项均已自然结束。pointSW='+str(rows[0]['report']['target_score'])+'，wellSW='+str(rows[1]['report']['target_score'])+'；差'+str(comparison['SW_delta'])+'。这是48→16内层SW，不是原第五折；原第五折未评估，旧80.274135保持。训练与监控轨迹及诊断已保存，无自动补折。报告results/deep_learning/sw_group_generalization_v14_20260928/FINAL_'+a.phase+'.md。\n'
            for p in [STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md']:
                with p.open('a',encoding='utf-8') as f:f.write(note)
            for row in rows:
                p=ROOT/'outputs'/row['id']
                for filename in ['README.md','PROJECTS.md']:
                    with (p/filename).open('a',encoding='utf-8') as f:f.write(note)
    active=[]
    for proc in psutil.process_iter(['pid','name','cmdline']):
        try:
            p=proc.info;argv=p.get('cmdline') or []
            if 'python' in (p.get('name') or '').lower() and any('pure_dl/' in x.replace('\\','/') and Path(x.replace('\\','/')).name.startswith(('train_','benchmark_')) for x in argv):active.append(p['pid'])
        except(psutil.NoSuchProcess,psutil.AccessDenied):pass
    print(json.dumps(dict(all_complete=done,runs=[dict(mode=r['mode'],status=r['status'],report=r['report'],diagnostic=r.get('diagnostic')) for r in rows],
        comparison=result.get('comparison'),timing=result.get('timing'),active_training_pids=active),ensure_ascii=False))

if __name__=='__main__':main()
