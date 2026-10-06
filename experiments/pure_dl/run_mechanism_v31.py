"""Explicit finite launch and status; no automatic queue or other-fold continuation."""
from pathlib import Path
from datetime import datetime
import argparse,ast,json,subprocess,sys
from launch_champion_posterior_v27 import jobs,append_once
ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/mechanism_checks_v31_20260929'
MODES={'POR':('tanh','anchored'),'SW':('affine','pwl4')}


def record():
    rows=[]
    for target in MODES:
        out=ROOT/'outputs'/f'pure_dl_mechanism_v31_{target}_fold5'
        state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else dict(status='not_launched')
        report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
        rows.append(dict(target=target,id=out.name,state=state,report=report))
        print('STATE',target,json.dumps(state,ensure_ascii=False),flush=True)
    result=dict(observed_at=datetime.now().astimezone().isoformat(),all_complete=all(r['report'] and r['state']['status']=='complete' for r in rows),rows=rows)
    (STUDY/'progress.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    path=ROOT/'docs/deep_learning/atlas/curation.json';cur=json.loads(path.read_text(encoding='utf-8'))
    for row in rows:
        t=row['target'];key=row['id'];rep=row['report'];state=row['state']
        scores=[a['target_score'] for a in rep['arms']] if rep else [s['best'] for s in state.get('modes',{}).values()]
        source='outputs/'+key+('/fold_5_report.json' if rep else '/runtime_status.json')
        cur.setdefault('nodes',{})[key]=dict(title='v31 '+t+('同范围路由重参数化' if t=='POR' else '单调连续分布修正'),stage='architecture',
            status='recorded' if rep else state['status'],primary=dict(value=max(scores),kind='diagnostic',scope='原第五折 · '+t+'单通道',source=source) if scores else None,
            summary='固定当前v30完整预测器；两项预定义匹配对照，原评分/数据不变。',
            conclusion='；'.join(f"{a['mode']}选定{a['target_score']:.9f}@{a['best_epoch']}，更新后{a['best_after_update']['target_score']:.9f}" for a in rep['arms']) if rep else '尚未完成，不宣称收益',
            evidence=[source,'results/deep_learning/mechanism_checks_v31_20260929/AGENTS.md'])
        parent='pure_dl_coadapt_v30_POR_fold5' if t=='POR' else 'pure_dl_split_route_v29_SW_fold5'
        if not any(e.get('source')==parent and e.get('target')==key for e in cur.setdefault('edges',[])):
            cur['edges'].append(dict(source=parent,target=key,type='evolution',note='用户采纳诊断后的有限机制验证',evidence=source))
    tmp=path.with_suffix('.v31.tmp.json');tmp.write_text(json.dumps(cur,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
    return result


def setup():
    protocol=(STUDY/'AGENTS.md').read_text(encoding='utf-8')
    initial='# v31进度\n\n目标：验证POR饱和参数化和SW连续分布表达机制。\n\n已完成：固定四项实验协议与训练入口。\n\n计划：按目标显式运行；记录初始/更新后/最终结果；采用目标磁盘重载、实际组合和图谱。\n\n未完成：真实训练收益与独立泛化确认。\n\n达成后：保留有效配方；跨曲线特征仅在后续授权下进行，当前不补折、不改旧交付。\n'
    if not (STUDY/'PROJECTS.md').exists():(STUDY/'PROJECTS.md').write_text(initial,encoding='utf-8')
    marker='## v31用户采纳诊断机制验证'
    note=marker+'\n基于80.336211945752，仅POR同边界tanh/anchored和SW affine/pwl4共四项。PERM冻结，原9曲线/数据/Q50/80-18/max2/解码不变。新目标变换计入完整密度Jacobian；不得将参数化变化误算成初始收益。不补折、旧交付保留。维护mechanism_checks_v31_20260929/PROJECTS.md。'
    for p in [ROOT.parent/'AGENTS.md',ROOT/'AGENTS.md',ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md']:
        append_once(p,marker,note)
    for t,modes in MODES.items():
        out=ROOT/'outputs'/f'pure_dl_mechanism_v31_{t}_fold5'
        for p in [out]+[out/m for m in modes]:
            p.mkdir(parents=True,exist_ok=True)
            if not (p/'README.md').exists():(p/'README.md').write_text(protocol+'\n\n本目录：'+t+'/'+p.name+'；尚未证实增益。\n',encoding='utf-8')
            if not (p/'PROJECTS.md').exists():(p/'PROJECTS.md').write_text(initial,encoding='utf-8')
        if not (out/'AGENTS.md').exists():(out/'AGENTS.md').write_text(protocol,encoding='utf-8')
        append_once(ROOT/'EXPLORATIONS.md',out.name,'- [v31 '+t+'诊断机制验证](outputs/'+out.name+'/README.md)：当前强模型上的有限参数化/分布对照。')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--targets',nargs='*',choices=tuple(MODES),default=[]);ap.add_argument('--status',action='store_true');a=ap.parse_args()
    if a.status:record();return
    setup()
    if len(a.targets)!=len(set(a.targets)):raise ValueError('Duplicate targets')
    if len(jobs())+len(a.targets)>2:raise RuntimeError('GPU jobs would exceed two')
    for t in a.targets:
        out=ROOT/'outputs'/f'pure_dl_mechanism_v31_{t}_fold5'
        if (out/'launch.json').exists() or (out/'config.json').exists():raise RuntimeError('Already launched '+t)
        module='por_reparam_v31.py' if t=='POR' else 'sw_warp_v31.py'
        for f in ['train_mechanism_v31.py',module]:ast.parse(Path(__file__).with_name(f).read_text(encoding='utf-8-sig'))
    for t in a.targets:
        out=ROOT/'outputs'/f'pure_dl_mechanism_v31_{t}_fold5'
        command=[sys.executable,'-X','utf8','-u',str(Path(__file__).with_name('train_mechanism_v31.py')),'--target',t,'--out',str(out)]
        with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=err,stdin=subprocess.DEVNULL,close_fds=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        info=dict(target=t,pid=proc.pid,started_at=datetime.now().astimezone().isoformat(),command=command)
        (out/'launch.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        for path in [STUDY/'PROJECTS.md',out/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md']:
            with path.open('a',encoding='utf-8') as f:f.write('\n\n## 显式启动\n'+json.dumps(info,ensure_ascii=False)+'\n没有自动队列。\n')
        print('LAUNCHED',json.dumps(info,ensure_ascii=False),flush=True)
    record()

if __name__=='__main__':main()
