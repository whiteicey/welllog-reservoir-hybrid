"""Launch explicitly requested finite target groups; never queue or auto-resume."""
from pathlib import Path
from datetime import datetime
import argparse,ast,json,subprocess,sys
from launch_champion_posterior_v27 import jobs,append_once
ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/crosscurve_features_v32_20260929'
TARGETS=('POR','PERM','SW')
MODES=('marginal','crosscorr')


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def record():
    rows=[]
    for t in TARGETS:
        out=ROOT/'outputs'/f'pure_dl_crosscurve_v32_{t}_fold5'
        state=read(out/'runtime_status.json') if (out/'runtime_status.json').exists() else dict(status='not_launched')
        report=read(out/'fold_5_report.json') if (out/'fold_5_report.json').exists() else None
        rows.append(dict(target=t,id=out.name,state=state,report=report))
        print('STATE',t,json.dumps(state,ensure_ascii=False),flush=True)
    result=dict(observed_at=datetime.now().astimezone().isoformat(),rows=rows,
        all_complete=all(r['report'] is not None and r['state']['status']=='complete' for r in rows))
    (STUDY/'progress.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    path=ROOT/'docs/deep_learning/atlas/curation.json';cur=read(path)
    for row in rows:
        t=row['target'];key=row['id'];rep=row['report'];state=row['state']
        values=[a['target_score'] for a in rep['arms']] if rep else [v['best'] for v in state.get('modes',{}).values()]
        source='outputs/'+key+('/fold_5_report.json' if rep else '/runtime_status.json')
        cur.setdefault('nodes',{})[key]=dict(title='v32 '+t+'跨曲线相关特征',stage='architecture',status='recorded' if rep else state['status'],
            primary=dict(value=max(values),kind='diagnostic',scope='原第五折 · '+t+'单通道',source=source) if values else None,
            summary='冻结当前模型，18维已有幅值派生对照/18维局部跨曲线相关；同结构同初始化。',
            conclusion='；'.join(f"{a['mode']}更新后{a['best_after_update']['target_score']:.9f}，保留{a['target_score']:.9f}@{a['best_epoch']}" for a in rep['arms']) if rep else '尚未完成，不能判定增益',
            evidence=[source,'results/deep_learning/crosscurve_features_v32_20260929/METHOD.md'])
        parent='pure_dl_mechanism_v31_POR_fold5' if t=='POR' else 'pure_dl_split_route_v29_'+t+'_fold5'
        if not any(e.get('source')==parent and e.get('target')==key for e in cur.setdefault('edges',[])):
            cur['edges'].append(dict(source=parent,target=key,type='evolution',note='当前有效模型固定，新增信息的同容量对照',evidence=source))
    tmp=path.with_suffix('.v32.tmp.json');tmp.write_text(json.dumps(cur,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
    return result


def setup():
    protocol=(STUDY/'AGENTS.md').read_text(encoding='utf-8')
    marker='## v32跨曲线局部相关特征对照'
    note=marker+'\n继续授权下，本批从v31 TOTAL80.336513436816开始，三目标marginal/crosscorr六项，同结构同初始化；原模型冻结，仅有限小头读取18维相关或已有统计冗余对照。原数据/Q50/评分/80-18/max2不变；暂不加显式交互，不补折、不改交付。维护crosscurve_features_v32_20260929/PROJECTS.md。'
    for p in [ROOT.parent/'AGENTS.md',ROOT/'AGENTS.md',ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md']:
        append_once(p,marker,note)
    for t in TARGETS:
        out=ROOT/'outputs'/f'pure_dl_crosscurve_v32_{t}_fold5'
        for p in [out]+[out/m for m in MODES]:
            p.mkdir(parents=True,exist_ok=True)
            if not (p/'README.md').exists():(p/'README.md').write_text(protocol+'\n\n本目录：'+t+'/'+p.name+'。尚无实测收益。\n',encoding='utf-8')
            if not (p/'PROJECTS.md').exists():(p/'PROJECTS.md').write_text((STUDY/'PROJECTS.md').read_text(encoding='utf-8'),encoding='utf-8')
        if not (out/'AGENTS.md').exists():(out/'AGENTS.md').write_text(protocol,encoding='utf-8')
        append_once(ROOT/'EXPLORATIONS.md',out.name,'- [v32 '+t+'跨曲线特征](outputs/'+out.name+'/README.md)：六对三尺度相关特征与冗余幅值视图的同结构对照。')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--targets',nargs='*',choices=TARGETS,default=[]);ap.add_argument('--status',action='store_true');a=ap.parse_args()
    if a.status:record();return
    setup()
    if len(set(a.targets))!=len(a.targets):raise ValueError('Duplicate target')
    if len(jobs())+len(a.targets)>2:raise RuntimeError('Maximum two GPU target processes')
    for name in ['crosscurve_v32.py','train_crosscurve_v32.py']:
        ast.parse(Path(__file__).with_name(name).read_text(encoding='utf-8-sig'))
    for t in a.targets:
        out=ROOT/'outputs'/f'pure_dl_crosscurve_v32_{t}_fold5'
        if (out/'launch.json').exists() or (out/'config.json').exists():raise RuntimeError('Do not repeat launch '+t)
    for t in a.targets:
        out=ROOT/'outputs'/f'pure_dl_crosscurve_v32_{t}_fold5'
        command=[sys.executable,'-X','utf8','-u',str(Path(__file__).with_name('train_crosscurve_v32.py')),'--target',t,'--out',str(out)]
        with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
            p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=err,stdin=subprocess.DEVNULL,close_fds=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        info=dict(target=t,pid=p.pid,started_at=datetime.now().astimezone().isoformat(),command=command)
        (out/'launch.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        for path in [STUDY/'PROJECTS.md',out/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md']:
            with path.open('a',encoding='utf-8') as f:f.write('\n\n## 显式启动\n'+json.dumps(info,ensure_ascii=False)+'\n其余目标不排队。\n')
        print('LAUNCHED',json.dumps(info,ensure_ascii=False),flush=True)
    record()


if __name__=='__main__':main()
