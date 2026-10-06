"""Set up a finite v28 protocol; launch only explicitly named target groups."""
from pathlib import Path
from datetime import datetime
import argparse,ast,json,subprocess,sys
from launch_champion_posterior_v27 import jobs,append_once

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/champion_route_v28_20260929'
PREVIOUS=ROOT/'results/deep_learning/all_heads_v26_20260928'
TARGETS=('POR','PERM','SW')
MODES=('joint','conditional')
METHOD='''# v28：保留80.30预测器，修正内部专家占比

## 探索重点
v26修正的是每个专家内部输出。本批不改变每个专家给出的默认概率、8个分量内部权重、位置或尺度，改动两位专家在最终混合中的占比。原路由网络也固定，新增一个小型路由残差。

## 两项匹配设计
每目标joint/conditional两项，共六项；所有候选10273参数，结构与初始张量完全相同。输入两专家隐藏128+128维与63维幅值=319维，319→32→1；输出零初始化。产生标量delta=0.25*tanh(u/0.25)，给原路由logits增加[delta/2,-delta/2]，使log(a1/a2)仅改变delta。
设原专家默认概率g_e，原路由a_e。原最终默认概率G=sum_e a_e*g_e，条件份额b_e=a_e*(1-g_e)/sum_k a_k*(1-g_k)。
joint用新a'同时更新G'与b'；conditional使用同一个新b'，但保持原G精确不变。两组的条件路由映射公式相同，只改变是否让默认质量同时变化。默认概率固定不代表最终输出默认值的数量固定，原效用解码仍比较全部动作。

## 训练协议
父配方v26原第五折80.300730612039，已有效的POR v25位置头、PERM v26门控头与SW v26权重头全部保留。加载器复用v27的父预测器构建接口，不加载或运行v27训练的新头。
保持原9曲线206维主输入+63幅值、原64训练井/16验证井、578891/144665点、Q50及训练侧统计、单覆盖256格/800候选动作和默认动作、原own-target似然/3。AdamW lr3e-4、cosine到1e-6、wd.01、clip2，seed20261010，新增初始化seed20261104，80轮上限/pat18/min10/64步/batch64。
不同目标私有骨干，不共享可训练参数；同目标两候选只复用冻结前向，各自优化器、调度、早停及权重。最多2GPU任务，显式分批，无自动队列或补折。

## 已实现与验证范围
零输出起点必须全量复现原对应列；检查实际梯度、旧状态冻结和选定参数恢复。conditional的最终门控与各专家分量位置/尺度逐元素固定，在最终选定模型和更新后模型上做少量训练窗口检查。采用的正增量模型从磁盘独立重建，实际三列合并调用原评分。

## 收益及判断规则
开始前尚未证实。同时报告排除第0轮的训练后最高分、最终选定分和conditional-joint差值。固定原模型是两项共享的基线；该对照检验更新范围，不把不同额外结构的所有变化归给一个因子。一个反复开发的第五折、六项选模，不作独立泛化结论。负结果不扫修正幅度、宽度或种子。

## 不继续的方向与记录
不重启冷启动、扩容、重复后验小头、旧ABD或其他折，不改原交付。当前研究只做六项路由比较。维护各README、PROJECTS、EXPLORATIONS和图谱；完整目标>82未完成。所有last_state含当前参数/优化器/调度器/RNG/早停，仅显式恢复。
'''


def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def record():
    rows=[]
    for t in TARGETS:
        out=ROOT/'outputs'/f'pure_dl_champion_route_v28_{t}_fold5'
        state=load(out/'runtime_status.json') if (out/'runtime_status.json').exists() else {'status':'not_launched'}
        report=load(out/'fold_5_report.json') if (out/'fold_5_report.json').exists() else None
        rows.append(dict(target=t,id=out.name,state=state,report=report))
        print('STATE',t,json.dumps(state,ensure_ascii=False),flush=True)
    data=dict(observed_at=datetime.now().astimezone().isoformat(),rows=rows,
        all_complete=all(r['report'] and r['state']['status']=='complete' for r in rows),
        reference_scores=load(PREVIOUS/'CURRENT_RECIPE.json')['scores'])
    (STUDY/'progress.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    here=ROOT/'docs/deep_learning/atlas';path=here/'curation.json';cur=load(path)
    for row in rows:
        t=row['target'];key=row['id'];rep=row['report'];state=row['state']
        values=[a['target_score'] for a in rep['arms']] if rep else [v['best'] for v in state.get('modes',{}).values()]
        best=max(values) if values else None
        source='outputs/'+key+('/fold_5_report.json' if rep else '/runtime_status.json')
        conclusion='尚未完成，不能判定增益。'
        if rep:conclusion='；'.join(f"{a['mode']}更新后{a['best_after_update']['target_score']:.9f}，选定{a['target_score']:.9f}@{a['best_epoch']}" for a in rep['arms'])
        cur.setdefault('nodes',{})[key]=dict(title='v28 '+t+'内部专家路由',stage='architecture',status='recorded' if rep else state['status'],
            primary=dict(value=best,scope='原第五折 · '+t+'单通道（含第0轮）',kind='diagnostic',source=source) if best is not None else None,
            summary='10273参数的联合/仅条件路由对照；v26全部专家输出和既有修正固定。',conclusion=conclusion,
            evidence=[source,'results/deep_learning/champion_route_v28_20260929/METHOD.md'])
        parent='pure_dl_champion_shape_v25_amplitude_fold5' if t=='POR' else 'pure_dl_all_heads_v26_'+t+'_fold5'
        if not any(x.get('source')==parent and x.get('target')==key for x in cur.setdefault('edges',[])):
            cur['edges'].append(dict(source=parent,target=key,type='evolution',note='原模型上只改变内部专家占比',evidence=source))
    path.write_text(json.dumps(cur,ensure_ascii=False,indent=2),encoding='utf-8')
    return data


def setup():
    STUDY.mkdir(parents=True,exist_ok=True)
    (STUDY/'METHOD.md').write_text(METHOD,encoding='utf-8')
    current='# v28进度\n\n目标：当前80.30模型上的有限路由优化。\n\n已完成：模型、训练代码及六项协议。\n\n计划：各目标joint/conditional独立训练和选模，复现初始功能，实际验证组合。\n\n未完成：实测训练收益和独立泛化证据；其他折不启动。\n\n目标达成后：固定正收益配方，由用户授权决定完整OOF确认；旧交付不覆盖。\n'
    if not (STUDY/'PROJECTS.md').exists():(STUDY/'PROJECTS.md').write_text(current,encoding='utf-8')
    marker='## v28当前80.30的路由更新范围对照'
    note=marker+'\n继续优化授权下，本批预定义三目标joint/conditional六项；只新增内部路由log-odds小头，原专家和v25/v26头冻结。原数据/Q50/解码/80-18/max2保持，显式分批，无其他折、冷启动、扩容和交付覆盖。详见champion_route_v28_20260929/PROJECTS.md。'
    for p in [ROOT.parent/'AGENTS.md',ROOT/'AGENTS.md',ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md']:
        append_once(p,marker,note)
    for t in TARGETS:
        out=ROOT/'outputs'/f'pure_dl_champion_route_v28_{t}_fold5';out.mkdir(exist_ok=True)
        if not (out/'AGENTS.md').exists():(out/'AGENTS.md').write_text((STUDY/'AGENTS.md').read_text(encoding='utf-8'),encoding='utf-8')
        for p in [out]+[out/m for m in MODES]:
            p.mkdir(exist_ok=True)
            if not (p/'README.md').exists():(p/'README.md').write_text(METHOD+'\n目标/目录：'+t+'/'+p.name+'\n',encoding='utf-8')
            if not (p/'PROJECTS.md').exists():(p/'PROJECTS.md').write_text(current,encoding='utf-8')
        append_once(ROOT/'EXPLORATIONS.md',out.name,'- ['+t+'联合/仅条件专家路由](outputs/'+out.name+'/README.md)：保留v26基座，更新范围匹配对照。')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--targets',nargs='*',choices=TARGETS,default=[])
    ap.add_argument('--status',action='store_true');args=ap.parse_args()
    if args.status:record();return
    setup()
    if len(set(args.targets))!=len(args.targets):raise ValueError('Duplicate targets')
    if len(jobs())+len(args.targets)>2:raise RuntimeError('Maximum two GPU jobs')
    for name in ['champion_route_v28.py','train_champion_route_v28.py']:
        ast.parse(Path(__file__).with_name(name).read_text(encoding='utf-8-sig'))
    for t in args.targets:
        out=ROOT/'outputs'/f'pure_dl_champion_route_v28_{t}_fold5'
        if (out/'launch.json').exists() or (out/'config.json').exists():raise RuntimeError('Already started '+t)
    for t in args.targets:
        out=ROOT/'outputs'/f'pure_dl_champion_route_v28_{t}_fold5'
        command=[sys.executable,'-X','utf8','-u',str(Path(__file__).with_name('train_champion_route_v28.py')),'--target',t,'--out',str(out)]
        with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
            p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=err,stdin=subprocess.DEVNULL,close_fds=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        info=dict(target=t,pid=p.pid,started_at=datetime.now().astimezone().isoformat(),command=command)
        (out/'launch.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        for p in [STUDY/'PROJECTS.md',out/'PROJECTS.md',ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md']:
            with p.open('a',encoding='utf-8') as f:f.write('\n\n## 显式启动\n'+json.dumps(info,ensure_ascii=False)+'\n其余目标不排队。\n')
        print('LAUNCHED',json.dumps(info,ensure_ascii=False),flush=True)
    record()


if __name__=='__main__':main()
