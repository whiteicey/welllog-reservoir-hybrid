"""Finite launch/status interface for existing-head joint refinement."""
from pathlib import Path
from datetime import datetime
import argparse,ast,json,subprocess,sys
from launch_champion_posterior_v27 import jobs,append_once
ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/champion_coadapt_v30_20260929'
PREVIOUS=ROOT/'results/deep_learning/champion_split_route_v29_20260929'
TARGETS=('POR','PERM','SW')
MODES=('route_only','coadapt')
METHOD='''# v30：已有有效修正头与路由协同微调

## 探索重点与预定义比较
从v29固定三目标模型80.335082954320开始，不增加新的学习模块。每目标route_only/coadapt两个候选，共六项；全部从已选模型原参数开始，采用同一低学习率重新初始化优化器，不能称作旧训练轨迹续跑。
route_only允许v28的10273参数路由与v29的20546参数路由更新，共30819；coadapt另外更新已有效的POR v25幅值位置修正12816参数，PERM v26门控12354参数，SW v26分量权重12816参数。因此coadapt可训练参数分别43635/43173/43635。两组网络结构和初始参数一致，只变更新范围。

## 计算与冻结
原双骨干、输入校准、原路由和原线性/point32输出固定。每目标只复用冻结骨干的中间特征，候选的完整读出和已有路由分别复制，避免任一候选改变另一候选；重新调用原读出保持运算顺序，POR仍按raw_mean+(point32+amplitude)计算，不把浮点加法顺序改变误算为效果。旧读出复制体仅作隔离计算，不引入新的模型功能；初始全144665点必须与v29一致，否则停止。
coadapt训练的参数来自现有已学模块，修正界仍是v25/v26/v28/v29原设定，没有额外放宽。原骨干与被固定的读出参数及缓冲区在终场逐项验证。训练后的单因子不变性检查只用固定训练窗口。

## 协同目标
每目标的完整预测似然L(theta_route,theta_factor)保持原实现。route_only优化theta_route并固定theta_factor；coadapt共同优化二者。学习率均3e-5、cosine到1e-6、AdamW wd0.01，路由和因子分别clip2，路由组的裁剪口径两候选相同。相同窗口seed20261010、原64井训练/16井验证、578891/144665点、Q50、九曲线206输入+63幅值、80轮/patience18/min10、batch64和64步。

## 评价与证据边界
原单覆盖256格/800动作、原评分不变，训练只读训练标签。单目标最佳与停止状态独立，最终每目标只选一个固定完整候选并实际组合。所有候选报告第0轮、训练后最高分和最终选定分。只有共同训练相对route_only的分差才用于讨论协同更新的附加效果；相对v29还包含了重新开始微调的收益。
同一第五折长期用于开发，六项选择并未独立确认，结果不代表新井泛化保证。不训练其他折、不开旧队列，不覆盖已有交付。负结果不再展开宽度、修正界或学习率网格。

## 记录
原子last_state含当前参数/优化器/调度器/RNG/早停；仅显式恢复。每目标候选维护README与PROJECTS，研究清单登记EXPLORATIONS并同步图谱。源代码champion_coadapt_v30.py/train_coadapt_v30.py，来源映射trainer_provenance.json。继承v28训练循环而不修改历史文件。
'''


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def record():
    rows=[]
    for t in TARGETS:
        out=ROOT/'outputs'/f'pure_dl_coadapt_v30_{t}_fold5'
        state=read(out/'runtime_status.json') if (out/'runtime_status.json').exists() else dict(status='not_launched')
        report=read(out/'fold_5_report.json') if (out/'fold_5_report.json').exists() else None
        rows.append(dict(target=t,id=out.name,state=state,report=report))
        print('STATE',t,json.dumps(state,ensure_ascii=False),flush=True)
    data=dict(observed_at=datetime.now().astimezone().isoformat(),rows=rows,
        all_complete=all(r['report'] is not None and r['state']['status']=='complete' for r in rows),reference_scores=read(PREVIOUS/'CURRENT_RECIPE.json')['scores'])
    (STUDY/'progress.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    path=ROOT/'docs/deep_learning/atlas/curation.json';cur=read(path)
    for r in rows:
        t=r['target'];key=r['id'];rep=r['report'];state=r['state']
        vals=[a['target_score'] for a in rep['arms']] if rep else [a['best'] for a in state.get('modes',{}).values()]
        source='outputs/'+key+('/fold_5_report.json' if rep else '/runtime_status.json')
        detail='尚未完成，不能判定收益'
        if rep:detail='；'.join(f"{a['mode']}更新后{a['best_after_update']['target_score']:.9f}，选定{a['target_score']:.9f}@{a['best_epoch']}" for a in rep['arms'])
        cur.setdefault('nodes',{})[key]=dict(title='v30 '+t+'已有模块协同微调',stage='architecture',status='recorded' if rep else state['status'],
            primary=dict(value=max(vals),scope='原第五折 · '+t+'单通道',kind='diagnostic',source=source) if vals else None,
            summary='v29同函数热启动，不增加容量；仅路由对比路由+有效因子共同更新。',conclusion=detail,
            evidence=[source,'results/deep_learning/champion_coadapt_v30_20260929/METHOD.md'])
        parent='pure_dl_split_route_v29_'+t+'_fold5'
        if not any(e.get('source')==parent and e.get('target')==key for e in cur.setdefault('edges',[])):
            cur['edges'].append(dict(source=parent,target=key,type='evolution',note='已有有效模块共同微调；无新学习支路',evidence=source))
    temp=path.with_suffix('.v30.tmp.json');temp.write_text(json.dumps(cur,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)
    return data


def setup():
    (STUDY/'METHOD.md').write_text(METHOD,encoding='utf-8')
    initial='# v30进度\n\n目标：在不增加容量条件下优化已有头与路由。\n\n已完成：模型/训练入口及六项协议。\n\n计划：显式分批训练，固定每目标方案后实际组合；采用模型独立重建。\n\n未完成：训练成绩、独立泛化确认。\n\n达成后：保留正收益研究配方，再由用户授权决定完整OOF；旧交付不变。\n'
    if not (STUDY/'PROJECTS.md').exists():(STUDY/'PROJECTS.md').write_text(initial,encoding='utf-8')
    marker='## v30已有有效模块协同微调'
    note=marker+'\n本批从v29 TOTAL80.335083出发，三目标route_only/coadapt六项；无新增学习模块，保留主骨干，仅路由或路由与已有效输出修正共同更新。统一3e-5、分组clip2，原数据/Q50/评分/80-18/max2不变。无其他折、旧队列或交付覆盖。维护champion_coadapt_v30_20260929/PROJECTS.md。'
    for p in [ROOT.parent/'AGENTS.md',ROOT/'AGENTS.md',ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md']:
        append_once(p,marker,note)
    for t in TARGETS:
        out=ROOT/'outputs'/f'pure_dl_coadapt_v30_{t}_fold5';out.mkdir(exist_ok=True)
        if not (out/'AGENTS.md').exists():(out/'AGENTS.md').write_text((STUDY/'AGENTS.md').read_text(encoding='utf-8'),encoding='utf-8')
        for p in [out]+[out/m for m in MODES]:
            p.mkdir(exist_ok=True)
            if not (p/'README.md').exists():(p/'README.md').write_text(METHOD+'\n本目录：'+t+'/'+p.name+'\n',encoding='utf-8')
            if not (p/'PROJECTS.md').exists():(p/'PROJECTS.md').write_text(initial,encoding='utf-8')
        append_once(ROOT/'EXPLORATIONS.md',out.name,'- ['+t+'现有输出修正与路由协同](outputs/'+out.name+'/README.md)：同v29起点、有限更新范围对照。')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--targets',nargs='*',choices=TARGETS,default=[]);ap.add_argument('--status',action='store_true');a=ap.parse_args()
    if a.status:record();return
    setup()
    if len(set(a.targets))!=len(a.targets):raise ValueError('Duplicate target')
    if len(jobs())+len(a.targets)>2:raise RuntimeError('Maximum two GPU jobs')
    for name in ['champion_coadapt_v30.py','train_coadapt_v30.py']:
        ast.parse(Path(__file__).with_name(name).read_text(encoding='utf-8-sig'))
    for t in a.targets:
        out=ROOT/'outputs'/f'pure_dl_coadapt_v30_{t}_fold5'
        if (out/'launch.json').exists() or (out/'config.json').exists():raise RuntimeError('Already launched '+t)
    for t in a.targets:
        out=ROOT/'outputs'/f'pure_dl_coadapt_v30_{t}_fold5'
        command=[sys.executable,'-X','utf8','-u',str(Path(__file__).with_name('train_coadapt_v30.py')),'--target',t,'--out',str(out)]
        with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
            p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=err,stdin=subprocess.DEVNULL,close_fds=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        info=dict(target=t,pid=p.pid,started_at=datetime.now().astimezone().isoformat(),command=command)
        (out/'launch.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        for path in [STUDY/'PROJECTS.md',out/'PROJECTS.md',ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md']:
            with path.open('a',encoding='utf-8') as f:f.write('\n\n## 显式启动\n'+json.dumps(info,ensure_ascii=False)+'\n其他目标不排队。\n')
        print('LAUNCHED',json.dumps(info,ensure_ascii=False),flush=True)
    record()


if __name__=='__main__':main()
