"""Finite v29 launch/status and incremental evidence metadata."""
from pathlib import Path
from datetime import datetime
import argparse,ast,json,subprocess,sys
from launch_champion_posterior_v27 import jobs,append_once

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/champion_split_route_v29_20260929'
PREVIOUS=ROOT/'results/deep_learning/champion_route_v28_20260929'
TARGETS=('POR','PERM','SW')
MODES=('coupled','split')
METHOD='''# v29：默认概率与条件密度的专家路由解耦

## 探索重点
从已验证v28 TOTAL80.324397278674出发，固定全部父预测器参数，新增两条有界路由增量。动机：v28 PERM默认段改善但数值段退步；检验分开学习默认概率与条件密度占比是否减少该取舍。三目标各coupled/split，共六项，事前固定。

## 结构与匹配边界
每条支路读取两专家128+128隐藏维和63幅值维，319→32→1，共10273参数；两个不同随机输入投影合计20546参数。候选间完整初始化相同，旧v28路由保留且冻结，新输出投影为零。
每支路输出d=0.125*tanh(u/0.125)。coupled将两条支路的平均d用于默认与密度路由；split分别使用d0与d1。两条网络都实际计算、训练，没有闲置占位参数。总容量相同，但coupled让两个任务共同使用两支路，split让每个任务使用一支路，是容量分配与梯度耦合的整体比较，不能只归因于梯度冲突。

设冻结v28的默认路由logits为lA∈R2、条件密度路由为lC∈R2；二者可不同。令t(d)=[d/2,-d/2]。新默认路由aA=softmax(lA+t(dA))，默认概率G=sum_e aA_e*g_e。新条件路由aC=softmax(lC+t(dC))，非默认条件份额b_e=aC_e*(1-g_e)/sum_k aC_k*(1-g_k)。最终P=G*默认点质量+(1-G)*sum_e b_e*f_e。g_e和每个f_e全部固定。新增±0.125是相对v28的增量界，不是原模型到本轮的总修正界。

## 已实现与验证
原函数零增量一致性由实际144665点评估检查，失败停止。原v28、v25/v26、两骨干及路由全部冻结；只新支路接受梯度。每候选训练内恢复最佳，最终新采用目标另从磁盘重建并实际组合。小批训练输入验证split默认输出对密度支路梯度为零、密度输出对默认支路梯度为零；各分量位置尺度不变。未通过的检查如实记失败。

## 协议
沿用原64/16分折、578891/144665点、训练侧预处理/Q50、九曲线206主输入+63幅值，原目标似然/3和单覆盖256格/800动作。与v28同AdamW3e-4、cosine→1e-6、wd.01、clip2，80轮/pat18/min10、batch64、64步/轮、窗口seed20261010。新头seed20261105和20261106；不同目标使用私有模型。同目标候选只复用冻结计算，各自优化器/早停/参数。最多2GPU，显式启动，不自动排队。

## 收益与不再展开
启动前尚未证实。报告训练后最高分与包含第0轮的最终分；对照范围变化与继续增加一个小头的收益分开描述。未做跨种子/独立验证，不能将反复开发第五折视为独立证据。旧冷启动/扩容/后验残差负路线不恢复。本批负候选不扫边界、宽度和种子，原交付及其他四折不动。

## 记录
PROJECTS.md维护目标/已做/计划/未做/目标后事项；每候选README保留结果，登记EXPLORATIONS并增量更新演进图谱。last_state包括当前参数、优化器、调度、RNG和早停，恢复必须显式指定。trainer_provenance.json记载新训练器源于v28及精确修改范围。
'''


def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def record():
    rows=[]
    for t in TARGETS:
        out=ROOT/'outputs'/f'pure_dl_split_route_v29_{t}_fold5'
        state=load(out/'runtime_status.json') if (out/'runtime_status.json').exists() else dict(status='not_launched')
        report=load(out/'fold_5_report.json') if (out/'fold_5_report.json').exists() else None
        rows.append(dict(target=t,id=out.name,state=state,report=report))
        print('STATE',t,json.dumps(state,ensure_ascii=False),flush=True)
    data=dict(observed_at=datetime.now().astimezone().isoformat(),rows=rows,
        all_complete=all(r['report'] is not None and r['state']['status']=='complete' for r in rows),
        reference_scores=load(PREVIOUS/'CURRENT_RECIPE.json')['scores'])
    (STUDY/'progress.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    path=ROOT/'docs/deep_learning/atlas/curation.json';cur=load(path)
    for r in rows:
        t=r['target'];key=r['id'];rep=r['report'];state=r['state']
        values=[a['target_score'] for a in rep['arms']] if rep else [v['best'] for v in state.get('modes',{}).values()]
        source='outputs/'+key+('/fold_5_report.json' if rep else '/runtime_status.json')
        conclusion='尚未完成'
        if rep:conclusion='；'.join(f"{a['mode']}:更新后{a['best_after_update']['target_score']:.9f}，选定{a['target_score']:.9f}@{a['best_epoch']}" for a in rep['arms'])
        cur.setdefault('nodes',{})[key]=dict(title='v29 '+t+'路由解耦',stage='architecture',status='recorded' if rep else state['status'],
            primary=dict(value=max(values),scope='原第五折 · '+t+'单通道',kind='diagnostic',source=source) if values else None,
            summary='固定v28，20546参数增量：两支路共同路由/默认与密度分别路由。',conclusion=conclusion,
            evidence=[source,'results/deep_learning/champion_split_route_v29_20260929/METHOD.md'])
        parent='pure_dl_champion_route_v28_'+t+'_fold5'
        if not any(e.get('source')==parent and e.get('target')==key for e in cur.setdefault('edges',[])):
            cur['edges'].append(dict(source=parent,target=key,type='evolution',note='完整保留v28，分离默认与条件密度增量',evidence=source))
    tmp=path.with_suffix('.v29.tmp.json');tmp.write_text(json.dumps(cur,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
    return data


def setup():
    (STUDY/'METHOD.md').write_text(METHOD,encoding='utf-8')
    current='# v29进度\n\n目标：在80.324397上减少默认与数值路由的相互牵制。\n\n已完成：模型、有限六项协议和继承训练器。\n\n计划：显式分批训练、选择、恢复与真实组合。\n\n未完成：真实收益和独立验证。\n\n目标完成后：固定有益配方再讨论完整OOF；此轮不补折，旧交付保留。\n'
    if not (STUDY/'PROJECTS.md').exists():(STUDY/'PROJECTS.md').write_text(current,encoding='utf-8')
    marker='## v29继续当前有效模型：有限路由解耦'
    note=marker+'\n用户继续优化，本批固定v28 TOTAL80.324397，三目标coupled/split六项。全部父权重冻结，新路由增量±0.125，同20546参数和初始函数；原数据/Q50/评分/80-18/max2保持。不恢复冷启动/扩容，不补折或覆盖交付。维护champion_split_route_v29_20260929/PROJECTS.md。'
    for p in [ROOT.parent/'AGENTS.md',ROOT/'AGENTS.md',ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md']:
        append_once(p,marker,note)
    for t in TARGETS:
        out=ROOT/'outputs'/f'pure_dl_split_route_v29_{t}_fold5';out.mkdir(exist_ok=True)
        if not (out/'AGENTS.md').exists():(out/'AGENTS.md').write_text((STUDY/'AGENTS.md').read_text(encoding='utf-8'),encoding='utf-8')
        for p in [out]+[out/m for m in MODES]:
            p.mkdir(exist_ok=True)
            if not (p/'README.md').exists():(p/'README.md').write_text(METHOD+'\n目录：'+t+'/'+p.name+'\n',encoding='utf-8')
            if not (p/'PROJECTS.md').exists():(p/'PROJECTS.md').write_text(current,encoding='utf-8')
        append_once(ROOT/'EXPLORATIONS.md',out.name,'- ['+t+'默认/数值路由解耦](outputs/'+out.name+'/README.md)：v28函数保持增量，同容量对照。')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--targets',nargs='*',choices=TARGETS,default=[]);ap.add_argument('--status',action='store_true');a=ap.parse_args()
    if a.status:record();return
    setup()
    if len(set(a.targets))!=len(a.targets):raise ValueError('Duplicate target')
    if len(jobs())+len(a.targets)>2:raise RuntimeError('Maximum two GPU jobs')
    for name in ['champion_split_route_v29.py','train_champion_split_route_v29.py']:
        ast.parse(Path(__file__).with_name(name).read_text(encoding='utf-8-sig'))
    for t in a.targets:
        out=ROOT/'outputs'/f'pure_dl_split_route_v29_{t}_fold5'
        if (out/'launch.json').exists() or (out/'config.json').exists():raise RuntimeError('Do not repeat launch '+t)
    for t in a.targets:
        out=ROOT/'outputs'/f'pure_dl_split_route_v29_{t}_fold5'
        command=[sys.executable,'-X','utf8','-u',str(Path(__file__).with_name('train_champion_split_route_v29.py')),'--target',t,'--out',str(out)]
        with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=err,stdin=subprocess.DEVNULL,close_fds=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        info=dict(target=t,pid=proc.pid,started_at=datetime.now().astimezone().isoformat(),command=command)
        (out/'launch.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        for p in [STUDY/'PROJECTS.md',out/'PROJECTS.md',ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md']:
            with p.open('a',encoding='utf-8') as f:f.write('\n\n## 显式启动\n'+json.dumps(info,ensure_ascii=False)+'\n其他目标不排队。\n')
        print('LAUNCHED',json.dumps(info,ensure_ascii=False),flush=True)
    record()


if __name__=='__main__':main()
