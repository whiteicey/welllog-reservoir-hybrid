"""Explicitly launch selected target jobs or observe all 11 fixed candidates."""
from pathlib import Path
from datetime import datetime
import argparse,ast,json,subprocess,sys
import psutil
from launch_latent_gate_v17 import active_jobs,append_once

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/all_heads_v26_20260928'
FACTORS={'POR':('gate','weight','scale'),'PERM':('gate','weight','location','scale'),'SW':('gate','weight','location','scale')}
METHOD='''# v26：保留当前强模型，覆盖三个目标的输出头

## 固定研究起点
POR为v25 amplitude第36轮，PERM/SW为保留v8私有双专家。现有原第五折总分80.295581058278；单通道77.520401442467/89.347502734408/73.622384767129。原骨干、校准、路由和所有已有输出参数完全冻结。无冷启动、扩容或已关闭路线重启。

## 预定义11项
POR gate/weight/scale；PERM、SW各gate/weight/location/scale。每项是独立双专家输出修正：输入原128维隐藏特征与63维幅值侧统计，191→32→1或8，门控候选12354参数，其他12816参数。输出维度随实际因子不同，这不是等输出维度的纯结构比较。新增输出投影为零。POR已采纳的位置修正保持，不再叠加重复位置候选。
每个专家完整输出包括默认logit g、8维混合权重logits w、8维位置mu、8维原始尺度s。新头读入h属于R^128和a属于R^63，产生u。gate/weight/scale使用delta=0.25*tanh(u/0.25)；location使用delta=0.25*sigma*tanh(u/(0.25*sigma))，sigma=softplus(s)+0.025。每个候选仅改变对应专家因子。尺度修正约束的是原始参数，不是直接把标准差限制为正负0.25；softplus仍确保标准差合法。门控改变会通过两专家生存质量影响最终合成混合权重。

## 训练及匹配范围
复用v25幅值输入和v21缓存/Q50，原64/16井、九曲线206主特征、256窗口/中央128点监督，原POR Student-t/PERM及SW Gaussian混合似然/3和256格/800动作单覆盖解码。其他目标标签不进入本目标损失。
每项新头独立AdamW3e-4、wd.01、clip2、cosine80×64到1e-6，max80/patience18/min10，batch64×64步。相同目标同轮次的训练窗口相同。多个候选只复用不可变的冻结推理结果，不共享可训练参数，不叠加梯度或预测。三个目标各自使用私有源模型。各候选独立选最佳轮次、独立早停；组内未停候选继续，不恢复已停候选。

## 完成及风险边界
每目标初始化的所有144665预测必须与现有配方一致。真实反向确认新头梯度和旧参数梯度为None，训练结束比对全部原权重。保存每项最佳/最佳更新后小头和预测，整组完整当前优化器/调度/RNG/早停支持显式恢复。
最后按每目标的完整候选最高分选择一个，或保留原函数；不把同目标多个因子得分直接相加，不按井/点用真值选择。新采用目标从磁盘独立重建源+小头，未改变目标复用既有验证预测。实际组合三列调用原metric，报告正/负结果和第0轮回退。
最多2个normal CUDA进程，目标显式分批，无自动排队、其他折或交付覆盖。11项和此前第五折多轮选择引入开发选择偏差，正差不等于独立泛化确认。无宽度/学习率/边界系数扫描。
'''


def load(p):return json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else None


def launch(targets):
    if not targets or len(set(targets))!=len(targets):raise ValueError('Explicit unique target list required')
    running=active_jobs()
    if len(running)+len(targets)>2:raise RuntimeError(f'Maximum2 GPU jobs; active={running}')
    for t in targets:
        if t not in FACTORS:raise ValueError(t)
        if (ROOT/'outputs'/f'pure_dl_all_heads_v26_{t}_fold5').exists():raise RuntimeError(f'{t} exists; inspect instead of duplicate launch')
    for name in ('champion_all_heads_v26.py','train_all_heads_v26.py'):
        ast.parse(Path(__file__).with_name(name).read_text(encoding='utf-8'))
    (STUDY/'METHOD.md').write_text(METHOD,encoding='utf-8')
    rules=(STUDY/'AGENTS.md').read_text(encoding='utf-8')
    marker='## v26 最新授权：三个目标的不同输出头均可优化'
    policy=marker+'\n用户明确不必局限某个头。当前直接从v25组合80.295581出发，预定义POR gate/weight/scale，PERM和SW各gate/weight/location/scale共11项；全部旧权重冻结，幅值条件小头独立训练/早停。每目标只采用一个完整候选并实际合并，不直接相加不同因子。max80/pat18/max2正常GPU，无冷启动/扩容/旧队列或其他折，维护all_heads_v26_20260928/PROJECTS.md。'
    for p in (ROOT.parent/'AGENTS.md',ROOT/'AGENTS.md'):append_once(p,marker,policy)
    records=[]
    for target in targets:
        out=ROOT/'outputs'/f'pure_dl_all_heads_v26_{target}_fold5';out.mkdir()
        (out/'AGENTS.md').write_text(rules,encoding='utf-8')
        (out/'README.md').write_text(METHOD+'\n本组：'+target+'，候选：'+','.join(FACTORS[target])+'。启动不等于获得收益。\n',encoding='utf-8')
        initial_progress='# '+target+'进度\n目标：优化当前私有目标的输出因子。\n已做：冻结起点与预定义候选。\n计划：独立训练、最佳恢复、真实组合。\n未做：本组收益和最终验证。\n达成后：有收益才采用，不自动追加或补折。\n'
        (out/'PROJECTS.md').write_text(initial_progress,encoding='utf-8')
        for factor in FACTORS[target]:
            arm=out/factor;arm.mkdir()
            (arm/'README.md').write_text(f'# {target}/{factor}\n\n'+METHOD+'\n本项只改变'+factor+'，独立参数与早停。\n',encoding='utf-8')
            (arm/'PROJECTS.md').write_text(initial_progress,encoding='utf-8')
        command=[sys.executable,'-X','utf8','-u',str(Path(__file__).with_name('train_all_heads_v26.py')),'--target',target,'--out',str(out)]
        with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
            p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,close_fds=True,
                               creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        item=dict(target=target,pid=p.pid,started_at=datetime.now().astimezone().isoformat(),command=command)
        (out/'launch.json').write_text(json.dumps(item,ensure_ascii=False,indent=2),encoding='utf-8');records.append(item)
        append_once(ROOT/'EXPLORATIONS.md',out.name,f'- [v26 {target}多输出头](outputs/{out.name}/README.md)：固定现有私有模型，{len(FACTORS[target])}项独立小头；启动时无收益结论。')
        print('LAUNCHED',json.dumps(item,ensure_ascii=False),flush=True)
    entry='## '+datetime.now().astimezone().isoformat()+' v26显式分批启动\n'+json.dumps(records,ensure_ascii=False)+'\n仅显式指定目标，剩余目标不排队，最多2GPU。'
    for p in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md'):
        with p.open('a',encoding='utf-8') as f:f.write('\n\n'+entry+'\n')


def observe(wait,record):
    if wait:
        processes=[]
        for target in FACTORS:
            s=load(ROOT/'outputs'/f'pure_dl_all_heads_v26_{target}_fold5/runtime_status.json')
            if s and s['status']=='running':
                try:
                    p=psutil.Process(s['pid'])
                    if any('train_all_heads_v26.py' in a for a in p.cmdline()):processes.append(p)
                except(psutil.NoSuchProcess,psutil.AccessDenied):pass
        psutil.wait_procs(processes,timeout=min(max(wait,0),40))
    rows=[]
    for target,factors in FACTORS.items():
        out=ROOT/'outputs'/f'pure_dl_all_heads_v26_{target}_fold5'
        status=load(out/'runtime_status.json') or dict(status='not_started');arms=[]
        for f in factors:
            arm=out/f;history=load(arm/'validation_history.json') or []
            report=load(arm/'fold_5_report.json')
            current=status.get('factors',{}).get(f,{})
            arms.append(dict(factor=f,final=report,current=current,initial=load(arm/'initial_verification.json'),
                best_after_update=max(history,key=lambda r:r['target_score']) if history else None,
                last=history[-1] if history else None,gradient_scope=load(arm/'gradient_scope.json')))
        failure=load(out/'failure.json')
        row=dict(target=target,id=out.name,status=status,arms=arms,final=load(out/'fold_5_report.json'),failure=failure)
        if failure:row['stderr_tail']=(out/'train.err').read_text(encoding='utf-8',errors='replace')[-2800:]
        rows.append(row)
    result=dict(observed_at=datetime.now().astimezone().isoformat(),rows=rows,all_complete=all(r['final'] is not None for r in rows),
        completed_arms=sum(a['final'] is not None for r in rows for a in r['arms']),declared_arms=11,
        complete_five_folds=False,remaining_folds_started=False)
    (STUDY/'progress.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if record:
        path=ROOT/'docs/deep_learning/atlas/curation.json';g=load(path)
        for r in rows:
            scores=[(a['final']['target_score'] if a['final'] else a['current'].get('best')) for a in r['arms']]
            scores=[x for x in scores if x is not None];best=max(scores) if scores else None
            source='outputs/'+r['id']+('/fold_5_report.json' if r['final'] else '/runtime_status.json')
            g.setdefault('nodes',{})[r['id']]=dict(title='v26 '+r['target']+'多输出头',stage='architecture',
                status='recorded' if r['final'] else r['status']['status'],
                primary=dict(value=best,scope='原第五折 · '+r['target']+'单通道',kind='diagnostic',source=source) if best is not None else None,
                summary='冻结现有私有源模型，幅值条件gate/weight/location/scale小头独立优化；POR保留已采纳位置修正。',
                conclusion='完成；逐头正负结果及实际总分见all_heads_v26_20260928/FINAL.md' if r['final'] else '尚未完成，不声明新增收益',
                evidence=[source,'outputs/'+r['id']+'/README.md'])
            parent='pure_dl_champion_shape_v25_amplitude_fold5' if r['target']=='POR' else 'pure_dl_independent_v8_'+r['target']+'_dual80_fold5'
            edge=dict(source=parent,target=r['id'],type='evolution',note='保留已有权重，新头各自只改变一个专家输出因子')
            if not any(e['source']==edge['source'] and e['target']==edge['target'] for e in g.setdefault('edges',[])):g['edges'].append(edge)
        path.write_text(json.dumps(g,ensure_ascii=False,indent=2),encoding='utf-8')
    compact={k:v for k,v in result.items() if k!='rows'}
    compact['rows']=[]
    for r in rows:
        compact['rows'].append(dict(target=r['target'],status=r['status'],failure=r['failure'],
            arms=[dict(factor=a['factor'],initial=a['initial'],last_epoch=a['last']['epoch'] if a['last'] else None,
                best_after_update={k:a['best_after_update'][k] for k in ('epoch','target_score','reference_replacement_total')} if a['best_after_update'] else None,
                final={k:a['final'][k] for k in ('target_score','reference_replacement_total','best_epoch','stopped_epoch','stop_reason')} if a['final'] else None) for a in r['arms']],
            **({'stderr_tail':r['stderr_tail']} if 'stderr_tail' in r else {})))
    print(json.dumps(compact,ensure_ascii=False,indent=2))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('action',choices=('launch','status'));p.add_argument('--targets',nargs='+',choices=tuple(FACTORS))
    p.add_argument('--wait',type=int,default=0);p.add_argument('--record',action='store_true');a=p.parse_args()
    if a.action=='launch':launch(a.targets)
    else:observe(a.wait,a.record)
