"""Explicit launch and observation for the two bounded champion refinements."""
from pathlib import Path
from datetime import datetime
import argparse, ast, json, subprocess, sys
import psutil
from launch_latent_gate_v17 import active_jobs, append_once

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT/'results/deep_learning/champion_shape_v25_20260928'
MODES = ('amplitude','shape')
METHOD = '''# v25：冻结80.274135，仅作输入条件位置修正

## 目标与收敛范围
用户要求回到80.27，停止无用思路。本批终止冷启动/扩容/专家训练机制的自动延伸，只做两项直接热启动候选。原冠军与交付不动，没有v24后续容量任务。

## 两项匹配实验
每个POR专家的128维隐藏特征与63维侧输入拼接，经191→32→8的小MLP产生位置建议；最终新增位置修正为0.25*sigma*tanh(proposal/(0.25*sigma))。sigma是原固定分量标准差。新输出投影和偏置为零；所有1290490个旧参数和缓冲区冻结，包括原线性位置与point32残差。只训练12816个新增参数，总参数1303306。
amplitude读取v20幅值统计侧视图；shape读取同维局部标准化形态。新头、初始化、预算与全部其他设置一致，仅输入内容不同。两组都是原80.274135出发，绝对增益对原模型计算，特征表示增益以shape减amplitude计算。原数据统计不重新拟合，shape特征在弱模型上曾有开发收益，但强模型上的收益尚未证实。

## 保持不变
主干仍只读取原206维；两个专家、校准、路由和原读出不更新。新的侧视图只进入POR位置分量；默认logit、混合权重和尺度按设计相同，并实测检查。最终默认值动作仍可能因位置分布和效用竞争变化。PERM/SW沿用各自已验证私有模型和预测。
原64/16井、Q50、九曲线、256点窗口/中央128监督、POR Student-t混合似然/3、256格/800动作单覆盖解码不变。新头固定lr3e-4，AdamW wd.01、clip2、cosine80×64到1e-6、bs64、每轮64步、max80/pat18/min10。模型整体eval、仅新头训练；没有新dropout。输入侧视图中9/51/201窗口不跨井，全局常数由64训练井拟合。

## 完成与停止标准
训练前核对所有144665点初始预测。保存当前模型/优化器/调度器/RNG/早停/最佳和更新后最佳；检查原参数不变和新梯度。所有分数按原验证行序实际组合计算。若无超过原模型的收益，保留原模型且不追加宽度/窗口/系数扫描。不得用第0轮回退宣称训练收益。
本批只两任务，normal CUDA/max2，无其他折、旧任务或交付更改。第五折反复开发，局部增益不冒称独立泛化/完整OOF。维护PROJECTS、README/EXPLORATIONS和图谱，失败/未完成如实记录。
'''


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else None


def launch():
    if active_jobs(): raise RuntimeError('Active trainers: inspect before starting')
    for mode in MODES:
        if (ROOT/'outputs'/f'pure_dl_champion_shape_v25_{mode}_fold5').exists():
            raise RuntimeError('Output exists; never overwrite or duplicate')
    for name in ('champion_shape_v25.py','train_champion_shape_v25.py'):
        ast.parse(Path(__file__).with_name(name).read_text(encoding='utf-8'))
    rules=(STUDY/'AGENTS.md').read_text(encoding='utf-8')
    (STUDY/'METHOD.md').write_text(METHOD,encoding='utf-8')
    marker='## 最新方向：回到80.274135，停止无收益替换路线'
    policy=marker+'\n用户明确要求回到80.27继续改进。冷启动替换、继续扩容及专家训练方式的后续实验停止延伸，不恢复既有队列。本批v25只做amplitude/shape两项小型POR位置修正；所有原冠军参数固定，只训练12816新参数，原数据/Q50/分折/解码不变。无其他折和交付覆盖，维护champion_shape_v25_20260928/PROJECTS.md。'
    for path in (ROOT.parent/'AGENTS.md',ROOT/'AGENTS.md'):
        append_once(path,marker,policy)
    records=[]
    for mode in MODES:
        out=ROOT/'outputs'/f'pure_dl_champion_shape_v25_{mode}_fold5';out.mkdir()
        for name,text in [('AGENTS.md',rules),('README.md',METHOD+'\n本项：'+mode+'。启动不等于取得收益。\n'),
            ('PROJECTS.md',(STUDY/'PROJECTS.md').read_text(encoding='utf-8'))]:
            (out/name).write_text(text,encoding='utf-8')
        command=[sys.executable,'-X','utf8','-u',str(Path(__file__).with_name('train_champion_shape_v25.py')),'--mode',mode,'--out',str(out)]
        with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
            process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,close_fds=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        record=dict(mode=mode,pid=process.pid,started_at=datetime.now().astimezone().isoformat(),command=command)
        (out/'launch.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8');records.append(record)
        append_once(ROOT/'EXPLORATIONS.md',out.name,f'- [v25 {mode}](outputs/{out.name}/README.md)：80.274135固定旧权重，小型位置头热启动；收益待实测。')
        print('LAUNCHED',json.dumps(record,ensure_ascii=False),flush=True)
    entry='## '+datetime.now().astimezone().isoformat()+' v25显式启动\n'+json.dumps(records,ensure_ascii=False)+'\n仅两项直接在冠军上改进；停止冷启动/扩容延伸，无自动追加。'
    for p in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md'):
        with p.open('a',encoding='utf-8') as stream:stream.write('\n\n'+entry+'\n')


def observe(wait=0,record=False):
    if wait:
        processes=[]
        for mode in MODES:
            s=read(ROOT/'outputs'/f'pure_dl_champion_shape_v25_{mode}_fold5/runtime_status.json')
            if s and s['status']=='running':
                try:
                    p=psutil.Process(s['pid'])
                    if any('train_champion_shape_v25.py' in a for a in p.cmdline()):processes.append(p)
                except(psutil.NoSuchProcess,psutil.AccessDenied):pass
        psutil.wait_procs(processes,timeout=min(max(wait,0),40))
    rows=[];configs=[]
    for mode in MODES:
        out=ROOT/'outputs'/f'pure_dl_champion_shape_v25_{mode}_fold5'
        cfg=read(out/'config.json');configs.append(cfg)
        history=read(out/'validation_history.json') or []
        row=dict(mode=mode,id=out.name,status=read(out/'runtime_status.json'),initial=read(out/'initial_verification.json'),
            final=read(out/'fold_5_report.json'),latest=history[-1] if history else None,
            best_after_update=max(history,key=lambda r:r['target_score']) if history else None,
            gradient_scope=read(out/'gradient_scope.json'),failure=read(out/'failure.json'))
        if row['failure']:row['stderr_tail']=(out/'train.err').read_text(encoding='utf-8',errors='replace')[-3000:]
        rows.append(row)
    result=dict(observed_at=datetime.now().astimezone().isoformat(),rows=rows,all_complete=all(r['final'] is not None for r in rows),
        complete_five_folds=False,remaining_folds_started=False)
    if all(configs):
        differences=[k for k in sorted(configs[0]) if configs[0][k]!=configs[1][k]]
        assert set(differences)=={'mode','input_view'},differences
        result['config_differences']=differences; result['same_initial_state_and_trainable_parameters']=True
    (STUDY/'progress.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if record:
        graphpath=ROOT/'docs/deep_learning/atlas/curation.json';graph=read(graphpath)
        for r in rows:
            done=r['final'];status=r['status'] or {'status':'not_started'}
            score=done['target_score'] if done else status.get('best_target_score')
            source='outputs/'+r['id']+('/fold_5_report.json' if done else '/runtime_status.json')
            graph.setdefault('nodes',{})[r['id']]=dict(title='v25冠军位置修正 '+r['mode'],stage='architecture',
                status='recorded' if done else status['status'],
                primary=dict(value=score,scope='原第五折 · POR单通道',kind='diagnostic',source=source) if score is not None else None,
                summary='固定80.274135全部旧权重，新12816参数只修正POR位置，幅值/形态侧视图匹配对照。',
                conclusion=(f"完成POR={score:.9f}，最佳第{done['best_epoch']}轮，实际总分见研究assembled_report。" if done else '尚未完成，不能报告新增收益'),
                evidence=[source,'outputs/'+r['id']+'/README.md'])
        edges=[dict(source='pure_dl_independent_v8_POR_dual80_fold5',target=rows[0]['id'],type='evolution',note='直接热启动冠军，冻结原函数，仅训练新增位置头'),
            dict(source=rows[0]['id'],target=rows[1]['id'],type='control',note='同头同参数同初始函数，只改变63维侧视图')]
        for e in edges:
            if not any(x['source']==e['source'] and x['target']==e['target'] for x in graph.setdefault('edges',[])):graph['edges'].append(e)
        graphpath.write_text(json.dumps(graph,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('action',choices=('launch','status'));p.add_argument('--wait',type=int,default=0);p.add_argument('--record',action='store_true')
    a=p.parse_args()
    if a.action=='launch':launch()
    else:observe(a.wait,a.record)
