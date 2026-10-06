"""Prepare and explicitly launch only the two predeclared component-query arms.
No polling, queue, restart, fullCV, or mutation of historical models/trainers.
"""
from pathlib import Path
from datetime import datetime
import argparse,ast,hashlib,json,subprocess,sys
import psutil

CODE=Path(__file__).resolve().parent
ROOT=CODE.parents[1]
STUDY=ROOT/'results/deep_learning/dual8222_component_query_v7_20260928'
MODES=['center','local']
PREFIX='pure_dl_dual8222_v7_query_'
PLAN='''# 分量查询式交叉注意力：精确参数匹配的局部记忆对照

## 目标与探索重点
在已验证80.274135337791的双专家预测器上，给两专家各8个POR混合分量分别构造查询，共16个查询。查询包含专家隐藏向量、分量身份、当前预测的位置/log标准差/log混合权重，读取局部特征记忆后输出位置修正。
本批与v6的重要差异：整个旧预测器（含线性位置头、旧非线性修正）全部固定，只训练新模块，避免原线性读出继续漂移。该范围变化与查询机制同时发生，不能通过v7-v6直接归因；本批center/local是严格同条件比较。

## 预先固定的两个方案
- center：34个记忆槽位，17个相对偏移×2个内部专家；所有槽位的特征来自当前点，相对位置和有效性掩码与local一致。旧骨干特征本身已含上下文，因此不称“完全不使用上下文”。
- local：相同结构、参数、初始化和掩码，槽位读取当前点前后8点、两位专家的真实隐藏特征；不跨井。
两组均14081个新参数，32维、2头×16维，16查询×34键；共享读出MLP为32->64->32，再逐分量输出一个标量。局部性只约束新模块的直接读取，不是完整模型总感受野。
新增修正delta=(0.25*sigma)*tanh(r/(0.25*sigma))，sigma来自旧模型且固定。输出层零初始化；原80.274135应逐元素复现。该界只约束新增修正，不是历轮总修正上限。

## 训练合同与已实现验证
固定原九曲线/206特征/原第五折/seed20261010/训练Q50/原单覆盖256格800动作；只有64训练井标签参与损失。保留原POR BCE+非默认NLL除3，不改目标。新模块AdamW lr3e-4余弦至1e-5，wd0.01，clip2，max80/patience18/min10、batch64、64步/轮。
两个新实验优化器从头开始，不是上次训练续跑。所有旧参数与PERM/SW在每轮/最终验证中检查不变。初始回放、首批梯度与最佳恢复在正式训练中检查，不另开预检任务。
每轮保存原子last_state，包括当前参数/优化器/调度器/RNG/早停/最佳状态。只有显式--resume才恢复，未经实测不声称等价性。训练报告保留运行、失败、早停和完成状态。

## 收益与关闭标准
启动前尚无收益证据。center与local以同容量对照验证是否需要邻近点；原80.274135作为冻结无新增支路基准。第0轮最佳等于无新训练收益，不能把恢复原函数说成attention成果。
本批仅两项，不根据中途结果追加半径、头数、学习率、宽度或损失搜索。两个完成后恢复固定参数，比较正式第五折与逐井/默认人口结果；若无收益继续保留原v5。第五折反复开发，不当独立测试；严格>80.5和完整OOF>82仍未达到。旧ABD/其他折保持暂停，交付不变。

## 方法依据
缩放点积与遮挡语义参考PyTorch官方scaled_dot_product_attention文档；本实现用显式float32 QK/softmax/AV，无dropout。可学习输出查询借鉴Transformer解码器设计思路，非DETR或其他论文的完整复现。
'''


def trainers():
    found=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            v=p.info;a=v.get('cmdline') or []
            if 'python' in (v.get('name') or '').lower() and 'pure_dl' in ' '.join(a) and any(Path(x.replace('\\','/')).name.startswith(('train_','benchmark_')) for x in a):found.append(v)
        except (psutil.AccessDenied,psutil.NoSuchProcess):pass
    return found


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--launch',choices=MODES,nargs='*',default=[]);args=ap.parse_args()
    if len(args.launch)!=len(set(args.launch)):raise ValueError('Duplicate mode')
    target=CODE/'train_dual8222_query_v7.py'
    if args.prepare:
        if target.exists() or STUDY.exists():raise RuntimeError('Refuse study overwrite')
        source=CODE/'train_dual8222_attention_v6.py'
        text=source.read_text(encoding='utf-8-sig')
        old_optim="    common=[p for n,p in model.named_parameters() if p.requires_grad and '.rich_location.' not in n]\n    rich=[p for n,p in model.named_parameters() if p.requires_grad and '.rich_location.' in n]\n    groups=[{'params':common,'lr':3e-5}]\n    if rich:groups.append({'params':rich,'lr':3e-4})\n    optimizer=torch.optim.AdamW(groups,weight_decay=.01)"
        old_cfg="         'location_readout':'old v5 residual frozen;common linear locations;new2block64wide joint readout',\n         'new_readout_learning_rate':3e-4,'new_readout_blocks':2,'head_width':64,'attention_heads':4,'attention_radius':16,\n         'attention_implementation':'explicit float32 dot products,relative bias,padding mask,no dropout',\n         'torch_version':str(torch.__version__),\n         'residual_bound_fraction':0.25 if args.mode!='linear' else None,\n         'local_kernel':None,\n         'comparison':'same80.274135;linear budget control;deepMLP86496 versus attention86888;392parameter difference',"
        new_cfg="         'location_readout':'all old predictor parameters fixed;new per-component query adapter only',\n         'component_queries':16,'query_descriptors':['clipped_mean','clipped_logstd','clipped_logmix'],\n         'head_width':32,'attention_heads':2,'attention_radius':8,'memory_slots':34,\n         'memory_control':'same slots,mask,relative encodings;center repeats current point,local reads neighboring points',\n         'attention_implementation':'explicit float32 dot products and masked softmax,no dropout',\n         'torch_version':str(torch.__version__),'residual_bound_fraction':.25,\n         'comparison':'identical14081parameters and initialization;only source of memory feature tokens differs',"
        edits=[
            ('Complex/attention readout comparison from verified80.274135; original factors frozen.','Component-query readout comparison from verified80.274135; entire previous predictor frozen.',1),
            ('Three predeclared modes; launch explicitly, at most two simultaneously.','Two predeclared modes; explicit launch, at most two simultaneously.',1),
            ('from dual8222_attention_head_v6 import Network,MODES','from dual8222_component_query_v7 import Network,MODES',1),
            ("Path(__file__).with_name('dual8222_attention_head_v6.py'),","Path(__file__).with_name('dual8222_component_query_v7.py'), Path(__file__).with_name('train_dual8222_attention_v6.py'),",1),
            ("'learning_rate':3e-5","'learning_rate':3e-4",1),
            (old_cfg,new_cfg,1),(old_optim,'    optimizer=torch.optim.AdamW(params,lr=3e-4,weight_decay=.01)',1),
            ("assert all(n.startswith('heads.0.') for n,p in model.named_parameters() if p.requires_grad)","assert all('.component_readout.' in n for n,p in model.named_parameters() if p.requires_grad)",1),
            ("'fixed_all_gates_unchanged':True,","'fixed_all_gates_unchanged':True,'entire_previous_predictor_unchanged':True,",1),
        ]
        for old,new,n in edits:
            if text.count(old)!=n:raise RuntimeError('Unexpected source edit count: '+old)
            text=text.replace(old,new)
        ast.parse(text);ast.parse((CODE/'dual8222_component_query_v7.py').read_text(encoding='utf-8'))
        target.write_text(text,encoding='utf-8');STUDY.mkdir()
        (STUDY/'trainer_derivation.json').write_text(json.dumps({'source':source.name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'target':target.name,'edits':edits},ensure_ascii=False,indent=2),encoding='utf-8')
        agents='# 目标与约束\n在原双专家上验证分量query读出的增量。遵守原输入/原分折/原评分/80-18/最多2GPU，旧源和交付不可覆盖，无自动队列或补折。维护PROJECTS.md，记录目标、已做、计划、未做及达成后可做的事，每批完成后更新。\n'
        project='# 项目进度\n目标：检验分量查询与局部记忆的结构价值。\n已完成：来源核对、两组同参数对照实现。\n计划：正式训练、最佳恢复、匹配差值及必要保存重载。\n未完成：尚无新增精度或独立泛化证据，续跑等价性未实测。\n达成后：固定结构和通道策略，再验证泛化；当前不启动其他折。\n'
        for p in [STUDY]+[ROOT/'outputs'/(PREFIX+m+'80_fold5') for m in MODES]:
            if p!=STUDY:p.mkdir(exist_ok=False)
            for fn,value in [('README.md',PLAN),('AGENTS.md',agents),('PROJECTS.md',project)]:
                (p/fn).write_text(value,encoding='utf-8')
        for m in MODES:
            (ROOT/'outputs'/(PREFIX+m+'80_fold5')/'runtime_status.json').write_text(json.dumps({'status':'planned','mode':m,'atlas_current_trial':False}),encoding='utf-8')
        rule='\n\n## 2026-09-28 当前有界优化：分量query读出v7\n继续优化本批预定义center/local两项，完全冻结80.274135旧预测器（含线性位置与旧支路），仅训练分量查询交叉注意力模块。两方案结构、参数、初始化、掩码一致，仅记忆读取当前点或邻近点不同。原输入/分折/评分/Q/80-18/max2GPU保持；新模块lr3e-4，不扩大本批候选。无旧ABD重启/补折，维护PROJECTS与独立记录。\n'
        for p in [ROOT.parent/'AGENTS.md',ROOT/'AGENTS.md',CODE/'AGENTS.md']:
            with p.open('a',encoding='utf-8') as f:f.write(rule)
        with (ROOT/'EXPLORATIONS.md').open('a',encoding='utf-8') as f:
            for m in MODES:f.write('\n- [分量query v7 '+m+'](outputs/'+PREFIX+m+'80_fold5/README.md)：完全冻结原读出、同参数局部记忆对照，真实收益待结果。\n')
    if not STUDY.exists():raise RuntimeError('Prepare study first')
    active=trainers()
    if len(active)+len(args.launch)>2:raise RuntimeError('GPU concurrency limit: '+str(active))
    for m in args.launch:
        out=ROOT/'outputs'/(PREFIX+m+'80_fold5')
        if (out/'config.json').exists() or (out/'launch.json').exists():raise RuntimeError('Duplicate run: '+m)
    launches=[]
    for m in args.launch:
        out=ROOT/'outputs'/(PREFIX+m+'80_fold5')
        command=[sys.executable,'-u',target.as_posix(),'--out',out.as_posix(),'--mode',m]
        with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
            p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=err,stdin=subprocess.DEVNULL,close_fds=True,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        row={'mode':m,'pid':p.pid,'command':command,'started_at':datetime.now().astimezone().isoformat()}
        (out/'launch.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
        launches.append(row);print('LAUNCHED',json.dumps(row,ensure_ascii=False),flush=True)
    if launches:
        old=json.loads((STUDY/'launches.json').read_text()) if (STUDY/'launches.json').exists() else []
        (STUDY/'launches.json').write_text(json.dumps(old+launches,ensure_ascii=False,indent=2),encoding='utf-8')
        note='\n\n## '+datetime.now().astimezone().isoformat()+' 分量query v7启动\n同80.274135，旧预测器全部固定。实际启动'+str([(a['mode'],a['pid']) for a in launches])+'；仅本批两项，没有旧任务或补折。初始/训练/最终收益待真实结果，详见results/deep_learning/dual8222_component_query_v7_20260928。\n'
        for p in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md']:
            with p.open('a',encoding='utf-8') as f:f.write(note)

if __name__=='__main__':main()
