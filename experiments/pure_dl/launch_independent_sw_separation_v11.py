"""Create the isolated v11 trainer and explicitly launch its two fixed arms."""
from __future__ import annotations
import ast,json,subprocess,sys
from pathlib import Path
from datetime import datetime
import psutil

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT.parent
CODE=ROOT/'experiments/pure_dl'
STUDY=ROOT/'results/deep_learning/independent_sw_separation_v11_20260928'
MODES=('shared','split')

METHOD='''# SW状态与数值：共享表示和独立表示的匹配对照v11

## 探索重点
上一轮两个冷启动模型在真实默认段提高，但非默认段下降。用户要求继续。本轮只检验SW内部两类任务的表示连接方式，不改变POR/PERM，不增补其他折。

## 预先固定的方案
shared和split都包含两套不同随机种子的TCN编码器。每套207->48，四个depthwise5门控残差块，dilation1/2/4/8，末层LayerNorm48。两套各39840参数；门控48->1和密度48->256共12593参数，总计92273，全部使用，没有闲置参数。
shared：两个头同时读取h=(E0(x)+E1(x))/2。split：门控只读E0，连续密度只读E1。两个候选每个同名参数的初始化逐字节一致，头初始权重为零，bias来自训练井默认率与软直方图，初始预测相同。分支0和分支1本身不同初始化，避免共享组变成两份始终相同的副本。
推理无标签依赖、无逐井/逐点选模型。两套编码器完整从头训练，不载入旧骨干或旧预测。原训练侧特征、坐标、支持格及Q继续沿用，不能说整条pipeline不依赖此前训练产物。

## 训练及验收
原九曲线生成206特征+窗口相对深度，原64/16井分组和行序，原144665点第五折，单覆盖256格/800候选动作解码。
两个候选用同一SW损失：mean(Q*(BCE(default)+(1-default)*soft2binCE))/3，数值监督只在真实非默认训练点有效。相同AdamW6e-4、wd.01、cosine80*64->1e-5、max80/pat18/min10、batch64、64steps/epoch、seed20261010；独立窗口生成器同种子。梯度裁剪按固定的[encoder0+gate]和[encoder1+density]两组各clip2，两候选一致，避免split中跨任务的裁剪缩放耦合。共享组的两套编码器仍接收两项监督，split中对应梯度必须严格隔离。checkpoint按整个SW官方分数统一选择，未另外搜索门控/数值最佳轮次。
与v10相比，宽度、并行分支数及裁剪分组均不同，因此机制归因以v11shared/split匹配对照为准，v10只作历史参考。
每次运行保存最佳完整模型与原子last_state（参数、优化器、调度器、RNG、早停状态），显式恢复且不自动重试。训练后在固定128个训练窗口上测BCE/CE编码器梯度方向及跨分支梯度，不用验证标签诊断梯度，不更新参数。单次小样本梯度观察不能解释全部验证因果。
每项报告恢复后的目标分；超过旧SW才进入真正三独立模型合并重载。若无胜出，保留已有验证组合，不把新模型冷启动改善冒充新增最佳。第五折严格>80.5和完整OOF>82仍未达成，无自动补折。

## 四项记录
探索重点：共享表征的梯度耦合与数值表征独立性。
已经探索及验证：模型/训练接口已实现；运行和最终结果以报告为准。
收益：启动时尚未证实，报告绝对SW、相对共享对照和旧SW的差值。
不继续的方向：首批只这两个候选；无证据不扩大宽度、核长、梯度算法或目标权重搜索。负结果只约束当前模型/数据/训练设置。
'''
AGENTS='''# 目标与约束
目标：在SW内比较默认状态与连续值共享/独立表征；POR/PERM保留私有模型。
遵守原输入、井划分、评分、80/18与最多2GPU；旧数据、模型、交付、训练源代码不改，无自动补折。
维护PROJECTS.md，写清已做、计划、未完成、目标及达成后事项，每批更新。记录独立README和图谱，区分启动、进行中、完成与全OOF成绩。
'''
PROJECTS='''# 进度

目标：用同参数连接方式对照验证SW内部解耦是否改善非默认段及总分。
已完成：核实v10两模型结束且未超旧SW；实现两套TCN共享/分离读出与训练接口。
计划：显式启动两候选，完成共同评分、每井/状态误差分解和训练侧梯度诊断。
未完成：本批训练收益、磁盘重载、独立新井泛化、完整OOF>82。
达成后：固定有益模型再评估推广；不自动补折，不覆盖旧交付。
'''


def active():
    result=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            info=p.info;args=info.get('cmdline') or []
            if 'python' in (info.get('name') or '').lower() and 'base64' not in ' '.join(args):
                if any('pure_dl/' in s.replace('\\','/') and Path(s.replace('\\','/')).name.startswith(('train_','benchmark_')) for s in args):result.append(info)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return result


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if active():raise RuntimeError('Active training exists; inspect instead of launching a third job')
    dirs=[ROOT/'outputs'/f'pure_dl_independent_sw_v11_{mode}80_fold5' for mode in MODES]
    destination=CODE/'train_independent_sw_separation_v11.py'
    if any(p.exists() for p in dirs) or destination.exists() or STUDY.exists():
        raise RuntimeError('New study already exists; inspect state, never duplicate launch')
    # Exact, bounded transformation into a NEW trainer. The v10 file is immutable.
    trainer=(CODE/'train_independent_sw_scratch_v10.py').read_text(encoding='utf-8-sig')
    trainer=trainer.replace('independent_sw_scratch_v10','independent_sw_separation_v11')
    old='from independent_sw_separation_v11 import Network, ARCHITECTURES, objective'
    assert trainer.count(old)==1
    trainer=trainer.replace(old,old+', training_gradient_diagnostic')
    old="'input_features': 206, 'relative_depth_features': 1, 'hidden_width': 64,"
    assert trainer.count(old)==1
    trainer=trainer.replace(old,"'input_features': 206, 'relative_depth_features': 1, 'hidden_width': 48,")
    old="""'structure': ('four gated depthwise5 residual blocks, dilations1/2/4/8, receptive61'
                         if args.architecture == 'tcn' else
                         'one bidirectional GRU64 per direction, packed valid segments,128to64 residual'),"""
    assert trainer.count(old)==1
    trainer=trainer.replace(old,"""'structure': 'two distinct48dimTCNs;four gated depthwise5 blocks each;dilations1/2/4/8',
           'within_SW_representation': 'both heads see mean(E0,E1)' if args.architecture=='shared' else 'gate seesE0; density seesE1',
           'parameter_count_matched': 92273,""")
    trainer=trainer.replace("'gru_compute': 'packed recurrent block float32;other blocks bfloat16 AMP',", "'compute': 'botharms_bfloat16AMP', 'gradient_clipping': 'separatelyencoder0+gate_and_encoder1+density',")
    old='    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=.01)'
    assert trainer.count(old)==1
    trainer=trainer.replace(old,"    assert sum(p.numel() for p in model.parameters()) == 92273\n    clip_groups = model.clipping_groups()\n"+old)
    old='            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)'
    assert trainer.count(old)==1
    trainer=trainer.replace(old,'            for parameters in clip_groups:\n                torch.nn.utils.clip_grad_norm_(parameters, 2.)')
    old="    report = {'fold': 5, 'screening_fold': 5, 'target': 'SW', 'architecture': args.architecture,"
    assert trainer.count(old)==1
    trainer=trainer.replace(old,"    diagnostic = training_gradient_diagnostic(model, data)\n    atomic_json(out/'training_gradient_diagnostic.json', diagnostic)\n"+old)
    old="'gradient_scope_verified': proof is not None, 'selected_backbone_change_L2': change,"
    assert trainer.count(old)==1
    trainer=trainer.replace(old,old+"\n              'gradient_diagnostic': diagnostic,")
    ast.parse(trainer)
    ast.parse((CODE/'independent_sw_separation_v11.py').read_text(encoding='utf-8-sig'))
    destination.write_text(trainer,encoding='utf-8')
    STUDY.mkdir(parents=True)
    for name,text in [('METHOD.md',METHOD),('AGENTS.md',AGENTS),('PROJECTS.md',PROJECTS)]:
        (STUDY/name).write_text(text,encoding='utf-8')
    for p in dirs:
        p.mkdir()
        for name,text in [('README.md',METHOD),('AGENTS.md',AGENTS),('PROJECTS.md',PROJECTS)]:
            (p/name).write_text(text,encoding='utf-8')
    rule='\n\n## 2026-09-28 当前继续：SW状态/数值表征解耦v11\n本批预定义shared/split两个匹配候选，各有两套独立48维TCN、同初始化和92273参数；仅改变两个头读共享均值还是各自编码器。训练与解码相同，按整个SW官方分选checkpoint。原输入/分折/Q/80-18/max2保持，POR/PERM独立且不变；训练侧梯度诊断不读验证标签，无自动搜索/旧任务/补折。维护PROJECTS、README与图谱。\n'
    for p in [WORK/'AGENTS.md',ROOT/'AGENTS.md',ROOT/'experiments/AGENTS.md',CODE/'AGENTS.md']:
        with p.open('a',encoding='utf-8') as f:f.write(rule)
    launches=[]
    for mode,out in zip(MODES,dirs):
        args=[sys.executable,'-X','utf8','-u',destination.as_posix(),'--architecture',mode,'--out',out.as_posix()]
        with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
            proc=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=err,stdin=subprocess.DEVNULL,close_fds=True,
                                   creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        row={'mode':mode,'pid':proc.pid,'command':args,'started_at':datetime.now().astimezone().isoformat(),'id':out.name}
        (out/'launch.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
        launches.append(row)
    (STUDY/'launches.json').write_text(json.dumps(launches,ensure_ascii=False,indent=2),encoding='utf-8')
    note='\n\n## '+datetime.now().astimezone().isoformat()+' SW解耦v11启动\n仅启动'+str([(r['mode'],r['pid']) for r in launches])+'。两候选同参数预算，实际分数待核实。无旧任务或补折；完整OOF>82未完成。记录results/deep_learning/independent_sw_separation_v11_20260928。\n'
    for p in [WORK/'PROJECTS.md',ROOT/'PROJECTS.md']:
        with p.open('a',encoding='utf-8') as f:f.write(note)
    with (ROOT/'EXPLORATIONS.md').open('a',encoding='utf-8') as f:
        for row in launches:f.write('\n- [SW内部共享/分离 '+row['mode']+'](outputs/'+row['id']+'/README.md)：同参数初始化，仅变连接方式；启动时无分数。\n')
    print(json.dumps({'launched':launches,'trainer':str(destination)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
