"""Prepare and explicitly launch the second bounded dual8222 head study.

Creates new files only. Original trainers and completed experiments are untouched.
The optional launch starts exactly two specified runs; there is no queue/retry.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import psutil

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]
ARMS = ['linear','cross80']
NAMES = {a:'pure_dl_dual8222_v2_'+a+'80_fold5' for a in ARMS}
COMMON = '''# 双专家独立头第二轮：线性归因与跨专家读出

## 目标与探索重点
在保留的local-context-router双专家80.221861@52基础上，补原线性读出的独立训练对照，并检验同一目标的读出能否从两个内部专家的联合隐藏特征获益。旧ABD路线不恢复。

## 预先固定的比较
- linear：只复制原头的线性行、分成三个独立目标优化；19350可训练参数，不增加非线性结构。
- cross80：每个目标把两专家128维特征拼为256维，经256->80->50的零输出残差同时修正两专家的25维参数。独立目标之间不共享可训练参数，内部输入路由仍冻结。93180可训练参数。
- 已有pointwise：两位专家分别128->80->25，总93420参数。cross80比其少240参数（约0.257%），是近参数预算的交互结构比较，不宣称完全等参数或唯一因果解释。
- 新两组均从原80.221861函数初始化，不从前轮POR最佳头热启动；原线性行继续允许训练。必须检查初始预测再解释分数。

## 已实现内容和训练合同
继承首轮训练器的原第五折/seed20261010/独立窗口RNG/固定训练Q(epoch50)/每目标AdamW/3e-5余弦至1e-5/80轮patience18/min10/batch64/64步。仅64训练井标签参与损失，原单覆盖256格/800动作解码不改。保留连续分布族、两个骨干、校准与内部路由。无外部模型投票，无重新拟合预处理，无新辅助损失。
每轮原子保存当前参数、优化器、调度器、RNG、早停和最佳状态；只有显式--resume才恢复。尚未做续跑等价性实测。

## 收益状态
准备阶段：尚无新成绩。结果以各目录实际history与final report为准，通道候选不是最终共同恢复。第五折>80.5且明确收益才可考虑后续，当前补折依然暂缓。

## 不继续的方向
不扩大核长/宽度随机扫描，不用同井标签选行或选井；若跨专家读出不优于已有逐点头，记录其负结果。线性控制用于判断前轮非线性是否真的有增量。
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--launch',action='store_true')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    original = CODE/'train_dual8222_heads80.py'
    target = CODE/'train_dual8222_heads_v2.py'
    if target.exists() or any((ROOT/'outputs'/NAMES[a]).exists() for a in ARMS):
        raise RuntimeError('Round two files already exist; no overwrite or repeat launch')
    text = original.read_text(encoding='utf-8-sig')
    sha = hashlib.sha256(original.read_bytes()).hexdigest()
    changes = [
        ('from dual8222_independent_heads import Network','from dual8222_head_structure_v2 import Network',1),
        ("choices=['pointwise', 'local5']","choices=['linear', 'cross80']",1),
        ('    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 93420',
         '    trainable_parameters = Network.parameter_count(args.architecture)\n    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == trainable_parameters',1),
        ("'dual8222_independent_heads.py'", "'dual8222_head_structure_v2.py', 'train_dual8222_heads80.py'",1),
        ("'model':'dual8222_independent_heads'","'model':'dual8222_head_structure_v2'",1),
        ("'head_width':80 if args.architecture=='pointwise' else 77", "'head_width':0 if args.architecture=='linear' else 80",1),
        ("'local_kernel':1 if args.architecture=='pointwise' else 5", "'local_kernel':1",1),
        ("'trainable_parameters':93420", "'trainable_parameters':trainable_parameters",2),
        ("'trainable_parameters_per_target':31140, 'residual_parameters_per_expert_target':12345",
         "'trainable_parameters_per_target':trainable_parameters//3, 'residual_parameters_per_target':trainable_parameters//3-6450",1),
        ("'copied source linear rows plus zero-output residual; original internal router fixed'",
         "'copied source linear rows; cross80 adds zero-output two-expert residual; original router fixed'",1),
        ("'original128 per expert, no added LayerNorm or dropped amplitude;2experts'",
         "'linear:original128 per expert;cross80 also sees concatenated256;no added normalization'",1),
        ("'equal parameter budget; pointwise width80 versus local5 width77, not pure context-only intervention'",
         "'linear controls independent training;cross80 is93180params versus old pointwise93420;near-budget internal expert interaction'",1)
    ]
    for old,new,count in changes:
        if text.count(old) != count:
            raise RuntimeError(f'Unexpected source occurrence count {count}: {old}')
        text = text.replace(old,new)
    ast.parse(text)
    ast.parse((CODE/'dual8222_head_structure_v2.py').read_text(encoding='utf-8'))
    target.write_text(text,encoding='utf-8')
    assert hashlib.sha256(original.read_bytes()).hexdigest() == sha
    study = ROOT/'results/deep_learning/dual8222_round2_20260928'
    study.mkdir(exist_ok=False)
    (study/'trainer_derivation.json').write_text(json.dumps({'source':original.name,'source_sha256':sha,
        'target':target.name,'edits':[{'from':a,'to':b,'count':c} for a,b,c in changes]},ensure_ascii=False,indent=2),encoding='utf-8')
    (study/'README.md').write_text(COMMON,encoding='utf-8')
    rule = ('\n\n## 2026-09-28 第二轮独立头优化\n用户继续优化的授权限定在当前双专家。先补linear-only对照，另做每目标联合两专家隐藏特征的cross80头；保持原来源、数据、Q、评分和训练预算。已有逐点/局部结果冻结不改。两任务上限，无旧ABD重启和补折。更新PROJECTS、实验README与演进图谱。\n')
    for path in [ROOT.parent/'AGENTS.md',ROOT/'AGENTS.md',CODE/'AGENTS.md']:
        with path.open('a',encoding='utf-8') as f:f.write(rule)
    for arm in ARMS:
        out = ROOT/'outputs'/NAMES[arm]
        out.mkdir()
        (out/'README.md').write_text(COMMON+'\n本目录方案：'+arm+'\n',encoding='utf-8')
        (out/'AGENTS.md').write_text('# 实验约束\n目标：当前双专家独立输出头优化。遵守根目录数据/评分/预算/2GPU约束，不启动旧路线或补折。维护PROJECTS.md，逐批记录已做、计划、未做、目标及达成后的工作；保留可恢复状态和验证证据，旧产物不覆盖。\n',encoding='utf-8')
        (out/'PROJECTS.md').write_text('# 进度\n目标：可复现地改善双专家输出头。已完成结构实现与来源锁定；待实际训练、逐通道比较及最终恢复；达成后固定方案评估跨折，当前不补折。\n',encoding='utf-8')
    launches = []
    if args.launch:
        active=[]
        for p in psutil.process_iter(['pid','name','cmdline']):
            try:
                d=p.info; argv=d.get('cmdline') or []
                if 'python' in (d.get('name') or '').lower() and 'pure_dl' in ' '.join(argv) and any(Path(x.replace('\\','/')).name.startswith(('train_','benchmark_')) for x in argv):
                    active.append(d)
            except (psutil.NoSuchProcess,psutil.AccessDenied): pass
        if active: raise RuntimeError('Training slots not clear; no pair launched: '+str(active))
        for arm in ARMS:
            out=ROOT/'outputs'/NAMES[arm]
            command=[sys.executable,'-u',target.as_posix(),'--out',out.as_posix(),'--architecture',arm]
            with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
                p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=err,stdin=subprocess.DEVNULL,close_fds=True,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
            row={'architecture':arm,'pid':p.pid,'command':command,'started_at':datetime.now().astimezone().isoformat()}
            (out/'launch.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
            launches.append(row)
            print('LAUNCHED',json.dumps(row,ensure_ascii=False),flush=True)
    (study/'launches.json').write_text(json.dumps(launches,ensure_ascii=False,indent=2),encoding='utf-8')
    note='\n\n## 2026-09-28 双专家第二轮结构优化\n新增linear-only对照和cross80联合内部专家特征头。源80.221861，固定原单覆盖/256格/800动作、独立目标/80-18/3e-5/Q50。前轮源码与结果不改。已启动任务数='+str(len(launches))+'；PID='+','.join(str(x['pid']) for x in launches)+'。初始/梯度/真实收益待核对，不预称成功。结果在results/deep_learning/dual8222_round2_20260928，目标>80.5与完整OOF>82仍未完成，未补折。\n'
    for path in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md']:
        with path.open('a',encoding='utf-8') as f:f.write(note)
    with (ROOT/'EXPLORATIONS.md').open('a',encoding='utf-8') as f:
        for arm in ARMS: f.write('\n- [双专家第二轮 '+arm+'](outputs/'+NAMES[arm]+'/README.md)：独立线性归因或联合专家读出；尚未取得完成结果。\n')
    print('PREPARED',target.as_posix(),flush=True)

if __name__ == '__main__':
    main()
