"""Explicit bounded continuation after completed linear/cross readout comparison."""
from __future__ import annotations
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import psutil

CODE=Path(__file__).resolve().parent
ROOT=CODE.parents[1]
ARMS=['gate_only','density_only']
COMMON='''# 双专家读出因素分离：默认状态门控和连续分布

## 探索重点与启动证据
已完成linear80.253777、cross80.255113，新增容量相对线性仅微弱差异。POR真实非默认段改善与默认段退步同时出现。下一步固定网络前向形式，只改变哪些输出参数能更新，定位可复用收益；不是标签分组选择预测，也不改变数据人口。

## 预先固定方案
- gate_only：只有两个专家、三个目标的默认门控线性行可训练，共774参数；24维密度线性行固定。
- density_only：只有24维密度线性行可训练，共18576参数；门控线性行固定。
- 公平对照：本轮已完成linear，19350参数全部可训练；从同一个80.221861函数开始、同输入/seed/窗口/Q/目标/学习率/预算/评价。不同参数数量是本次更新范围消融本身，不是参数预算匹配结构比较。
- 仍训练原完整双专家每通道BCE+条件NLL。门控变化也影响非默认专家混合份额；密度变化也可能改变最终最优默认动作。因此冻结门控参数不等于冻结最终默认输出数量。

## 已实现及约束
继承原80/18/min10、固定Q50、3e-5余弦、独立目标优化与选模、原单覆盖256格800动作。只有内部两专家，没有外部B/D。保留原始门控/密度张量数值，首轮初始回放及最终固定因子不变均须实测。
原子last_state保留当轮参数、优化器、调度器、RNG与停止状态。恢复时重新施加因子冻结掩码；未经实测不宣称中断重跑等价。暂停旧ABD不动，不补折、不改交付。

## 收益
尚未启动/未验证。完成后各通道分别比较，固定来源实际共同推理；第五折>80.5和完整OOF>82均未完成。以最终report更新，不把初始成绩当收益。

## 不继续的方向
不因联合读出0.0013分就扩大网络；若两部分独立训练也不能改善，就记录限制，不无界扫描优化器。默认/非默认误差分层仅诊断，不作为推理时真实标签条件。
'''


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    for name in ['pure_dl_dual8222_v2_linear80_fold5','pure_dl_dual8222_v2_cross8080_fold5']:
        p=ROOT/'outputs'/name
        if not (p/'fold_5_report.json').exists():raise RuntimeError('Predefined preceding comparison not finished')
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            v=p.info;args=v.get('cmdline') or []
            if 'python' in (v.get('name') or '').lower() and 'pure_dl' in ' '.join(args) and any(Path(x.replace('\\','/')).name.startswith(('train_','benchmark_')) for x in args):
                raise RuntimeError('Another training process is active; no new pair launched')
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    src=CODE/'train_dual8222_heads_v2.py';dest=CODE/'train_dual8222_factors80.py'
    if dest.exists():raise RuntimeError('Refuse to overwrite trainer')
    text=src.read_text(encoding='utf-8')
    edits=[
        ('from dual8222_head_structure_v2 import Network','from dual8222_factorized_readout import Network',1),
        ("choices=['linear', 'cross80']","choices=['gate_only','density_only']",1),
        ("'dual8222_head_structure_v2.py', 'train_dual8222_heads80.py'","'dual8222_factorized_readout.py', 'dual8222_head_structure_v2.py', 'train_dual8222_heads80.py', 'train_dual8222_heads_v2.py'",1),
        ("'model':'dual8222_head_structure_v2'","'model':'dual8222_factorized_readout'",1),
        ("'head_width':0 if args.architecture=='linear' else 80","'head_width':0",1),
        ("'residual_parameters_per_target':trainable_parameters//3-6450","'total_copied_output_parameters':19350",1),
        ("'copied source linear rows; cross80 adds zero-output two-expert residual; original router fixed'","'copied source linear rows;only the selected gate or density factor is trainable'",1),
        ("'linear:original128 per expert;cross80 also sees concatenated256;no added normalization'","'original128 per expert;no residual feature branch'",1),
        ("'linear controls independent training;cross80 is93180params versus old pointwise93420;near-budget internal expert interaction'","'linear full-update control versus gate-only or density-only update mask;otherwise matched training and evaluation'",1),
        ('for j in range(3): model.heads[j].requires_grad_(active[j])','model.set_trainable_targets(active)',1),
        ("    report = {'fold':5,", "    model.verify_fixed_factor()\n    report = {'fixed_output_factor_unchanged':True, 'fold':5,",1)
    ]
    for old,new,n in edits:
        if text.count(old)!=n:raise RuntimeError('Unexpected source for '+old)
        text=text.replace(old,new)
    ast.parse(text)
    dest.write_text(text,encoding='utf-8')
    study=ROOT/'results/deep_learning/dual8222_round2_20260928'
    (study/'factor_trainer_derivation.json').write_text(json.dumps({'source':src.name,'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
        'target':dest.name,'edits':[{'from':a,'to':b,'count':n} for a,b,n in edits]},ensure_ascii=False,indent=2),encoding='utf-8')
    rule='\n\n## 2026-09-28 双专家继续优化：门控与密度分离\nlinear/cross两项已完成，依当前继续优化授权，在同一双专家和原线性头上预先定义gate_only与density_only的参数更新范围对照；评分/数据/种子/Q/预算不变。只启动此两项，无旧路线或补折。各输出因子冻结需最终校验；分别记录通道收益，维护PROJECTS。\n'
    for p in [ROOT.parent/'AGENTS.md',ROOT/'AGENTS.md',CODE/'AGENTS.md']:
        with p.open('a',encoding='utf-8') as f:f.write(rule)
    launched=[]
    for arm in ARMS:
        out=ROOT/'outputs'/('pure_dl_dual8222_v3_'+arm+'80_fold5')
        out.mkdir(exist_ok=False)
        (out/'README.md').write_text(COMMON+'\n本目录：'+arm+'\n',encoding='utf-8')
        (out/'AGENTS.md').write_text('# 目标与约束\n仅优化当前双专家的指定输出因子，遵守根目录原数据/原评分/80-18/最多2GPU限制。维护PROJECTS.md，记录已做、计划、未做、目标及完成后工作，每批更新。原模型、旧实验和交付不可覆盖；无自动补折。\n',encoding='utf-8')
        (out/'PROJECTS.md').write_text('# 进度\n已实现因素分离、已记录假设。待真实初始验证、自然早停和最终固定三头回放。目标第五折>80.5、完整OOF>82尚未完成；完成后再讨论固定配置的泛化验证，当前不补折。\n',encoding='utf-8')
        command=[sys.executable,'-u',dest.as_posix(),'--out',out.as_posix(),'--architecture',arm]
        with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
            p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=err,stdin=subprocess.DEVNULL,close_fds=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        row={'architecture':arm,'pid':p.pid,'started_at':datetime.now().astimezone().isoformat(),'command':command}
        (out/'launch.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
        launched.append(row);print('LAUNCHED',json.dumps(row,ensure_ascii=False),flush=True)
        with (ROOT/'EXPLORATIONS.md').open('a',encoding='utf-8') as f:
            f.write('\n- [双专家因素分离 '+arm+'](outputs/'+out.name+'/README.md)：与独立线性全更新比较，真实收益待最终结果。\n')
    (study/'factor_launches.json').write_text(json.dumps(launched,ensure_ascii=False,indent=2),encoding='utf-8')
    note='\n\n## 双专家因素分离启动\n已完成linear80.253777和cross80.255113；前轮新增非线性仅微弱差异。POR非默认收益部分被默认段退步抵消，故在原线性头上比较只更新门控与只更新密度。PID='+','.join(str(x['pid']) for x in launched)+'。2GPU上限、原评分/数据/预算不变，等待真实训练及最终固定因子验证；无旧ABD或补折启动。\n'
    for p in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md']:
        with p.open('a',encoding='utf-8') as f:f.write(note)

if __name__=='__main__':main()
