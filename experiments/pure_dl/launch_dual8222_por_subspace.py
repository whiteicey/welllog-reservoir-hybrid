"""Prepare a bounded three-arm POR study and launch only explicitly named arms.

No scheduler, automatic next stage, old experiment resume, or full-CV launch.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import psutil

ROOT=Path(__file__).resolve().parents[2]
CODE=Path(__file__).resolve().parent
STUDY=ROOT/'results/deep_learning/dual8222_por_subspace_20260928'
MODES=['mean_only','mean_scale','density_all']
PREFIX='pure_dl_dual8222_v4_por_'
README='''# POR分布子空间：从已验证80.257638配方热启动

## 目标和固定对照
本批依照继续优化授权，预先限定三个对照：mean_only只更新POR的8个分量位置读出；mean_scale更新位置与原始尺度读出；density_all更新混合权重、位置和尺度，作为同起点的全部密度参数更新对照。没有第四个参数扫描。三者可训练参数分别2064/4128/6192。
所有对照从上一轮固定配方80.257638624609开始：POR密度第1轮、PERM门控第34轮、SW原来源第52轮。只训练POR；PERM、SW预测全程不变。两个骨干、输入校准和内部路由冻结；原九曲线/206特征/原第五折/单覆盖256格800动作不变。

## 实现和待验证内容
参数按门控、混合权重、位置、尺度真实拆分，不训练的行使用buffer，避免仅屏蔽梯度却受到AdamW权重衰减。保留原卷积输出形状与内部双专家分布合并。
每组独立新优化器与调度器：AdamW lr3e-5、wd.01、cosine80*64至1e-5；max80/patience18/min10/bs64/64步。独立CUDA窗口RNG为20261010；固定源epoch50Q，损失沿用POR BCE+非默认NLL后除3。新优化起点相同，但不是原训练的无缝续跑；记录新增训练预算。
初始须逐元素复现当前固定配方。最终恢复最佳POR头，检查PERM/SW、所有固定因子和骨干张量均不变。每轮保存current heads、optimizer、scheduler、RNG、停止计数、best heads/history用于显式恢复；未实测中断恢复等价性。

## 收益
启动前尚未验证。初始80.257638不是新增收益；日志最佳也不能充当最终共同恢复。两组对比不能从不同起点减分归因，因此补同热启动density_all对照。所有成绩为反复开发的第五折，无独立泛化结论。

## 停止和后续边界
最多2个GPU任务，3组分批明确启动，不设置自动队列。旧ABD与补折保持暂停，交付不变。若3组无增益，记录有限更新空间的负结果，不扩展宽度/LR扫描。只有第五折严格>80.5且明确增益后再讨论下一阶段；完整OOF>82尚未达成。
'''


def active_trainers():
    active=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            d=p.info;args=d.get('cmdline') or []
            if 'python' in (d.get('name') or '').lower() and 'pure_dl' in ' '.join(args):
                if any(Path(a.replace('\\','/')).name.startswith(('train_','benchmark_')) for a in args):active.append(d)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return active


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser()
    ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--launch',choices=MODES,nargs='*',default=[])
    args=ap.parse_args()
    if len(args.launch)!=len(set(args.launch)):raise ValueError('Duplicate mode')
    if args.prepare:
        if STUDY.exists():raise RuntimeError('Study already exists')
        for mode in MODES:
            if (ROOT/'outputs'/(PREFIX+mode+'80_fold5')).exists():raise RuntimeError('Run already exists')
        for name in ['dual8222_por_subspace.py','train_dual8222_por_subspace80.py']:
            ast.parse((CODE/name).read_text(encoding='utf-8'))
        STUDY.mkdir()
        (STUDY/'README.md').write_text(README,encoding='utf-8')
        agents='# 目标与约束\n仅优化既有双专家的POR分布子空间，保留其他目标与原评分；遵守最多2GPU、原数据隔离和预算。维护PROJECTS.md，记录已做、计划、未做、目标及达成后的工作，每批更新。禁止自动队列、旧路线复活、补折和覆盖交付。\n'
        projects='# 实验进度\n目标：在80.257638固定配方上获得可复现的POR读出增益。\n已做：来源核对，三个有界匹配对照已实现。\n计划：实际训练并最终共同恢复，核对冻结因子和PERM/SW，再作配对比较。\n未做：当前尚无训练分数、无恢复等价性验证、无完整OOF提升。\n达成后：固定有益更新子空间，进一步评估泛化；当前不自动补折，完整OOF>82仍未完成。\n'
        for name,body in [('AGENTS.md',agents),('PROJECTS.md',projects)]:
            (STUDY/name).write_text(body,encoding='utf-8')
        for mode in MODES:
            out=ROOT/'outputs'/(PREFIX+mode+'80_fold5');out.mkdir()
            for name,body in [('AGENTS.md',agents),('PROJECTS.md',projects),('README.md',README+'\n本目录：'+mode+'\n')]:
                (out/name).write_text(body,encoding='utf-8')
            (out/'runtime_status.json').write_text(json.dumps({'status':'planned','atlas_current_trial':False,'mode':mode}),encoding='utf-8')
        rule='\n\n## 2026-09-28 本批有界POR子空间实验\n继续优化授权用于三个预先固定对照：mean_only、mean_scale、density_all，全部从已验证80.257638选定头热启动，仅POR更新；PERM/SW/双骨干/路由/Q/原评分冻结。独立新优化器和80/18预算，不冒称旧轨迹续跑。最多2GPU，显式分批启动，不开自动队列/旧ABD/补折。按真实最终恢复记录结果并维护PROJECTS、README与图谱。\n'
        for path in [ROOT.parent/'AGENTS.md',ROOT/'AGENTS.md',CODE/'AGENTS.md']:
            with path.open('a',encoding='utf-8') as f:f.write(rule)
        for path in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md']:
            with path.open('a',encoding='utf-8') as f:f.write('\n\n## POR位置/尺度匹配优化准备\n三个限定对照已准备；同80.257638起点，真实训练与最终收益待验证。PERM/SW固定，不恢复旧路线，不补折。详细计划：results/deep_learning/dual8222_por_subspace_20260928/README.md。\n')
        with (ROOT/'EXPLORATIONS.md').open('a',encoding='utf-8') as f:
            for mode in MODES:f.write('\n- [POR分布子空间 '+mode+'](outputs/'+PREFIX+mode+'80_fold5/README.md)：同固定头热启动，等待实际训练结果。\n')
    if not STUDY.exists():raise RuntimeError('Prepare the predeclared study first')
    active=active_trainers()
    if len(active)+len(args.launch)>2:raise RuntimeError('Exceeds two GPU jobs: '+str(active))
    for mode in args.launch:
        out=ROOT/'outputs'/(PREFIX+mode+'80_fold5')
        if (out/'config.json').exists() or (out/'launch.json').exists():raise RuntimeError('No duplicate launch: '+mode)
    launches=[]
    for mode in args.launch:
        out=ROOT/'outputs'/(PREFIX+mode+'80_fold5')
        cmd=[sys.executable,'-u',(CODE/'train_dual8222_por_subspace80.py').as_posix(),'--out',out.as_posix(),'--mode',mode]
        with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
            p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=err,stdin=subprocess.DEVNULL,close_fds=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        row={'mode':mode,'pid':p.pid,'command':cmd,'started_at':datetime.now().astimezone().isoformat()}
        (out/'launch.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
        launches.append(row);print('LAUNCHED',json.dumps(row,ensure_ascii=False),flush=True)
    if launches:
        existing=json.loads((STUDY/'launches.json').read_text(encoding='utf-8')) if (STUDY/'launches.json').exists() else []
        (STUDY/'launches.json').write_text(json.dumps(existing+launches,ensure_ascii=False,indent=2),encoding='utf-8')
        note='\n\n## POR子空间实际启动\n本次明确启动 '+str([(r['mode'],r['pid']) for r in launches])+'；共3个预先限定对照，无自动队列。初始复现、训练结果与最终验证仍待实际日志。\n'
        for path in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md']:
            with path.open('a',encoding='utf-8') as f:f.write(note)


if __name__=='__main__':main()
