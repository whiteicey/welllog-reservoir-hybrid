"""Explicit two-arm continuation after the completed independent-v8 study."""
from __future__ import annotations
import argparse, ast, json, subprocess, sys
from pathlib import Path
from datetime import datetime
from launch_independent_channels_v8 import trainers

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]
STUDY = ROOT/'results/deep_learning/independent_sw_v9_20260928'
PREFIX = 'pure_dl_independent_sw_v9_'
MODES = ('ce', 'ce_utility')
PLAN = '''# SW独立时序卷积分布读出：匹配目标函数对照

## 探索重点
先完成断线遗留的PERM_single与三模型共同恢复；v8六项结果全部选择第0轮，真实独立三模型TOTAL80.274135337791。当前继续推进中，优先给SW引入新的私有输入编码器与非参数分布读出，不再只改POR的8个分量位置。

SW模型包含自己独有的冻结双专家先验和一条全新可训练TCN。新TCN直接读取206维曲线特征+相对深度，宽64；四个门控残差块使用深度卷积kernel5、dilation1/2/4/8，输出256个密度格的对数权重修正。新分支直接感受野61点；旧先验的感受野更大。最终零初始化确保起点接近原函数，真实初始差异须记录。log密度修正限制为2*tanh(raw/2)。该界是预声明的保护，不视为已证实最优。

## 预先固定的两项
ce：真实非默认SW的双邻格软交叉熵+0.01*KL(prior||new)。
ce_utility：在相同损失上增加0.1*(-log(sum p_k*K_SW(center_k,y)))；它是训练侧条件容差效用代理，不等于直接优化最终argmax得分。两组网络、参数量、初始化、抽样顺序、Q、优化器和推理解码完全相同，仅效用项系数0与0.1不同。
新分支端到端训练。旧SW先验与门控冻结，POR/PERM是各自独立模型，不参与本批训练；各模型无共享Parameter。原始门控固定不保证最终默认动作数量固定。

## 已经探索与验证
当前已实现两个候选的模型和训练入口，尚无新成绩。固定原九曲线/206特征、64训练井16验证井、578891训练有效点/144665验证点、原单覆盖256格800候选动作。只训练SW监督，不读取测试标签。训练时保留原Q50，不更新动态质量表。
AdamW lr3e-4/weight_decay0.01，cosine80*64到1e-5，batch64/64步，最多80轮、patience18/min10；种子20261010。第0轮允许保留原模型。两任务最多并行2项，无自动队列；保存完整last_state，只有显式resume。
首两步检查新TCN梯度与先验无梯度，最终恢复并检查冻结权重；最终跨候选比较和磁盘重载由总结阶段执行。浮点概率下限1e-12可能造成极小初始变化，不能默称逐元素一致；以initial_verification为准。

## 取得的收益
尚未证实。每项只记录SW分与固定POR/PERM替换诊断分；最终固定一种SW模型并磁盘重载之后才报告实际三模型结果。不能用运行中最大值冒称完成结果。

## 不继续的方向及边界
六项v8的全骨干微调没有超越各自初始函数，暂不重复相同低学习率和初始化的宽网格。V9改变SW结构与训练目标，和v8的跨阶段差异不能全归于单一因素；ce对ce_utility才是匹配监督消融。
固定第五折已反复用于开发，不把本次选择当独立测试；>80.5及完整OOF>82尚未达成，不自动补折、不恢复旧ABD、不覆盖原交付。只能在这两项结束后决定下一步，未定义更多候选。
'''


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--launch', nargs='*', choices=MODES, default=[])
    args = ap.parse_args()
    if len(args.launch) != len(set(args.launch)):
        raise ValueError('Duplicate launch')
    if args.prepare:
        if STUDY.exists():
            raise RuntimeError('Study already exists')
        for fn in ('independent_sw_distribution_v9.py', 'train_independent_sw_v9.py'):
            ast.parse((CODE/fn).read_text(encoding='utf-8'))
        assert (ROOT/'outputs/pure_dl_independent_v8_selected_fold5/fold_5_report.json').exists()
        dirs = [STUDY] + [ROOT/'outputs'/(PREFIX+m+'80_fold5') for m in MODES]
        if any(p.exists() for p in dirs):
            raise RuntimeError('Refuse overwrite')
        for p in dirs:
            p.mkdir()
            (p/'README.md').write_text(PLAN, encoding='utf-8')
            (p/'AGENTS.md').write_text('# 目标和约束\nSW私有时序卷积分布模型，仅ce和ce_utility两个匹配候选；其余目标独立且冻结。遵循原数据/分折/评分和最多2GPU，保留失败记录。维护PROJECTS.md，每批更新目标、已做、计划、未做与完成后的工作。旧模型/交付不覆盖，无自动补折或队列。\n', encoding='utf-8')
            (p/'PROJECTS.md').write_text('# 实验进度\n目标：验证SW独立TCN分布读出与训练容差代理的增量。\n已完成：v8全部六项和实际三模型恢复；v9两项实现及预定义。\n计划：实际训练、梯度与最佳恢复、匹配目标比较、固定三模型评价。\n未完成：v9成绩、独立泛化、完整OOF>82。\n达成后：固定配方与成本评价；补折仍暂缓，旧交付保持。\n', encoding='utf-8')
        for m in MODES:
            (ROOT/'outputs'/(PREFIX+m+'80_fold5')/'runtime_status.json').write_text(json.dumps({'status': 'planned', 'mode': m, 'target': 'SW', 'atlas_current_trial': False}), encoding='utf-8')
        rule='\n\n## 2026-09-28 继续推进：独立SW分布模型v9\nv8六项与三模型恢复完成后，本批预定义ce/ce_utility两项：SW私有冻结双专家先验+直接读取原206特征的可训练TCN概率格修正；只差训练容差效用项0/0.1，其余原数据/分折/评分/Q/80-18/max2保持。仅新TCN训练，POR/PERM独立且不更新，不把当前设计扩成自动搜索或补折。维护PROJECTS/README/EXPLORATIONS/图谱与真实失败记录。\n'
        for p in [ROOT.parent/'AGENTS.md', ROOT/'AGENTS.md', ROOT/'experiments/AGENTS.md', CODE/'AGENTS.md']:
            with p.open('a', encoding='utf-8') as f:
                f.write(rule)
        with (ROOT/'EXPLORATIONS.md').open('a', encoding='utf-8') as f:
            for m in MODES:
                f.write('\n- [独立SW时序分布v9 '+m+'](outputs/'+PREFIX+m+'80_fold5/README.md)：匹配概率拟合与容差效用监督；结果待验证。\n')
        cur_path = ROOT/'docs/deep_learning/atlas/curation.json'
        cur = json.loads(cur_path.read_text(encoding='utf-8'))
        for m in MODES:
            name = PREFIX+m+'80_fold5'
            cur['nodes'][name] = {'title': 'SW私有TCN分布·'+m, 'summary': '冻结SW私有先验，新TCN直接读输入并修正256格密度；匹配监督对照。',
                                  'conclusion': '尚未训练；单目标分数不得当TOTAL。', 'stage': 'loss', 'highlight': True}
            cur['edges'].append({'source': 'pure_dl_independent_v8_SW_dual80_fold5', 'target': name,
                                 'type': 'evolution', 'note': '冻结私有SW先验，新建可训练时序分支。', 'evidence': 'outputs/'+name+'/README.md'})
        cur_path.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding='utf-8')
    if not STUDY.exists():
        raise RuntimeError('Prepare first')
    if len(trainers())+len(args.launch) > 2:
        raise RuntimeError('GPU process limit would be exceeded')
    for m in args.launch:
        p = ROOT/'outputs'/(PREFIX+m+'80_fold5')
        if (p/'config.json').exists() or (p/'launch.json').exists():
            raise RuntimeError('Refuse duplicate launch '+m)
    started = []
    for m in args.launch:
        p = ROOT/'outputs'/(PREFIX+m+'80_fold5')
        cmd = [sys.executable, '-u', (CODE/'train_independent_sw_v9.py').as_posix(), '--mode', m, '--out', p.as_posix()]
        with (p/'train.log').open('wb') as log, (p/'train.err').open('wb') as err:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=err, stdin=subprocess.DEVNULL,
                                    close_fds=True, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        row = {'mode': m, 'pid': proc.pid, 'started_at': datetime.now().astimezone().isoformat(), 'command': cmd}
        (p/'launch.json').write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding='utf-8')
        started.append(row)
        print('LAUNCHED', json.dumps(row, ensure_ascii=False), flush=True)
    if started:
        (STUDY/'launches.json').write_text(json.dumps(started, ensure_ascii=False, indent=2), encoding='utf-8')
        note='\n\n## '+datetime.now().astimezone().isoformat()+' SW独立TCN分布v9启动\n已显式启动'+str([(x['mode'],x['pid']) for x in started])+'。只比较两个预定义目标，未有新收益证据；旧POR/PERM、交付和原数据不变，无补折。记录见results/deep_learning/independent_sw_v9_20260928。\n'
        for p in [STUDY/'PROJECTS.md', ROOT/'PROJECTS.md', ROOT.parent/'PROJECTS.md']:
            with p.open('a', encoding='utf-8') as f:
                f.write(note)


if __name__ == '__main__':
    main()
