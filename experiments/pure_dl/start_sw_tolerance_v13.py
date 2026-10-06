"""Explicit bounded launch of the two v13 arms, with persistent study records."""
from pathlib import Path
from datetime import datetime
import argparse, json, os, subprocess, sys
import psutil

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/independent_sw_tolerance_v13_20260928'
MODES=('likelihood','tolerance_mass')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--launch',nargs='+',choices=MODES,required=True); args=ap.parse_args()
    active=[]
    for proc in psutil.process_iter(['pid','name','cmdline']):
        try:
            v=proc.info; argv=v.get('cmdline') or []
            if 'python' in (v.get('name') or '').lower() and any('pure_dl/' in a.replace('\\','/') and Path(a.replace('\\','/')).name.startswith(('train_','benchmark_')) for a in argv):
                active.append(v)
        except (psutil.NoSuchProcess,psutil.AccessDenied): pass
    if len(active)+len(args.launch)>2:
        raise RuntimeError('At most two GPU jobs; active='+str([p['pid'] for p in active]))
    for mode in args.launch:
        p=ROOT/'outputs'/('pure_dl_independent_sw_v13_'+mode+'80_fold5')
        if (p/'launch.json').exists() or (p/'config.json').exists():
            raise RuntimeError('Explicit new launch would duplicate '+str(p))
    STUDY.mkdir(parents=True,exist_ok=True)
    rules='# 目标与约束\n\n本目录仅记录SW私有双专家的容差质量损失对照。保留原数据、分折、种子、评分、三通道隔离、最多2GPU与80/18/min10预算。旧模型和交付不覆盖，无自动补折或重启。维护PROJECTS.md，记录目标、已做、计划、未做及目标完成后的选项；每批操作完成后更新。\n'
    (STUDY/'AGENTS.md').write_text(rules,encoding='utf-8')
    method='''# SW私有双专家：容差概率质量监督v13

## 探索重点
当前SW较强参考73.622384767129，独立三模型TOTAL80.274135337791。v12已在本轮读取前完成，较弱TCN两项都以第0轮为最佳；不重复它们，不把既有任务当本轮启动。
本轮在强SW私有双专家上比较proper hurdle NLL与NLL+0.5*(-log tolerance coverage)。默认质量与连续概率格都参与辅助项。它是有偏决策代理，保留proper NLL避免只学覆盖；不等于推理时argmax的精确风险。

令g为标量默认概率，p为256维连续概率，s_r为原物理单位格中心，y为SW真值。
K(a,y)=max(0,1-|a-y|/[0.05(|y|+1e-5)])。
C=g*K(99.9,y)+(1-g)*sum_r p_r*K(s_r,y)。
L_cov=mean[Q*(-log(max(C,1e-8)))]/3，L=L_like+alpha*L_cov。alpha分别0和0.5，预先固定不扫描。
连续p从原Gaussian混合分布的CDF差分得到；新增实现保留梯度，原conditional_probability仅用于推理。显式比较首个真实训练批次的概率前向与原推理函数，并验证辅助梯度到达门控、密度和SW私有骨干。

## 已经探索的内容与验证
同一独立SW checkpoint、同网络/参数/初始化、同随机窗口和两个AdamW参数组。每组全SW模型训练，POR/PERM不加载到训练图。源模型曾在64训练井多任务预训练，属于单任务热启动，不声称从头训练。
使用原9曲线/206特征/相对深度/原第五折、seed20261010，固定源epoch50Q；最大80轮、patience18/min10、batch64、64步每轮。骨干lr1e-5/输出lr3e-5、cosine5120步至1e-6、wd0.01、clip2。两组都计算相同辅助量，只在loss系数上不同。
推理保留原单覆盖、256格与800候选动作+默认值，无阈值或分辨率选择。每轮保存原子last_state，含模型、优化器、调度器、RNG和早停状态，仅显式resume。

## 取得的收益
启动时尚未证实。分开报告初始、更新后最佳、最后选定恢复成绩。最终TOTAL仅在实际固定通道组合后报告；常规替换TOTAL是固定旧POR/PERM预测的诊断。当前开发折反复选择，不能称独立验证。

## 不值得继续的点及证据边界
本批仅两个预定义模式。不重复已完成的v12，不扫描alpha、学习率或窗口。不因单次负结果否定所有容差监督或独立模型。第五折严格>80.5及完整OOF>82目标仍未达成；不自动补折、不改旧交付。
'''
    (STUDY/'METHOD.md').write_text(method,encoding='utf-8')
    (STUDY/'PROJECTS.md').write_text('# 进展\n\n目标：检验强SW私有骨干上的容差概率质量损失能否改善固定第五折。\n已完成：读取旧实验和规则，新损失与训练器已实现；v12两项已结束且无增益。\n计划：显式运行两个匹配候选，读取原始评分与误差分解，记录最终恢复。\n未完成：实际训练收益和跨井泛化验证。\n目标完成后：固定候选后再依授权考虑其他折，当前不补折、不覆盖交付。\n',encoding='utf-8')
    runs=[]
    for mode in args.launch:
        out=ROOT/'outputs'/('pure_dl_independent_sw_v13_'+mode+'80_fold5'); out.mkdir(parents=True,exist_ok=True)
        (out/'AGENTS.md').write_text(rules,encoding='utf-8')
        (out/'README.md').write_text(method+'\n本目录模式：'+mode+'。\n',encoding='utf-8')
        (out/'PROJECTS.md').write_text('# 进展\n\n目标：完成本模式训练和实际恢复。\n已完成：实现与协议。\n计划：最多80轮按SW分选模。\n未完成：本次实测成绩、收益和最后重载。\n达成后：与匹配对照比较，当前不补折或改交付。\n',encoding='utf-8')
        argv=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_tolerance_mass_v13.py'),'--mode',mode,'--out',str(out)]
        with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
            child=subprocess.Popen(argv,cwd=str(ROOT),stdout=log,stderr=err,stdin=subprocess.DEVNULL,
                close_fds=True,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        run=dict(mode=mode,pid=child.pid,started_at=datetime.now().astimezone().isoformat(),command=argv)
        (out/'launch.json').write_text(json.dumps(run,ensure_ascii=False,indent=2),encoding='utf-8')
        runs.append(run)
    note='\n\n## '+datetime.now().astimezone().isoformat()+' 强SW容差质量v13启动\n本轮显式启动'+str([(x['mode'],x['pid']) for x in runs])+'。使用SW私有双专家73.622385同起点，区别仅辅助损失系数0/0.5。此前v12已完成无收益，未重复启动。原数据/分折/评分/预算保持；不恢复旧任务、不补折，真实收益待读取。记录results/deep_learning/independent_sw_tolerance_v13_20260928。\n'
    for p in [ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md',STUDY/'PROJECTS.md']:
        with p.open('a',encoding='utf-8') as f:f.write(note)
    with (ROOT/'EXPLORATIONS.md').open('a',encoding='utf-8') as f:
        f.write('\n- [强SW私有双专家容差质量监督v13](results/deep_learning/independent_sw_tolerance_v13_20260928/METHOD.md)：同起点NLL/容差质量对照，启动时收益未证实。\n')
    print(json.dumps(runs,ensure_ascii=False))


if __name__=='__main__':main()
