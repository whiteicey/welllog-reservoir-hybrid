"""Explicitly launch one fixed-gate/private-density arm, without a queue."""
from datetime import datetime
import argparse,json,subprocess,sys
from sw_frozen_gate_utility_v84 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v84 固定默认网络与私有SW数值分布

## 探索重点与固定对照
当前保留全人口系数1第五TOTAL80.382093291214/SW73.793310952488。v83均衡效用数值SW+0.487570但默认-0.468565、总体负；这可能含门控偏移，不证明密度独立改善。本批固定原默认预测网络，检验私有密度似然 versus 加均衡效用，两项同初始函数/容量/数据/预算。
两份网络均加载保留v82第25轮原64训练井模型，不重置输出。固定网络eval/no_grad，只有私有密度的骨干/FiLM/输出进优化器；其129个gate输出参数未用于损失或推理。单个SW默认原子/8Gaussian预测分布，联合原效用argmax；没有模型投票或依真实人口选择预测器。固定门控不保证默认点评分固定，条件密度仍参与动作选择，需实际完整SW评价。
旧v62已试过此前v56来源的固定默认/私有密度原似然且无整体收益。本批新条件为当前联合效用来源与明确人口tradeoff，不把结构当首次新发现，不恢复旧队列。

## 已探索、训练与验证
实现完成，启动无新分数；直接训练。初始全部144665点须与当前SW函数相同，训练实际第二batch记录私有body/output/FiLM梯度、辅助密度梯度及冻结gate无梯度；记录固定网络当前/最佳hash不变。两项都计算同一均衡效用图，唯一损失差是系数0/1；原BCE固定无可更新gate，原数值NLL/Q50不改。人口份额.5/.5，比例只用原64训练井计数，人口平均权重1，验证人口不参与拟合。
原9曲线206/输入井bank/分折/seed20261010/1024上下文128中心、私有分支40%DropKey、256支持800动作+默认99.9/评分不改，固定POR/PERM。新AdamW3e-5/wd.01/clip2、cos80*64至1e-5/max80/pat18/min10/b64/64steps；是同checkpoint新优化，不是旧轨迹续跑。原子last含当前/最佳、优化器/scheduler/RNG/停止状态，仅显式恢复。

## 收益、关闭条件与未完成
收益尚未证实。按完整SW官方分选checkpoint，报告默认/数值/各井/真实三列和同损失对照；正候选原始数据磁盘重建。若两项都恢复起点或无有效匹配增益，停止此结构/损失组合，不继续扫LR或人口份额。数值点改善必须同时报告默认和整体代价；不以局部子群正差当总体收益。未完成80轮/其他四折不能写成完整验证。
无亮点权重完成后先清单再删除，原正模型/来源/交付保留。最多2GPU，观察实际显存后第二项显式启动；不足则串行，无自动队列/旧任务/其他折。单项耗时和同工作量总墙钟/并发/显存分开，加载/预热/训练/验证/保存/启动间隔计入，准备和图谱收尾另计。维护PROJECTS目标/已做/待做/未做/目标后工作及README/EXPLORATIONS/图谱。
严格原第五>80.5且明确收益后考虑其余四折，最终完整原分组OOF>82仍未完成；目标后独立井确认和新目录交付。
'''
RULES='''# v84目标与约束
仅固定原SW默认网络、私有密度两损失对照；同保留v82函数初始化，默认网络eval/no_grad、不进优化器；人口比例训练井拟合。原9曲线206/分折/Q50/评分/normal GPU/max2/80-18/min10保持，新lr3e-5。单个默认原子+连续密度，不投票或人口选模型。维护PROJECTS和独立探索/图谱；原子完整状态显式恢复，负权重清单删除，正来源/交付不改；无其他折/旧队列。第五>80.5非完整OOF>82完成。
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_sw_frozen_gate_utility_v84_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md',RULES),('README.md',TEXT),('PROJECTS.md',TEXT),('PLAN.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_frozen_gate_utility_v84.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v84 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','sw_frozen_gate_utility_v84_20260930','- [v84 SW固定默认/私有密度](results/deep_learning/sw_frozen_gate_utility_v84_20260930/README.md)：似然/均衡效用，同当前完整函数热启动；启动无新成绩。')
    append(ROOT.parent/'AGENTS.md','## v84 SW分布分工范围','## v84 SW分布分工范围\n'+RULES)
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
