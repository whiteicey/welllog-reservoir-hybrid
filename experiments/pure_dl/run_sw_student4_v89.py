"""Single explicit Student-t4 candidate; no queue or remaining folds."""
from datetime import datetime
import json,subprocess,sys
from sw_student4_v89 import ROOT,STUDY
from run_retrieval_smooth_v36 import jobs,append

RULES='# v89目标与约束\n仅新SW Student-t4候选，同839481参数、v56训练折内骨干/FiLM和训练SW先验原始权重初始化；当前Gaussian v82 a100作历史对照，旧共享t3负路线不重启。std/scale一致，t4 NLL和CDF同步，原9曲线206/输入bank/分折/seed/Q50/官方支持动作不改。效用1/DropKey0.4/1e-4/80-18/min10/bs64/64步/cos到1e-5，normal/max2GPU。初始函数不强行等于Gaussian。原子完整last、训练后最佳state/pred、显式恢复；维护PROJECTS/独立记录/图谱，负权重清单删除、正源/交付保留，无其他折/旧队列。原第五严格>80.5且明确收益后考虑其余四折，完整原OOF>82未完成。\n'
TEXT='''# v89 SW Student-t4条件密度

探索重点：Gaussian薄尾假设可能限制当前私有SW密度；测试固定df4，scale=std/sqrt2，NLL和概率积分同族。网络839481参数和原组件mean/std初始化不变。t4 CDF=1/2+3w/4-w³/4，w=(edge-mean)/sqrt((edge-mean)²+2std²)；无穷尾在算术前替换，再恢复常数，保持可导有限梯度。

固定对照：历史v82 a100 Gaussian SW73.793310952488/当前固定POR-PERM组合80.382093291214。同原v56第11轮训练折内骨干/FiLM、训练64井输出先验重置、raw参数状态/1e-4/80-18/效用1/40%DropKey；不加载Gaussian已训练权重。原支持256/动作800+默认99.9/官方核/exp逆变换/Q50/原9曲线206与输入bank/seed20261010保持。分布族导致初始函数不同，不套用全预测相等断言。POR/PERM不训练，最后实际三列评价。

历史负证据：旧共享三通道global-cosine+t3，SW73.120707746865对Gaussian73.207488167083为-0.086780420218、TOTAL79.606136235786对79.989866973822负。该源码/权重/队列不复活；新假设限当前私有SW/已有效效用与DropKey配方的df4，不声称重尾已有效。

已探索与验证：记录建立时未训练，收益尚未证实。直接训练，无独立预检/审查。实际第二batch记录全骨干/FiLM/输出及效用gate/density梯度；raw初始hash、源/评分/预处理、原144665行序与最佳恢复核对。正候选原始输入磁盘重建。另保存排除epoch0的训练后最佳完整state/pred，不能按选定轮数重训补证明。

收益和不继续：完整结果分开报告对历史Gaussian绝对/默认/数值/各井差；不能把t4从自身较差先验训练提高当作超越Gaussian。无正绝对或固定同容量对照收益则关闭本df4/来源/预算，不继续扫df或scale。只覆盖本原始初始化配方，不泛化全部重尾模型。

已做：新分布、训练与评估入口。计划：仅此项实际训练、组合、正重载、负清单与图谱。未做：真实收益/其他折/独立泛化。新AdamW1e-4/wd.01/clip2/max80/pat18/min10/bs64/64步，cos80*64到1e-5；原子全状态/优化器/scheduler/RNG/停止，恢复显式。一个GPU进程，最多2，记录加载预热/训练验证保存/总墙钟和allocated/reserved，准备与收尾另计。目标后：第五严格>80.5且明确收益再考虑其余四折，原完整OOF>82后新目录交付与独立井确认；旧交付保持。
'''

def main():
    if len(jobs())>=2:raise RuntimeError('Maximum two GPU jobs')
    out=ROOT/'outputs/pure_dl_sw_student4_v89_SW_student4_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir();out.mkdir()
    for name,txt in [('AGENTS.md',RULES),('README.md',TEXT),('PROJECTS.md',TEXT),('PLAN.md',TEXT)]:
        (STUDY/name).write_text(txt,encoding='utf-8')
        if name!='PLAN.md':(out/name).write_text(txt,encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_student4_v89.py'),'--mode','student4','--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v89 SW Student-t4显式启动'
    for p in (ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md',STUDY/'PROJECTS.md'):append(p,marker,marker+'\n'+json.dumps(launch))
    for p in (ROOT/'AGENTS.md',ROOT.parent/'AGENTS.md'):append(p,'## v89当前有限范围','## v89当前有限范围\n'+RULES)
    append(ROOT/'EXPLORATIONS.md','sw_student4_v89_20260930','- [v89 SW Student-t4](results/deep_learning/sw_student4_v89_20260930/README.md)：当前私有SW分布族，同raw初始化/容量/预算/效用1/40%DropKey；历史Gaussian对照，未有新成绩。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
