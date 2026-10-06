"""Explicit two-arm population allocation study; no automatic queue."""
from datetime import datetime
import argparse,json,subprocess,sys
from sw_population_utility_v83 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v83 SW效用监督人口平衡

## 探索重点与同条件对照
v82系数1当前原第五TOTAL80.382093291214/SW73.793310952488，优于.5仅TOTAL+0.001956689615；默认SW+0.050181618258，数值-0.096831032031，6井胜/10负。本批检验联合效用的人口分配，不宣称已证实默认占比是退步原因。
预定义balanced/numeric75两项，默认/数值份额.5/.5和.25/.75；只改变联合效用点权重。先用原64训练井SW原点标签计数，权重=份额/训练人口比例，使训练人口平均权重1。没有第五验证人口拟合，原Q50另乘且不改，BCE/NLL不加人口权重，总辅助系数1、温度.05固定。比较已完成全人口系数1；既有系数0和.5仅历史机制证据。
同839481参数、v56第11轮训练折内骨干/FiLM加原64井输出先验重置，全SW网络训练；不加载v82训练后权重。原9曲线206/输入井bank/分折/seed20261010/1024上下文128中心/40%DropKey/原256支持800动作+默认官方argmax保持。固定POR/PERM并实际三列评分；没有按人口/行挑选模型。

## 已探索与验证
已实现，启动无新分数。直接训练，第二batch记录门控/密度/骨干真实梯度与训练人口配置，不做另一次数值预检或独立审查。实际初始状态/全点预测须与全人口1对照一致；新训练AdamW1e-4/wd.01/clip2/cos80*64至1e-5，max80/pat18/min10/b64/64步。原子last含优化器/scheduler/RNG/停止状态，显式恢复。原训练侧logSW反变换及有限CDF尾代码不改。
最终报告对全人口1的总体/默认/数值及各井差，以及对旧无效用的历史机制差；正候选144665点磁盘重建后实际三列评价。只有第五开发折，不把初步/消融写成完整五折收益。

## 实际收益、不继续与未完成
收益尚未证实；若两项都无匹配收益，结束此份额网格并换结构，不继续临时调份额。即使数值点改善也必须报告默认和总体代价，固定一个完整SW模型。正模型保留，无新增亮点checkpoint完成后按清单删除，源/config/日志/紧凑证据保留；旧.5/1及交付不改。
最多2GPU且资源足够才显式双并发，不开队列或旧任务/其他折。单项耗时和同工作量总墙钟/并发/显存分开，加载预热训练验证保存与启动间隔计入，准备和图谱收尾另计。维护PROJECTS目标/已做/待做/未做/目标后工作、独立README/EXPLORATIONS和图谱。
严格原第五>80.5且明确收益再考虑其余四折；完整原分组OOF>82仍未完成。目标后独立井确认及新目录交付。
'''
RULES='''# v83目标与约束
仅SW效用人口份额balanced/numeric75，训练64井拟合比例，BCE/NLL/Q50/系数1/温度.05和原输入/评分不变；单模型推理，没有验证人口拟合或行级选择。839481全SW参数、同v56骨干/输出重置/40%DropKey、max80/pat18/min10/normal/max2GPU，原子状态且显式恢复。维护PROJECTS及独立记录/图谱；正模型和历史交付不改，负权重按清单删除。无其他折/旧队列，严格第五>80.5不是完整OOF>82完成。
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_sw_population_utility_v83_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md',RULES),('README.md',TEXT),('PROJECTS.md',TEXT),('PLAN.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_population_utility_v83.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v83 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','sw_population_utility_v83_20260930','- [v83 SW效用人口平衡](results/deep_learning/sw_population_utility_v83_20260930/README.md)：balanced/numeric75对全人口系数1，训练井拟合份额权重，启动无新成绩。')
    append(ROOT.parent/'AGENTS.md','## v83 SW人口效用范围','## v83 SW人口效用范围\n'+RULES)
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
