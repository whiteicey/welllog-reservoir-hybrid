"""Launch only one explicitly selected arm; no queue and no other folds."""
from datetime import datetime
import argparse,json,subprocess,sys
from sw_utility_strength_v82 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v82 SW联合效用强度有限比较

## 探索重点与预定义对照
v81系数.5相对同批0使SW+0.051945916102，11/16井改善；实际组合TOTAL80.380136601599，相对旧组合+0.016553980753。原九曲线和原评分不变，正候选全144665点磁盘重建一致。这是原第五开发折增益，不是完整OOF或独立泛化证明。
本批仅两项a025/a100，系数.25/1，与已经完成的v81 .5比较；系数0的v81 likelihood仅作机制基线。先定义系数网格，再读新验证结果；不继续临时插入点。假设：较弱系数减少代理对似然拟合的影响，较强系数提高最终动作相关监督；两者都可能退步。
全部同839481参数、v56第11轮训练折内迁移骨干/FiLM、原64井SW输出先验重置，不加载v81训练后权重。全SW网络更新，40%DropKey/原1024上下文/128中心/原Q50、种子20261010和验证144665点不变。共同似然与可微联合效用图，一次网络前向，唯一训练差异是系数。历史.5初始状态/预测和关键配置须匹配；CUDA warn_only，不宣称历史重训轨迹逐位一致。

## 数据、损失与预算
只用当前SW训练标签，不用其他目标预测或验证标签训练；固定POR/PERM在验证时实际组合。BCE+8Gaussian NLL与soft动作温度.05联合收益，正负无限CDF尾常数处理沿用v80r1，物理SW用原训练侧归一化反变换。推理保留原256支持/800动作+默认99.9/官方argmax，辅助代理不是精确argmax。
新AdamW1e-4/wd.01/clip2/cos80*64至1e-5，max80/pat18/min10/b64/64steps；按完整SW官方分保存最佳。原子last保存优化器、scheduler、RNG及停止状态，显式恢复。最多2GPU且观察资源后显式启动第二项；不足则串行，不开自动队列。

## 已探索、收益与不继续
启动无新成绩；实际训练记录初始函数及第二batch梯度。最终区分相对.5的系数收益、相对0的机制收益、原第五组合绝对分；报告默认/数值和各井差，不加多个同目标候选，不按行挑选模型。新正候选磁盘重建再实际三列评分。若两边均不优于.5，停止此原温度/架构的系数扫描，保留已有效.5，不泛化全部效用学习。仅相对0重复旧机制但低于.5不当新系数收益。
不保留无新增亮点的checkpoint；完成后先登记删除清单，再删除best/last，保留source/config/日志/紧凑结果。旧来源、.5正模型、交付不改。每批维护PROJECTS、README、EXPLORATIONS和演进图谱。
单项耗时与同工作量总墙钟分开，记录实际并发/显存；加载/预热/训练/验证/保存与启动间隔计入，代码准备和图谱收尾另计，无串行对照不宣称加速比。
目标仍为原第五严格>80.5且明确收益后考虑其余四折，最终完整原分组OOF>82。无其他折/旧任务/自动后续。目标后独立井确认与新目录交付。
'''
RULES='''# v82目标与约束
仅SW系数.25/1对既有.5；原训练侧v56骨干加重置输出先验，839481全参数、40%DropKey，其他数据/评分/Q/seed不变。max80/pat18/min10/normal GPU/max2，无其他折或旧队列/交付覆盖。只读训练标签；原子完整状态，显式恢复。维护PROJECTS目标、已做/待做/未做/目标后工作；独立README、EXPLORATIONS与图谱。负权重按清单删除，原正模型保留。第五严格>80.5不等于完整OOF>82完成。
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_sw_utility_strength_v82_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md',RULES),('README.md',TEXT),('PROJECTS.md',TEXT),('PLAN.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_utility_strength_v82.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v82 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','sw_utility_strength_v82_20260930','- [v82 SW效用强度](results/deep_learning/sw_utility_strength_v82_20260930/README.md)：原同起点/预算/40%DropKey，系数.25/1对已验证.5；启动无成绩。')
    append(ROOT.parent/'AGENTS.md','## v82 SW效用强度范围','## v82 SW效用强度范围\n'+RULES)
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
