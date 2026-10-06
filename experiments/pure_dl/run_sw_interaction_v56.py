"""Explicitly launch matched additive/multiplicative full-network candidates."""
from datetime import datetime
import argparse,json,subprocess,sys
from sw_interaction_v56 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v56 SW内部特征交互

## 探索重点与固定对照
v54采样及v55权重扰动都没有超过共同起点，停止比例/rho扫描。本次改网络内部表达：长程注意力块中，两条隐藏分支相加与相乘是否有差异。原206已含九曲线井内秩、ZIQR、QNORM，因此检查后放弃重复添加秩特征的初步构想，没有启动该无新信息方案。
additive/multiplicative两个839481参数私有SW模型，从v46SW第9轮继承TCN、注意力、原归一化及整井FiLM；四个注意力块的FF层全替换为共同新初始化结构。每层96→512拆成a,b两支各256维，融合为(SiLU(a)+b)/sqrt(2)或SiLU(a)*b，两组均经过同一形状LayerNorm256→Linear256→96。只有组合运算不同，完整初始参数数值相同；有效函数表达不同正是变量，不把同参数量当同函数。
四层共同增加2560参数，相对原836921为839481。旧25行SW输出（默认/密度权重/位置/尺度）全部重置；输出权重零，偏置由同64训练井SW标签先验初始化，所以初始144665点预测相同。源函数不是本次起点，不声称源成绩73.680964被精确保留；新先验到训练成绩的变化也不等于超过现有模型。
全网络（旧TCN/注意力/FiLM、新FF、输出头）均训练，不是冻结旧预测器补输出小头。两个模型各自私有权重、无预测投票，不改变源SW或原POR/PERM模型。源骨干同开发折选模的局限保持，不称独立泛化证明。

## 数据、训练与复现
原9曲线/206、Q50、训练64井/第五验证16井144665点/seed20261010、1024上下文/中心128、原8Gaussian+BCE损失与256支持点/800动作+默认评分保持。只有SW训练标签进入损失，训练侧预处理/网格/质量不变；整井输入库无标签/井ID特征，验证输入仅用于自身推理，无外部权重或新增数据。
统一AdamWlr3e-4/wd.01/clip2、cosine80*64至1e-5，max80/pat18/min10/b64/64steps，按本模型SW官方分选最佳并直接恢复；新优化器非旧轨迹续跑。原子last_state含当前/最佳网络、优化器/调度器、RNG与早停，显式恢复。最多2GPU，无自动队列/旧任务/其他折/交付覆盖。CUDA注意力/插值反向warn_only，不声称重新训练逐位一致。

## 已做、待做与后续边界
已实现四层交互模块及两项训练器，开始前无成绩。直接真实训练，记录新FF与旧骨干/输出/FiLM实际梯度、初始参数/先验、源哈希、曲线和显存，不做额外预检/独立审查。结束恢复最佳、实际固定三列评分，有价值候选磁盘重建；局部匹配差与相对现有SW73.680964及TOTAL80.342772分开报告。
待完成实际对照、收益结论、负权重清理和PROJECTS/README/EXPLORATIONS/atlas。无乘性交互增益则不扫分支宽度/层数，不泛化所有门控网络无效。若有局部收益但低当前模型，可保留机制候选；完整原第五严格>80.5且明显收益才考虑其余折。完整原五折OOF>82、独立井确认、新目录交付仍未完成。
单项与双并发总墙钟/显存分开记录，含加载/输入库/预热/启动间隔，代码与收尾另计。实际曲线明显无望可提前终止，未完成对照不写成最终机制收益。
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_sw_interaction_v56_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md','# v56规则\n仅additive/multiplicative两项，四个内部FF替换/同参数初始/新SW头；其余源v46迁移，全模型只用SW损失训练。原数据/Q/评分/80-18/max2保持，无其他折/旧队列/交付覆盖。维护PROJECTS目标/已做/未做/后续。\n'),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_interaction_v56.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v56 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','sw_interaction_v56_20260929','- [v56 SW内部特征交互](results/deep_learning/sw_interaction_v56_20260929/README.md)：四个FF层同参数初始的加法/乘法对照，SW输出全重置，其余SW来源迁移、全模型训练；开始前无成绩。')
    append(ROOT.parent/'AGENTS.md','## v56 SW内部特征交互','## v56 SW内部特征交互\n本批additive/multiplicative两项，四个FF层同参数初始、两支256维加/乘、公共LayerNorm；其余v46SW迁移，输出全重置且全模型训练。原数据/Q/评分/80-18/max2，无旧任务/其他折/交付覆盖，维护sw_interaction_v56_20260929。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
