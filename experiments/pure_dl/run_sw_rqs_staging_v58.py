"""Explicit paired density-adaptation schedule; no automatic queue."""
from datetime import datetime
import argparse,json,subprocess,sys
from sw_rqs_staging_v58 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v58 新加法骨干与平滑密度的适配

## 探索重点与固定对照
当前v56加法骨干SW73.713975628494、TOTAL80.354325927816，v57并行扩容无收益。本批组合两条已保留机制：v56加法骨干与v48有局部插值收益的RQS条件密度；不继续扩容、线性CDF或分位数量/尾质量扫描。
joint/headstart同840771参数、相同完整初始状态与RQS函数。继承v56第11轮TCN/注意力/加法FF/归一化/整井FiLM以及默认概率头，原高斯密度换为34输出的33分位RQS。新密度零权重，偏置由64训练井非默认SW经验分位初始化；这是新密度起点，不是父模型的原预测。
两组唯一预定义变量是训练顺序：joint首轮全网络共同训练；headstart前5轮只更新4386参数的新密度output，body/norm/condition/gate全部固定，第6轮全解冻。两组统一AdamWlr1e-4，无分支高学习率，优化器从一开始包含全部参数，但冻结期旧参数没有梯度/权重衰减/Adam状态更新。全局cosine时钟均从第1轮开始；headstart骨干少5轮更新，这是顺序机制的一部分，不冒称核心更新量相等。
不加载旧v48 RQS输出权重或预测，不混合模型；复用其原CDF/解析密度/指数尾实现，不再重复数学公式自检。历史v48平滑对照收益并不保证和新骨干组合后有收益，本轮必须看完整实测。

## 数据、训练与复现约束
原9曲线/206、Q50、原训练64井/第五验证16井144665点/seed20261010、1024窗口/中心128保持。原标签变换、256支持点/800动作加默认值与评分保持；RQS CDF差分并归一化得到网格概率。损失仅Q加权SW默认BCE与非默认解析NLL除3，默认行不计连续NLL。井输入库无标签/井ID特征，验证库仅自身推理；来源同开发折选模局限如实保留。
共同max80/pat18/min10/b64/64steps、AdamWlr1e-4/wd.01/clip2、cosine80*64至1e-5。每模型自身第0轮与训练轮共同选最佳，采用当前三通道配方另作判断。原子last_state保存全部模型、优化器/调度/RNG/早停，恢复时按当前轮次重设冻结阶段；仅显式恢复，最多2GPU，无旧任务/其他折/交付覆盖。CUDA注意力/插值反向warn_only，不宣称重新训练逐位一致。

## 已完成、待做与不继续边界
已实现新骨干接入RQS及两阶段训练，开始前无成绩。直接训练，记录首个实际第二步与第6轮解冻后的各组梯度、活跃参数量、初始函数和源哈希；结束恢复144665点最佳，实际三列评分，有价值候选磁盘重建。旧密度公式已验证且未改，不做重复数值预检/独立审查。
待完成两项对照、同起点最终收益和相对当前绝对差、单项/总墙钟/显存、负权重清理、PROJECTS/README/EXPLORATIONS/atlas更新。若先适配无局部收益，不扫描冻结轮数或学习率；若组合整体低于当前模型，不将新密度初始到训练的变化当新冠军。未完成比较不写最终收益。
第五严格>80.5且明显收益后才考虑其他折；完整原五折OOF>82、独立井确认和新目录交付尚未完成。耗时包含加载/输入库/预热/启动间隔，代码和最终重建另计。实际明显无望可提前结束。
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_sw_rqs_staging_v58_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md','# v58规则\n仅joint/headstart两个同初始RQS模型，v56加法骨干和默认头保留、新密度训练侧先验；直接全训练或先密度5轮再全训练。原数据/Q/评分/80-18/max2，无其他折/旧队列/交付覆盖，维护PROJECTS目标/已做/未做/后续。\n'),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_rqs_staging_v58.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v58 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','sw_rqs_staging_v58_20260929','- [v58 新SW骨干与平滑密度](results/deep_learning/sw_rqs_staging_v58_20260929/README.md)：同初始RQS，直接全训练或先密度5轮再解冻，全网络来源v56；开始前无成绩。')
    append(ROOT.parent/'AGENTS.md','## v58 新SW骨干与平滑密度','## v58 新SW骨干与平滑密度\n本批joint/headstart同初始840771参数RQS，v56加法骨干/默认头保留，先全训练或密度单独5轮后全解冻；统一lr1e-4、原数据/Q/评分/80-18/max2，无其他折/旧队列/交付覆盖，维护sw_rqs_staging_v58_20260929。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
