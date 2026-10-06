"""Explicit single launch for each of two matched final-block candidates."""
from datetime import datetime
import argparse,json,subprocess,sys
from por_interval_v64 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v64 POR物理容差质量监督

## 探索重点与固定对照
复用v63 point逐点内部适配器的已完成对照，不重复相同训练；该对照39888参数，最高含起点POR77.551936801885，训练后77.46023686328418。新候选仍是同18个rank8逐点适配器、相同初始张量和v31完整函数，仅加入训练目标。当前组合80.354325927816。
损失=原POR Student-t3混合似然+0.5*区间质量负对数。对训练非default真实POR y，用官方正收益区间半径0.08*(abs(y)+1e-5)，将左右物理边界经原log1p/median/scale映射，用原df3 CDF求16分量条件质量P；辅助=-log(max(P,1e-8))。default辅助为0，两项均原Q50加权、按所有监督点平均/3。它鼓励落入容差区间，不是官方三角效用的精确优化，也不改变验证动作/评分。原BCE/NLL保留，无验证标签训练。
原df3 CDF代码含no_grad的推理helper不能直接用于训练；新模块复用同解析公式的可微表达。POR变换是log1p、阈值8%，不套用SW的log与5%。比例0.5固定，未开展扫描。

## 边界与已做
网络完全复用v63 point：旧权重冻结、锚点权重固定且输入梯度可传播，0.1单块新增界、39888新参数；原9曲线/206+63侧视图/训练64井/第五16井/144665验证点/Q50/seed20261010/256中心128/原解码不变。AdamW3e-4/wd.01/clip2、cosine80*64至1e-5、max80/pat18/min10/b64/64steps。完整原子训练状态，显式续跑；本轮仅一个新任务，历史对照不恢复。
已实现候选，未有成绩。起点/新增参数哈希/优化预算与v63 point比对，运行时真实梯度/固定权重/初始与最佳144665点及三列评分验证。保留候选才额外原始输入磁盘重建，不做重复数学预检或独立审查。允许CUDA反向非确定，历史控制不是同步双跑或逐位重训保证。

## 收益、未做与不继续
收益尚未证实。待新候选训练、与已完成v63 point配对、当前绝对差、保留/清理、PROJECTS/EXPLORATIONS/atlas。若含初始选择无收益，不扫本区间宽度或辅助系数；仅限当前POR适配器与目标，不泛化所有效用监督。
单项新运行与历史对照耗时分别报告，不能把两次不同批次写成同次双并发墙钟。记录本轮实际墙钟含加载/预热/训练/评估，代码与收尾另计。无自动队列/其他折/交付覆盖，第五严格>80.5且明确收益再考虑补折；完整原五折OOF>82和独立确认未完成。目标达成后可独立验证、新目录交付。
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if jobs():raise RuntimeError('This pair is explicitly serial after prior concurrent OOM')
    out=ROOT/f'outputs/pure_dl_por_interval_v64_POR_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md','# v64约束\n仅interval候选，复用v63 point历史控制，只改POR8percent区间质量辅助0.5；原预测器冻结，单任务，原数据/评分/Q50/80-18/max2保持。维护PROJECTS.md目标/已做/待做/后续，保留旧交付，无其他折或旧队列。\n'),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_por_interval_v64.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=p.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v64 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','por_interval_v64_20260929','- [v64 POR容差质量监督](results/deep_learning/por_interval_v64_20260929/README.md)：同v63 point结构/初始化，仅加入训练区间质量项，复用既有控制。')
    append(ROOT.parent/'AGENTS.md','## v64 POR容差质量监督','## v64 POR容差质量监督\n本批仅POR interval；同v63 point结构和初始化，复用其完整控制，新增训练条件容差质量项0.5；单任务，原数据/Q50/评分/80-18/max2保持。无其他折、旧队列或交付覆盖，维护por_interval_v64_20260929。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
