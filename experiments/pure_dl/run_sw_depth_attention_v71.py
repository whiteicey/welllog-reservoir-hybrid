"""Explicit base versus parallel128 growth after the adopted v56 gain."""
from datetime import datetime
import argparse,json,subprocess,sys
from sw_depth_attention_v71 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v71 SW深度关系注意力

## 探索重点和预定义对照
本轮先核对旧特征：原206已有差分/导数，directional_depth_features.py及已关闭试验覆盖1m/4m物理深度方向统计，故不重复该输入特征。改在当前v56 SW注意力内部学习深度关系，冻结全部原模型，仅4层×4头×8系数=128新参数。
symmetric仅用物理深度差绝对值，directional保留上下方向。均先以10m尺度tanh压缩，8个固定归一化RBF基（各自[0,1]/[-1,1]等距，带宽为一个基间距），偏置是0.5*tanh系数的凸组合，绝对值≤.5。只差关系几何；同参数值/数量及零偏置完整起点，不声称函数空间相同。
原始深度仅用于窗口中心化后的token均值与两两差，平移不改变偏置；不恢复绝对深度、DEVI/AZIM/BIT/CASE输入或井ID。无跨井token，原valid key掩码保留。原注意力加上带偏置与零偏置两次相同浮点mask实现之差，零偏置时差为0，避免改变原布尔mask实现的初始数值函数。此实现每块三次attention，真实开销需记录，不能宣传为免费。

## 数据训练边界与已做
原9曲线/206输入/训练64井预处理/Q50/第五16井144665点/seed20261010/1024窗口中心128/原Gaussian SW损失/官方解码评分保持。原v56全部参数冻结eval，但新偏置经原冻结网络的输入梯度保持；仅128系数AdamW3e-3/wd.01/clip2、cosine80*64至1e-5，max80/pat18/min10/b64/64steps。零系数不重置原预测，回退不是学习收益。POR/PERM不改。
已实现，未训练/无成绩。直接运行真实四组系数梯度、原权重不变、初始与最佳144665点及三列评分验证，源哈希固定；正候选保留前磁盘重建，无独立数值预检/审查。原子完整last_state包含优化器/调度/RNG/停止，仅显式恢复。

## 收益、待做与不继续
两项训练/配对及绝对差、清理/PROJECTS/EXPLORATIONS/图谱待完成。若directional无最终配对收益，不扫距离尺度/RBF数量；若两者都无当前收益，不继续偏置幅度扫描。仅限当前粗token物理关系，不泛化全部位置编码。
最多2GPU，按实际显存决定第二项启动，不自动队列；记录单项与实际总墙钟、并发和显存，包含加载/预热/三次attention训练/评估，代码和收尾另计。负检查点先清单后删，保留原正模型/交付。第五严格>80.5且明显收益后才考虑其他折，完整原五折OOF>82未完成，无其他折/旧任务/交付修改。目标后独立确认、新目录交付。
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_sw_depth_attention_v71_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md','# v71规则\n仅symmetric/directional同128参数深度注意力偏置，原v56全冻结；深度只作相对两两距离，原数据/Q/评分/80-18/max2保持。无其他折/旧队列/交付修改，维护PROJECTS目标/已做/未做/后续。\n'),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_depth_attention_v71.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v71 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','sw_depth_attention_v71_20260929','- [v71 SW深度注意力](results/deep_learning/sw_depth_attention_v71_20260929/README.md)：同起点128参数，距离绝对值/方向对照；启动时无成绩。')
    append(ROOT.parent/'AGENTS.md','## v71 SW深度注意力','## v71 SW深度注意力\n本批symmetric/directional两项，v56全冻结只训128距离偏置系数，深度仅用于相对差，原数据/Q/评分/80-18/max2；无旧任务/其他折/交付修改，维护sw_depth_attention_v71_20260929。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
