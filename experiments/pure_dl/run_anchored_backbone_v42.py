"""Explicit single launch for each of two matched final-block candidates."""
from datetime import datetime
import argparse,json,subprocess,sys
from anchored_backbone_v42 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v42 SW末端残差块与原权重约束

## 探索重点、假设及对照
从原第五折TOTAL80.336513436816、SW73.663082797066出发。冻结头小改多次无收益，v39仅norm和v40井不变性也无收益；本次允许已有末端残差块卷积修改局部表征，并检验参数原点约束能否减少微调退化。不是网络扩容或冷启动，不延伸分布族小头。
free与anchored均只训练两个私有专家的dec0全部原参数，其余骨干/头/路由/幅值小头固定。前向仍对这些残差块可微，eval/dropout状态保持。相同可训练参数、初始化、窗口、优化器；唯一差别为原权重约束系数0或1。
约束=各参数张量 mean((p-p0)^2)/max(mean(p0^2),1e-4) 的平均，p0为本训练折已训练原模型权重。不是伪标签，不读验证标签产生监督；无原预测蒸馏。对每个张量做RMS归一化，固定0.01 RMS下限，初始惩罚为0。
历史重叠窗口一致性对两个偏移窗口同时反传、第五折80.1022不达标；本次不重复该机制。v39使用norm而本次使用完整dec0卷积块，两个实验不能视为仅一个因素的对照；本次free/anchored才是匹配消融。

## 训练与选择
原64训练井/16验证井、9曲线206主特征及原63幅值侧特征、Q50、SW NLL/3、原解码和评分不变。80最大轮/pat18/min10/b64/64步；AdamW lr1e-4/cosine80*64至1e-5、wd0、clip2，两组相同。每组独立优化器/早停/按官方SW选最佳；保留第0轮回退。保存原子完整训练状态，恢复需显式启动。
最多2GPU进程，无自动队列。分别记录单项耗时、峰值显存和相同工作量总墙钟。原POR/PERM保持独立且不训练；最后固定3模型实际组合评分。

## 已做、收益及待做
已实现模块/训练器，收益尚未证实；启动后记录真实快照，结束记录更新后最高和含原模型回退的分数，不能将第0轮回退视作训练收益。
若两项均无收益，不继续本系数/学习率网格扫描；结论只限当前SW起点/更新块与固定配置。
目标仍为第五折严格>80.5、有明确收益后再考虑其余折，完整原五折OOF>82未完成。本批无其他折、旧队列重启或交付覆盖。维护PROJECTS.md、独立README、EXPLORATIONS和图谱。目标达成后的可选工作为新井独立验证与交付，当前不启动。
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU jobs')
    out=ROOT/f'outputs/pure_dl_anchored_backbone_v42_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md','# v42约束\n仅SW dec0的free/anchored同容量对照，原数据/评分/Q50/80-18/max2保持。维护PROJECTS.md目标/已做/待做/后续，保留旧交付，无其他折或旧队列。\n'),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_anchored_backbone_v42.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=p.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v42 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','anchored_backbone_v42_20260929','- [v42 SW末端残差块原权重约束](results/deep_learning/anchored_backbone_v42_20260929/README.md)：free/anchored同更新范围；只改参数原点约束。')
    append(ROOT.parent/'AGENTS.md','## v42末端残差块','## v42末端残差块\n继续当前80.33优化，本批SW free/anchored两项；同dec0可训练范围，仅原权重约束系数0/1不同，原数据/Q50/评分/80-18/max2保持。无其他折、旧队列或交付覆盖，维护anchored_backbone_v42_20260929。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
