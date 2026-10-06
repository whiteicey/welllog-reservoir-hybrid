"""Explicit admission-controlled launch of one SW normalization scope."""
from datetime import datetime
import argparse,json,subprocess,sys
from backbone_norm_v39 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v39 SW骨干归一化仿射更新

## 探索重点
当前第五折TOTAL80.336513436816，SW73.663082797066。v37/v38冻结骨干的新输出头未能改善，转而更新当前私有双专家骨干已有归一化层的scale/bias。
decoder_norm仅两个专家dec0/dec1/dec2中的GroupNorm/LayerNorm仿射参数；all_norm更新两个专家所有骨干归一化仿射参数。两个范围的参数量不同，是更新范围对照，不是同容量结构消融。无新增参数、无宽度扩展、无冷启动。两者均从同一已验证权重出发。
原卷积/注意力投影/头/路由/校准权重固定；其计算对骨干特征保持可微。移除旧推理捕获的detach/no_grad路径，保留原前向数值与eval/dropout模式。原输入、相对深度、63幅值侧视图、路由和解码保持。

## 固定训练条件
仅SW自身原NLL/3，Q50不改，原64训练井/16第五验证井144665点，原9曲线/206特征。batch64、每轮64步、max80/patience18/min10；AdamW lr1e-4至cosine80*64末端1e-5，无norm权重衰减，clip2。所有归一化都是GroupNorm/LayerNorm，不更新任何批运行统计，也不在验证上更新参数。
两候选各一个GPU进程，最大并发2；独立优化器/早停/完整原子状态。每轮原官方SW分选最佳，最后恢复并与固定POR/PERM实际组装。无需所有目标同时提高。

## 已完成、待做及收益
已实现可微SW骨干读出和两个显式更新范围；待运行结果。收益尚未证实。首批真实训练记录各可训练参数梯度，最终检查冻结参数与最佳预测恢复，不另开预检/审查流程。
不再延伸v37全因子NLL小头和v38直接奖励小头的扫描；本次负结果若出现，只限本更新范围与学习率设置。
最终完整OOF>82未完成；严格第五>80.5且明显收益才考虑其余折。本批无其他折、旧任务或交付覆盖。维护PROJECTS.md/README/EXPLORATIONS/图谱；训练恢复须显式启动，不自动排队。
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',required=True,choices=MODES);a=ap.parse_args()
    active=jobs()
    if len(active)>=2:raise RuntimeError(('max2',active))
    out=ROOT/f'outputs/pure_dl_backbone_norm_v39_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md','# v39 工作约束\n仅两个SW归一化仿射更新范围，无额外任务。维护PROJECTS.md已做/计划/未做/目标/后续；遵守根原数据/评分/80-18/max2和交付保留要求。\n'),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for n in ('AGENTS.md','README.md','PROJECTS.md'):(out/n).write_text((STUDY/n).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_backbone_norm_v39.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=p.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v39 '+a.mode+' 显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','backbone_norm_v39_20260929','- [v39 SW骨干归一化更新](results/deep_learning/backbone_norm_v39_20260929/README.md)：固定原头与卷积，只更新解码端或全骨干norm仿射；参数量不同。')
    append(ROOT.parent/'AGENTS.md','## v39骨干内部更新','## v39骨干内部更新\n继续80.33授权，本批SW decoder_norm/all_norm两项，已有归一化仿射可训练、旧卷积/头/路由固定；原数据/Q50/评分/80-18/max2保持。不扩容、不冷启动、不补折，维护backbone_norm_v39_20260929记录。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
