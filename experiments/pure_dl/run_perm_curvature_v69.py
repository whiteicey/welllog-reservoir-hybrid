"""Explicit base versus parallel128 growth after the adopted v56 gain."""
from datetime import datetime
import argparse,json,subprocess,sys
from perm_curvature_v69 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v69 PERM分位间距曲率约束

## 探索重点与固定对照
从v67正候选RQS第24轮PERM88.827250出发；v68冲突投影无最终收益，不继续其阈值扫描。新假设是分位间距过度起伏可能追逐训练样本，固定增加间距平滑惩罚。仅一项curvature，复用v68 sum已完成对照（含起点88.827250、训练后最高88.807819），不重训未改控制。
相同838211参数、完整函数、初始张量、优化器和两次autograd。惩罚为相邻32个正分位间距log后的二阶差分平方均值，只对训练非default PERM点有效，按原Q加权并对全部监督点平均；总目标原BCE/NLL+0.05*惩罚/3。仍按sum直接相加门控与（密度+惩罚）梯度，不做投影。此项只约束分位间距起伏，不是对物理密度曲率的精确积分，也不强制密度单峰。

## 数据训练边界和已做
原9曲线/206输入、64训练井拟合/Q50、第五16井144665点、seed20261010、1024窗口中心128、原RQS公式/解码/评分不变，仅PERM标签训练；整井输入库无标签/井ID，验证井只供自身推理。全网络AdamW3e-5/wd.01/clip2，cosine80*64至1e-5，max80/pat18/min10/b64/64steps。历史控制不是同步双跑或逐位重训保证，比较初始状态/预测及配置相同。
已实现，未训练/无成绩；直接运行实际梯度、来源哈希、初始/最佳144665点和三列评分核验，正候选保留前磁盘重建。无独立预检/审查。原子完整last_state保存网络/优化器/调度/RNG/停止，只显式恢复；原v67模型保留。

## 收益、待做与不继续
待新候选训练、与历史控制配对及当前绝对差、保留/清理/PROJECTS/EXPLORATIONS/图谱。预固定系数0.05；若无最终配对或源模型增益，不继续系数/差分阶数扫描，只限当前PERM间距约束，不泛化所有密度正则化。初始回退不算训练收益。
本轮一个新任务，历史对照耗时单列；计入加载/预热/训练/评估，代码/收尾另计，不把历史并发墙钟算作本轮双跑。最多2GPU、无队列/其他折/旧交付修改。负模型先清单后删除。第五严格>80.5且明显收益后才考虑其他折，完整原五折OOF>82仍未完成。目标后独立确认和新目录交付。
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if jobs():raise RuntimeError('One new candidate only, no concurrent current study control')
    out=ROOT/f'outputs/pure_dl_perm_curvature_v69_PERM_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md','# v69规则\n仅curvature新候选，复用v68 sum历史控制；同v67完整起点增加0.05间距曲率惩罚，原数据/Q/评分/80-18/max2。无其他折/队列/交付修改，维护PROJECTS目标/已做/未做/后续。\n'),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_perm_curvature_v69.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v69 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','perm_curvature_v69_20260929','- [v69 PERM间距曲率](results/deep_learning/perm_curvature_v69_20260929/README.md)：复用v68 sum控制，仅新curvature；启动时无成绩。')
    append(ROOT.parent/'AGENTS.md','## v69 PERM间距曲率','## v69 PERM间距曲率\n本批只训curvature，同v67 RQS函数及v68 sum优化，新增0.05间距二阶差分惩罚；原数据/Q/评分/80-18/max2，无旧队列/其他折/交付覆盖，维护perm_curvature_v69_20260929。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
