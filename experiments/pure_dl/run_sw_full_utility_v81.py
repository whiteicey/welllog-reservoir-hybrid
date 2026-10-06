"""Explicitly launch matched additive/multiplicative full-network candidates."""
from datetime import datetime
import argparse,json,subprocess,sys
from sw_full_utility_v81 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='# v81 SW完整学习阶段的联合效用\n\n## 探索重点与固定对照\n当前独立通道组合TOTAL80.363582620846，SW73.740423322864。v80r1固定骨干热启动仅拟合3225输出参数，两项均恢复起点；更新后最高joint比likelihood仅+0.002902171987，尚非绝对改善。停止该固定头LR/温度/系数扫参。本批将这条窄局部线索放到独立的完整网络学习阶段；不加载v80训练后权重。\n预定义likelihood/joint两项，同839481参数、同v56 additive第11轮骨干/FiLM、完整SW输出重置为原64训练井先验；这是训练折内迁移，不是全随机网络，也不是v77最佳权重续训。保留已有效40% DropKey（4层、头/查询共享mask，全丢回退valid），每批一次网络前向。两项都计算相同似然与可微联合动作效用，唯一差是辅助系数0/.5；新共同图控制必须训练，历史v77 p40只作额外绝对比较，不冒称CUDA轨迹逐位重现。\n\n## 损失、数据与预算\n原Q50加权SW BCE+8Gaussian NLL除3，联合项仍mean Q*(1-softmax(U/.05)所给真动作收益)/3，默认/数值点皆参与。可微Gaussian区间质量沿用v80r1，正负无限边界先有限替代再设置CDF常数尾；不会误用推理no_grad函数。物理真值按原训练侧logSW归一化反变换；不使用验证标签或其他目标预测训练。平滑动作代理受动作网格密度影响，不等于精确优化官方argmax。\n原9曲线206、原64训练井/第五16井144665点、Q50/seed20261010、1024上下文中心128、256支持/原800动作+默认/评分不改。完整SW骨干、FiLM和输出均可更新，固定POR/PERM。新AdamW1e-4/wd.01/clip2、cos80*64至1e-5、max80/pat18/min10/b64/64steps。原子完整state/RNG，仅显式恢复。推理无DropKey/温度/辅助损失，仍原argmax；新训练用fold5_screening_805。\n\n## 已探索、收益与不继续\n已实现，启动没有新成绩。直接训练，实际初始参数/预测需两项相同且与旧同起点p40一致；实际第二batch记录四组网络梯度与辅助门控/密度梯度，按完整SW官方分选checkpoint。结束核对source/评分/训练侧归一化哈希、最佳恢复与真实三列评分；正候选磁盘重建。分别报告joint-likelihood、当前绝对分、默认/数值子群、各井差与含起点/更新后最高。开发折反复选择不当独立泛化证据。\n若joint没有配对收益，不继续该完整DropKey配方的辅助温度/系数扫参，保留旧正模型；不能把共同训练收益归给辅助项，也不泛化全部效用学习。若control单独超当前模型也按真实绝对改进记录。负/重复权重按清单删除，source/config/紧凑结果保留；旧来源/正控制/交付不改。\n最多2GPU，实测资源后显式启动第二项，无自动队列。单项与同工作量总墙钟/并发/显存记录，加载预热训练验证保存启动间隔计入，作者准备与收尾另计。维护PROJECTS/README/EXPLORATIONS/图谱。\n固定第五严格>80.5且明确收益才考虑其他折，完整原五折OOF>82未达成；无旧队列/其他折/交付覆盖。目标后独立井验证与新目录交付。\n'
RULES='# v81规则\n仅likelihood/joint，同v56迁移骨干与重置SW先验，不加载v77/v80训练后权重。全839481参数更新，40%训练DropKey保持，唯一损失对照为联合效用系数0/.5，温度.05。原9曲线206/分折/Q50/评分/80-18/max2不变，无其他折/旧队列/交付覆盖；维护PROJECTS目标、已做、待做、未做及目标后工作。\n'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_sw_full_utility_v81_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md',RULES),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_full_utility_v81.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v81 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','sw_full_utility_v81_20260930','- [v81 SW全网络联合效用](results/deep_learning/sw_full_utility_v81_20260930/README.md)：likelihood/joint同v56迁移及重置先验，全839481参数与40%DropKey，辅助效用权重0/.5；初始无成绩。')
    append(ROOT.parent/'AGENTS.md','## v81 SW全网络联合效用','## v81 SW全网络联合效用\n'+RULES)
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
