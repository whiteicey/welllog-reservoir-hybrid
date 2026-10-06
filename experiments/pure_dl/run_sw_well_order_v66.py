"""Explicit base versus parallel128 growth after the adopted v56 gain."""
from datetime import datetime
import argparse,json,subprocess,sys
from sw_well_order_v66 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v66 SW整井层序残差

## 探索重点与预定义对照
当前v56 SW73.713975628494、组合80.354325927816。旧整井编码对128个深度均匀采样点仅取64维隐表示的均值/标准差/最大值，忽略顺序。本轮冻结整个原预测器，仅加85216参数整井条件残差。先经原冻结point编码得到128×64，再Conv1d64→32/kernel3/GELU，汇总32均值和32标准差，Linear64→1216给六个FiLM层输出0.1*tanh增量。旧FiLM保留，新增输出权重/偏置全零，初始预测不变；原±0.5界加新增±0.1，整体界不冒称仍为±0.5。
预定义permuted/ordered两项，同参数、全部张量与初始化；permuted固定同一随机排列128点、ordered原深度排列。排列只由seed20261166产生，所有训练和验证井共享，不读标签/井ID；只改变邻近点组合，信息集合、pooling/核宽度相同。不混入其他模型预测。

## 数据训练边界与已做
原9曲线/206输入、64训练井预处理/Q50、第五16井144665点、seed20261010、1024随机窗口中心128、原Gaussian+BCE SW损失/解码/官方评分保持。全井输入库无标签，验证井仅自己的推理输入；原骨干/输出头/FiLM全冻结但新调制到输出的梯度通路保留。仅新支路AdamW3e-4/wd.01/clip2，cosine80*64至1e-5，max80/pat18/min10/b64/64steps。相同dropout eval，原函数纳入选择，回退不是学习收益。
已实现，启动时无成绩。直接训练，实际检查新参数梯度、旧权重不变、源哈希、初始/最佳144665点及最终三列分数，不做独立预检/审查。保留新模型前原始输入磁盘重建。完整原子last_state保存网络/优化器/调度/RNG/早停，仅显式恢复。

## 待做、收益与不继续
两项实际训练、同容量顺序收益和绝对差、负权重清理、图谱及PROJECTS待完成。先启动一项观察真实显存余量，再决定第二项并发或串行；max2不是默认2，不自动队列。分别记录单项和同等任务总墙钟，含加载/预热/训练/评估，代码/最终重建另计。
若ordered没有最终配对收益，不扫本支路核宽度/残差幅度；只限当前稀疏采样层序编码，不泛化所有地层顺序信息。第五严格>80.5且明确收益后才考虑补折，完整原五折OOF>82和独立确认未完成，无旧队列/其他折/交付覆盖。维护PROJECTS/EXPLORATIONS/图谱；目标完成后独立确认、新目录交付。
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_sw_well_order_v66_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md','# v66规则\n仅permuted/ordered，冻结原v56预测器，新增同参数整井层序FiLM残差，原数据/Q/评分/80-18/max2保持；维护PROJECTS目标/已做/未做/后续。无其他折、旧队列或交付修改。\n'),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_well_order_v66.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v66 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','sw_well_order_v66_20260929','- [v66 SW整井层序残差](results/deep_learning/sw_well_order_v66_20260929/README.md)：permuted/ordered同初始函数，原模型冻结，只训练新增条件支路；启动时无成绩。')
    append(ROOT.parent/'AGENTS.md','## v66 SW整井层序残差','## v66 SW整井层序残差\n本批permuted/ordered两项，冻结v56全预测器，只训练85216参数整井FiLM残差；同初始化，只差128点邻接次序。原数据/Q/评分/80-18/max2，无旧队列/其他折/交付覆盖，维护sw_well_order_v66_20260929。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
