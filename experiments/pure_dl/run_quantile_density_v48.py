"""Explicit launch of the two conditional quantile density families."""
from datetime import datetime
import argparse,json,subprocess,sys
from quantile_density_v48 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v48 条件分位密度：线性与平滑样条

## 探索重点与假设
v47任务记忆无收益，当前保留v46 TOTAL80.342771761689、SW73.680963725271。本批更换连续输出分布机制：v46共享骨干/整井FiLM/默认状态头热启动，原8高斯混合密度删除，换为神经网络预测33个有序分位点；全骨干与新头均训练。原默认概率仍与连续密度共同解码，不混合旧预测或按标签选点。
两项预定义linear/rqs，同838211参数和同初始状态。新密度头34输出：中点、总范围、32个间距logits，softplus范围与softmax间距保证有序，间距下界.001。固定CDF高度从.005至.995等距；两端用斜率匹配的指数尾延伸至全实数，不硬裁标签。linear在相邻分位点间用线性CDF；rqs用单调有理二次CDF，内节点导数由相邻割线斜率调和平均确定，不加额外参数。
公式来源：[Durkan等，Neural Spline Flows (2019)，附录A公式19/22](https://arxiv.org/html/1906.04032)。本项目固定CDF高度、调和导数与指数尾是明确的适配，不是论文coupling-flow基准复现。按CDF对标签的导数计算完整密度负对数；在原变换标签空间建模，原标签变换不变。
假设更灵活的条件密度能刻画SW连续值形状，而非继续增加输入记忆。对照可检验同分位参数下插值平滑性的收益；不假定样条优于混合分布。先报告新分布初始成绩，再报告训练后最佳与旧SW差，不能把初始化差当训练收益。

## 初始化与数据边界
骨干/FiLM/默认头加载v46自身原训练折权重，属于热启动，非外部预训练。新密度零权重、偏置来自原64训练井非默认标签的经验分位数（不加权初始分位；训练损失仍用原Q50）。新分布不保持原函数；最佳候选从自身第0轮及实际训练轮中选择，是否替换旧模型单独判定。当前井输入库无标签/井ID，16验证井库只用于本井推理；标准化、标签变换、网格和Q均复用原训练侧来源。
固定原第五144665点，seed20261010，原9曲线206输入，1024窗口/中心128，原256支持点与800动作加默认值。全CDF质量在原支持边界相减并归一，与原网格规则一致。训练仍为Q加权默认BCE加非默认密度NLL，除3保持原尺度；无效点或默认行不贡献连续损失。

## 训练、已做和待做
已实现两种CDF/密度、完整骨干训练、原评分解码和显式启动器；启动前无成绩。两项均max80/pat18/min10/b64/64步，AdamWlr1e-4/wd.01/clip2、cosine80*64至1e-5，原子last_state保存优化器/调度/RNG/停机状态，显式恢复。最多2GPU，无自动队列。
直接实际训练，记录初始/训练快照/最终恢复及真实梯度；新增分布的CDF单调性和概率合法性随真实评分检查。结尾实际三列评分；新正候选才增加全144665点磁盘重建，原已验证预测不重复跑。CUDA注意力/插值反向warn_only，不声称重新训练逐位确定。
待做两项训练、同容量结果比较、实际组合、紧凑诊断、负权重清理和图谱更新。若均不提高，不继续扫分位点数或尾部质量。单项耗时与双并发总墙钟/显存分开，数据加载/预热/输入库计入；代码编写与最终重载另计。
原第五严格>80.5并有明确收益后才考虑其他折；完整OOF>82未完成。无其他折/旧队列/交付覆盖；维护PROJECTS目标/已做/未做/后续，独立README、EXPLORATIONS及atlas。目标达成后的独立井确认和新目录交付尚未开始。
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_quantile_density_v48_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for n,txt in [('AGENTS.md','# v48规则\n仅linear/rqs两项33分位密度，同参数；全骨干训练，原数据/Q/评分/80-18/max2不变。新密度初始化不保持父函数，不虚报训练收益。维护PROJECTS目标/已做/待做/后续，无补折或交付覆盖。\n'),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/n).exists():(STUDY/n).write_text(txt,encoding='utf-8')
    out.mkdir()
    for n in ('AGENTS.md','README.md','PROJECTS.md'):(out/n).write_text((STUDY/n).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_quantile_density_v48.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=p.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v48 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','quantile_density_v48_20260929','- [v48 条件分位密度](results/deep_learning/quantile_density_v48_20260929/README.md)：同33分位参数的线性与有理二次CDF，全骨干训练，非函数保持的新密度头；启动时无成绩。')
    append(ROOT.parent/'AGENTS.md','## v48条件分位密度','## v48条件分位密度\n本批linear/rqs两项33分位密度，同参数、原v46骨干/FiLM/默认头热启动，密度训练侧分位初始化，非原函数；全骨干训练。原数据/Q/解码评分/80-18/max2，无其他折/旧队列/交付覆盖，维护quantile_density_v48_20260929。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
