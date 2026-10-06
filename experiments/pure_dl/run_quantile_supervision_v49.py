"""Explicit two-arm auxiliary quantile supervision study."""
from datetime import datetime
import argparse,json,subprocess,sys
from quantile_supervision_v49 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v49 平滑分位密度的直接分位监督

## 探索重点与预先固定对照
v48 RQS相对同容量linear有局部SW+1.082068，但其绝对SW73.428509530063仍低v46的73.680963725271。按用户接受消融收益的规则，本批携带RQS机制，研究其训练监督，不扫描节点数或尾概率。
两项likelihood/quantile_aux从同一个v48第4轮完整模型出发，同838211参数、原函数和初始状态。两组均保留原默认BCE与RQS密度NLL；quantile_aux额外加0.5×pinball，likelihood系数为0。pinball定义为33个固定概率点(.005至.995)的分位损失均值乘2，仅非默认训练行参与。此为有限分位点监督，不称为精确CRPS，也不使用验证标签、旧预测或其他通道标签作为训练目标。
假设密度NLL主要约束落入标签附近的局部分位区间，而直接分位损失能给整条分位曲线提供监督；是否有实际收益以同容量完成对照为准。骨干、FiLM、默认头、分位头全部训练，没有新增学习模块或冻结小头。两组推理都是同一个RQS CDF与指数尾，默认概率与连续概率共同走原官方动作解码。

## 数据、初始化与预算
当前全局配方仍是v46 TOTAL80.342771761689，不能把v48热启动源的低分当作新SOTA；同时分别报告本批相对热启动、匹配对照及当前最优的差值。原第五折144665点、64训练井/16验证井、9曲线206输入、Q50、1024窗口/中心128、原256支持点与800动作加默认值保持。输入井库没有标签或井编号特征，验证井库仅用于该井推理。
新优化器共同lr3e-5、AdamWwd.01/clip2、cosine80*64至1e-5；80轮上限/patience18/min10/b64/64步，seed20261010。较v48降低微调学习率但两组完全相同，这不是原优化器轨迹续跑，不重训选定轮数。初始完整144665点应精确复现v48训练后保留预测；不重新拟合新密度经验先验。
最大2GPU，显式启动无自动队列，原子last_state保存当前/最佳权重、优化器、调度器、RNG及早停状态；显式恢复。若显著落后可基于真实曲线提前结束。CUDA注意力/插值反向warn_only，不声称重新训练逐位确定。

## 已做、待做与收益边界
已实现两个损失和训练器，开始前无新成绩。直接训练，记录真实梯度、初始与训练快照；结束恢复最佳/实际三列评分。若新候选有保留价值才追加必要的磁盘重建；不重跑v48已通过且未改的CDF公式检查。
待做两项实际训练、同容量对照、热启动与当前最优差值区分、负权重清理、PROJECTS/README/EXPLORATIONS/atlas更新。耗时分别报告单项和双并发总墙钟；加载、输入库、预热计入，编写与最终重载另计。无未完成结果冒称最终收益。
如果直接分位项没有同容量正收益，不继续扫描其权重或分位点数。只关闭当前配置，不推断所有分位监督都无效。有局部收益可携带到独立机制，但原第五严格>80.5且明确收益才考虑其他折；完整OOF>82未完成。无其他折/旧队列/交付覆盖；目标达成后的独立井确认和新目录交付尚未开始。
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_quantile_supervision_v49_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for n,txt in [('AGENTS.md','# v49规则\n仅likelihood/quantile_aux两项，v48 RQS完整同起点，全骨干训练；仅直接分位损失系数0/.5不同。原数据/Q/评分/80-18/max2不变，无补折/交付覆盖。维护PROJECTS目标/已做/未做/后续。\n'),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/n).exists():(STUDY/n).write_text(txt,encoding='utf-8')
    out.mkdir()
    for n in ('AGENTS.md','README.md','PROJECTS.md'):(out/n).write_text((STUDY/n).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_quantile_supervision_v49.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=p.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v49 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','quantile_supervision_v49_20260929','- [v49 平滑分位密度的直接监督](results/deep_learning/quantile_supervision_v49_20260929/README.md)：v48同检查点，似然与似然加0.5分位损失对照；全骨干训练，初始无新成绩。')
    append(ROOT.parent/'AGENTS.md','## v49平滑密度直接监督','## v49平滑密度直接监督\n本批likelihood/quantile_aux两项，同v48 RQS第4轮起点，原似然加0或0.5直接分位损失，全骨干训练。共同新lr3e-5/80-18/max2，原数据/Q/评分不变，无其他折/旧队列/交付覆盖；维护quantile_supervision_v49_20260929。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
