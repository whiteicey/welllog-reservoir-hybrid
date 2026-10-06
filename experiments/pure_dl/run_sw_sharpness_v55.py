"""Explicit matched optimizer trials on the current private SW predictor."""
from datetime import datetime
import argparse,json,subprocess,sys
from sw_sharpness_v55 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v55 当前独立SW模型的权重扰动优化

## 探索重点与预定义对照
v54分层抽样两组均回退，停止比例扫描。本批adamw/sam同836921参数、同v46 SW第9轮完整函数/头/骨干起点，仅改变参数更新方式。原组合TOTAL80.342771761689，SW73.680963725271保持。
adamw正常一次前反向与AdamW更新；sam每步同一批训练窗口计算g，沿rho*g/(||g||2+1e-12)扰动全部有梯度参数，rho固定0.05；在扰动位置重新计算梯度，精确复制恢复原权重后clip2并作一次AdamW更新。两次前向沿用相同随机状态，不把扰动权重保存为模型。复用已有sam_sequence_step.py，其finally恢复逻辑保持不变，不重复历史解析自检。
两组每轮64次优化器更新，SAM两遍前反向，非等计算量比较；记录单项和批次总墙钟、显存与实际closure次数。原网络没有BatchNorm运行统计；当前LayerNorm和输入井编码均随全模型训练。扰动所用标签/损失只有原64训练井SW，无验证梯度或测试标签。
历史pure_dl_sam_sequence曾在旧1000641参数多目标模型跑部分折，结果不能支持本配置收益，也不恢复该旧队列。这里是在当前单目标、长井段/整井条件模型上的独立对照，不声称首次使用SAM，不扫描rho网格。方法来源：[原论文](https://arxiv.org/abs/2010.01412)、[官方项目](https://github.com/google-research/sam)。本地PyTorch实现，不加载官方预训练权重。

## 数据、目标与训练约束
恢复原批次有效点均值Q50*(SW BCE+Gaussian NLL)/3，不使用v54窗口均值或分层采样。两组均原64井均匀训练锚点、相同seed20261010/采样生成器；原9曲线206、1024上下文/中心128、原第五16井144665点、原256支持点/800动作+默认解码及评分不改。训练侧预处理/质量/网格保持，输入井库只含该井输入，不含标签/井ID特征。源模型来自同开发折选模，不能声称独立新井验证。
共同新AdamWlr3e-5/wd.01、cosine80*64至1e-5、max80/pat18/min10/b64/64步，最佳官方SW选模且允许恢复第0轮。骨干/FiLM/输出均训练，三目标模型保持独立，其他目标仅固定列用于组合。原子last_state保存当前/最佳网络、优化器/调度器/RNG/早停；显式恢复，最多2GPU，无自动队列/旧任务/其他折/交付覆盖。

## 已完成、待做与不继续边界
已实现两组训练与复用SAM更新；开始前无成绩。直接真实训练，首个实际第二步记录参数扰动L2、两遍损失和最终梯度，每轮记录实际closure次数，不追加数值预检/独立审查。结束检查初始参数/预测相同、原始权重恢复的已知实现依赖、源码哈希、144665点最佳恢复与三列实际评分。有价值候选磁盘重建；无价值权重清理，源码/配置/成绩及删除清单保留。
待做实际训练结果、相对同容量对照和当前成绩的差、真实额外计算成本、PROJECTS/README/EXPLORATIONS/atlas更新。若SAM无含起点的选模收益，不扩展rho或优化器组合；仅限本热启动/尺度/数据，不泛化全部锐度优化无效。若训练后两组均差于起点，小差异不算可采用的收益。实际曲线明显无望可提前结束。
单项和并发总墙钟包括加载/输入库/预热/启动间隔，代码/最终磁盘重建另计。第五严格>80.5且有明确收益后才考虑其余折；完整原五折OOF>82、独立井确认与新目录交付仍未完成。
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_sw_sharpness_v55_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md','# v55规则\n仅adamw/sam两个v46SW同起点全网络对照，原批次点均损失/Q/数据/评分，固定SAMrho.05两遍更新。80-18/max2，无旧任务/其他折/交付覆盖。维护PROJECTS目标/已做/未做/后续并分别记录计算成本。\n'),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_sharpness_v55.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v55 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','sw_sharpness_v55_20260929','- [v55 SW权重扰动优化](results/deep_learning/sw_sharpness_v55_20260929/README.md)：同v46SW完整起点，AdamW与固定rho.05两遍SAM，全网络原损失；开始前无成绩，非等计算量。')
    append(ROOT.parent/'AGENTS.md','## v55 SW权重扰动优化','## v55 SW权重扰动优化\n本批adamw/sam两项，同v46SW完整函数/全网络训练；原批次点均SW损失，固定rho.05两遍SAM与AdamW对照，精确恢复扰动后更新。原数据/Q/评分/80-18/max2不变，无旧任务/其他折/交付覆盖，维护sw_sharpness_v55_20260929并记录额外成本。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
