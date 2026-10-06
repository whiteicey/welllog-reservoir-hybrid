"""Explicit base versus parallel128 growth after the adopted v56 gain."""
from datetime import datetime
import argparse,json,subprocess,sys
from sw_transition_v65 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v65 SW相邻状态监督

## 探索重点与预定义对照
从当前v56 additive SW73.713975628494、组合80.354325927816出发，不继续POR容差损失或内部适配器扫描。固定detached/joint两项，同一完整SW骨干和初始函数，新增相同8257参数训练辅助头（256→32→1）。输入为相邻归一化隐藏特征的均值和绝对差，标签为两点SW default状态是否不同。辅助头不参与推理，绝不把真实状态作为模型输入。
仅改变辅助损失是否反传到骨干；detached组同样训练辅助头但detach特征，joint组联合反传。原SW主损失相同；辅助0.1*平衡BCE，正/负类各自Q加权损失均值后平均/3。只取中心128内两端均有标签且同井相邻原行的点对；井界、无效/未标注、重复填充不参与。标签仅原64训练井。

## 已做与训练边界
原9曲线/206输入、训练井拟合预处理、输入整井库不读标签/ID、原第五16井144665点、Q50、seed20261010、1024随机窗口中心128、原Gaussian+BCE/评分/解码保持。全部骨干lr3e-5，新辅助头lr3e-4；两参数组独立clip2以免detach对照因辅助梯度影响主梯度裁剪。AdamW wd.01、cosine80*64至1e-5、max80/pat18/min10/bs64/64steps。完整初始状态相同；将原函数纳入最佳选择，回退不算学习收益。
已实现，尚未训练/无成绩；实际训练中记录辅助到特征的梯度joint>0/detached=0。完整原子last_state包含网络/优化器/调度/RNG/停止状态，仅显式恢复。先串行，最近双骨干并发OOM后不把max2视作默认并发；本轮不需要双骨干，但保守控制总显存。

## 待做、收益和不继续
直接运行两项，比较最终选定SW与训练后最高SW，实际固定POR/PERM后三列评分。保留正候选前磁盘原始输入重建，源哈希/初始预测/最佳恢复作为结果证据，不增加独立预检或审查。若joint无最终配对收益，停止当前状态差辅助系数/头宽度扫描；结论只限此目标，不泛化全部辅助学习。
分别记录单项耗时与串行总墙钟，含加载/预热/训练/评估，代码和收尾另计。负权重清理前写删除清单。第五严格>80.5且明确收益后才考虑补折；完整原五折OOF>82未完成，无旧队列/其他折/交付覆盖。维护PROJECTS、EXPLORATIONS和图谱；目标完成后独立确认及新目录交付。
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if jobs():raise RuntimeError('Explicit serial execution for this study')
    out=ROOT/f'outputs/pure_dl_sw_transition_v65_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md','# v65规则\n仅detached/joint两项，同完整起点和辅助头；只改变SW相邻状态辅助到骨干梯度。原数据/Q/评分/80-18/max2，显式串行；无其他折/交付覆盖。维护PROJECTS目标/已做/未做/后续。\n'),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_transition_v65.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v65 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','sw_transition_v65_20260929','- [v65 SW状态转移辅助](results/deep_learning/sw_transition_v65_20260929/README.md)：detached/joint同初始网络，只差训练辅助到骨干梯度；启动时无成绩。')
    append(ROOT.parent/'AGENTS.md','## v65 SW相邻状态辅助','## v65 SW相邻状态辅助\n本批detached/joint两项，完整v56预测器和新辅助头同初始化，只差辅助到骨干梯度；原数据/Q/评分/80-18/max2，显式串行。无旧任务/其他折/交付覆盖，维护sw_transition_v65_20260929。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
