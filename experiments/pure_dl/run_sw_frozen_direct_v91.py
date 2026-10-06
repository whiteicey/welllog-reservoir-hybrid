"""Single process shares frozen SW extraction for two explicit heads."""
from datetime import datetime
import json,subprocess,sys
from sw_frozen_direct_v91 import ROOT,STUDY
from run_retrieval_smooth_v36 import jobs,append

RULES='# v91目标与约束\n仅当前v82 Gaussian SW冻结全模型，参考/本井两独立2425参数零线性头。原FiLM用本井，两个code均计算，共享冻结推理；无DropKey训练随机化。原9曲线206/输入bank/分折/Q50/官方评分/Gaussian效用1/80-18/min10/bs64/64步/1e-4cos到1e-5不变。normal/max2GPU，本批一个进程，两头各自优化/早停/选模。原子last含current/best/优化器/scheduler/RNG/停止，恢复显式。维护PROJECTS/独立记录/图谱，清理计划先落盘。无其他折/旧队列/交付覆盖，第五严格>80.5且明确收益，完整OOF>82未完成。\n'
TEXT='# v91 冻结当前Gaussian的全井直接头\n\n探索重点：v90同容量well-reference SW+0.004915085801，完整两组都低于Gaussian，原模型保持。把这个局部证据放到当前v82 a100完整函数上，冻结全部839481原参数，仅新增2425参数零Linear96→25，避免全骨干重学和先验重置损失。两组仅新头读取本井code或训练64井参考code不同，同初始化/容量/梯度图/计算；原FiLM两组均本井，reference只用训练测井输入均值。不会加载v90训练后负绝对权重。\n\n对照：当前SW73.793310952488、固定POR/PERM TOTAL80.382093291214。预定义well-reference训练后最佳差与相对当前绝对差分别报告。初始完整144665动作须同当前；原9曲线206+深度/原64-16/Q50/seed20261010/256支持800动作及默认99.9不变，BCE+GaussianNLL+联合效用1/3，温度.05。冻结base eval，DropKey不激活；与v90全网络DropKey训练不同，不能直接以v90-v91差归因。两头各独立AdamW1e-4/wd.01/clip2/cos80*64到1e-5/max80pat18min10/batch64/64步。共享冻结提取，每组对自身SW选模/早停。\n\n已探索与验证：建立时未训练，收益尚未证实。实际训练检查新头与效用梯度、冻结无梯度/最终权重无变化、源/评分/预处理、初始全预测和最佳恢复。正绝对或完整公平局部收益重载头与父模型原始输入；实际三通道组合，不能将epoch0恢复当新收益。无需独立预检/重复Q。\n\n收益与不继续：完成后报告绝对/匹配、默认/数值/各井与实际停止；若无绝对和well-reference训练后收益，关闭当前冻结Gaussian/共享96code/零线性25头/1e-4/预算组合，不临时扫宽度/LR/强度。局部正可带入隔离新假设，但非SOTA或独立泛化。原子完整last，恢复显式；训练后最佳state/pred保留至对照评估。\n\n已做：两有限头和共享训练入口。计划：实际训练/组合/必要重载/清理/图谱。未做：真实收益/其他折/独立泛化。一个GPU进程，实际加载预热/训练验证保存/单项共享工作量/总墙钟和显存分别记录，不把并发上限当默认。目标后：原第五严格>80.5且明确收益再考虑其余四折，完整OOF>82后新目录交付和独立井确认；旧交付保持。\n'

def main():
    if len(jobs())>=2:raise RuntimeError('Maximum two GPU jobs')
    out=ROOT/'outputs/pure_dl_sw_frozen_direct_v91_SW_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir();out.mkdir()
    for name,txt in [('AGENTS.md',RULES),('README.md',TEXT),('PROJECTS.md',TEXT),('PLAN.md',TEXT)]:
        (STUDY/name).write_text(txt,encoding='utf-8')
        if name!='PLAN.md':(out/name).write_text(txt,encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_frozen_direct_v91.py'),'--target','SW','--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v91 SW冻结全井直接头显式启动'
    for p in (ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md',STUDY/'PROJECTS.md'):append(p,marker,marker+'\n'+json.dumps(launch))
    for p in (ROOT/'AGENTS.md',ROOT.parent/'AGENTS.md'):append(p,'## v91当前有限范围','## v91当前有限范围\n'+RULES)
    append(ROOT/'EXPLORATIONS.md','sw_frozen_direct_v91_20260930','- [v91 SW冻结全井直接头](results/deep_learning/sw_frozen_direct_v91_20260930/README.md)：固定当前Gaussian函数，仅参考/本井直接头2425参数，共享冻结计算，未有新成绩。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
