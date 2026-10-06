"""Launch only one explicitly selected arm; no queue and no other folds."""
from datetime import datetime
import argparse,json,subprocess,sys
from sw_sam_v93 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='# v93 SW两次计算匹配的SAM优化\n\n探索重点：v92完整POR低LR效用两项负（排除起点joint-likelihood -0.029103、同轮2升16降/均值-0.040396），结束该组合。回到当前SW有效全训练Gaussian/效用1/DropKey.4，在839481参数同架构上改变优化几何。SAM提出同时压低训练损失及其参数邻域损失；本批仅rho0控制与rho.05两项。来源论文：https://arxiv.org/abs/2010.01412；官方仓库：https://github.com/google-research/sam。不下载released模型或额外数据。此处是自编PyTorch一阶L2 SAM加AdamW变体，不冒称论文原数据复现。\n\n固定对照：两项从原v56训练折内骨干/FiLM、原64井SW训练先验重置开始，不加载v82训练后模型；当前SW73.793310952488/固定POR-PERM TOTAL80.382093291214只作绝对历史对照。两项都每次更新两次前后向，完全同batch、回放同DropKey随机态，首梯度全局未clip L2归一化扰动，第二损失梯度用于AdamW更新，更新前从clone精确恢复参数（不用减扰动近似）。rho0与.05计算工作量/更新步/窗口人口匹配；第二forward结束恢复首forward之后RNG，随机流每更新只消费一轮DropKey。不把新增计算当免费，历史单次AdamW与重跑两次控制不声称逐位CUDA训练相同。每更新最后梯度clip2，wd.01/1e-4/cos80*64到1e-5/max80pat18min10/b64/64steps。\n\n原9曲线206+相对深度/训练侧全井128输入bank/原64-16/578891训练/144665第五/seed20261010/Q50/评分、GaussianNLL+效用1/3、温度.05/256支持800动作+default99.9完全不改。只有自身SW训练标签；POR/PERM固定，不使用外部预测/投票/未知测试标签。学习全SW网络，无新参数。SAM无验证梯度或额外Q拟合。\n\n已探索与验证：建立时未训练，收益未证实。正常初始化raw state/旧头/初始动作hash对原Gaussian，实际第二batch记录全网络/效用梯度、请求与实际扰动L2和同mask，原子last在完整更新后保存；保持best/更新后best/optim/sched/RNG/停止，恢复显式。最终核对源/预处理/评分和全144665恢复、实际三列，正绝对或完整固定匹配收益需要两侧磁盘重载。无独立数值预检/审查/重复原Q。\n\n收益与不继续：预定义sam005-control完整训练后最佳差与相对历史Gaussian绝对差分开报告，补同轮轨迹/默认数值/各井。自身较差先验改善不是超越当前。无两种收益则关闭本rho.05/原始SW来源/AdamW/DropKey.4/预算变体，不临时扫rho/LR/ASAM；不泛化全部SAM。\n\n已做：两次计算匹配的有限方案。计划：显式分批训练、真实组合/正重载/清理/图谱。未做：真实收益/其他折/独立泛化。normal最多2GPU，以实测显存启第二项；记录单项数据预热/两次前后向/验证保存、同工作量总墙钟/并行/allocated/reserved，准备和收尾另计。清理计划先落盘、保留正源/匹配依赖/旧交付。目标后：第五严格>80.5且明确收益再考虑其余四折，原完整OOF>82之后新目录交付和独立井验证，无自动旧队列/其他折。\n'
RULES='# v93目标与约束\n仅SW同839481参数/原v56训练折内骨干FiLM+训练先验，两次forward/backward rho0/SAM.05同计算控制，原Gaussian/效用1/DropKey.4/训练窗口和回放mask完全相同。首未clip全局L2扰动、次梯度更新，clone精确恢复，RNG每更新消费一个forward；原9曲线206/输入bank/64-16/Q50/评分/1e-4/80-18/min10/b64/64steps/cos不改。只SW标签，无released权重/其他预测；normal/max2GPU，实测分批，原子完整last/显式恢复。维护PROJECTS/独立记录/EXPLORATIONS/图谱；删除计划先落盘，正源/旧交付保留。第五严格>80.5且明确收益后才考虑其他折，原OOF>82未完成。\n'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_sw_sam_v93_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md',RULES),('README.md',TEXT),('PROJECTS.md',TEXT),('PLAN.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_sam_v93.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v93 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','sw_sam_v93_20260930','- [v93 SW两次计算匹配SAM](results/deep_learning/sw_sam_v93_20260930/README.md)：原Gaussian全SW，同raw初始化/容量/两次计算，rho0/.05；原输入/Q50/评分不改，启动尚无收益。')
    
    for path in (ROOT/'AGENTS.md',ROOT.parent/'AGENTS.md'):append(path,'## v93 SW完整模型效用范围','## v93 SW完整模型效用范围\n'+RULES)
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
