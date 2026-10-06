"""Launch only one explicitly selected arm; no queue and no other folds."""
from datetime import datetime
import argparse,json,subprocess,sys
from sw_residual_readout_v85 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='# v85 SW最终输出残差结构\n\n## 探索重点与固定对照\n当前独立路线原第五TOTAL80.382093291214/SW73.793310952488。v84r1固定默认/私有密度两项18轮均恢复起点，关闭该组合。本批回到完整SW网络的全人口效用系数1与40%DropKey，预定义point64/local64两个最终输出残差头，对既有线性头v82 alpha1作历史对照。\n两者保留原128→25线性头；新增128→64/GELU/64→25支路，最终投影零初始化。local64多64通道depthwise卷积3点，初始中心恒等；同井窗口原valid遮掉无效邻居。point新增9881参数，local新增10137参数，差256，近容量而非完全同参数。构造支路用CPU fork_rng保护原随机状态；初始两个预测函数与原线性先验一致，新增隐藏层初始化相同，完整新增状态因局部卷积不同而不强行判相同。\n全部从原v56第11轮训练折内骨干/FiLM及原64井SW先验重置开始，不加载v84负模型或v82训练后权重。全网络训练，只有读出结构改变。原9曲线206/输入井bank/分折/seed20261010/Q50/1024上下文128中心/原256支持800动作+默认/评分不变，固定POR/PERM实际三列评分。\n\n## 已探索与实际验证\n已实现，启动无新成绩。原BCE+8GaussianNLL和系数1/tau.05可微联合效用，无人口重权；训练侧目标归一化和有限CDF尾不变。新AdamW1e-4/wd.01/clip2/cos80*64至1e-5，max80/pat18/min10/b64/64steps，按整个SW官方分选择最佳。原子last含当前/最佳、优化器/scheduler/RNG/停止状态，显式恢复。\n直接训练，不另做数值预检/独立审查。实际第二batch记录主body/condition/output和新增hidden/project/局部卷积的梯度；实际初始基本状态hash和全144665点预测与历史线性头一致。历史CUDA重训不宣称逐位轨迹相同。正候选原始输入磁盘重建，再实际组合三列。\n\n## 收益、不继续与未完成\n收益尚未证实。分别报告local-point结构差、两项对线性原模型的绝对差、默认/数值/各井及真实三列；不叠加不同头，不按行挑预测器。若仅匹配结构有收益而未超当前绝对分，按用户增益规则保留该机制并如实区分；未训练完/仅第五不写完整OOF收益。\n若两项均无绝对或匹配结构增益，关闭该小读出批次，不无界扩头或扫参数。无亮点权重完成后清单删除，源/config/日志/紧凑证据保留；原正模型/交付不改。最多2GPU且显存足够才第二项显式启动，不开队列/旧任务/其他折。单项与同工作量总墙钟/并发/显存分开，加载预热训练验证保存和间隔计入，准备/图谱收尾另计。维护PROJECTS目标/已做/待做/未做/目标后工作及README/EXPLORATIONS/图谱。\n严格原第五>80.5且明确收益后考虑其余四折，完整原分组OOF>82仍未完成。目标后独立井确认和新目录交付。\n'
RULES='# v85目标与约束\n仅SW point64/local64小输出残差头，原骨干/全人口效用1/40%DropKey/数据/Q50/评分/normal/max2保持；原64训练先验重置，新1e-4/80-18/min10。新增投影0、局部中心恒等、有效窗口遮罩，fork_rng保护原随机态，容量差如实记录。原子状态显式恢复；维护PROJECTS/独立探索/图谱，负权重清单删除，原正来源/交付不改，无其他折/旧队列。第五>80.5非完整OOF>82完成。\n'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_sw_residual_readout_v85_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md',RULES),('README.md',TEXT),('PROJECTS.md',TEXT),('PLAN.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_residual_readout_v85.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v85 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','sw_residual_readout_v85_20260930','- [v85 SW残差输出头](results/deep_learning/sw_residual_readout_v85_20260930/README.md)：原同起点/预算/40%DropKey，point64/local64对历史线性头，同原骨干/全人口效用1/40%DropKey；启动无成绩。')
    append(ROOT.parent/'AGENTS.md','## v85 SW残差输出头范围','## v85 SW残差输出头范围\n'+RULES)
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
