"""Launch only one explicitly selected arm; no queue and no other folds."""
from datetime import datetime
import argparse,json,subprocess,sys
from sw_sam_radius_v94 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='# v94 SW有界SAM半径对照\n\n探索重点：v93两次计算匹配SAM.05完整收益：SW73.895436429384 vs rho0 73.785877274491，+0.109559154892；相对旧有效Gaussian+0.102125476896，当前独立三通道TOTAL80.417837208128。两组原始输入重载一致，只有第五折，不冒称全OOF。本批固定rho.025/rho.1两项，检验已有效扰动半径的范围；无无限优化器搜索。\n\n对照预定义：两项各自对已完成.05的SW/TOTAL绝对差，和已完成rho0的同容量机制差；同轮轨迹分开报告。两项仍从原v56骨干FiLM/训练64井SW先验重置，不从v93训练后权重热启动；839481参数/全网络/Gaussian/全人口效用1/DropKey.4/同batch两次前后向，首未clip全局L2扰动、第二梯度AdamW，clone精确恢复，RNG每更新消费一forward不改。数学路径和数据不变，新增第二pass RNG终态核对只读不消耗随机数；peak保存到last并跨恢复取max，非改变优化机制。\n\n原9曲线206+相对深度/原训练侧全井128输入bank/64-16训练578891验证144665/seed20261010/Q50/官方评分/256支持800动作+default99.9不变。仅SW标签，固定当前POR/PERM，不用其他模型预测、released权重、额外数据或未知测试标签。新AdamW1e-4/wd.01/最终clip2/cos80*64到1e-5/max80pat18min10/b64/64steps，原softaction温度.05。CUDAwarn_only不声称逐位训练复现。原子last保存current/best/更新后best/optimizer/scheduler/RNG/早停及跨段peak；恢复显式。\n\n已探索与验证：建立时未训练，尚无新成绩/收益。正常初始state/旧头/动作hash对齐原Gaussian；实际第二batch记录全网络和效用梯度/扰动半径/RNG终态。完整结果核对源/预处理/评分/原144665最佳恢复与真实三列；有绝对或完成公平机制收益的候选必要磁盘重载。无独立预检/审查/重复Q。单项从自身先验提高不等于超越.05。\n\n收益和不继续：两项若都不超过.05，保留既有效.05及已核验控制；按对rho0是否还有亮点区别保留或删除，不以绝对低于SOTA抹去公平正收益。若两项无新的绝对增益，结束这两个半径的范围，不继续无界扫rho/LR/ASAM，转向独立机制；结论限此网络/来源/损失/预算。开发折和同轮诊断非独立泛化。\n\n已做：两有限半径协议及入口。计划：显式串行训练、组合/正重载/负和重复状态清理/图谱。未做：真实收益、其他折、独立泛化。上一批双并发CUDA OOM/cuBLAS失败、串行恢复完成；本批强制一进程，各项显式启动，不自动队列、不关闭其他应用，原最多2GPU上限保留。记录单项加载预热/训练验证保存、同等两项总墙钟/启动间隔/实际并行和跨段allocated/reserved，准备与收尾另计；不把过去逻辑run生命周期重叠当并发。\n\n清理完整计划bytes/hash先落盘再删除，留一份正/匹配依赖的最佳full state；当前正源/旧交付保持。第五严格>80.5且明确收益后才考虑其余四折，各折train-only拟合/折内训练来源，不把第五权重用于另折验证。完整原OOF>82未完成；目标后独立井确认和新目录交付。\n'
RULES='# v94目标与约束\n仅SW SAM半径.025/.1两项，相同839481参数/v56训练折骨干FiLM+训练先验重置，历史已完成rho.05/0作对照；不加载v93已训练权重。原9曲线206/输入bank/64-16/Q50/评分/Gaussian/效用1/DropKey.4/两次计算/RNG回放/clone恢复/1e-4/80-18/min10/b64/64steps/cos不改，仅自身SW标签。强制一个GPU进程、每项显式启动，normal/最多2GPU，不自动队列/关闭别的应用。原子完整last含跨恢复peak，恢复显式；维护PROJECTS/独立记录/EXPLORATIONS/图谱，清理计划先落盘，正源和交付保留。第五严格>80.5且有收益后才考虑其他折，原OOF>82未完成，无旧任务或覆盖交付。\n'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if jobs():raise RuntimeError('v94 explicit serial execution: a trainer is active')
    out=ROOT/f'outputs/pure_dl_sw_sam_radius_v94_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md',RULES),('README.md',TEXT),('PROJECTS.md',TEXT),('PLAN.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_sam_radius_v94.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v94 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','sw_sam_radius_v94_20260930','- [v94 SW有界SAM半径](results/deep_learning/sw_sam_radius_v94_20260930/README.md)：已验证.05收益后的.025/.1，原始初始化和两次计算匹配；原输入/Q50/评分不改，尚无新收益。')
    
    for path in (ROOT/'AGENTS.md',ROOT.parent/'AGENTS.md'):append(path,'## v94 SW完整模型效用范围','## v94 SW完整模型效用范围\n'+RULES)
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
