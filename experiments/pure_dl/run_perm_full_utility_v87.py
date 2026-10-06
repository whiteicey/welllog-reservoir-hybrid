"""Launch only one explicitly selected arm; no queue and no other folds."""
from datetime import datetime
import argparse,json,subprocess,sys
from perm_full_utility_v87 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='# v87 PERM完整私有网络与联合动作效用\n\n## 探索重点和固定对照\n当前独立三通道原第五折TOTAL80.382093291214，PERM89.396724049366。v86 POR冻结专家/只路由两项无绝对或匹配收益，停止其具体范围；SW全网络效用已有收益，为跨目标验证提供假设，不能先验认为PERM也有效。\n本批预定义likelihood/joint两项，均从v8 PERM双专家+v26 gate+v28 joint+v29 coupled当前完整函数初始化，旧epoch2路由和所有依赖不改。无额外参数，不重置预测或先验。两组完整私有backbone、内部输入路由、PERM读出与保留的小头更新，解除旧no_grad/detach接口；旧其他目标packed输出行仍是buffer，不进入优化器，不用其他目标训练标签或预测。\n两组只有训练效用系数0/1不同，计算同一个损失图。原Q50的PERM BCE+GaussianNLL /3，加Q50的soft动作遗憾/3；温度0.05。PERM官方核max(0,1-|log10动作-log10真值|)，不能套用POR/SW相对容差；log10归一化和训练median/scale不改，默认0.01。原256支持/800几何动作+默认/单覆盖解码保持。可微Gaussian CDF以安全无穷尾处理，512点梯度checkpoint控制激活显存；仅训练目标改变，动作softmax是原argmax的代理。\n\n## 已探索与验证\n当前已实现、尚未启动，无新成绩。直接训练，无独立数值预检/审查。真实初始全144665点预测应与原PERM精确相同；实际第二batch记录两专家、输入路由、原读出修正/v28/v29全部梯度与效用gate/density梯度，确保不只是接口声称可训练。完整buffer不变、来源hash、最佳恢复和实际原三列评价在结束核对；有收益模型原始输入磁盘重建。\n原9曲线206+既有训练侧幅值63、原64/16井/578891训练/144665验证/seed20261010/Q50保持。新AdamW骨干与输入路由1e-5、读出和小头3e-5，wd0.01/分组clip2；80上限/pat18/min10/bs64/64步，cos80*64/eta1e-5，因此body1e-5固定在下界、head3e-5渐降1e-5。新优化器而非旧训练轨迹续跑，dropout随完整模型训练模式恢复，评估关闭；两组同初始化及seed/window策略，不宣称CUDA逐位轨迹相同。\n原子last完整state、优化器、scheduler、RNG、早停、当前/最佳，恢复显式授权。\n\n## 收益和不继续\n收益尚未证实。最终独立报告joint-likelihood、对当前绝对差、默认/数值/各井以及固定POR/SW实际TOTAL。按整个PERM官方分选择checkpoint；不按行挑模型或用未知测试标签。匹配机制增益和绝对超越分别报告，不把epoch0恢复当新收益。两项都无收益则关闭本warm-start/全网络/LR/系数1/温度组合，不临时无限扫权重；适用范围不能泛化所有PERM效用训练。\n\n## 已做、计划、未做及目标后\n已做：新模型/可导捕获/训练接口、两项固定协议。计划：显式两项训练、实际组合、正重载、负清单、图谱。未做：真实成绩、独立泛化、其他折；当前不补折。最多2GPU训练，先运行一项，依据实测GPU空闲和峰值决定第二项是否并发，不设自动队列。记录单项训练墙钟、两项同工作量总墙钟、串并行、GPU allocated/reserved与全卡快照，并区分预热/数据/验证/保存/准备收尾。\n目标后：原第五严格>80.5且明确收益再考虑其余四折，完成原OOF>82后新目录交付与独立井确认。原冠军、正对照和旧交付不变。\n'
RULES='# v87目标与约束\n仅PERM current v29完整私有函数的likelihood/joint两项，所有PERM骨干/内部输入路由/读出及有效小头可训练，原其他目标输出行仍是冻结buffer。原206/已有幅值63/分折/训练侧统计/Q50/评分不改，默认0.01和log10官方效用、原256支持800动作解码保持。无新增容量，低LR body1e-5/head3e-5、新80/18/min10/bs64/64步/cos到1e-5，原子last与显式恢复。normal最多2GPU，以实测显存选择串/并行；无旧队列、其他折或覆盖交付。维护PROJECTS/独立探索/EXPLORATIONS/图谱，负权重清单删除，正源保留。第五严格>80.5和明确收益才考虑补折，完整原OOF>82未完成。\n'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_perm_full_utility_v87_PERM_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md',RULES),('README.md',TEXT),('PROJECTS.md',TEXT),('PLAN.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_perm_full_utility_v87.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v87 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','perm_full_utility_v87_20260930','- [v87 PERM完整模型联合效用](results/deep_learning/perm_full_utility_v87_20260930/README.md)：当前完整私有PERM，似然/效用0/1同起点容量；原数据/Q50/评分不改，启动尚无收益。')
    
    for path in (ROOT/'AGENTS.md',ROOT.parent/'AGENTS.md'):append(path,'## v87 PERM完整模型效用范围','## v87 PERM完整模型效用范围\n'+RULES)
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
