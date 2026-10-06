"""One explicit GPU process, two independent heads; no queue or other folds."""
from datetime import datetime
import json,subprocess,sys
from perm_grid_readout_v88 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import append,jobs

RULES='# v88目标与约束\n仅PERM冻结当前v29完整私有函数的概率格读出，两项同37185参数/同初始化，只差效用0/1。原9曲线206+已有幅值63/训练侧统计/原第五/seed/Q50/官方评分与支持动作保持。新head1e-4/80-18/min10/bs64/64步/cos到1e-5，其他目标不训练。normal/max2GPU，本批一进程共享冻结计算。原子完整可训练状态与冻结依赖，显式恢复，另存训练后最佳state/pred。维护PROJECTS/独立探索/图谱；负权重清单删除，正源/交付不改，无其他折/旧队列。原第五严格>80.5且明确收益后考虑其他折，原完整OOF>82未完成。\n'
TEXT='''# v88 PERM条件概率格读出

探索重点：停止v87完整微调，冻结当前PERM89.396724049366模型，独立小MLP64读取自己的双专家隐藏256和已有幅值63。319→64→257，共37185学习参数；原模型eval/no_grad，无其他目标预测/标签或外部投票。原输入/训练拟合/折/seed/Q50保持。

固定对照：likelihood/joint同参数同初始，只差联合官方效用0/1；BCE+双邻支持格软标签CE /3，以及Q50动作softmax遗憾/3，温度0.05。标签仅训练PERM的原log10归一化值；256支持/800几何动作+默认0.01/官方log误差不改。新分布是旧概率格的乘性修正，log质量界0.5、默认logit界0.25；零输出保持旧函数及prior总质量。原先验零质量格不能新增质量，这是本试验的表达限制，不能泛化全部概率格学习。

已探索与验证：建协议时未启动，无新成绩。直接训练，无独立预检/审查；真实初始144665预测应等于原函数，第二batch检查可训练头与效用实际梯度；结束核对原冻结状态/source/最佳恢复/真实三列。保存整体最佳及排除epoch0的训练后最佳完整head/pred，匹配收益候选磁盘重建。PERM之外两列固定。

收益判据：分别报告当前绝对变化与同容量训练后最佳joint-likelihood；完整比较结束后，相对正收益可保留机制及两组验证依赖，即使低于当前最高。不把epoch0恢复写成新收益，不把第五折写成全OOF。若均无绝对或匹配收益，停止此有界乘性格方案，不无限扫界/温度。

已做：新模型、概率监督与独立训练入口。计划：两头真实训练、组合、正收益重载、负清理及图谱。未做：真实收益、其他折与独立泛化。新AdamW1e-4/wd.01/clip2/max80/pat18/min10/bs64/64步/cos80*64到1e-5，原子head/opt/scheduler/RNG/停止及冻结来源；只显式恢复。两臂共享冻结计算，不能伪造单臂墙钟；逐轮活跃臂、过程总墙钟、加载预热/验证保存和GPU allocated/reserved记录，准备/收尾另计。

目标后：第五严格>80.5且明确收益再考虑原其他四折，完整原OOF>82后新目录交付/独立井确认。旧交付与正模型保留，无自动队列。
'''

def main():
    if len(jobs())>=2:raise RuntimeError('Maximum two GPU jobs')
    out=ROOT/'outputs/pure_dl_perm_grid_readout_v88_PERM_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir();out.mkdir()
    for name,txt in [('AGENTS.md',RULES),('README.md',TEXT),('PROJECTS.md',TEXT),('PLAN.md',TEXT)]:
        (STUDY/name).write_text(txt,encoding='utf-8')
        if name!='PLAN.md':(out/name).write_text(txt,encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_perm_grid_readout_v88.py'),'--target','PERM','--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,started_at=datetime.now().astimezone().isoformat(),command=cmd,candidates=list(MODES),shared_frozen_compute=True,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v88 PERM概率格显式启动'
    for p in (ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md',STUDY/'PROJECTS.md'):append(p,marker,marker+'\n'+json.dumps(launch))
    for p in (ROOT/'AGENTS.md',ROOT.parent/'AGENTS.md'):append(p,'## v88当前有限范围','## v88当前有限范围\n'+RULES)
    append(ROOT/'EXPLORATIONS.md','perm_grid_readout_v88_20260930','- [v88 PERM条件概率格](results/deep_learning/perm_grid_readout_v88_20260930/README.md)：冻结当前私有函数，两项同37185学习参数，格CE/效用0或1；启动未有新成绩。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
