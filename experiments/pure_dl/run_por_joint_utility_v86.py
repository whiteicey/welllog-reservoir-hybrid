"""Explicit finite POR comparison; both independent heads share frozen extraction."""
from datetime import datetime
import json,subprocess,sys
from por_joint_utility_v86 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

RULES='# v86目标与约束\n仅POR既有anchored路由30819参数的likelihood/joint两项；原专家/骨干/位置头冻结，无扩容或旧路线。原9曲线206+已有幅值侧63/训练侧拟合/Q50/原第五折/seed/评分/Student-t3原解码保持。normal GPU/max2/80-18/min10/b64/64步；新AdamW3e-5、cos80*64到1e-5，优化器重置。原子完整状态显式恢复，维护PROJECTS/独立记录/图谱；负权重清单删除，正来源/交付保留，无其他折或自动队列。第五严格>80.5且明确收益后再考虑补折，原完整OOF>82未完成。\n'
TEXT='''# v86 POR似然与联合动作效用

## 探索重点与固定对照
SW输出MLP64/局部3点都未取得收益，停止该结构批次，保留当前独立路线TOTAL80.382093291214。转向POR77.551936801885的v31 anchored完整函数，同30819个既有路由参数、同初始化、同输入/损失图/随机窗口；原专家、骨干、位置支路、锚点冻结，只有路由优化器更新，无新容量。
预定义likelihood/joint两项：BCE+Student-t3 NLL（原Q50、除3）与同一似然加系数1的联合动作效用（Q50、除3），动作softmax温度0.05。两组均计算效用图，系数0/1为唯一组间差异。先验和起点不重置，来源v31 anchored第7轮；原优化器不加载。
POR使用标准差参数化的t3 CDF、训练归一化log1p的逆expm1、官方相对容差0.08、默认0.1。原256支持格/800几何动作+默认/原singlecoverage解码完全保持；可微CDF先替换无穷尾再恢复常数，512点梯度checkpoint减少中间显存。效用是argmax代理，不能把它称为精确官方优化。

## 已探索与实际验证
本记录建立时未启动，收益尚未证实。训练直接运行，无独立数值预检或审查。实际起点全144665预测和当前POR一致；实际第二batch记录似然总梯度和效用独立梯度，防止效用无梯度。最终检查冻结状态、最佳恢复、来源hash、原行序和真实固定三列评价；正候选再原始输入磁盘重建。
原64训练/16验证井、578891训练点、144665第五折、seed20261010、原9曲线206及既有幅值63、Q50保持，统计不重新看验证标签；训练仅POR标签，不使用其他目标真值或模型预测。PERM/SW固定现有选择。新AdamW3e-5/wd.01/clip2/max80/pat18/min10/b64/64步，cos80*64至1e-5，这是新优化轨迹；当前主模型和历史脚本不改。原子last包含两个头/各自优化器/调度器/RNG/停止状态，恢复显式授权。

## 收益与不值得继续
最终分别报告两项对当前POR绝对差、joint-likelihood匹配差、默认/数值/每井变化和三列TOTAL。若匹配损失有稳定实测增益但未超过当前绝对分，保留机制并如实区分，不把epoch0恢复算新收益。若两项均无训练收益/匹配效用收益，则关闭此源/路由范围/温度/系数，而不临时扫系数或延长轮数。当前尚无结论，不泛化所有效用学习。

## 已做、计划、未做与目标后
已做：独立协议、模型/损失/训练入口。计划：只运行这两个头，实际评分/正候选重建/负权重清单/图谱。未做：真实收益、其他折与独立泛化；当前不补折。一个GPU进程共享冻结提取、两头依次独立更新，并非两个GPU进程，最多2GPU限制保持；单臂不能从共享总耗时伪造，逐轮活跃臂/过程墙钟/预热数据和显存如实记录。目标后：达到第五严格>80.5且明确收益再决定原其他折，完成原OOF>82后新目录交付和独立井验证；旧模型/交付保留。
'''

def main():
    if len(jobs())>=2:raise RuntimeError('Maximum two GPU jobs')
    out=ROOT/'outputs/pure_dl_por_joint_utility_v86_POR_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True);out.mkdir()
    for name,content in [('AGENTS.md',RULES),('README.md',TEXT),('PROJECTS.md',TEXT),('PLAN.md',TEXT)]:
        (STUDY/name).write_text(content,encoding='utf-8')
        if name!='PLAN.md':(out/name).write_text(content,encoding='utf-8')
    sources=STUDY/'sources';sources.mkdir()
    for name in ('por_joint_utility_v86.py','train_por_joint_utility_v86.py','run_por_joint_utility_v86.py'):
        (sources/name).write_bytes((ROOT/'experiments/pure_dl'/name).read_bytes())
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_por_joint_utility_v86.py'),'--target','POR','--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=p.pid,started_at=datetime.now().astimezone().isoformat(),command=cmd,
        candidates=list(MODES),shared_frozen_extraction=True,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v86 POR联合动作效用显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
        append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    for path in (ROOT/'AGENTS.md',ROOT.parent/'AGENTS.md'):
        append(path,'## v86当前有限范围','## v86当前有限范围\n'+RULES)
    append(ROOT/'EXPLORATIONS.md','por_joint_utility_v86_20260930',
        '- [v86 POR联合动作效用](results/deep_learning/por_joint_utility_v86_20260930/README.md)：当前v31 anchored同30819路由的似然/效用0/1匹配对照，原专家冻结；启动尚无新成绩。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
