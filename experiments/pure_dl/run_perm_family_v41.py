"""Explicit two-family PERM study; one process shares frozen backbone forward."""
from datetime import datetime
import json,subprocess,sys
from perm_family_v41 import ROOT,STUDY
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v41 PERM条件分布族

## 探索重点
固定原第五折TOTAL80.336513436816，PERM89.396724049366，比较Gaussian与Student-t3条件混合分布。历史POR用t3、SW有t3对照；历史PERM有CRPS/MAE损失尝试。本次是当前独立PERM强预测器的同参数分布族对照，不恢复旧训练。
两头各11824参数，319→32→48，仅修正最终16分量的权重、位置、尺度；默认概率及所有旧模型固定。两候选同初始化、同训练窗、同优化器，唯一不同是高斯还是df3 Student-t密度及其对应CDF。Student scale=标准差/sqrt(3)，两候选初始分量均值/标准差相同，分布形状不同。

## 初始效果与训练收益分离
高斯第0轮逐点复现父预测。t3第0轮是改变分布族的实测结果，不冒称函数相同。始终单独记录原父模型回退、候选初始分、候选最佳和训练后最高；不把第0轮差异算成训练收益。
训练在原标准化log10 PERM空间做对应NLL/3，Q50与默认/缺失掩码保持。复用已验证高斯CDF与df3 Student CDF，不改原256格支持、原800动作加默认动作及官方评分。默认概率固定不意味着最终默认动作选择完全不变。
新权重/rawscale增量界.25，位置增量界为父分量标准差的.25；两者相同。原9曲线206主特征+既有63幅值侧视图不变。

## 训练预算与进度
80上限/pat18/min10/b64/64步，lr3e-4/AdamWwd.01/clip2/cosine80*64至1e-5。一个GPU进程共享冻结前向，两个头独立参数/优化/早停；总GPU任务最多2，不自动排队。使用实际最佳checkpoint，保存原子last_state供显式恢复。
已实现模块/训练器；实际收益待训练。完成后恢复候选、固定POR/SW实际三列评分；采用新PERM才另做磁盘重建。所有拟合只用原64训练井，第五折144665点用于原选模评分。
v37-v40无收益设置不继续扫描。本次若t3未显示收益，不扩大自由度/混合系数/解码网格搜索。完整OOF>82仍未完成；第五严格>80.5且明显收益才考虑其余折，本批不补折/重启旧任务/覆盖交付。
维护PROJECTS.md、独立README、EXPLORATIONS与图谱；结束后记录差值/负证据并清理无用权重。
'''

def main():
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU jobs')
    out=ROOT/'outputs/pure_dl_perm_family_v41_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md','# v41 工作约束\n仅两个PERM分布族对照，旧模型固定、原分折/数据/评分/80-18/max2保持。记录初始族差与训练差，维护PROJECTS.md目标/已做/未做/计划/后续。无自动恢复或补折，保留交付。\n'),('PROJECTS.md',TEXT),('README.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for n in ('AGENTS.md','PROJECTS.md','README.md'):(out/n).write_text((STUDY/n).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_perm_family_v41.py'),'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=p.pid,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,'## v41显式启动','## v41显式启动\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','perm_family_v41_20260929','- [v41 PERM分布族](results/deep_learning/perm_family_v41_20260929/README.md)：Gaussian/Student-t3同参数密度修正头，分开记录初始族变化与训练收益。')
    append(ROOT.parent/'AGENTS.md','## v41 PERM分布族','## v41 PERM分布族\n本批仅Gaussian/Student-t3同参数PERM条件密度头，旧预测器及默认概率固定；原log10标签/动作/评分/Q50/80-18/max2保持。初始t3变化与训练收益分开；不补折或修改交付，维护perm_family_v41_20260929。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
