"""Explicit base versus parallel128 growth after the adopted v56 gain."""
from datetime import datetime
import argparse,json,subprocess,sys
from perm_gradient_v68 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v68 PERM门控与密度梯度

## 探索重点与诊断依据
v67 RQS相对同容量linear最终+0.221226，已磁盘验证并保留；但PERM88.827250低于当前89.396724。现有诊断默认95533点分数-0.422583、数值49132点-0.855092，折算TOTAL贡献各约-0.098/-0.102，不能归因只有单侧出错。本轮验证两项监督在共享表征上的梯度冲突，仅作为机制假设，不把诊断当证明。
固定sum/project两项，同v67 RQS最佳第24轮完整函数、838211参数，无密度重置。两组都分别计算g_class=grad(Q*BCE平均/3)、g_density=grad(Q*NLL平均/3)，因此计算路径和工作量匹配。sum直接相加；project仅在共享body/norm/FiLM全局点积d<0时，使用原梯度对称去除冲突投影：a'=a-d/(||b||²+1e-12)*b，b'=b-d/(||a||²+1e-12)*a，再相加；非冲突直接相加。gate/output私有参数不投影，保持原自身梯度。不是修改损失权重或逐点选模型。

## 数据与训练边界、已做
原9曲线/206输入、训练64井/第五16井144665点、Q50、seed20261010、1024窗口中心128、原RQS正确密度/门控/官方评分和解码保持，仅PERM训练标签。输入整井库无标签/ID；不使用验证标签计算梯度或投影。原参数全可训练，重置AdamW3e-5/wd.01/clip2，cosine80*64至1e-5，max80/pat18/min10/b64/64steps。初始源模型纳入最佳，回退不算新学习收益；各组独立停止和checkpoint。
已实现，未训练/无新成绩。直接运行，实际梯度记录冲突比例/cosine与投影系数、原始初始和最佳144665点、同初始哈希/来源哈希/三列分数核验。正候选额外原始输入磁盘重建，不做独立预检/审查。原子完整last_state含当前/最佳、优化器/调度/RNG/停止状态，仅显式恢复。

## 收益、待做与不继续
待两项完整配对、实际绝对差、保留/清理和PROJECTS/EXPLORATIONS/图谱。若project无最终配对收益，不扫当前投影幅度/阈值；只限当前PERM的两项损失，不泛化所有梯度优化。保留原v67正候选，不因新负结果删除其依赖。正配对收益低于SOTA仍可保留；暂时领先不算最终收益。
最多2GPU，观察显存后显式启动第二项，无自动队列。两次autograd开销计入单项与真实总墙钟，记录并发和显存；加载/预热/训练/评估计入，代码和最终重建另计。负权重先清单后删除。第五严格>80.5且明显收益后才考虑其他折，完整原五折OOF>82未完成；无旧任务/其他折/交付修改。目标后独立确认及新目录交付。
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_perm_gradient_v68_PERM_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md','# v68规则\n仅PERM sum/project，v67 RQS完整同起点，同两次autograd，原数据/Q/评分/80-18/max2保持；不启动其他折/旧队列或改交付。维护PROJECTS目标/已做/未做/后续。\n'),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_perm_gradient_v68.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v68 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','perm_gradient_v68_20260929','- [v68 PERM梯度冲突](results/deep_learning/perm_gradient_v68_20260929/README.md)：同v67 RQS完整函数，sum/project共享梯度对照；启动时无成绩。')
    append(ROOT.parent/'AGENTS.md','## v68 PERM梯度冲突','## v68 PERM梯度冲突\n本批sum/project两项，同v67 RQS源模型，只改门控/密度共享梯度冲突处理；原数据/Q/评分/80-18/max2，不补折/旧队列/交付覆盖，维护perm_gradient_v68_20260929。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
