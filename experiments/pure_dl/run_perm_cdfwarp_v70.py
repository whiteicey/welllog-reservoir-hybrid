"""Explicit single launch for each of two matched final-block candidates."""
from datetime import datetime
import argparse,json,subprocess,sys
from perm_cdfwarp_v70 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v70 当前冠军PERM条件CDF校准

## 探索重点与固定对照
转回当前冠军PERM v29 coupled89.396724049366，冻结全部双专家/原读出/路由，不继续RQS负优化设置。仅新增10306参数小头，读取两个128维隐藏及已有63幅值侧特征；Conv319→32→2，affine中间恒等，nonlinear中间GELU，同参数张量与零输出初始化。两组原始完整预测相同。
头产生u/v=0.25*tanh(output)，变换G(F)=F+uF(1-F)+vF(1-F)(1-2F)。端点G(0)=0/G(1)=1；导数1+u(1-2F)+v(1-6F+6F²)，因两个基导数绝对值≤1且|u|/|v|≤.25，导数≥.5。训练新连续密度logpdf=原Gaussian mixture logpdf+log G'(F)；默认门控原值冻结。不是验证分数校准或标签作输入，两个神经头仅训练64井PERM似然/Q50。
验证沿原256格概率累计作G变换，并以原概率+校正CDF差的残差形式保持零头逐位一致；原800动作/default/评分不变。端点校正为0、总质量守恒，浮点误差clamp非负，不再次归一化破坏零头身份。无需改变原混合分量位置或尺度。

## 训练边界与已做
原9曲线/206输入及已有63侧特征/原64训练井和第五16井144665点/seed20261010/Q50/256随机窗口中心128不变，无额外特征数据或其他模型投票。原模型完全冻结eval，新增头AdamW3e-4/wd.01/clip2、cosine80*64至1e-5、max80/pat18/min10/b64/64steps。初始模型纳入选择，回退不算新收益；完整原子last_state保存头/优化器/调度/RNG/停止，仅显式恢复。
已实现，启动时无成绩，直接训练。实际新头梯度/原模型状态不变、初始/最佳144665点、来源哈希和三列评分作为结果证据；保留正候选前原始输入磁盘重建，不做独立数值预检/审查。

## 收益、待做与不继续
待两项实测/配对差/当前绝对差、负权重清理、PROJECTS/EXPLORATIONS/图谱。若非线性无最终配对收益，不扫本头宽度/激活；若两者都无收益，不扫当前多项式阶数/幅度。结论只限这个条件CDF变换，不泛化全部分布校准。第0轮回退不算学习收益，开发折仍需独立确认。
最多2GPU，显存允许时双跑，无自动队列；分别记录单项及两项实际总墙钟、并发与显存，含加载/预热/训练/评估，代码/收尾另计。先清单后删负权重，保留所有既有正候选/交付。第五严格>80.5且明显收益后才考虑其他折，完整原五折OOF>82未完成；无其他折/旧任务/交付覆盖。目标后独立确认、新目录交付。
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_perm_cdfwarp_v70_PERM_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md','# v70规则\n仅PERM affine/nonlinear两项，同冠军完整起点和参数；只训练有界CDF变换小头，原骨干/头/路由冻结。原数据/Q/评分/80-18/max2，无其他折/队列/交付修改；维护PROJECTS目标/已做/未做/后续。\n'),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_perm_cdfwarp_v70.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=p.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v70 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','perm_cdfwarp_v70_20260929','- [v70 PERM条件CDF校准](results/deep_learning/perm_cdfwarp_v70_20260929/README.md)：冻结冠军原模型，affine/nonlinear同参数零初始校准；启动时无成绩。')
    append(ROOT.parent/'AGENTS.md','## v70 PERM条件CDF校准','## v70 PERM条件CDF校准\n本批affine/nonlinear两个10306参数头，冻结当前v29 PERM全预测器，学习有界正导数CDF变换；原数据/Q/评分/80-18/max2，无其他折/旧队列/交付修改，维护perm_cdfwarp_v70_20260929。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
