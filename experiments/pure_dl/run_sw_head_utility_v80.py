"""Explicitly launch matched additive/multiplicative full-network candidates."""
from datetime import datetime
import argparse,json,subprocess,sys
from sw_head_utility_v80 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='# v80 SW输出头联合效用对照\n\n## 探索重点与历史依据\n当前v77 p40 SW73.740423322864，独立三通道组合TOTAL80.363582620846。v78全网络低LR微调和v79输入适配均无收益，停止对应扫描。本批固定完整当前骨干，只训练25维原输出线性层（3225参数），比较likelihood/joint两项同初始化。\n历史v12用旧TCN类别分布、全网络训练：两项均回退73.093512781564起点；更新后最高joint72.963080210424 versus likelihood72.941747620362，差+0.021332590061。该结果不是绝对改进，只支持一次有界迁移。本批更换为当前已有效Gaussian模型与固定骨干头拟合，不能冒称历史重跑或已证实新收益。\n\n## 损失与相同条件\n两项都计算原Q50加权SW BCE+Gaussian NLL（除3）。从本模型8Gaussian和默认概率形成原257格联合概率，再乘完全原样的801左右去重动作效用矩阵；训练动作策略softmax(U/0.05)。真标签反变换exp(z*原训练侧scale+median)，默认取99.9；每动作真收益max(0,1-|a-y|/(0.05*(|y|+1e-5)))。辅助项mean Q*(1-策略期望收益)/3。\nlikelihood为L_like+0*L_joint，joint为L_like+0.5*L_joint；共同完整图计算/遍历，唯一差是辅助权重0/.5。门控和Gaussian密度均须在实际训练获得辅助梯度。所有标签、Q及归一化仅原64训练井；不加载其他目标预测到训练损失，不按验证标签构造效用目标。这个平滑动作代理受动作网格密度影响，并不等于精确argmax官方分数。\n完整v77第25轮同起点、同839481总参数，原权重除output.weight/bias外全部冻结且eval（训练不丢key）。原9曲线206、原64/16井/第五144665点/seed20261010/1024上下文中心128/256支持与原800动作+默认/原评分保持。两组新AdamW3e-5/wd.01/clip2/cos80*64至1e-5/max80-pat18-min10-b64-64steps，优化器重置；原子完整状态/RNG，仅显式恢复。推理不使用softmax温度或辅助项，保留原argmax解码。新源使用fold5_screening_805资格函数。\n\n## 已探索、收益与不继续\n已实现，启动尚无新成绩；直接训练，不另作数值预检或审查。实际初始预测恢复完整父函数，新输出层梯度、旧层无梯度/权重逐位冻结、辅助损失两条路径、最佳恢复及真实三列评分需记录。报告含起点与更新后最高、joint-likelihood、各自相对当前绝对差、默认/数值子群；开发折选择不当独立泛化证明。\njoint若无最终配对收益，不扫描当前温度/辅助权重；两项都無绝对改进也不扩大相同头拟合LR扫描。正候选磁盘重建，负或重复checkpoint先写清单后删；来源/历史正候选/交付保留。最多2GPU，根据实测资源显式启动第二项，无自动队列。单项与同工作量总墙钟、并发、显存记录，加载/预热/训练/验证/保存/启动间隔计入，代码与收尾分开。\n维护PROJECTS/README/EXPLORATIONS/图谱。固定原第五严格>80.5且明确收益才考虑其他折，完整原五折OOF>82仍未完成；无旧队列/补折/交付覆盖。目标后独立井验证与新目录交付。\n'
RULES='# v80规则\n仅likelihood/joint，同完整v77 SW起点和3225可训练输出参数；旧骨干/FiLM/norm冻结且eval，唯一对照是联合效用辅助权重0/.5，温度固定.05。只训练侧SW标签，原推理解码/数据/Q50/分折/评分/80-18/max2不变。无旧队列/其他折/交付覆盖；维护PROJECTS目标、已做、待做、未做及目标后工作。\n'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_sw_head_utility_v80_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md',RULES),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_head_utility_v80.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v80 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','sw_head_utility_v80_20260930','- [v80 SW输出头联合效用](results/deep_learning/sw_head_utility_v80_20260930/README.md)：likelihood/joint完整v77同起点，冻结骨干仅3225输出参数，辅助效用权重0/.5；初始无成绩。')
    append(ROOT.parent/'AGENTS.md','## v80 SW输出头联合效用','## v80 SW输出头联合效用\n'+RULES)
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
