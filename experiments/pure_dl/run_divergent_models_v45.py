"""Launch one explicitly selected new architecture; maximum two GPU processes."""
from datetime import datetime
import argparse,json,subprocess,sys
from divergent_models_v45 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v45：离开冻结小头路线，两个新私有预测器

## 目标与用户方向
用户明确认为此前过于保守，要求发散。当前目标仍是原第五折严格>80.5、有明显收益再考虑其余四折，完整原五折OOF>82。本批重新开放两个独立新架构的从随机初始化训练，不加载旧网络权重、不以复现80.33初始函数为要求；原模型只是保留的替换基线。

## 两条预定义路线
1. SW long_section：原206特征+相对深度，1024点上下文，中央128点监督。96维逐点/局部多尺度支路，16点分块得到64个token，四层四头全局注意力，再把全局信息融合回128维逐点细节。直接输出SW门控和8分量Gaussian(logSW)密度；整个骨干和输出头训练。
2. POR curve_axial：仅GR/PE/SP/CAL/AC/DEN/CNL/RXO/RT九个基础曲线的原训练侧标准化特征+相对深度，512点上下文，中央128点监督。8点一块，每块9个曲线token；三层交替曲线轴/深度轴四头注意力，64维token，融合回逐点128维读出。直接输出POR门控和8分量Student-t3(log1pPOR)密度。
POR同时随机遮住25%的曲线×8点块，输入显式遮挡标识，细节支路也使用同一遮挡输入；同次前向联合训练原目标NLL+.1*曲线重建SmoothL1。重建只来自训练井真实输入，不产生储层伪标签，验证输入不做预训练或在线适配。九曲线子集是单独记录的新输入变体；旧206基线和所有旧源码保持不变，无DEVI/AZIM/BIT/CASE。
两条同时改变架构/上下文/目标，POR另改变输入表达和训练任务；这是不同研究路线，不宣称同容量单因素消融。若获得亮点，后续另开针对性消融。

## 与历史区别
历史mask_pretrain是旧多目标256点U-Net先600步重建再40轮监督、历史首折81.754低于对照81.798，不能把普通掩码预训练当新点。本次POR采用九曲线二维轴向架构、同程遮挡增强与重建、512上下文、独立目标80轮；不原样重跑旧实验。历史v15潜变量/状态空间、v16 SW TCN/latent均保留；本次也不是仅给原骨干加FFT或微小冻结头。

## 训练与隔离
两者只使用原64训练井自己的目标标签，原第五16井144665点仅验证选checkpoint；原分组/行序/seed20261010/官方指标/原物理256支持与800候选动作保持。复用已核验的原训练侧标准化、Q50和解码网格；从旧文件读取的metadata仅特征名和网格，任何state_dict不加载到新模型。初始化的默认比例/8位置先验只由训练标签计算。
max80/pat18/min10、batch64、64步/轮，lr6e-4 AdamWwd.01、clip2、cosine80*64到1e-5。网络容量以实际参数量和显存报告为准。两个进程最多并发2，先看第一项真实训练显存再启动第二项，不做单独benchmark。原子完整last_state含优化器/调度器/RNG/停止状态；用户恢复须显式启动。额外支持有证据的stop_requested.json在完成轮次边界早停并恢复最佳。
验证窗口长度按新架构改变，中央128点覆盖与原行序不变，所有窗口限制在同一井；解码batch16对两者固定，报告真实全144665点。训练预算相同但计算量不同，分别记录总墙钟/单项/预热与处理、显存。

## 已做、计划、未做及后续
已实现两种完整新模型、训练器和新窗口评价；收益尚未证实。计划显式分批启动、观察真实学习曲线、恢复最佳并实际三目标组合；正候选做磁盘重建。若单路线明显负向则记录并更换，不退回连续小头微调。
未做：其他原折、新井独立验证、交付替换。完整目标完成后可做新目录打包和稳定性验证，当前不启动。保留旧基线/正对照/交付，清理无亮点模型，维护PROJECTS.md目标/已做/计划/未做/后续、README、EXPLORATIONS和atlas。
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    spec=MODES[a.mode];out=ROOT/f"outputs/pure_dl_divergent_v45_{spec['target']}_{a.mode}_fold5"
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for n,txt in [('AGENTS.md','# v45工作约束\n用户要求发散，本批仅两个新私有架构，不加载旧网络。原分折/训练隔离/评分/Q50/80-18/max2保持，新上下文和POR九曲线输入变体须明示。维护PROJECTS.md目标/已做/计划/未做/后续；无自动队列/补折/旧交付覆盖。\n'),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/n).exists():(STUDY/n).write_text(txt,encoding='utf-8')
    out.mkdir()
    for n in ('AGENTS.md','README.md','PROJECTS.md'):(out/n).write_text((STUDY/n).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_divergent_models_v45.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=p.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v45 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','divergent_models_v45_20260929','- [v45 发散新架构](results/deep_learning/divergent_models_v45_20260929/README.md)：SW1024长井段全局/局部网络，POR九曲线×深度轴向网络联合重建；从随机网络完整训练。')
    append(ROOT.parent/'AGENTS.md','## v45用户要求发散','## v45用户要求发散\n用户否定持续冻结小头修补，明确要求发散。本批重新开放两个新私有网络的随机初始化完整训练：SW1024长段全局局部、POR512九曲线轴向联合重建。原标签/分折/评分/Q50/80-18/max2保持；新输入变体与上下文明示，无旧任务重启/补折/交付覆盖，维护divergent_models_v45_20260929。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
