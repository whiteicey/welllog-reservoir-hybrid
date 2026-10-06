"""Explicitly launch the two private-POR initialization controls."""
from datetime import datetime
import argparse,json,subprocess,sys
from por_transfer_v53 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v53 POR整井骨干：随机与跨目标迁移

## 探索重点与固定对照
当前组合TOTAL80.342771761689，其中POR77.551936801885。v50的SW到PERM迁移有局部匹配收益，本批检验其能否推广到POR，不继续v52池化扫描。
预定义scratch/sw_transfer两个836921参数的POR私有网络：1024点长井段Transformer/TCN共享骨干、128点整井输入编码FiLM、POR默认门控与8个Student-t3分量混合密度。scratch全部骨干随机；sw_transfer加载v46自身原64训练井训练得到的SW骨干/LayerNorm/FiLM。两组均完整重置输出25行（门控/权重/位置/尺度），使用相同训练井POR标签拟合初始偏置、权重为零；不得保留SW输出或投票。
本轮变量是骨干初始化来源，所以两个完整初始参数哈希理应不同；输出头初始哈希与完整144665点初始预测应相同。所有参数均训练、拥有私有复制，不会更新源SW模型。用相同结构/预算验证SW训练学到的输入表征是否能迁移到POR，不把不同旧架构的比较单独归因初始化。
历史v34是训练侧辅助预测配合冻结小头，不是本次完整骨干权重迁移。本次不声称跨目标学习首次出现。来源SW曾按同一个开发折早停选模，非独立新井验证；继续使用开发折的局限如实保留。

## 数据与监督边界
源权重只在原64训练井标签上优化，验证井从未进入参数拟合。当前两个任务训练损失仅原64井POR标签，源训练历史使用SW标签是已声明的监督迁移，不是外部发布权重或测试标签。随机组不读取网络源权重；共同加载的数据缓存仅复用原训练侧标准化/网格/Q。
原9曲线/206特征、Q50、固定原第五144665点/原井集合/seed20261010、中心128及原256支持点/800动作加默认解码保持。输入记忆是当前井测井曲线，无井ID特征或标签。训练锚点仅64井，16验证井输入库仅该井推理。

## 训练、已做与未做
已经实现两项私有POR初始化/训练器，开始前无新成绩。两组统一AdamWlr3e-4/wd.01/clip2/cosine80*64至1e-5，80上限/patience18/min10/b64/64步；完整网络训练，按POR官方分选最佳。源检查点不是本次起点函数，不把初始先验成绩当迁移训练收益。
显式两任务、最多2GPU，无自动队列/恢复；原子last_state含当前/最佳模型、优化器、调度/RNG/早停。基于明显落后的实际曲线允许提前结束。训练中记录真实梯度/初始预测，结束恢复最佳与实际三通道组合；有价值候选额外磁盘重建。未改的旧公式不重复预检；CUDA注意力/插值反向warn_only，不宣称重新训练逐位确定。
待做实际训练、完成后的匹配差及相对当前POR绝对差、原组合实际评分、负权重清理、PROJECTS/EXPLORATIONS与atlas更新。两项可先后显式启动但最多2GPU，单项和总墙钟/显存分开，含加载/预热/输入库，代码与收尾另计。
若迁移无正收益，不扫描冻结层数或学习率；结论仅限当前SW来源与此结构。若有局部匹配收益可带入新独立方案，但完整原第五严格>80.5且明确收益后才考虑其余折。完整OOF>82未完成，不补其他折/重启旧队列/覆盖交付。目标达成后独立井确认、新目录交付尚未开始。
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_por_transfer_v53_POR_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for n,txt in [('AGENTS.md','# v53规则\n仅scratch/sw_transfer两个私有POR模型，原64训练井SW权重迁移与随机对照；输出头全部重置，全骨干训练。原数据/Q/评分/80-18/max2保持，无补折/交付覆盖。每批维护PROJECTS目标/已做/未做/后续。\n'),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/n).exists():(STUDY/n).write_text(txt,encoding='utf-8')
    out.mkdir()
    for n in ('AGENTS.md','README.md','PROJECTS.md'):(out/n).write_text((STUDY/n).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_por_transfer_v53.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=p.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v53 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','por_transfer_v53_20260929','- [v53 POR整井骨干迁移](results/deep_learning/por_transfer_v53_20260929/README.md)：随机与原训练井SW表征初始化对照，重置POR全输出头、全骨干训练；开始前无成绩。')
    append(ROOT.parent/'AGENTS.md','## v53 POR整井骨干迁移','## v53 POR整井骨干迁移\n本批scratch/sw_transfer两个私有POR模型，同结构与输出先验，比较随机骨干和同64训练井SW表征；输出全重置，无旧预测投票，全骨干仅POR损失。lr3e-4/80-18/max2，原数据/Q/评分不变，无其他折/旧队列/交付覆盖；维护por_transfer_v53_20260929。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
