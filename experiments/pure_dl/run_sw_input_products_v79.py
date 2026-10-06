"""Explicitly launch matched additive/multiplicative full-network candidates."""
from datetime import datetime
import argparse,json,subprocess,sys
from sw_input_products_v79 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='# v79 SW显式曲线输入交互对照\n\n## 探索重点与相同条件\nv78低学习率全网络微调/独立掩码没有收益，两项回到原函数；不继续该分支。本批冻结当前v77 p40 SW完整模型（SW73.740423322864，组合TOTAL80.363582620846），只在207→96 stem的GELU之前加4320参数的45→96无偏置零初始化适配器。原9曲线206维、相对深度和所有旧层不删改。\n两项marginal/cross同参数与完整初始预测：提取原206输入里GR/PE/SP/CAL/AC/DEN/CNL/RXO/RT对应的已按原训练井处理的数值，z=tanh(x/3)；marginal用九曲线各自2~6次幂共45维，cross用含对角的上三角45个z_i*z_j。仅新特征基不同，不新增统计拟合或标签特征；不是v32局部相关统计输出头，也不是v56隐藏FF乘法。假设显式幅值交互能用小参数适应SW；控制检验相同新增容量的单曲线非线性。\ndelta=0.25*tanh(Wfeature/0.25)，每个stem激活修正有界；这不是原测井值或SW预测的物理界。旧参数/原FiLM/旧输出全部冻结且eval模式，训练也不再丢旧attention key；只对新输入适配器反传SW自身BCE+8Gaussian NLL，固定POR/PERM。两组原模型、适配器初始张量和窗口RNG一致。新AdamW3e-4/wd.01/clip2、cos80*64至1e-5、max80/pat18/min10/b64/64steps，优化器重置。\n原64训练井/第五16井144665点、Q50、seed20261010、1024上下文中心128、256支持/800动作解码不变。原子完整状态/RNG，只显式恢复；无外部数据/未知标签/其他折。\n\n## 已探索、收益与不继续\n已实现，启动没有新成绩；直接训练。实际训练初始评价须恢复全部原预测，不单独数值预检或代码审查。记录新层梯度、旧层无梯度、原权重逐位冻结、最佳恢复和实际三列评分；正候选磁盘重建。分别报告cross-marginal及各自相对父模型的差、含起点/训练后最佳成绩，开发折反复选择不当独立泛化证据。\ncross若无固定消融收益不继续该乘积基的阶数/组合细扫；两项若都无绝对收益也不继续此stem适配器幅度/容量细扫，不泛化所有特征工程。只保留有效候选，负或重复checkpoint先写清单后删除；完整来源/历史正候选/旧交付保留。最多2GPU，根据实际资源启动第二项，无自动队列；单项和同等工作量总墙钟、并发/显存均记录，加载/预热/训练/评价/保存/启动间隔计入，代码与收尾分开。\n维护PROJECTS/README/EXPLORATIONS/图谱。严格原第五>80.5且明确收益才考虑其余折，完整原五折OOF>82仍未达成；目标后独立井确认和新目录交付。\n'
RULES='# v79规则\n仅marginal/cross两项，同完整v77 SW起点，旧模型参数冻结且eval，只训练4320参数零初始化有界输入适配器。45维单曲线多项式或跨曲线乘积基，无新增统计拟合/数据/标签特征。原9曲线206/分折/Q50/评分/80-18/max2保持，无旧队列/其他折/交付覆盖；持续维护PROJECTS目标、已做、待做、未做及目标后工作。\n'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_sw_input_products_v79_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md',RULES),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_input_products_v79.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v79 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','sw_input_products_v79_20260930','- [v79 SW输入乘积特征](results/deep_learning/sw_input_products_v79_20260930/README.md)：marginal/cross冻结完整v77，4320参数同初始化，只差45维特征基；初始无成绩。')
    append(ROOT.parent/'AGENTS.md','## v79 SW输入乘积特征','## v79 SW输入乘积特征\n'+RULES)
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
