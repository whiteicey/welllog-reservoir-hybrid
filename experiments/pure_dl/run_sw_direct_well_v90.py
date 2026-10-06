"""Explicit two-arm launches, staged by measured GPU headroom; no queue."""
from datetime import datetime
import argparse,json,subprocess,sys
from sw_direct_well_v90 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append
RULES='\n# v90目标与约束\n仅SW Gaussian参考/本井直接读出两项，同841906参数；原v56骨干FiLM和训练先验，全SW训练。原FiLM两组均本井，新零输出96→25头只差reference/well来源，两种code都计算。原9曲线206/输入bank/64-16/seed/Q50/评分/效用1/DropKey.4/80-18/min10/bs64/64步，normal/max2GPU，依据实测显存分批。原子完整last和训练后最佳，恢复显式。维护PROJECTS/独立记录/图谱；负项先落盘清理计划再删。无其他折/旧队列/交付覆盖，第五严格>80.5且明确收益、原OOF>82仍未完成。\n'
TEXT="""# v90 SW全井编码直接读出

探索重点：现有全井FiLM只影响隐藏表示，新增同一96维全井code直达Gaussian门控/权重/位置/尺度25输出，检验是否帮助SW精细数值。reference/well同841906参数、新增2425，同v56训练折内权重及SW训练先验；零输出使初始函数一致。共享原编码器，旧FiLM均用本井。两code均计算；新头仅输入来源不同。本井bank仅测井输入，reference仅64训练井位置均值，无验证拟合/ID/标签银行。

对照预定义：well-reference完整同容量训练后最佳SW/TOTAL差，与原Gaussian v82 alpha1 SW73.793310952488/固定POR-PERM TOTAL80.382093291214绝对差分开。使用原始v56权重/输出先验，不加载v82训练后模型。原9曲线206+深度/原第五144665/seed20261010/Q50/官方256支持800动作及默认99.9不变，Gaussian NLL+全人口效用1/3，温度.05，DropKey.4，全部SW与新头更新。新AdamW1e-4/wd.01/clip2/cos80*64到1e-5/max80pat18min10batch64steps64。

已探索与验证：建立时未训练，尚无新成绩或收益。直接训练；正常初始化核对原基本state/旧头hash和初始动作hash，实际第二batch记录骨干/FiLM/旧头/新头/效用梯度。训练后按自身SW选最佳，恢复所有144665；正绝对或完整匹配收益必要磁盘重载。POR/PERM固定，真实组合评价；不把先验改善当超越当前。

收益与不继续：等本轮结束后报告绝对和匹配收益、默认/数值/各井及实际停止。若两项均不超旧Gaussian且well-reference无正收益，关闭本零输出线性全井直接支路/共享编码/当前来源与预算；不临时扫头宽/LR/强度，不泛化全部全井方法。只有完整公平局部收益可作为下一次独立假设，不能冒称SOTA或五折收益。

已做：两个有限方案。计划：显式分批训练、实际组合/必要重载/清理/图谱。未做：真实收益、其他折、独立泛化。最多2GPU进程，根据实测显存启动第二项，无自动队列。记录单项加载预热/训练验证保存、同工作量总墙钟、重叠和allocated/reserved，准备与收尾另计。恢复需显式；原子last含current/best/优化器/scheduler/RNG/停止及训练后最佳state/pred。目标后：第五严格>80.5且明确收益后考虑其余四折，完整原OOF>82后独立井验证与新目录交付，旧模型/交付保留。
"""
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum two GPU jobs')
    out=ROOT/f'outputs/pure_dl_sw_direct_well_v90_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True);out.mkdir()
    for name,txt in [('AGENTS.md',RULES),('README.md',TEXT),('PROJECTS.md',TEXT),('PLAN.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
        if name!='PLAN.md':(out/name).write_text(txt,encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_direct_well_v90.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v90 SW全井直接读出显式启动 '+a.mode
    for p in (ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md',STUDY/'PROJECTS.md'):append(p,marker,marker+'\n'+json.dumps(launch))
    for p in (ROOT/'AGENTS.md',ROOT.parent/'AGENTS.md'):append(p,'## v90当前有限范围','## v90当前有限范围\n'+RULES)
    append(ROOT/'EXPLORATIONS.md','sw_direct_well_v90_20260930','- [v90 SW全井直接读出](results/deep_learning/sw_direct_well_v90_20260930/README.md)：参考/本井同容量零初始直接支路，原FiLM不变，匹配与绝对收益分别报告，未有新成绩。')
    print(json.dumps(launch),flush=True)
if __name__=='__main__':main()
