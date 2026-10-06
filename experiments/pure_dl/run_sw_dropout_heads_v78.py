"""Explicitly launch matched additive/multiplicative full-network candidates."""
from datetime import datetime
import argparse,json,subprocess,sys
from sw_dropout_heads_v78 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='# v78 SW跨注意力头失活掩码对照\n\n## 探索重点与相同条件\n当前v77 p40独立通道组合TOTAL80.363582620846，SW73.740423322864。40%失活在16井8胜8负，较20%数值子群+0.286177但默认-0.111979；绝对收益很小。本批从完整p40第25轮函数出发，比较shared/per_head两项：4层各四头共用随机key掩码，或每头独立掩码。假设独立头保留更丰富的联合上下文覆盖，减少单次全头遗漏；可能增加训练噪声，收益未知。\n839481参数、初始完整状态与预测相同；仅改变同BHt随机张量是否广播头0，每层两组消耗相同BHt随机数。每头全丢回退原valid、padding保持、掩码对查询共享。验证无失活、一次完整前向，未新增融合或改变解码。不是旧1e-4轨迹续跑：两组均新AdamW3e-5/wd.01/clip2，cos80*64至1e-5，优化器重置。旧v77不能直接作为新随机序列的同训练控制，所以本批shared也重训。\n原9曲线206、64训练井/第五16井144665点/Q50/seed20261010、1024随机上下文中心128、SW自身BCE+8Gaussian NLL、256支持/800动作、80/18/min10/b64/64steps不改。全部SW参数更新，POR/PERM固定；只用训练标签，输入侧井上下文不含井ID或标签。完整原子状态/RNG，仅显式恢复。\n\n## 已探索、收益与不继续\n已实现；启动时尚无新成绩。直接训练，初始完整预测检查属于实际训练评估，不单独预检或启动审查。记录训练真实梯度、最优恢复、最高训练后得分和含起点回退的分数，结束实际组合验证。分别报告per_head-shared、各自相对完整起点、默认/数值子群及井差；开发折反复选择不当独立泛化证明。\nper_head若无最终配对收益，关闭当前头间掩码拆分的概率扫描；若shared仍有绝对收益可独立保留，不把共同低学习率微调收益归给独立掩码。只保存有绝对/明确配对收益的候选；负checkpoint先写清单后删除，来源与旧交付保留。最多2GPU，实测资源后显式启动第二项，无自动队列；记录单项及同工作量总墙钟、并发/显存，加载预热训练验证保存与启动间隔计入，代码/收尾分开。\n未启动旧任务或其他折。固定第五严格>80.5且明确收益才考虑补其余折；完整原五折OOF>82仍未达成。维护PROJECTS/README/EXPLORATIONS/图谱；目标完成后独立井确认与新目录交付。\n'
RULES='# v78规则\n仅shared/per_head两项，完整v77 p40同起点、同839481参数与BHt随机消耗，只改变头间掩码共享方式。新AdamW3e-5/80-18；原9曲线/206/Q50/分折/评分/max2保持。无旧队列/其他折/交付覆盖。持续维护PROJECTS：目标、已做、待做、未做及目标后工作。\n'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_sw_dropout_heads_v78_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md',RULES),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_dropout_heads_v78.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v78 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','sw_dropout_heads_v78_20260930','- [v78 SW头间失活掩码](results/deep_learning/sw_dropout_heads_v78_20260930/README.md)：shared/per_head完整v77同起点，新3e-5微调，只差掩码头间共享；初始无成绩。')
    append(ROOT.parent/'AGENTS.md','## v78 SW头间失活掩码','## v78 SW头间失活掩码\n'+RULES)
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
