"""Explicitly launch matched additive/multiplicative full-network candidates."""
from datetime import datetime
import argparse,json,subprocess,sys
from sw_dropout_extend_v77 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='# v77 SW失活强度向上扩展\n\n## 探索重点与固定对照\nv76的20% DropKey已被采用：SW73.731635814592，当前三通道TOTAL80.360506992951，相对上一开发配方+0.006181。本批明确仅p30/p40（30%/40%），复用v76已完成20%作为控制，同时保留v74零失活证据；不重复控制训练，不启动无界搜索。\n\n## 相同初始化与训练条件\n839481参数，同v56 additive第11轮骨干/FiLM与重置SW训练侧先验。只复用v76实现，不加载v76训练后权重；改四层attention的随机key阈值。与20%同随机抽样数、seed和窗口顺序，mask共享头/查询，全丢回退valid，padding不变。验证无失活，一次完整前向，8Gaussian和原256支持/800动作不变。两新项与历史控制初始参数/预测需最终核对一致。\n原9曲线206、64训练井/第五16井144665点、Q50、seed20261010、1024随机上下文中心128、SW自身BCE+Gaussian NLL、全部网络训练；POR/PERM固定。AdamW1e-4/wd.01/clip2、cos80*64至1e-5、max80/pat18/min10/b64/64steps。完整原子状态/RNG、显式恢复。无其他数据、外部预训练或验证拟合。\n\n## 已做、收益与未做\n已实现，启动无新成绩。直接训练记录实际资源/梯度/曲线，结束核对历史控制同配置、最佳全点恢复及真实三列评分；正候选原始输入磁盘重建。报告相对20%、零失活、当前组合的实际差与默认/数值子群，不能将开发折收益当独立井证明。\n若两项都不超过20%，停止这轮向上强度扩展及密集概率细扫；不泛化所有正则化。原20%与10%正候选保留，负权重按清单删除。维护PROJECTS/README/EXPLORATIONS/图谱。最多2GPU，按实测资源显式启动；当前电源Online，仍记录实际单项/总墙钟及环境，不把上轮跨夜停顿或电池供电耗时当可比速度。加载预热训练评估与启动间隔计入，代码/收尾另计。\n无旧队列/其他折/交付覆盖。第五严格>80.5且明确收益才考虑其余折；完整原五折OOF>82未完成，目标后独立井确认和新目录交付。\n'
RULES='# v77规则\n仅p30/p40两项，同原训练起点/配置，只改30%/40% DropKey，复用v76的20%控制，不加载其训练后权重。原数据/Q/评分/80-18/max2，无其他折/旧队列/交付覆盖；维护PROJECTS目标/已做/未做/后续。\n'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_sw_dropout_extend_v77_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md',RULES),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_dropout_extend_v77.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v77 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','sw_dropout_extend_v77_20260930','- [v77 SW失活强度扩展](results/deep_learning/sw_dropout_extend_v77_20260930/README.md)：p30/p40同原训练起点，复用20%控制，不加载其训练后权重；初始无成绩。')
    append(ROOT.parent/'AGENTS.md','## v77 SW失活强度扩展','## v77 SW失活强度扩展\n'+RULES)
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
