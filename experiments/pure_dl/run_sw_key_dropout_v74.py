"""Explicitly launch matched additive/multiplicative full-network candidates."""
from datetime import datetime
import argparse,json,subprocess,sys
from sw_key_dropout_v74 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='# v74 SW长程键随机失活\n\n## 探索重点与证据\nv73两项全网络重训均第4轮见顶，此后训练损失下降但验证SW降低，提示检验长程上下文过拟合。先检查旧Q50权重：原578891训练点仅1755降权；SW默认/数值点平均Q为0.999930/0.991344。历史v37等权小头也未超起点，本轮不重复该质量权重路线；这不证明所有Q机制无效。\n\n## 预定义对照\n本批control/dropkey各839481参数，无新增模块。加载v56 additive第11轮全部骨干/FiLM，旧25维输出重置64训练井SW先验，全部网络训练。原4个长程attention层在训练时各对每样本64个key独立采样10%丢弃，mask对头/查询共享，softmax在剩余key重新归一化；若全被丢弃恢复原有效mask。原padding/query mask保持。control也消耗相同随机抽样但不用drop mask，训练窗口有独立RNG且完全一致。两组初始张量和先验预测相同，不保留父完整函数。验证无任何dropkey，旧Gaussian解码不变。\n\n## 数据和预算\n原9曲线206、训练64井/第五16井144665点/Q50、seed20261010、1024随机上下文中心128、原SW BCE+Gaussian NLL、原256支持/800动作保持。只用SW训练标签，POR/PERM固定，预处理与井输入库无验证拟合，无外部权重数据。AdamW1e-4/wd.01/clip2、cos80*64至1e-5、max80/pat18/min10/b64/64steps；原子完整状态/RNG，显式恢复。最大2GPU、明确启动，不开其他折/旧队列或改交付。\n\n## 已做、收益与未做\n已实现，启动时没有新成绩。直接训练记录有效梯度、初始参数/预测和来源哈希，结束最佳恢复及真实三列评分，正候选原始输入磁盘重建，负权重记清单删除。比较最终配对差与原SW73.713976绝对差；训练中暂时领先不是最终收益。若dropkey无收益，不扫当前比例或施加层数，结论不泛化全部正则化。最终见FINAL.md。\n记录单项与同工作量总墙钟、并发、显存；加载/预热/训练/评价和启动间隔计入，代码和收尾另计。维护PROJECTS/README/EXPLORATIONS/图谱。第五严格>80.5且明确收益才考虑其他折，完整原五折OOF>82未完成；目标后独立井验证和新目录交付。\n'
RULES='# v74规则\n仅control/dropkey两项；同v56骨干+重置输出先验，无新增参数，训练注意力key丢弃0/10%，验证完整。原数据/Q/评分/80-18/max2，不开其他折/旧队列或改交付。维护PROJECTS目标/已做/未做/后续。\n'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_sw_key_dropout_v74_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md',RULES),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_key_dropout_v74.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v74 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','sw_key_dropout_v74_20260929','- [v74 SW长程键随机失活](results/deep_learning/sw_key_dropout_v74_20260929/README.md)：control/dropkey同模型，训练key丢弃0/10%，验证无丢弃，启动无成绩。')
    append(ROOT.parent/'AGENTS.md','## v74 SW长程键随机失活','## v74 SW长程键随机失活\n'+RULES)
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
