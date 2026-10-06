"""Explicit base versus parallel128 growth after the adopted v56 gain."""
from datetime import datetime
import argparse,json,subprocess,sys
from sw_upsample_v61 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v61 SW神经深度还原

## 探索重点与固定对照
当前v56加法SW73.713975628494，组合TOTAL80.354325927816。连续冻结RQS读出与Brier辅助未收益，切换完整骨干的信息还原结构，不继续其宽度/损失系数扫描。
原模型把每16点一个的96维token固定线性插值到1024点。本轮增加3104参数的rank16 query/key模块，对每点对应的左右两个粗token重新分配插值权重。coarse_query读原线性插值粗特征；local_query读插值前的精细TCN特征，其他相同。两者842585参数，完整参数相同，query权重与偏置全零，key同seed20261161；初始函数应与v56第11轮逐点一致。新骨干不是多模型预测投票。
位置计算与原align_corners=False相同；对原两点权重的log加入q·k/4。实现返回原线性结果+两种softmax权重差乘token，零query时校正严格零；1e-12下限只用于log，边缘同一token索引不会引入外井信息。无效token依原模型置零，最终保留原valid掩码。query/key输入无参数LayerNorm；查询来源是唯一对照变量，不改变token数量或网络主宽度。

## 数据与训练边界
原9曲线/206输入、64训练井/第五16验证井144665点/seed20261010、Q50、1024随机锚点窗口/中心128、原8Gaussian+BCE SW损失、256支持点/800动作+默认评分保持。损失只用SW训练标签，整井输入库无标签/ID特征，验证井输入仅自身推理。POR/PERM冻结其选择，不共用目标损失。
全网络可训练，两组新AdamWlr3e-5/wd.01/clip2，cosine80*64至1e-5，max80/pat18/min10/b64/64steps。初始模型纳入最佳选择，回退不算训练收益。原子last_state包含当前/最佳模型、优化器/调度/RNG/早停，只显式续跑。最多2GPU，无旧队列/其他折/交付覆盖。

## 已做、收益与待做
已实现同起点查询来源对照，启动前无训练成绩、收益尚未证实。直接训练，记录新query/key及普通骨干/头/FiLM实际梯度；原始初始预测、源哈希、最佳恢复与三列实际评分作为最终证据，不增加独立审查/数学预检。保留正候选需磁盘原始输入重建。
待完成两项训练、同参数差及相对当前绝对差、负权重清理、PROJECTS/EXPLORATIONS/atlas。只保留有匹配收益或超当前目标的模型；完整开发折差不等于独立新井收益。

## 不继续条件与目标
若local_query无最终匹配收益，不扫描本插值rank/邻居数/查询温度；只限当前两点重分配结构，不否定全部神经上采样。若明显落后，可按证据提前结束，仍保存紧凑结果和负结论。
分别记录单项耗时、同等两任务总墙钟、并发峰值与显存；加载/输入库/预热/评估计入，代码和最终磁盘重建另计。第五严格>80.5且明确收益后才考虑其他折；完整原五折OOF>82、独立井确认和新目录交付仍未完成，目标达成后可继续独立确认。
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_sw_upsample_v61_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md','# v61规则\n仅coarse_query/local_query两项，同v56完整函数；新rank16两点插值模块共3104参数，查询来源不同，全网络共同lr3e-5训练。原数据/Q/评分/80-18/max2，无其他折/旧队列/交付覆盖。维护PROJECTS目标/已做/未做/后续。\n'),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_upsample_v61.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v61 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','sw_upsample_v61_20260929','- [v61 SW神经深度还原](results/deep_learning/sw_upsample_v61_20260929/README.md)：同v56完整函数，两点插值的粗/局部查询对照，全模型相同LR；开始前无成绩。')
    append(ROOT.parent/'AGENTS.md','## v61 SW神经深度还原','## v61 SW神经深度还原\n本批coarse_query/local_query两项，同v56加法SW函数，新增rank16两点插值query/key，query零初始；全模型共同lr3e-5训练。原数据/Q/评分/80-18/max2，无旧任务/其他折/交付覆盖，维护sw_upsample_v61_20260929。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
