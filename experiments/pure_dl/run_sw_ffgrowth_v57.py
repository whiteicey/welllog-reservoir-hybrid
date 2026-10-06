"""Explicit base versus parallel128 growth after the adopted v56 gain."""
from datetime import datetime
import argparse,json,subprocess,sys
from sw_ffgrowth_v57 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v57 有收益加法配置的有限扩容

## 探索重点与固定对照
当前v56加法SW73.713975628494，组合TOTAL80.354325927816；比v46微增，不代表独立新井收益。本轮沿已采用配置做一次容量对照，不继续失败的乘法结构。
base完整保留v56四个256维加法FF核心；grow128在每个核心旁并联一个128维加法FF，输出相加。额外分支96→256拆成128维a,b，(SiLU(a)+b)/sqrt(2)经过自身LayerNorm128和Linear128→96；额外输出权重/偏置全零，前层共同固定seed20261157随机初始化。原核心、全输出头、TCN/注意力/FiLM都从同v56第11轮加载，均可训练。
base839481参数，grow128989369参数，增加149888。是保留原256归一化并新增独立128归一化的并行容量，不冒称直接把单一LayerNorm宽度改成384。完整参数量/哈希不同是变量，公共核心参数哈希相同，完整144665点初始预测必须精确一致；原输出不重置。起点已计入最佳选择，不能将初始成绩当训练收益。
两组所有参数统一lr3e-5和同样clip2，新增分支没有额外高学习率，旧权重不冻结。观察扩容相对本轮base与相对当前SW/组合的差，不能把加法组历史微增推断成容量必然有益。

## 数据与训练边界
原9曲线/206、Q50、原64训练井/第五16验证井144665点/seed20261010、1024窗口/中心128、原8Gaussian+BCE批次点均损失、256支持点/800动作+默认评分保持。损失只用SW训练标签，输入井库无标签/井ID特征，验证输入仅自身推理，无外部权重或旧输出投票。
两组新AdamWlr3e-5/wd.01/clip2，cosine80*64至1e-5，max80/pat18/min10/b64/64steps。全网络训练，各自按官方SW选模/早停/恢复最佳；原子last_state含模型、优化器/调度器/RNG/早停，仅显式恢复。最多2GPU，无旧队列/其他折/交付覆盖；CUDA反向warn_only不声称重训逐位确定。

## 已做、待做、不继续与目标
已实现零输出并行增长及公共核心哈希，开始前无成绩。直接真实训练，记录新增FF实际梯度与普通骨干/头/FiLM梯度；结束核验初始公共核心/完整预测、源哈希、最佳恢复及三列实际评分。保留候选需磁盘重建，若收尾已核验同一权重则复用证据；不额外预检/独立审查。
待完成本轮两组、实际收益及耗时、负权重清理、PROJECTS/README/EXPLORATIONS/atlas维护。若扩容无同起点收益，停止这一路128/256分支宽度和层数扫描；结论仅限该初始化/共同LR/并行容量，不否定所有大模型。实际明显落后可提前结束。
单项与双并发总墙钟/显存分开，含加载/输入库/预热/启动间隔，代码和最终磁盘重建另计。第五严格>80.5且明确收益才考虑其他折，完整原五折OOF>82仍未完成；独立井确认、新目录交付尚未开始。
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_sw_ffgrowth_v57_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md','# v57规则\n仅base/grow128两项，同v56加法SW完整函数；新增四个零输出并行128FF，公共核心相同，全模型共同lr3e-5训练。原数据/Q/评分/80-18/max2，无其他折/旧队列/交付覆盖。维护PROJECTS目标/已做/未做/后续。\n'),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_ffgrowth_v57.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v57 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','sw_ffgrowth_v57_20260929','- [v57 SW加法层容量](results/deep_learning/sw_ffgrowth_v57_20260929/README.md)：同v56完整函数，base与四个零输出并行128FF扩容对照，全模型相同LR；开始前无成绩。')
    append(ROOT.parent/'AGENTS.md','## v57 SW加法层容量','## v57 SW加法层容量\n本批base/grow128两项，同v56加法SW函数，核心256不改、增加独立归一化128并行分支，输出零初始；全模型共同lr3e-5训练。原数据/Q/评分/80-18/max2，无旧任务/其他折/交付覆盖，维护sw_ffgrowth_v57_20260929。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
