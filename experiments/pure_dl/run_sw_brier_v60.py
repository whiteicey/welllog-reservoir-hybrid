"""Prepare a shared frozen-feature cache, then explicitly launch each readout."""
from datetime import datetime
import argparse,json,os,subprocess,sys,traceback
from sw_brier_v60 import ROOT,STUDY,CACHE,MODES,prepare
from run_retrieval_smooth_v36 import jobs,append
from train_dual8222_heads80 import atomic_json

TEXT='''# v60 SW概率平方损失对照

## 探索重点
从已保留v58 headstart第2轮SW73.708218开始；当前组合SW73.713976/TOTAL80.354326。v59读出扩展无最终收益，不再扫描。此次无新增模块，冻结body/norm/FiLM，只训练原gate和RQS密度，共4515参数（总840771）。两组参数、起点和样本相同，只差辅助项系数。
likelihood使用原Q加权BCE+连续NLL；brier额外加入0.5*(gate概率平方误差+16*非default条件256格Brier)。Brier为sum(p²)-2*p_true+1，与容差覆盖、分位pinball不同；类别按原归一化density_edges由训练SW标签确定，开放外侧尾部覆盖所有值。因子16固定，不扫描。它是离散概率训练目标，不声称等同官方效用或连续CRPS；保留原连续NLL。

## 数据与执行约束
原9曲线/206输入、64训练井/16验证井/原seed20261010、Q50及原解码不变。缓存冻结骨干1024上下文的中心128点，训练与验证分包；canonical固定训练上下文是两组共同变化，不将相对旧随机锚点绝对差全部归因于Brier。只读取训练包SW标签作损失；验证只选官方SW最佳。原完整起点预测及人口覆盖在缓存运行时检查，保留候选需原始输入144665点重新推理，不增加数学预检或独立审查。
新AdamW1e-4/wd.01/clip2，cosine80*64至1e-5，max80/pat18/min10/b64/64steps。门控和密度同训练；原子保存best及last完整optimizer/scheduler/RNG/早停状态，仅显式续跑。最大2GPU，无其他折/旧队列/交付覆盖。

## 已做、收益、未做与关闭条件
已实现匹配目标，尚未训练，收益尚未证实。待共享缓存、两项训练、配对差与当前绝对分、原始输入保留模型验证、清理负权重和缓存，更新PROJECTS/EXPLORATIONS/atlas。
若Brier无最终配对收益，关闭此256格平方损失系数扫描，不泛化所有概率评分无效。第0轮回退不算学习收益。开发折重复使用，不能写成独立新井收益；完整五折OOF>82、第五严格>80.5、独立确认与新目录交付未完成。
共享缓存准备与单项训练、双并发总墙钟分别记录；含加载/预热/评估，代码间隔如实单列，不把并发耗时相加当墙钟。目标达成后可独立确认及新目录交付。
'''

def main():
    ap=argparse.ArgumentParser();group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--mode',choices=MODES);a=ap.parse_args()
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md','# v60规则\n仅likelihood/brier，v58骨干冻结且门控/密度头可训练；原数据/Q/评分/80-18/max2，固定canonical训练缓存与验证分包，无其他折/旧队列/交付覆盖；维护PROJECTS目标/已做/未做/后续，保留前原始输入重建。\n'),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    if a.prepare:
        if jobs():raise RuntimeError('Prepare shared cache before starting this pair')
        event=dict(status='running',pid=os.getpid(),started_at=datetime.now().astimezone().isoformat())
        atomic_json(STUDY/'cache_status.json',event)
        try:prepare()
        except BaseException as exc:
            atomic_json(STUDY/'cache_status.json',event|dict(status='failed',error=repr(exc),traceback=traceback.format_exc()));raise
        atomic_json(STUDY/'cache_status.json',event|dict(status='complete',finished_at=datetime.now().astimezone().isoformat()))
        append(STUDY/'PROJECTS.md','## 共享缓存完成','## 共享缓存完成\n实际范围、时间、源函数核验见cache_manifest.json；两个候选尚未训练。')
        return
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    if not CACHE.exists() or not (STUDY/'cache_manifest.json').exists():raise RuntimeError('Prepare cache explicitly first')
    out=ROOT/f'outputs/pure_dl_sw_brier_v60_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_brier_v60.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    atomic_json(out/'launch.json',launch)
    marker='## v60 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','sw_brier_v60_20260929','- [v60 SW概率平方损失](results/deep_learning/sw_brier_v60_20260929/README.md)：v58固定骨干的共享canonical缓存，同原头似然/Brier损失；开始前无候选成绩。')
    append(ROOT.parent/'AGENTS.md','## v60 SW概率平方损失','## v60 SW概率平方损失\n本批likelihood/brier同840771总/4515门控密度参数及初始函数，v58骨干固定，共享canonical上下文缓存、只读训练包标签；原数据/Q/评分/80-18/max2，保留前原始输入重建，无其他折/旧队列/交付覆盖。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
