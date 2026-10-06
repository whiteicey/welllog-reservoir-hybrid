"""Prepare a shared frozen-feature cache, then explicitly launch each readout."""
from datetime import datetime
import argparse,json,os,subprocess,sys,traceback
from sw_cached_readout_v59 import ROOT,STUDY,CACHE,MODES,prepare
from run_retrieval_smooth_v36 import jobs,append
from train_dual8222_heads80 import atomic_json

TEXT='''# v59 完整RQS密度读出：仿射与非线性

## 探索重点和固定对照
v58先适配头有同起点+0.120651SW收益，最佳在仅密度第2轮；SW73.708218略低当前73.713976。本轮沿有效的新密度读出研究非线性表达，不继续骨干扩容或冻结时长扫描。
从v58 headstart第2轮完整模型出发，body/norm/FiLM/default gate全部固定。保留34输出原密度线性核并可训练，再加入128→64→34零输出分支；affine中间为恒等，silu中间为SiLU。两组851237总参数、14852可训练密度参数、初始参数和完整函数相同；新增最后层权重/偏置为零。仿射组是线性映射的过参数化，非线性组改变函数表达能力，不能把参数量相同误解为函数族相同。输出仍是完整RQS密度，默认头固定，不做旧预测投票。

## 缓存、数据与训练边界
共享缓存只保存固定骨干的输入推理结果，训练64井与验证16井分包。按原推理计划每128点一个中心块，骨干读1024上下文，保存中心128点归一化隐藏向量及冻结默认logit。训练采用固定canonical窗口，替代历史随机锚点上下文；这是两组共同变化，历史绝对差不能全归因于激活。对照仍原9曲线206、Q50、原井集合和144665验证点、原目标变换/256支持点/800动作及评分。
仅训练包的SW标签进入损失。缓存构建核对训练raw集合、验证覆盖和井隔离，并用完整144665点初始预测核对源模型；未使用验证标签拟合特征。验证密度头按原B×1024点行布局运行，外围填零不影响逐点头，避免改变矩阵形状造成初始数值差。候选保留前再做原始输入完整骨干推理验证，缓存分数不能替代它。
新AdamWlr1e-4/wd.01/clip2，cosine80*64至1e-5，max80/pat18/min10/b64个缓存中心块/64steps，seed20261010。全部密度参数共同训练，按官方SW选择最佳；原子last_state含头/模型、优化器/调度/RNG/早停，恢复须显式。最多2GPU，无旧队列/其他折/交付覆盖。

## 已做、待做、收益与关闭条件
已实现两种读出和共享缓存构建，尚无候选训练成绩。直接运行缓存与训练，运行中检查真实数据覆盖和冻结状态，不加重复数学预检/独立审查；RQS公式与原评分未改。待完成两项训练、同参数对照/绝对分数、保留模型原始输入重建、负检查点及可再生缓存清理、PROJECTS/README/EXPLORATIONS/atlas。
若SiLU无最终配对收益，不扫描隐藏宽度/激活函数；只限该固定骨干及RQS参数化，不否定其他非线性头。若有局部收益但低当前模型，可保留机制候选。第五严格>80.5且明显收益才考虑其他折；完整原五折OOF>82、独立井确认和新目录交付未完成。
共享缓存只构建一次，准备耗时与单项训练、两任务总墙钟分开，完整流程从准备到训练结束的实际墙钟另列；如有编写间隔如实区分，不把两项并发耗时相加冒充总墙钟。显存/预热/读取均记录，代码与最终重建单列。缓存可由保留源模型和数据重建，不作为新测试数据或精度证据。
'''

def main():
    ap=argparse.ArgumentParser();group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--mode',choices=MODES);a=ap.parse_args()
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md','# v59规则\n仅affine/silu两个完整RQS密度头对照，同参数初始；v58骨干/默认头冻结，固定canonical训练缓存与验证分包。原数据/Q/评分/80-18/max2，无其他折/旧队列/交付覆盖；维护PROJECTS目标/已做/未做/后续，保留前原始输入重建。\n'),('README.md',TEXT),('PROJECTS.md',TEXT)]:
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
    out=ROOT/f'outputs/pure_dl_sw_cached_readout_v59_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_cached_readout_v59.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    atomic_json(out/'launch.json',launch)
    marker='## v59 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','sw_cached_readout_v59_20260929','- [v59 完整RQS非线性密度头](results/deep_learning/sw_cached_readout_v59_20260929/README.md)：v58固定骨干的共享canonical缓存，同参数仿射/SiLU读出；开始前无候选成绩。')
    append(ROOT.parent/'AGENTS.md','## v59 完整RQS非线性密度头','## v59 完整RQS非线性密度头\n本批affine/silu同851237总/14852密度参数及初始函数，v58骨干/默认头固定，共享canonical上下文缓存、只读训练包标签；原数据/Q/评分/80-18/max2，保留前原始输入重建，无其他折/旧队列/交付覆盖。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
