"""Explicitly launch matched additive/multiplicative full-network candidates."""
from datetime import datetime
import argparse,json,subprocess,sys
from sw_dropout_rate_v76 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='# v76 SW随机键失活强度\n\n## 探索重点与对照\nv74单前向10% DropKey相对0%有SW+0.120507，保留为正机制；v75两前向监督范围没有收益，不继续该分支。本批只新训p05/p20，即5%和20%失活，复用v74已完成10%和0%紧凑证据作为历史控制，不重跑同配置。假设较低强度可减少数值预测下降，较高强度可增加泛化；不预设谁更好。\n\n## 相同条件与训练范围\n两项及v74均839481参数、同v56 additive第11轮骨干/FiLM、SW输出重置同64训练井先验，不加载v74训练后模型；仅随机key阈值不同。每批一次前向，同4层、相同随机抽样数与窗口独立RNG。mask对头/查询共享，全丢恢复valid，padding保持。验证一次完整上下文，无失活或融合。全部参数只由本模型SW BCE+Gaussian NLL更新，POR/PERM不动。\n原9曲线206、64训练井/第五16井144665点/Q50/seed20261010、1024随机上下文中心128、原256支持/800动作保持。新AdamW1e-4/wd.01/clip2、cos80*64至1e-5、max80/pat18/min10/b64/64steps。原子完整状态/RNG，显式恢复。历史控制数据/参数/初始化/预算需最终核对，CUDA非确定性warn_only不冒称重训逐位可复现。\n\n## 已做、待做与收益\n已实现两项，启动时没有新成绩。直接训练看实际梯度/资源与曲线，结束核对初始状态/预测哈希和历史控制同条件、最佳恢复、真实三列评分。分别报告相对10%、0%和当前SW73.713976的差值及默认/数值子群。仅超当前模型才替换组合；相对10%有新增配对收益者可保留，正模型先磁盘重建。未完成不得当收益，开发折不当独立新井验证。\n若两点都不优于10%，停止当前失活比例细扫，保留原v74正候选。负权重清理先写清单，保留源/配置/紧凑结果，更新PROJECTS/README/EXPLORATIONS/图谱。最多2GPU；这是单前向，与v75的两图吞吐不等同，先看实测资源再显式启动第二项，记录实际单项/批次总墙钟和显存。加载/预热/训练/评价/启动间隔计入，代码和收尾分开。\n不启动其他折/旧队列，不覆盖旧交付。第五严格>80.5且明确收益才考虑其他折，完整原五折OOF>82未完成；目标后独立井确认、新目录交付。\n'
RULES='# v76规则\n仅p05/p20两项，单前向5%/20% DropKey，复用历史v74的10%及0%控制，不重复训练。原数据/Q/评分/80-18/max2，不开其他折/旧队列或改交付。维护PROJECTS目标/已做/未做/后续。\n'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_sw_dropout_rate_v76_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md',RULES),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_dropout_rate_v76.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v76 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','sw_dropout_rate_v76_20260930','- [v76 SW失活强度](results/deep_learning/sw_dropout_rate_v76_20260930/README.md)：p05/p20同v74单前向配置，只改失活概率，复用10%/0%历史证据；初始无成绩。')
    append(ROOT.parent/'AGENTS.md','## v76 SW失活强度','## v76 SW失活强度\n'+RULES)
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
