"""Explicitly launch matched additive/multiplicative full-network candidates."""
from datetime import datetime
import argparse,json,subprocess,sys
from sw_patch_moments_v73 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='# v73 SW块内有序矩\n\n## 探索重点与对照\n原SW将16点局部96维表示平均为长程token，可能丢失块内有序形态。本批permuted/ordered保留均值，新增线性和二阶Legendre位置矩（按有效点中心化），拼接192维经零初始化Linear192→96无bias加回token。只区别新矩读取前是否固定打乱16个隐藏点及有效mask；局部路径、原token坐标不打乱。相同点集合、18432新参数、857913总参数、初始张量与先验函数；固定排列不额外随机采样。\n\n## 已做与训练边界\n已实现并真实启动。v56 additive第11轮骨干/注意力/FiLM/FF迁移，旧25维输出重置训练侧先验；不声称保留父函数。全网络仅SW标签训练，POR/PERM不动。初始化scope继承的random body是旧静态文案，以checkpoint哈希和initialization为准，实际是加载骨干；训练源/config保留。\n原9曲线206、64训练井/第五16井144665点、Q50、seed20261010、1024随机上下文中心128、原Gaussian BCE+NLL和256支持/800动作保持。整井输入库无标签/井ID，验证仅自身推理。新AdamW1e-4/wd.01/clip2、cos80*64至1e-5、max80/pat18/min10/b64/64steps；原子完整状态/RNG，仅显式恢复。最大2GPU，未开其他折/旧队列，不改交付。\n\n## 收益、待做与不继续\n开始时无成绩。直接训练核对实际梯度/相同初始参数与先验，结束实际三列组合评分；正候选磁盘重建后保留，负权重记清单删除。完整结果见FINAL.md（完成前不得声称收益）。有序矩相对打乱矩的配对收益和当前SW73.713976绝对收益分开；若无配对收益，停止矩阶数/patch大小扫描，不泛化全部顺序结构。\n记录单项与总墙钟、实际并发/显存，包含加载、预热、训练、评价及启动间隔，代码和收尾另计。第五严格>80.5且明确收益才考虑其他折，完整原五折OOF>82未完成，目标后独立确认和新目录交付。维护PROJECTS/README/EXPLORATIONS/图谱。\n\n本文件修复原启动说明经PowerShell管道传输的中文编码损坏；不改已训练的源码、配置、分数和时间。\n'
RULES='# v73 rules\nSame v56 additive backbone and reset SW output prior; within16-point centered moments ordered or fixed-permuted, same857913 parameters, all trained at1e-4.\nOriginal9curves/206 features, fixed original fold5, train-only fitting and Q50, max80/pat18/min10, maximum2GPU; no otherfolds, old queues or delivery replacement. Maintain PROJECTS: objective/completed/planned/outstanding/post-goal work.\n'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_sw_patch_moments_v73_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md',RULES),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_patch_moments_v73.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v73 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','sw_patch_moments_v73_20260929','- [v73 SW块内有序矩](results/deep_learning/sw_patch_moments_v73_20260929/README.md): Same v56 additive backbone and reset SW output prior; within16-point centered moments ordered or fixed-permuted, same857913 parameters, all trained at1e-4.')
    append(ROOT.parent/'AGENTS.md','## v73 SW块内有序矩','## v73 SW块内有序矩\nSame v56 additive backbone and reset SW output prior; within16-point centered moments ordered or fixed-permuted, same857913 parameters, all trained at1e-4.\nOriginal9curves/206 features, fixed original fold5, train-only fitting and Q50, max80/pat18/min10, maximum2GPU; no otherfolds, old queues or delivery replacement. Maintain PROJECTS: objective/completed/planned/outstanding/post-goal work.\n')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
