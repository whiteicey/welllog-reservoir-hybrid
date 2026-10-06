"""Explicit base versus parallel128 growth after the adopted v56 gain."""
from datetime import datetime
import argparse,json,subprocess,sys
from perm_observation_v72 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='# v72 PERM有限观测分辨率\n\n## 探索重点与对照\n延续v67 RQS相对linear的局部收益+0.221226，其PERM88.827250仍低于当前89.396724。检验点似然是否追逐窄密度尖峰。本批point/interval均从v67第24轮完整838211参数和函数开始，全部参数训练；只改平均密度观测半宽0或0.05训练归一化单位。两组同五点Gauss-Legendre与logsumexp，默认BCE不变。是固定建模假设与近似积分，不宣称真实测量精度或官方效用。\n\n## 已做与验证\n两项已完成，均第18轮早停、最佳第0轮。训练后最高PERM分别88.778070945782、88.771566640436，区间组配对差-0.006504305345；含起点配对差0，没有采用。当前TOTAL80.354325927816保持。原9曲线206、原训练64井/第五16井144665点、Q50、1024中心128、原RQS解码保持；仅PERM训练标签，新AdamW3e-5/wd.01/clip2/cos80*64至1e-5/max80/pat18/min10/b64/64steps。初始状态和全点预测同一、四组真实梯度/最佳恢复/源码哈希/三列评分通过，详情见FINAL.md。\n\n## 收益与不继续\n没有本轮收益；停止当前观测宽度/积分节点扫描，不泛化所有标签噪声建模。已删除本轮4个无收益/重复权重共37411436字节，保留v67正控制、原模型与交付。实际串行峰值1，单项108.972/117.610秒、总墙钟260.439秒含启动间隔；加载预热训练评价计入，代码与收尾另计。\n\n## 未完成与目标后\n无其他折或旧队列。第五严格>80.5与完整原五折OOF>82仍未达成；接下来换机制，目标后独立井验证和新目录交付。PROJECTS/README/EXPLORATIONS/图谱已维护。\n本文件修复原启动说明经PowerShell管道传输的中文编码损坏；不改已训练的源码、配置、分数和时间。\n'
RULES='# v72 rules\nSame v67 RQS epoch24 predictor; only observation halfwidth 0/.05 differs; own PERM loss, unchanged inference, all parameters trainable at 3e-5.\nOriginal9curves/206 features, fixed original fold5, train-only fitting and Q50, max80/pat18/min10, maximum2GPU; no otherfolds, old queues or delivery replacement. Maintain PROJECTS: objective/completed/planned/outstanding/post-goal work.\n'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_perm_observation_v72_PERM_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md',RULES),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_perm_observation_v72.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v72 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','perm_observation_v72_20260929','- [v72 PERM有限观测分辨率](results/deep_learning/perm_observation_v72_20260929/README.md): Same v67 RQS epoch24 predictor; only observation halfwidth 0/.05 differs; own PERM loss, unchanged inference, all parameters trainable at 3e-5.')
    append(ROOT.parent/'AGENTS.md','## v72 PERM有限观测分辨率','## v72 PERM有限观测分辨率\nSame v67 RQS epoch24 predictor; only observation halfwidth 0/.05 differs; own PERM loss, unchanged inference, all parameters trainable at 3e-5.\nOriginal9curves/206 features, fixed original fold5, train-only fitting and Q50, max80/pat18/min10, maximum2GPU; no otherfolds, old queues or delivery replacement. Maintain PROJECTS: objective/completed/planned/outstanding/post-goal work.\n')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
