"""Explicit base versus parallel128 growth after the adopted v56 gain."""
from datetime import datetime
import argparse,json,subprocess,sys
from perm_quantile_v67 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v67 PERM分位密度

## 探索重点与预定义对照
将既有SW平滑CDF的局部消融机制迁移至PERM；原v50 sw_transfer已保留，PERM89.050807574234，低于当前双专家89.396724049366。本轮linear/rqs两个私有PERM候选从同v50训练折来源加载骨干/FiLM/默认门控，只重置连续密度。新838211参数模型预测33个有序分位点，两端指数尾各质量.005。linear分段线性CDF，rqs有理二次CDF，斜率来自相邻割线调和均值；沿用v48已训练源码，不重复公式预检。
相同参数张量与分位先验、优化配置；插值函数不同，初始CDF/预测可能不同，不能冒称初始完整函数一致。密度先验只用64训练井非default PERM标签，保留原归一化目标；原父密度函数不保留。比较最终配对差和相对当前冠军绝对差，单独报告初始差，收益不由起点变化冒充。

## 数据训练边界与已做
原9曲线/206输入、训练侧预处理/Q50、第五16井144665点、seed20261010、1024随机窗口中心128、原256支持点/800动作+default官方评分不变。整井128输入库无标签/井ID，验证井仅自身推理。训练仅PERM BCE与正确新CDF导数负对数，所有骨干/门控/密度/FiLM可训练，POR/SW固定不更新。
统一AdamW1e-4/wd.01/clip2、cosine80*64至1e-5、max80/pat18/min10/b64/64steps；完整原子last_state含模型/优化器/调度/RNG/早停，仅显式恢复。已实现，启动时无成绩。运行实际梯度、来源哈希、同初始张量、全144665最佳恢复及三列评分核验；正候选保留前磁盘原始输入重建，不做独立数学预检/审查。

## 收益、待做与不继续
待两项训练及完整配对消融、当前绝对差、负模型清理、PROJECTS/EXPLORATIONS/图谱更新。rqs若最终优于同容量linear，即便低于当前冠军，保留为局部机制；不能称当前组合提高。若无配对收益，停止当前PERM分位数/尾质量扫描；结论不泛化其他目标。训练中的暂时领先不算最终收益。
先启动linear观察实际显存后启动rqs，最多2任务不自动队列，记录单项与两任务实际总墙钟及并发/显存，含加载/预热/训练/评估，代码/最终重建另计。负权重删除先写清单，源配置紧凑证据保留。第五严格>80.5且明确收益才考虑其他折，完整原五折OOF>82未完成；无旧任务/其他折/交付覆盖。目标完成后独立井确认、新目录交付。
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_perm_quantile_v67_PERM_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md','# v67规则\n仅PERM linear/rqs同容量密度对照，来源v50本训练折模型；只用PERM训练标签，原数据/Q/评分/80-18/max2，无其他折/旧队列/交付修改。维护PROJECTS目标/已做/未做/后续。\n'),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_perm_quantile_v67.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v67 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','perm_quantile_v67_20260929','- [v67 PERM分位密度](results/deep_learning/perm_quantile_v67_20260929/README.md)：同v50 PERM来源，linear/rqs同容量/初始参数，密度重置；启动时无成绩。')
    append(ROOT.parent/'AGENTS.md','## v67 PERM分位密度','## v67 PERM分位密度\n本批linear/rqs两项，同PERM v50来源骨干/门控，重置密度为训练侧33分位先验；原数据/Q/评分/80-18/max2，无旧队列/其他折/交付覆盖，维护perm_quantile_v67_20260929。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
