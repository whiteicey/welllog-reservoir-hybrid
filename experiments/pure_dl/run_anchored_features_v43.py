"""Explicit full-feature-body SW adaptation; no automatic queue."""
from datetime import datetime
import argparse,json,subprocess,sys
from anchored_features_v43 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v43 SW完整特征骨干：原权重约束迁移

## 探索重点与预定义对照
从当前TOTAL80.336513436816、SW73.663082797066开始。v42原点约束相对free训练后最高+0.002969806658，18/18对应轮次更高，但都低于原模型。保留这个减少退化的局部机制，检验它是否允许更充分的特征学习。
本批仅free/anchored两项。两个私有专家中除main_head/aux头之外的所有既有特征参数可训练，包含输入校准、局部/全局表征、编码/解码和逐点融合分支；原目标输出头、内部输入路由、v26/v28/v29修正权重保持冻结，计算对骨干仍可微。无新增容量，不冷启动，不读其他目标标签。
两组同初始化/可训练参数/训练窗口/优化器/评分，唯一差别仍为原点惩罚系数0/1：各参数张量 mean((p-p0)^2)/max(mean(p0^2),1e-4) 的平均。p0来自原训练折已训练模型，无伪标签或验证数据监督。
相对v42，更新范围更大且使用已有私有骨干微调的保守lr1e-5；不是宣称两批只有一个因素不同。因果对照仅本批free与anchored。v42只末端dec0的当前系数/LR扫描仍关闭。

## 固定条件
原9曲线206主特征与既有63幅值侧特征、64训练井/16原第五验证井、Q50、SW NLL/3、原解码/评分和seed20261010保持。80最大轮/patience18/min10/b64/64步；AdamW lr1e-5/cosine80*64至1e-6、wd0、clip2。降低整个调度比例，周期仍80轮。所有dropout保持原eval状态，不修改验证统计。
每项独立优化器、早停和最佳checkpoint，保留第0轮回退；原子last_state含参数/优化器/调度器/RNG/停止状态。每次恢复须显式启动。最多2GPU任务，显式启动两项，无自动队列。

## 已做、收益、计划与未做
已实现新模块和训练器；本批收益尚未证实。真实训练记录梯度范围与冻结状态，结束恢复最佳预测并固定POR/PERM作实际三列评分，正候选另做磁盘重建。总墙钟与单项耗时/显存分开记录，加载/预热/处理计入。
若无新增或匹配消融收益，不继续扫描本完整骨干的约束系数/LR。旧v8端到端训练同时更新输出头/路由且起点不同，本批不冒充完全复跑。
完整原五折OOF>82目标未完成；第五折须严格>80.5且有明显收益才考虑其他折。本批无其他折、旧队列恢复或交付覆盖。维护PROJECTS.md的目标/已做/计划/未做/后续、README、EXPLORATIONS与图谱。目标完成后可选新井独立验证与新目录交付，当前未做。
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU jobs')
    out=ROOT/f'outputs/pure_dl_anchored_features_v43_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md','# v43工作约束\n仅SW完整特征骨干free/anchored匹配对照，头/路由固定。原数据/评分/Q50/80-18/max2保持。维护PROJECTS.md已做/目标/未做/计划/后续；无其他折/旧队列/交付覆盖。\n'),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_anchored_features_v43.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=p.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v43 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','anchored_features_v43_20260929','- [v43 SW完整特征骨干原权重约束](results/deep_learning/anchored_features_v43_20260929/README.md)：继承v42局部减少退化证据，仅free/anchored同容量匹配；头和路由冻结。')
    append(ROOT.parent/'AGENTS.md','## v43完整特征骨干','## v43完整特征骨干\n继续当前80.33优化，SW free/anchored两项完整既有特征骨干微调、原头/路由固定；原点约束系数0/1，统一lr1e-5，原数据/Q50/评分/80-18/max2保持。无其他折、旧队列或交付覆盖，维护anchored_features_v43_20260929。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
