"""Explicit two-candidate whole-well conditioning study on the new SW backbone."""
from datetime import datetime
import argparse,json,subprocess,sys
from well_conditioned_v46 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v46 井级条件化：整井输入驱动内部参数

## 目标、探索重点与对照
继承用户“发散思路”的要求，从v45新SW长井段骨干出发研究井级条件网络，不回到冻结小头。v45自身第五SW73.664377810365、候选TOTAL80.336966691471只是数值微增，尚无明确新井收益。
两个匹配候选reference/well：同一个新井编码器读取128×206输入，经逐点64维MLP提取mean/std/max后得到96维井代码，生成六处中间表示的scale/bias：输入stem、四层全局注意力和细节融合。scale/bias界±.5，编码器末层零初始化，两组初始函数均为v45第12轮模型；骨干、密度输出头和新编码器全部训练，不冻结主模型。
well读取当前整口井按深度均匀抽取的128组测井输入；reference读取64训练井输入库的逐位置等井平均，所有样本相同。参数/初始化/窗口/损失/优化器相同，只有井级输入源不同。reference仍可学习静态调制，是容量和新增分支的控制。
历史paired_conditioned_mix仅在旧双专家的校准输入分支用已有井统计作逐点rank8混合；本次是新单模型、可训练整井输入编码及六处内部调制，不宣称井条件化概念首次出现，也不恢复旧任务。

## 数据边界
原9曲线206特征及1024局部窗口保持。每口井的库由原训练侧标准化输入在128个固定深度分位位置确定；没有储层标签、预测或可训练井编号embedding。井编号只作输入库索引，不进入任何网络特征。训练batch只取64训练井anchors及其输入库；16验证井库只在该井推理时使用，不参与参数或归一化统计拟合。reference平均仅用64训练井，不能混入16验证井。
整井输入可用于新井推理，不属于用验证标签监督或测试时优化。模型没有在线训练；完整输入库行号/训练井名单/哈希记录在input_context_manifest.json。除本批新增输入条件网络，原数据/原Q50/原标签变换/官方256支持及800动作保持。

## 训练计划
两组均从本训练折自有v45最佳权重热启动，新优化器统一lr1e-4 AdamWwd.01/clip2/cosine80*64至1e-5，80上限/pat18/min10/b64/64步。新调度不是继续v45旧轨迹，不以v45最佳轮数重训。原窗口中央128点监督、原第五144665点单覆盖、batch16解码。
独立进程与优化器，最多2GPU，显式启动，无自动队列；完整原子last_state。若明显负向可通过显式stop_requested.json在完整轮次边界结束，用户中断另保存，不自动恢复。

## 已做、收益、未做与后续
已实现井输入库、神经编码/调制、匹配对照和训练器；收益待真实训练。启动后记录初始原函数复现与真实梯度；结束恢复最佳、实际三列组合，新采用模型额外磁盘重建。参数量/单项及总墙钟/显存以实际运行记录为准，包含加载与预热。
若well不优于reference与旧SW，不密集扫采样点数/调制界；通过匹配对照的局部正收益可带入另一个独立机制，不把第0轮回退当训练收益。
目标第五严格>80.5且明显收益后才考虑其余折，完整原OOF>82未完成。当前不补折/重启旧队列/覆盖交付。维护PROJECTS.md目标/已做/计划/未做/后续、README、EXPLORATIONS和图谱；目标达成后的新井确认与新目录交付尚未启动。
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_well_conditioned_v46_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for n,txt in [('AGENTS.md','# v46约束\n仅reference/well两项整井输入条件网络，同新SW起点且全骨干训练。无井编号特征、标签入输入库或验证拟合；原数据/评分/Q50/80-18/max2保持。维护PROJECTS.md目标/已做/计划/未做/后续，不补折或覆盖交付。\n'),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/n).exists():(STUDY/n).write_text(txt,encoding='utf-8')
    out.mkdir()
    for n in ('AGENTS.md','README.md','PROJECTS.md'):(out/n).write_text((STUDY/n).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_well_conditioned_v46.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=p.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v46 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','well_conditioned_v46_20260929','- [v46 井级条件网络](results/deep_learning/well_conditioned_v46_20260929/README.md)：整井128点输入编码调制六层；固定训练井参考输入作为同容量对照，全部骨干训练。')
    append(ROOT.parent/'AGENTS.md','## v46井级条件网络','## v46井级条件网络\n本批reference/well两个匹配井条件候选，v45SW同起点，整井128点输入编码调制六层且全骨干训练。只输入曲线无标签/井ID特征，reference平均只64训练井；原数据/Q50/评分/80-18/max2保持，无其他折/旧队列/交付覆盖，维护well_conditioned_v46_20260929。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
