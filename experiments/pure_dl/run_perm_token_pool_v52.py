"""Explicit matched learned-depth-pooling study."""
from datetime import datetime
import argparse,json,subprocess,sys
from perm_token_pool_v52 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v52 PERM内容自适应深度汇聚

## 探索重点与固定对照
v51训练增强无最终收益，当前PERM89.396724/TOTAL80.342772保持。新假设：原骨干每16深度点平均为一个注意力token，可能弱化局部薄层信号；用内容相关汇聚检验这一结构限制，不继续增强概率/Beta/层扫描。
position/content两个完整私有PERM网络，同837305参数、同源SW第9轮骨干/FiLM、同完全重置的PERM25维输出。源与训练预算沿用v50已证实局部正收益的迁移配置，不从已过拟合PERM最后状态继续微调。增加96→4无偏置权重打分器，四组各24通道值，在每16有效深度点内softmax加权汇聚，再送原四层长程注意力。初始打分全零，对应各组有效点均值。
两组只改打分器输入：content读三层局部TCN后的96维测井内容；position读GELU(原位置编码(relative depth, window position))。四组值、下游结构、参数量、初始化、标签、优化器相同；位置输入也可学习非线性位置偏好。名义容量相同，但输入连通/有效表达能力不同是预设实验变量，不能单独归因于参数数目。空token权重与输出为零，有效掩码内归一。
全部骨干/FiLM/位置/打分器/输出同时训练，不是冻结模型的输出小补丁。旧v50等权汇聚作为历史参照，不重新跑同一对照；本批position/content是严格配对。与旧预测概率池化/多窗口投票不同，本次仅内部特征压缩，不混合旧模型输出。

## 数据与训练边界
源权重仅原64训练井SW监督训练；当前PERM两组输出全重置，损失仅PERM训练标签。原9曲线206、Q50、固定原第五144665点/seed20261010、1024窗口/中心128、原8Gaussian/默认头与256支持点/800动作加默认保持。输入井库无标签/井ID特征，验证井库只用于该井推理；无跨井增强。
同AdamWlr3e-4/wd.01/clip2、cosine80*64至1e-5，max80/pat18/min10/b64/64步。输出训练侧先验相同，不把先验到88分的学习当超越当前模型。完整原子last_state、显式恢复，最多2GPU无自动队列；实际明显落后可提前停止。CUDA注意力/插值反向warn_only，不声称重新训练逐位一致。

## 已做、待做与不继续边界
已实现四组掩码汇聚、两种打分输入及训练器；开始前无新成绩。直接真实训练，记录打分器梯度、权重范数及训练池化熵，区分实际学习与失活。结束核验同初始状态/先验预测、源哈希、最佳恢复和实际三列评分；有价值候选磁盘重建。不重复原密度检查。
待做两项完整对照、与同容量/历史v50/当前最优三种差值分开、负权重清理、PROJECTS/README/EXPLORATIONS/atlas更新。单项和双并发总墙钟/显存分开，含加载/预热/输入库/启动间隔；代码与最终重载另计。
若内容汇聚不优于位置对照，不扫描汇聚组数、步长或温度；只限此结构和训练配置，不能否定所有自适应token化。原第五严格>80.5且明确收益后才考虑其他折；完整OOF>82未完成。无旧任务/其他折/交付覆盖；独立井确认及新目录交付尚未开始。
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_perm_token_pool_v52_PERM_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for n,txt in [('AGENTS.md','# v52规则\n仅position/content两个同结构同初始PERM模型，改变内部16点四组汇聚打分输入；源SW迁移/输出全重置/全骨干训练。原数据/Q/评分/80-18/max2保持，维护PROJECTS目标/已做/未做/后续，无补折或交付覆盖。\n'),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/n).exists():(STUDY/n).write_text(txt,encoding='utf-8')
    out.mkdir()
    for n in ('AGENTS.md','README.md','PROJECTS.md'):(out/n).write_text((STUDY/n).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_perm_token_pool_v52.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=p.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v52 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','perm_token_pool_v52_20260929','- [v52 PERM内容深度汇聚](results/deep_learning/perm_token_pool_v52_20260929/README.md)：同迁移初始和全新PERM头，位置与内容四组汇聚对照，全骨干训练；开始前无成绩。')
    append(ROOT.parent/'AGENTS.md','## v52 PERM内容深度汇聚','## v52 PERM内容深度汇聚\n本批position/content两项，同SW迁移骨干与全重置PERM头，16点四组内部汇聚，打分输入位置或内容；全部参数训练。原数据/Q/评分/80-18/max2，无其他折/旧队列/交付覆盖，维护perm_token_pool_v52_20260929。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
