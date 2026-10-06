"""Explicit single-process, two-independent-head SW message study."""
from datetime import datetime
import json,subprocess,sys
from state_messages_v44 import ROOT,STUDY
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v44 SW默认状态神经消息

## 探索重点与假设
从原第五折TOTAL80.336513436816、SW73.663082797066开始。训练井相邻有效点SW状态切换率2.0538%，仅说明存在持续性，不是效果证据。连续分布固定，尝试用输入条件化的状态消息修正默认概率。
历史transition_loss_sequence只做过公式/资源原型并由用户取消，未正式训练；不得写作失败。本次是新预测模块，不恢复旧原型或队列。与历史邻接变化辅助损失不同，这次消息进入推理。

## 两项匹配对照
self/neighbor各10273参数（319→32→1），相同初始化、窗口和优化器，最终层零初始化。只用冻结的本目标两个专家隐藏特征和既有幅值侧特征预测耦合J；J[t]=.5*tanh((z[t-1]+z[t])/2)，仅连接同井有效邻点。正负耦合均可学习，不硬编码必须平滑。
每方向8次消息更新：m=2*atanh(tanh(J)*tanh((g_source+m_previous)/2))。neighbor读取相邻位置状态及前一步消息，self读取当前位置状态及自身前一步消息；两者边耦合输入相同，唯一差别是状态消息来源。两头均可利用原骨干的上下文，self不等于完全无上下文模型。
每方向消息绝对值≤1，门控总修正≤2；J初始为0，逐点复现原输出。固定8轮消息只是有限半径结构网络，不宣称整井精确CRF或Viterbi。conditional density原样传递；最终仍用原默认+连续动作官方期望效用解码，不作手工后处理或OOF逐点择优。

## 固定训练边界
仅训练SW门控BCE/3，Q50保持；原连续NLL固定，省略不会改变可训练梯度。不用相邻验证标签或伪标签，不改变原9曲线206主特征及63侧特征，64训练井/16原第五验证井、seed20261010不变。其他目标不训练。
80轮上限/pat18/min10/batch64/64步；lr3e-4 AdamWwd.01、clip2、cosine80*64至1e-5。两头独立优化/早停/最佳checkpoint；一个GPU进程共享固定骨干前向，最多2GPU进程，无自动队列。原子last_state保存完整状态，恢复需显式启动。

## 已完成、收益与待做
训练侧邻接诊断、两头代码已完成；收益尚未证实。真实训练后恢复最佳，核对冻结参数和原条件密度，固定POR/PERM实际三列评分；正候选再磁盘重建。记录单项与同工作量墙钟、显存，包含加载和预热。
若neighbor不优于self和原模型，不继续消息轮数/耦合界扫描；不能据此否定所有序列模型。v43完整骨干微调不再延伸。
第五折严格>80.5、完整原五折OOF>82仍未完成；本批不补其他折、不重启旧队列、不覆盖旧交付。每批更新PROJECTS.md目标/已做/计划/未做/可选后续、README、EXPLORATIONS和演进图谱。目标完成后的新井独立验证与新目录交付当前未做。
'''

def main():
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/'outputs/pure_dl_state_messages_v44_SW_fold5'
    if out.exists():raise FileExistsError(out)
    assert (STUDY/'training_state_diagnostic.json').exists()
    (STUDY/'README.md').write_text(TEXT,encoding='utf-8')
    append(STUDY/'PROJECTS.md','## v44预定义训练', '## v44预定义训练\n'+TEXT)
    out.mkdir()
    for n in ('AGENTS.md','README.md','PROJECTS.md'):(out/n).write_text((STUDY/n).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_state_messages_v44.py'),'--target','SW','--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=p.pid,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,'## v44显式启动','## v44显式启动\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','state_messages_v44_20260929','- [v44 SW默认状态消息](results/deep_learning/state_messages_v44_20260929/README.md)：原连续分布固定，self/neighbor匹配消息来源对照。')
    append(ROOT.parent/'AGENTS.md','## v44默认状态消息','## v44默认状态消息\n本批仅SW self/neighbor两项有限8步消息头，同容量/初始化，旧预测器和连续分布固定；原数据/Q50/评分/80-18/max2保持。原训练井状态诊断，非验证标签监督。无旧队列/其他折/交付覆盖，维护state_messages_v44_20260929。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
