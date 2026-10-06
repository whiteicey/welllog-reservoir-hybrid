"""Explicitly launch only the two declared v24 training jobs."""
from pathlib import Path
from datetime import datetime
import ast,hashlib,json,subprocess,sys
from launch_latent_gate_v17 import active_jobs,append_once

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/expert_supervision_v24_20260928'
MODES=('mixture','independent')
METHOD='''# v24：专家独立监督与混合责任监督的匹配实验

## 探索重点
在保持相同私有POR双专家、1290490参数、相同随机初始状态下，检验专家是否应该各自拟合全部POR监督，而不是只通过最终混合似然接收责任加权梯度。原冠军TOTAL80.274135保留；不增宽、不追加其他目标或深层辅助头。

## 两个预定义候选
mixture：专家损失为最终两专家混合的POR似然，计算该损失时detach路由权重；路由损失为detach两专家输出后的相同混合似然。
independent：专家损失为两个专家各自完整POR似然的均值；路由损失与mixture采用相同公式及梯度隔离。
两组专家和路由均用两个相同AdamW与独立clip2，学习率/调度相同。mixture合计标量为2倍混合损失，但每个参数组只接收一次对应梯度，未双倍更新专家或路由。不同候选的专家损失量值不可简单当作相同目标的优化程度。

## 公式与形状
点i的两专家完整预测似然为p_i1,p_i2，默认标签处是概率质量，连续标签处是密度；路由a_i属于R^2、非负且和为1。混合p_i=sum_e a_ie*p_ie。混合监督对专家e的梯度为-r_ie*grad(log p_ie)，r_ie=a_ie*p_ie/p_i；独立专家均值损失的系数为1/2。两者都乘原质量q_i/(3N)，N为有效监督点数，q_i为Q50标量。专家、路由参数不共享，stop-gradient只改变训练梯度路径，推理仍采用相同两专家混合。
模型输入B×256×206，另有B×256相对深度与有效掩码。两位专家各输出B×256×1默认logit和B×256×1×3×8分布原始参数；路由B×256×1×2；合成分布有16个Student-t3分量。

## 固定条件
原v23 base1共同随机权重seed20261010，两组相同初始全部144665点预测；没有加载旧网络权重。原206/68视图、校准、局部路由、注意力、多尺度卷积和point32头保持。使用原64/16井、v21缓存与Q50，预处理/目标变换/质量权重有训练历史，不能称完整pipeline零历史。
bs64×64步、max80/patience18/min10、AdamW6e-4、wd.01、cosine到1e-5；expert和router分别clip2。原256格/800动作/单覆盖解码不变。两组相同轮次抽样窗口相同，但CUDA warn_only不保证训练逐位可重复。
本批mixture是新训练的匹配对照。v23只有一个全局clip和优化器，与本轮不同；不将跨批差值全部归因于专家监督。原80.22还含多任务/深层辅助/动态Q，本轮仅检验监督形成机制的一部分。

## 验证及停止边界
首个实际训练批检查梯度隔离、两专家和路由非零梯度。每轮原子保存完整current/optimizers/schedulers/RNG/early-stop和最佳状态。最后新模型磁盘重载，固定PERM/SW数组按原行序合成并调用原metric，记录默认/非默认段与训练/验证、路由责任诊断。未改变目标的旧模型不重复GPU重载。
只有两项，最多2GPU，无自动队列、其他折或旧任务恢复，交付不改。第五折反复参与开发，早停/模型差值不作独立泛化证据。无增益继续保留80.274135；完整OOF>82与第五折>80.5尚未完成。
'''


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if len(sys.argv)!=1:raise RuntimeError('Launch exactly mixture and independent')
    if active_jobs():raise RuntimeError('Existing trainers; inspect before launching')
    for mode in MODES:
        if (ROOT/'outputs'/f'pure_dl_expert_supervision_v24_{mode}_fold5').exists():raise RuntimeError('Output already exists: '+mode)
    for name in ('expert_supervision_v24.py','train_expert_supervision_v24.py'):
        ast.parse(Path(__file__).with_name(name).read_text(encoding='utf-8'))
    source=Path(__file__).with_name('scratch_scaling_v23.py')
    if hashlib.sha256(source.read_bytes()).hexdigest()!='22926074f0b6839211f18e76c984934a03154e3890ed391914c05082d0c63a56':
        raise RuntimeError('v23 initialization source changed')
    rules=(STUDY/'AGENTS.md').read_text(encoding='utf-8')
    (STUDY/'METHOD.md').write_text(METHOD,encoding='utf-8')
    progress='# v24当前状态\n\n目标：固定POR双专家容量，比较专家监督机制。\n\n已做：确认v23全部结束，核对原梯度路径，完成两个候选的实现与固定协议。\n\n计划：两组训练、最佳恢复、实际三列评分与责任分解。\n\n未做：真实训练收益、最终重载、其他目标和全五折。\n\n达成后：只有超过冠军才更新当前研究配方，不自动补折或覆盖交付。\n'
    (STUDY/'PROJECTS.md').write_text(progress,encoding='utf-8')
    marker='## v24 固定容量的专家监督对照'
    append_once(ROOT.parent/'AGENTS.md',marker,marker+'\n本批仅mixture/independent，原尺寸1290490参数随机初始化，两组相同expert/router优化器和独立clip2；只改专家损失，路由都detach专家训练。原64/16井/Q50/解码/80-18/max2，不补折或恢复旧任务。维护expert_supervision_v24_20260928/PROJECTS.md。')
    records=[]
    for mode in MODES:
        out=ROOT/'outputs'/f'pure_dl_expert_supervision_v24_{mode}_fold5';out.mkdir()
        for name,value in [('AGENTS.md',rules),('README.md',METHOD+'\n本项目模式：'+mode+'；启动时尚无实测收益。\n'),('PROJECTS.md',progress)]:
            (out/name).write_text(value,encoding='utf-8')
        command=[sys.executable,'-X','utf8','-u',str(Path(__file__).with_name('train_expert_supervision_v24.py')),'--mode',mode,'--out',str(out)]
        with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
            proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,close_fds=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        record=dict(mode=mode,pid=proc.pid,started_at=datetime.now().astimezone().isoformat(),command=command)
        (out/'launch.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8');records.append(record)
        append_once(ROOT/'EXPLORATIONS.md',out.name,f'- [v24 {mode}](outputs/{out.name}/README.md)：同容量专家监督对照，启动时收益未证实。')
        print('LAUNCHED',json.dumps(record,ensure_ascii=False),flush=True)
    entry='## '+datetime.now().astimezone().isoformat()+' v24显式启动\n'+json.dumps(records,ensure_ascii=False)+'\n两项预定义任务，无其他候选/折或自动续跑。'
    for path in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md'):
        with path.open('a',encoding='utf-8') as stream:stream.write('\n\n'+entry+'\n')
    (STUDY/'execution_notes.json').write_text(json.dumps(dict(prior_read_only_command_quoting_failure=True,
        initial_connector_read_failures=2,connector_recovered_before_writing_or_launching=True,
        model_training_failures_at_launch=[]),ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
