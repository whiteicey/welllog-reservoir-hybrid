"""Explicitly launch one of two matched whole-well memory experiments."""
from datetime import datetime
import argparse,json,subprocess,sys
from well_memory_v47 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v47 任务查询的整井输入记忆

## 探索重点与预定义对照
父组合v46 TOTAL80.342771761689，SW73.680963725271。上一轮连续值部分提高、默认状态下降而相互抵消；本轮假设：不同输出任务需要按当前地层上下文读取不同整井信息。
历史v11两套TCN完全分离SW较共享低0.166031，且训练侧梯度不普遍冲突；本次不重新强制拆开两个完整骨干，也不声称已证明梯度冲突。保留v46共享骨干和整井FiLM，增加同一整井输入记忆和状态/数值两组查询。全部旧骨干、FiLM与新记忆同时训练，不是冻结骨干的小头扫描。
128个原206维整井输入样本加确定性深度分位位置，经207→96→64编码产生4头键/值；64个井段token分别产生状态查询和数值查询。两个读取结果经各自64→128零初始化投影、上采样加到原最终隐藏表示，再走原LayerNorm/输出参数。无旧预测投票，不用其他通道标签；密度/默认预测共同经过原官方动作解码。
两项预先固定：global_query先平均当前1024点窗口内有效token，再复制给各位置；section_query保留64个token各自的查询。global指当前窗口的全局查询，并非整井平均。两组同900985参数、相同初始权重、同记忆/损失/采样/优化器，只改变查询的空间变化。两组都可按任务分别读记忆；对照不能单独证明任务分离本身的贡献。
新增投影初始为零，保持v46原函数；原模型继续可训练。输入记忆仅来自当前井测井输入，所有标准化/Q和网格只用原64训练井；16验证井输入库仅用于该井推理。无井ID特征、标签记忆、测试时训练或验证标签拟合。

## 训练与验证边界
固定原第五折、seed20261010、9曲线206特征、Q50、1024上下文/中心128、原256支持点与800动作加默认值。80轮上限/patience18/min10/b64/每轮64步；新AdamW lr1e-4、wd.01、clip2，cosine80*64至1e-5。只在原64井采样训练。原子保存完整训练状态和最佳权重；只显式恢复。最多2GPU，无自动队列。
启动即真实训练，不安排独立预检/审查；初始父函数、梯度和覆盖证据在训练中记录。结尾恢复本折最佳，并在实际三列组合上评分；新正候选额外磁盘重载。训练warn_only确定性，CUDA注意力/插值反向可能非确定，不宣称重新训练逐位一致。

## 已做与待做
已实现模块、训练器和明确两项对照；未训练时成绩尚无。运行中只报告快照，结束记录最终最佳/停止轮次及负结果。待完成两项训练、实际组合、紧凑诊断、清理无收益权重和更新演进图谱。
收益先相对同起点v46及匹配候选分别报告，不把第0轮回退当训练收益。若两项没有收益，不扫查询维数/记忆点数/注意力头数；结论仅适用于当前热启动结构。严格第五>80.5且明确收益后才考虑其余折，完整OOF>82未完成；不覆盖交付。
总墙钟与单项耗时分开记录，说明数据加载/预热/输入库计入范围及最大并发和显存。目标达成后的独立新井验证/新目录交付尚未启动。维护本PROJECTS、根PROJECTS、EXPLORATIONS与atlas。
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_well_memory_v47_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for n,txt in [('AGENTS.md','# v47规则\n仅global_query/section_query两项；原井/数据/Q/评分不变，全共享骨干与任务输入记忆训练。输入库无标签/井ID。80/18/max2，无补折或交付覆盖。每批更新PROJECTS目标、已做、待做及后续。\n'),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/n).exists():(STUDY/n).write_text(txt,encoding='utf-8')
    out.mkdir()
    for n in ('AGENTS.md','README.md','PROJECTS.md'):(out/n).write_text((STUDY/n).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_well_memory_v47.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=p.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v47 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','well_memory_v47_20260929','- [v47 任务查询整井输入记忆](results/deep_learning/well_memory_v47_20260929/README.md)：同参数窗口平均查询与局部井段查询对照，全骨干训练；启动时无成绩。')
    append(ROOT.parent/'AGENTS.md','## v47任务查询整井记忆','## v47任务查询整井记忆\n本批global_query/section_query两项，v46SW同起点，共享全骨干可训练，状态/数值各自注意力读取整井输入库。仅改变查询是否随井段变化。原数据/Q/评分/80-18/max2，无其他折/旧队列/交付覆盖，维护well_memory_v47_20260929。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
