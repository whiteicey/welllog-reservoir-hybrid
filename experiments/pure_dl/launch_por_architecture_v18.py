"""Explicitly launch the two predefined POR experiments. No automatic queue."""
from pathlib import Path
from datetime import datetime
import argparse,ast,json,subprocess,sys
from launch_latent_gate_v17 import active_jobs,append_once

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/por_architecture_v18_20260928'
MODES=('tcn104','latent64')
METHOD='''# v18 POR私有模型：宽TCN与潜变量记忆

## 探索重点
v15/v16的全局记忆改善了SW非默认数值段，但未超过旧SW。本批将结构转向POR，检验它是否适合不同目标。仅两项，不改变其他通道。

## 模型和控制条件
TCN104：207→104，四个门控扩张卷积块，d=1/2/4/8，默认门控+8分量Student-t3输出。
latent64：207→64，局部卷积→16个64维潜变量读/两层处理/写回→局部卷积→相同形式的输出头，4头注意力。
TCN约158833参数，latent约159769参数（以config实数为准）；容量差约0.6%，但宽度、上下文组织同时改变，不能称仅有一个模块变化。两组初始输出使用相同训练先验，完整参数形状不相同。
输入207=九曲线固定206维特征+相对深度；窗口256点、监督中心128点。每点输出1门控+3×8混合参数。z=(log1p(POR)-m)/s；mu、std和pi每点均8维。std=softplus(raw)+0.025，Student-t自由度3、scale=std/sqrt(3)。训练目标mean[BCE(g,d)+(1-d)*NLL(z)]/3，Q=1。

## 数据与评价
复用v16原64井拟合的输入缓存；POR归一化、256支持格、CDF边缘及800候选动作公式沿用同64训练井的原模型元数据。只读取这些元数据，没有把原网络权重放进候选。POR训练标签只写入64训练井；验证标签仅用于早停和评分。完整标签行序与原第五折核对。
固定seed20261010、max80/pat18/min10、bs64、64步/轮、AdamW6e-4、wd0.01、cosine至1e-5、clip2。训练集完整推理仅最终最佳执行，避免每轮重复开销。PyTorch确定性warn_only，不保证跨训练逐位复现。
当前总分基线80.274135。运行中替换TOTAL是固定PERM/SW得分公式；训练结束后从磁盘重建新POR，用原验证顺序的固定PERM/SW预测实际合并评分。不会声称本轮又重载了未重载的旧模型。

## 取得的收益
启动时尚未证实；所有参数量、初始、运行中、最终恢复结果以独立config/history/report为准。

## 不继续的边界
不自动追加宽度、学习率、核长或种子扫描，不恢复旧ABD/其他折/自动队列，不替换旧交付。单一第五折已用于多轮开发，不把小幅结果称稳定泛化或完整OOF。
若无增益保留v5/v8，若有增益固定配方和核查三通道总分；后续补折仍保持暂缓。
'''


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--launch',choices=MODES,nargs='+',required=True)
    args=ap.parse_args()
    if len(set(args.launch))!=len(args.launch):raise ValueError('Duplicate mode')
    jobs=active_jobs()
    if len(jobs)+len(args.launch)>2:raise RuntimeError(f'GPU concurrency limit; existing={jobs}')
    if not (STUDY/'data_manifest.json').exists():raise RuntimeError('Prepare POR data first')
    for name in ('por_architecture_v18.py','por_data_v18.py','train_por_architecture_v18.py'):
        ast.parse(Path(__file__).with_name(name).read_text(encoding='utf-8'))
    for mode in args.launch:
        out=ROOT/'outputs'/f'pure_dl_por_architecture_v18_{mode}_fold5'
        if out.exists():raise RuntimeError(f'Existing trial, no overwrite: {out}')
    (STUDY/'METHOD.md').write_text(METHOD,encoding='utf-8')
    rules=(STUDY/'AGENTS.md').read_text(encoding='utf-8')
    records=[]
    for mode in args.launch:
        out=ROOT/'outputs'/f'pure_dl_por_architecture_v18_{mode}_fold5';out.mkdir()
        (out/'AGENTS.md').write_text(rules,encoding='utf-8')
        (out/'README.md').write_text(METHOD+'\n本目录模式：'+mode+'。\n',encoding='utf-8')
        (out/'PROJECTS.md').write_text('# 进度\n目标：POR骨干优化。\n已做：协议/代码/数据准备。\n计划：训练和恢复最佳。\n未做：实际收益和磁盘重载。\n完成后：与匹配候选及v5比较，原其他折与交付不变。\n',encoding='utf-8')
        command=[sys.executable,'-X','utf8','-u',str(Path(__file__).with_name('train_por_architecture_v18.py')),'--mode',mode,'--out',str(out)]
        with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
            proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,close_fds=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        row=dict(mode=mode,pid=proc.pid,started_at=datetime.now().astimezone().isoformat(),command=command)
        (out/'launch.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8');records.append(row)
        print('LAUNCHED',json.dumps(row,ensure_ascii=False),flush=True)
        append_once(ROOT/'EXPLORATIONS.md',out.name,f'- [v18 POR {mode}](outputs/{out.name}/README.md)：相近容量骨干对照；只训练POR，启动时未证实收益。')
    marker='## v18本轮POR完整骨干对照'
    append_once(ROOT.parent/'AGENTS.md',marker,marker+'\n继续优化的本批实施范围：tcn104/latent64两个从随机网络开始的POR私有模型，近容量匹配，共同Student-t3输出。保持原第五折/数据/评分/80-18/最多2GPU，固定其他目标，无旧任务或补折。结果与未完成项维护por_architecture_v18_20260928/PROJECTS.md。')
    entry='## '+datetime.now().astimezone().isoformat()+' v18显式启动\n'+json.dumps(records,ensure_ascii=False)+'\n仅两项预定义POR训练；未获得新的最终分数，未改旧模型/交付/其他折。'
    for p in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md'):
        with p.open('a',encoding='utf-8') as f:f.write('\n\n'+entry+'\n')


if __name__=='__main__':main()
