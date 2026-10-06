"""Explicitly launch the two predefined v16 candidates; no automatic queue."""
from pathlib import Path
from datetime import datetime
import json, subprocess, sys
import psutil

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/latent_fold5_v16_20260928'
MODES=('tcn','latent')
METHOD='''# v16：v15潜变量记忆的原第五折验证

## 探索重点
用户继续推进最近版本，当前目标是得到v15 latent的实际三通道总分。固定POR/PERM为已有私有模型，在原第五折比较从头训练的TCN与latent的SW。结构引用原实现，不改损失或宽度，不混入内层成绩。

## 协议与已实现内容
原64井训练、16井第五折验证，578891/144665个有效点。全部统计由原64训练井重拟合，Q=1，逐点BCE+非默认两格CE/3；输入206特征+相对深度，输出默认门控+256格连续概率。预定义TCN82113参数和latent174849参数，64维隐藏；latent为16个潜变量、四头注意力、两次潜空间处理及前后局部卷积。
网络与优化器都重新初始化，不加载内层48井权重。共享形状的输入和输出初始权重相同。80轮上限/patience18/min10，batch64、64步，AdamW6e-4、cosine至1e-5、wd0.01、clip2。依原第五折SW选择checkpoint并直接恢复。最多2GPU，本批两个任务显式启动，无自动队列。
输入、支持格、动作和解码流程对两候选相同；参数数目不同，属于整网方案比较，不能视作等容量单因素因果消融。旧v5使用不同预训练及Q历史，二者比较的是最终候选效用。

## 取得的收益
启动时尚未证实。每轮TOTAL是固定已验证POR/PERM加SW替换的诊断。结束后重载新SW和旧POR/PERM，实际合并144665行预测，报告候选总分和相对v5的变化；不可由内层76.59外推第五折总分。
第五折参与早停与选模，结果为开发集分数，无独立测试或完整OOF声明。严格>80.5及完整OOF>82不等于本批自动补折授权。

## 不继续方向与边界
本批只跑tcn/latent，不扫描种子、损失系数或扩大模型。若latent低于旧SW，则保留v5/v8配方；即使有收益，也只创建新研究配方，不覆盖旧交付或自动补其他四折。同规模TCN及删除记忆的重训消融尚未完成。

## 可追溯性
数据在本研究data_outer.npz，原井集合和验证行序已比对。训练源引用历史源码但不修改它们，当前状态含优化器/调度器/RNG/停止状态，显式resume。每批更新PROJECTS、实验README、EXPLORATIONS与图谱。
'''


def main():
    if len(sys.argv)!=1: raise RuntimeError('Only the two predefined modes can be launched')
    active=[]
    for proc in psutil.process_iter(['pid','name','cmdline']):
        try:
            info=proc.info;argv=info.get('cmdline') or []
            if 'python' in (info.get('name') or '').lower() and any('pure_dl/' in a.replace('\\','/') and Path(a.replace('\\','/')).name.startswith(('train_','benchmark_')) for a in argv):active.append(info['pid'])
        except(psutil.NoSuchProcess,psutil.AccessDenied):pass
    if active: raise RuntimeError('Two launches require zero existing project trainers: '+str(active))
    if not (STUDY/'data_outer_manifest.json').exists(): raise RuntimeError('Prepare original fold data first')
    outdirs=[ROOT/'outputs'/('pure_dl_latent_v16_'+m+'_fold5') for m in MODES]
    if any(p.exists() for p in outdirs): raise RuntimeError('Never duplicate or overwrite runs')
    rules=(STUDY/'AGENTS.md').read_text(encoding='utf-8')
    (STUDY/'METHOD.md').write_text(METHOD,encoding='utf-8')
    (STUDY/'PROJECTS.md').write_text('# 目标与进展\n\n目标：实际验证v15 latent的原第五折总分。\n已做：原64井重新拟合预处理，原验证144665行核对，两个训练入口实现。\n计划：两项从头训练，恢复最佳SW，再重载固定POR/PERM共同评分。\n未做：最终训练成绩、磁盘组合验证、相同容量消融及完整OOF。\n达成后：按实际收益保留新候选，原交付和其他四折保持不动。\n',encoding='utf-8')
    note='\n\n## v16本轮推进到原第五折\n继续推进最近架构候选：原样v15latent与TCN对照在原64井重新训练，Q=1，原第五折144665点选模。只读既有POR/PERM作为固定模型，结束后实际三通道合并验证。本批不运行内层或原其他四折，不覆盖旧交付。详细边界见latent_fold5_v16_20260928/AGENTS.md。\n'
    for p in (ROOT.parent/'AGENTS.md',ROOT/'AGENTS.md'):
        with p.open('a',encoding='utf-8') as f:f.write(note)
    runs=[]
    for mode,out in zip(MODES,outdirs):
        out.mkdir()
        (out/'AGENTS.md').write_text(rules,encoding='utf-8')
        (out/'README.md').write_text(METHOD+'\n本目录模式：'+mode+'。\n',encoding='utf-8')
        (out/'PROJECTS.md').write_text('# 进展\n\n目标：完成'+mode+'原第五折训练和恢复。\n已做：协议和模型定义。\n计划：训练、早停、磁盘验证。\n未做：真实收益与原三模型总分比较。\n达成后：与对照及旧方案比较，不自动补折。\n',encoding='utf-8')
        cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_latent_fold5_v16.py'),'--mode',mode,'--out',str(out)]
        with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
            proc=subprocess.Popen(cmd,cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=log,stderr=err,close_fds=True,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        record=dict(mode=mode,pid=proc.pid,started_at=datetime.now().astimezone().isoformat(),command=cmd)
        (out/'launch.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8');runs.append(record)
    entry='\n\n## '+datetime.now().astimezone().isoformat()+' v16原第五折启动\n实际显式启动'+str([(r['mode'],r['pid']) for r in runs])+'。不加载内层权重，原64/16井，固定其余通道，最终总分待核对。原旧任务、交付、其他四折不改。研究记录results/deep_learning/latent_fold5_v16_20260928。\n'
    for p in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
        with p.open('a',encoding='utf-8') as f:f.write(entry)
    with (ROOT/'EXPLORATIONS.md').open('a',encoding='utf-8') as f:f.write('\n- [v16潜变量架构原第五折与真实总分验证](results/deep_learning/latent_fold5_v16_20260928/METHOD.md)：固定POR/PERM，TCN与latent两候选重新训练，启动时无收益结论。\n')
    print(json.dumps(runs,ensure_ascii=False))


if __name__=='__main__': main()
