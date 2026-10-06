"""Explicit launch of at most two predefined backbone candidates; no queue."""
from pathlib import Path
from datetime import datetime
import argparse,json,subprocess,sys
import psutil
from architecture_zoo_v15 import ARCHITECTURES,DESCRIPTIONS

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/architecture_zoo_v15_20260928'
RULES='''# 骨干结构探索目标与约束

目标：按用户要求扩大网络结构探索，区别于冻结骨干后的微小输出修正。预定义adaptive、latent、state_space三种完整可训练私有骨干，首轮均单独预测SW，POR/PERM保持原私有模型。

使用v14固定48训练井/16监控井及严格训练侧预处理缓存，Q=1，损失和解码固定；当前阶段不触及原第五折。模型均从头训练，不读旧网络预测或标签记忆。上限80轮/patience18/min10/batch64/64步，最多2GPU任务，第三项需明确调用启动，无自动队列。

维护PROJECTS.md，说明目标、已做、计划、未做与目标完成后的选项，每批更新。维护独立README和演进图谱。分清内层SW、原第五折TOTAL与完整OOF。结构参数量不完全一致，不宣称纯单因素因果效应。不覆盖旧源码、模型或交付，不自动补原其他折。
'''
METHOD='''# v15：三类不同机制的完整网络结构

## 探索重点
从网络结构角度拓宽：自适应采样处理不同层厚与边界；潜变量注意力学习整窗背景；双向门控状态空间处理多时间尺度记忆。使用相同207维输入、64维隐藏宽度、完整门控+256格输出、原BCE/CE、优化器和80轮上限。保持三个业务目标独立，首轮在SW瓶颈上筛查，不把试验范围误写为三个通道都已训练。

## 结构
1. adaptive：四层content-conditioned fractional-depth sampler，5采样点/层，基础间距1/2/4/8。每点学习偏移和采样权重，线性插值读取邻域，接逐点FFN。所有层保留全分辨率。借鉴deformable convolution，不是原DCN的完整复现。
2. latent：前后局部卷积保留细节，16个64维潜变量通过四头cross-attention读整窗256点，在潜空间做2层self-attention，随后逐点query读回记忆。无旧冻结骨干或跨井标签缓存。借鉴Perceiver IO，并非论文完整复现。
3. state_space：四层双向对角指数记忆，每通道学习稳定tau>1，初始tau2..128。s_t=a*s_(t-1)+(1-a)*u_t，a=exp(-1/tau)，反向独立tau，内容输出门调节两向消息；FFT线性卷积长度2L，避免周期绕回。当前A不随输入变化，也没有完整S4D初始化，不标为Mamba或S4D复现。

## 统一实验与基线
复用v14按点TCN的冻结结果作为已完成基线：内层SW76.234930079686。候选重用其缓存/窗口生成/损失/评估函数。48训练井434270有效点，16监控井144621点。没有旧Q或旧标签先验泄漏；无外部预训练。共享形状的输入/输出层初始相同，骨干结构不同且参数量不同；统一宽度预算比较，不能仅以分数归因于参数量或某个子模块。

全部骨干与输出头训练；优化器AdamW6e-4，cosine80*64到1e-5，wd0.01，clip2，随机窗口seed20261010。初始输出为相同训练标签先验，专用骨干初始化seed20261027。训练轮次/耗时、显存、实际最佳轮次单独报告，不将模型名等同于实测收益。

## 已经探索与验证
启动时完成机制实现和协议；真实梯度、初始预测、最终恢复按实际报告填写，不预先声称通过。非线性采样/状态递推/注意力掩码需要结果检查，训练后只对有限固定窗口做padding测试；正面候选再做必要的完整重载。新增结构不改变损失或输出分布族，不把内层76.x与第五折73.x作差。

## 收益及继续条件
启动时尚未证实。本批仅三项；内层至少+0.05SW且9/16井提高视为值得细化的筛查信号，仍不等于独立泛化确认；不能自动补原四折。结构比较后才做参数匹配或机制关闭对照。旧三模型TOTAL80.274135保持，原第五折>80.5、完整OOF>82未达成。

## 不继续方向及证据边界
不重复前几轮仅POR位置头、冻结先验小修正或井权重扫描；已有curve-token/spectral分支不重新命名当新机制。三项都可以产生负结果，负结果只约束本设置。跨井检索、学习层段图、选择性SSM留为后续明确假设，不同时堆入本轮。

## 方法来源
DCN arXiv:1703.06211；Perceiver IO arXiv:2107.14795；Diagonal SSM参数化 arXiv:2206.11893。只引用结构启发，不据论文结果预告本数据成绩。
'''


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--launch',nargs='+',choices=ARCHITECTURES,required=True);a=ap.parse_args()
    if len(a.launch)!=len(set(a.launch)):raise RuntimeError('Duplicate launch')
    active=[]
    for proc in psutil.process_iter(['pid','name','cmdline']):
        try:
            d=proc.info;argv=d.get('cmdline') or []
            if 'python' in (d.get('name') or '').lower() and any('pure_dl/' in x.replace('\\','/') and Path(x.replace('\\','/')).name.startswith(('train_','benchmark_')) for x in argv):active.append(d['pid'])
        except(psutil.NoSuchProcess,psutil.AccessDenied):pass
    if len(active)+len(a.launch)>2:raise RuntimeError('max2 GPU tasks; active='+str(active))
    for mode in a.launch:
        if (ROOT/'outputs'/('pure_dl_architecture_v15_'+mode+'_inner')).exists():raise RuntimeError('Do not duplicate run '+mode)
    STUDY.mkdir(parents=True,exist_ok=True)
    if not (STUDY/'METHOD.md').exists():
        (STUDY/'AGENTS.md').write_text(RULES,encoding='utf-8');(STUDY/'METHOD.md').write_text(METHOD,encoding='utf-8')
        (STUDY/'PROJECTS.md').write_text('# 目标与进展\n\n目标：比较三种不同骨干机制，拓宽网络结构设计。\n已完成：历史与资源核对，三种结构和统一训练接口。\n计划：仅三项预定义内层SW实验，显式分批最多两任务。\n未完成：实测训练收益、机制消融、三目标扩展和原第五折验证。\n目标完成后：选择有证据的结构再进行匹配消融及其他独立通道实验，不自动补折或改交付。\n',encoding='utf-8')
        with (ROOT/'EXPLORATIONS.md').open('a',encoding='utf-8') as f:f.write('\n- [v15骨干架构探索：自适应采样/潜变量记忆/双向状态空间](results/deep_learning/architecture_zoo_v15_20260928/METHOD.md)：三种完整网络，严格48/16内层SW，启动时未证实收益。\n')
        for p in [ROOT.parent/'AGENTS.md',ROOT/'AGENTS.md']:
            with p.open('a',encoding='utf-8') as f:f.write('\n\n## v15用户要求拓宽网络结构：本轮执行范围\n用户要求从网络结构拓宽探索。本轮方案为三种完整可训练私有SW骨干：自适应采样、潜变量读写记忆、双向门控对角状态空间。固定v14内层48/16和损失/评分/Q=1/80-18/max2，预定义三项显式分批。当前不修改POR/PERM模型，不恢复旧任务或补折；构想与完成事实分开记录。\n')
    launches=[]
    for mode in a.launch:
        out=ROOT/'outputs'/('pure_dl_architecture_v15_'+mode+'_inner');out.mkdir()
        (out/'AGENTS.md').write_text(RULES,encoding='utf-8');(out/'README.md').write_text(METHOD+'\n本目录：'+mode+'\n',encoding='utf-8')
        (out/'PROJECTS.md').write_text('# 进展\n\n目标：检验'+mode+'完整骨干的未见井效果。\n已做：结构与实验定义。\n计划：训练与恢复最佳权重、比较基线。\n未做：最终成绩和必要验证。\n完成后：仅按证据细化，不自动补其他折。\n',encoding='utf-8')
        argv=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_architecture_zoo_v15.py'),'--architecture',mode,'--out',str(out)]
        with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
            proc=subprocess.Popen(argv,cwd=str(ROOT),stdout=log,stderr=err,stdin=subprocess.DEVNULL,close_fds=True,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        record=dict(architecture=mode,pid=proc.pid,started_at=datetime.now().astimezone().isoformat(),command=argv)
        (out/'launch.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8');launches.append(record)
    note='\n\n## '+datetime.now().astimezone().isoformat()+' v15结构候选启动\n显式启动'+str([(x['architecture'],x['pid']) for x in launches])+'；只做既定三骨干内层SW比较，初始与最终分数待实际记录。无原第五折或其他通道模型更新，无自动队列。\n'
    for p in [STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md']:
        with p.open('a',encoding='utf-8') as f:f.write(note)
    print(json.dumps(launches,ensure_ascii=False))

if __name__=='__main__':main()
