"""Construct the isolated v19 trainer from frozen v18 logic; explicitly launch two arms."""
from pathlib import Path
from datetime import datetime
import argparse,ast,hashlib,json,subprocess,sys
from launch_latent_gate_v17 import active_jobs,append_once

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/por_pyramid_v19_20260928'
MODES=('regional','global')
RULES='''# v19多尺度POR骨干：目标与约束

本批仅regional/global两项，使用完全相同的96维三尺度局部金字塔、跳接和可学习门控潜变量辅助路径。两者只改变读/处理/写注意力的区域或全窗口连通掩码；所有参数形状和初始化一致。区域组也能通过局部卷积、输入派生特征和多层通信获取较大上下文，不能称没有上下文。

原九曲线206特征+相对深度，窗口256、中央128监督，原第五折64/16井。复用v18的训练侧POR目标/变换元数据和v16输入缓存，Q=1，原8分量Student-t3头及256格/800动作单覆盖解码，不使用其他通道标签损失或旧网络权重。PERM/SW固定且独立。

normal CUDA，最多2任务，max80/patience18/min10，bs64、64步/轮、AdamW6e-4、wd.01、cosine至1e-5、clip2，按验证POR选择并恢复实际checkpoint。每轮原子保存模型/优化器/调度器/RNG和停止状态，显式恢复，无自动队列或补折。旧模型与交付不可覆盖。

维护PROJECTS.md，记录目标、已做、计划、未做和完成后选项；每批结束更新。独立README/EXPLORATIONS/图谱区分运行、单通道和实测总分；失败与负结果如实记录。相对v18不仅改结构也增加容量，仅regional/global是本批匹配对照。
'''
METHOD='''# v19：多尺度局部路径 + 门控记忆辅助

## 探索重点
v18约16万参数的POR模型非默认段泛化差较大。本轮检验两级下采样和高分辨率跳接是否提供更好的局部表征，并在这套固定结构上比较区域记忆和全窗口记忆。不给任何结构预先承诺收益。

## 结构
输入B×256×207，stem投影到96维。三个尺度长度256/128/64，各含两个dilation1/2的门控卷积；下采样为带有效掩码的2点平均。解码逐级最近邻上采样，与对应高分辨率特征拼接投影并残差相加，再用局部卷积细化。输出仍1门控+3×8的Student-t混合参数。
在64点粗尺度，额外读写16个64维潜变量；使用4头注意力和2层潜空间处理。辅助消息经64→96投影，由sigmoid(W_g LN(h)+b_g)逐点逐通道门控后加入粗尺度主路径。b_g=-2、W_g=0，起始门值约0.119；这是比例门控，不是对消息幅度的硬上界。
两组参数、初始state、训练窗口完全一致。regional中读取/写回只连接对应区域的粗尺度点，潜变量之间每层仅邻接通信；global读写/潜空间通信不作区域限制。密集注意力张量形状相同，但梯度和实际开销不承诺相同。

## 训练和评价
使用冻结v18训练循环的隔离副本，原文件不改。两模型全部参数从头训练，只读训练侧统计，不加载预训练网络。目标数据仍位于por_architecture_v18_20260928，输入缓存仍位于latent_fold5_v16_20260928；引用来源写入配置。
验证只在原144665点，按实际最佳checkpoint恢复。两个新POR磁盘重载后与既有PERM/SW预测按行序合并，实际调用原评分；不重复声称旧模型本轮再次GPU重载。

## 收益
启动时尚未证实。必须同时报告两新结构相互差异、相对v18和旧v5/v8的总分差；第0轮和更新后最佳分开。第五折已重复用于开发，不称独立泛化证明。

## 不继续和未做
仅本批两项；无宽度/学习率/种子网格，无第三组/其余四折/旧ABD重启。若两者均未超过旧配方，保留旧80.274135。不会把容量增长归因于单一记忆模块。尚未验证其他目标和完整OOF，严格80.5/>82目标保持。
'''


def replace_once(text,old,new):
    if text.count(old)!=1:raise RuntimeError('Unexpected source fragment: '+old[:100])
    return text.replace(old,new,1)


def make_trainer():
    parent=ROOT/'experiments/pure_dl/train_por_architecture_v18.py'
    text=parent.read_text(encoding='utf-8-sig')
    parent_hash=hashlib.sha256(parent.read_bytes()).hexdigest()
    expected='296079009a69ae04bfdfd55e1de75f36257beef5213cdc1ad41d6670d3def93a'
    if parent_hash!=expected:raise RuntimeError('Frozen v18 trainer changed; inspect before adapting')
    edits=[
        ('Bounded original-fold5 POR architecture training, private networks only.', 'v19 matched multiresolution POR communication experiment; private networks.'),
        ('from por_architecture_v18 import Network,MODES','from por_pyramid_v19 import Network,MODES'),
        ('from por_data_v18 import ROOT,STUDY,REFERENCE,load_data,Evaluator','from por_data_v18 import ROOT,STUDY as DATA_STUDY,REFERENCE,load_data,Evaluator\nSTUDY=ROOT/"results/deep_learning/por_pyramid_v19_20260928"'),
        ("torch.load(STUDY/'model_metadata.pt'","torch.load(DATA_STUDY/'model_metadata.pt'"),
        ("(STUDY/'data_manifest.json').read_text","(DATA_STUDY/'data_manifest.json').read_text"),
        ("(STUDY/'data_manifest.json').read_bytes","(DATA_STUDY/'data_manifest.json').read_bytes"),
        ("Path(__file__).with_name('por_architecture_v18.py'),Path(__file__).with_name('por_data_v18.py')","Path(__file__).with_name('por_pyramid_v19.py'),Path(__file__).with_name('por_architecture_v18.py'),Path(__file__).with_name('por_data_v18.py')"),
        ("network_module='por_architecture_v18'","network_module='por_pyramid_v19'"),
        ("width=model.width,latent_tokens=16 if args.mode=='latent64' else 0,attention_heads=4 if args.mode=='latent64' else 0,","width=model.width,latent_tokens=16,attention_heads=4,pyramid_lengths=[256,128,64],high_resolution_skips=2,\n        memory_connectivity=args.mode,data_study=str(DATA_STUDY.relative_to(ROOT)),"),
        ("parameter_count_approximately_matched=True,","parameter_matched=True,initial_state_sha256=hashlib.sha256(b''.join(k.encode()+v.detach().cpu().numpy().tobytes() for k,v in model.state_dict().items())).hexdigest(),")]
    for old,new in edits:text=replace_once(text,old,new)
    destination=ROOT/'experiments/pure_dl/train_por_pyramid_v19.py'
    if destination.exists():raise RuntimeError('Refuse overwrite trainer')
    ast.parse(text);destination.write_text(text,encoding='utf-8')
    (STUDY/'trainer_adaptation.json').write_text(json.dumps(dict(parent=str(parent.relative_to(ROOT)),parent_sha256=parent_hash,
        current=str(destination.relative_to(ROOT)),replacements=[dict(old=o,new=n) for o,n in edits],
        old_source_unchanged=True),ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--launch',nargs='+',choices=MODES,required=True);args=ap.parse_args()
    if len(set(args.launch))!=len(args.launch):raise ValueError('Duplicate mode')
    if active_jobs():raise RuntimeError('Existing training; no duplicate/competing launch')
    if len(args.launch)>2:raise RuntimeError('Maximum two jobs')
    if STUDY.exists():raise RuntimeError('New study already exists; inspect rather than overwrite')
    for mode in args.launch:
        if (ROOT/'outputs'/f'pure_dl_por_pyramid_v19_{mode}_fold5').exists():raise RuntimeError('Output exists')
    STUDY.mkdir(parents=True)
    (STUDY/'AGENTS.md').write_text(RULES,encoding='utf-8')
    (STUDY/'METHOD.md').write_text(METHOD,encoding='utf-8')
    (STUDY/'PROJECTS.md').write_text('# v19进度\n\n目标：多尺度POR表征及全局辅助路径对照。\n已做：规则/进程/原分数核对，新结构实现。\n计划：启动两组，恢复最佳、实际组合评分。\n未做：真实训练收益、最终重载、其他目标/折验证。\n完成后：保留有收益的开发配方；不自动补折，不覆盖交付。\n',encoding='utf-8')
    make_trainer()
    ast.parse((ROOT/'experiments/pure_dl/por_pyramid_v19.py').read_text(encoding='utf-8'))
    marker='## v19本轮POR多尺度与通信范围对照'
    append_once(ROOT.parent/'AGENTS.md',marker,marker+'\n当前继续的执行范围为regional/global两项，同一96维三尺度跳接骨干、同参数同初始状态，只改变辅助记忆的连通范围；保留原POR数据/头/评分/80-18/max2，无其他折或旧任务恢复。结果及未做项见por_pyramid_v19_20260928/PROJECTS.md。')
    records=[]
    for mode in args.launch:
        out=ROOT/'outputs'/f'pure_dl_por_pyramid_v19_{mode}_fold5';out.mkdir()
        for name,content in [('AGENTS.md',RULES),('README.md',METHOD+'\n本目录模式：'+mode+'\n'),('PROJECTS.md','# 进度\n目标：'+mode+'模式POR结构验证。\n已做：代码、来源与参数配置。\n计划：训练与最佳恢复。\n未做：收益和磁盘重载。\n达成后：比较同批对照和旧v5；不自动晋级。\n')]:
            (out/name).write_text(content,encoding='utf-8')
        command=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_por_pyramid_v19.py'),'--mode',mode,'--out',str(out)]
        with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
            proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,close_fds=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        item=dict(mode=mode,pid=proc.pid,started_at=datetime.now().astimezone().isoformat(),command=command)
        (out/'launch.json').write_text(json.dumps(item,ensure_ascii=False,indent=2),encoding='utf-8');records.append(item)
        print('LAUNCHED',json.dumps(item,ensure_ascii=False),flush=True)
        append_once(ROOT/'EXPLORATIONS.md',out.name,f'- [v19 POR {mode}](outputs/{out.name}/README.md)：同参数多尺度骨干，区域/全窗口辅助记忆；启动时收益未证实。')
    entry='## '+datetime.now().astimezone().isoformat()+' v19显式启动\n'+json.dumps(records,ensure_ascii=False)+'\n仅两项新POR训练，其他通道/折/交付保持不变；实际成绩待运行结果。'
    for p in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md']:
        with p.open('a',encoding='utf-8') as f:f.write('\n\n'+entry+'\n')

if __name__=='__main__':main()
