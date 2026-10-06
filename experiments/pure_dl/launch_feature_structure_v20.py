"""Explicit bounded launcher; frozen v18 training logic adapted into new file."""
from pathlib import Path
from datetime import datetime
import argparse,ast,hashlib,json,subprocess,sys
from launch_latent_gate_v17 import active_jobs,append_once

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/feature_structure_v20_20260928'
MODES=('amplitude_early','shape_early','shape_late')
METHOD='''# v20：输入形态与视图融合位置

## 探索重点
原206维已有幅值、井排名、滚动残差和斜率。新增侧视图检验局部波动尺度标准化是否比直接幅值统计更适合POR；第二个对照检验两种输入先独立编码是否更好。原206维保持，绝对幅值不会被整体替换。

## 特征
九曲线GR/PE/SP/CAL/AC/DEN/CNL/RXO/RT；GR/RXO/RT用log1p，其他保留原单位。仅训练井拟合插补常数与缩放，缺失在各井输入内插值。局部窗口9/51/201，中心窗口不跨井。
幅值视图每曲线7维：[x,mean9,mean51,mean201,std9,std51,std201]。
形态视图每曲线7维：三种(x-mean_w)/d_w，(mean9-mean51)/d51，(mean51-mean201)/d201，(x-x_lag5)/d51，(x-x_lag20)/d201；d_w=sqrt(std_w^2+(0.05*training_IQR_curve)^2)。所有是输入派生量，无标签。变化量按点间隔，不是每米梯度。
两种侧视图均63维，分别只用训练点拟合中位数/IQR并截断到[-8,8]。旧206维来自未修改的v16缓存。近常量区间的尺度下限削弱严格仿射不变性，不能把整个模型称井间不变。

## 三组控制
amplitude_early：原206+相对深度主投影，与63维幅值侧投影先平均，再交给两个独立局部编码器。
shape_early：仅将侧输入换成63维形态，其他不变；与第一组比较特征表示。
shape_late：分别编码主视图和形态视图，之后再融合；与第二组比较融合位置。
所有模型两路80维，每个编码器2个dilation1/2的卷积块，融合后再经dilation4/8两块。融合权重sigmoid(Linear([hA,hB]))，逐点逐通道混合。各参数形状和初始化完全相同，所有参数实际参与前向和训练。
相同8分量Student-t3 POR头、原256格/800动作/单覆盖解码。其他目标不共享参数，固定PERM/SW；总分只通过原验证行序的实际预测数组核算。

## 训练及记录
复用冻结v18训练循环的新文件，Q=1，80/18/min10、bs64、64步/轮、AdamW6e-4/weight_decay0.01、cosine至1e-5、clip2。只用64训练井标签，原第五折用于早停与选择；三者从随机网络开始。完整原子状态可显式恢复，无自动队列。
与旧v5比较涉及预训练/视图/容量差异，不能当作单因素效应。三组的两个相邻比较才是预定义消融。原第五折已反复用于开发，得分及井重采样仅属开发证据。

## 收益和不继续边界
启动时未证实收益。预定义三项之后不追加网格、不补其他折、不覆盖交付。无增益保留80.274135；局部形态或晚融合负结果不推广为整个方法家族无效。当前没有独立盲测或完整五折结论。
'''

def replace_once(text,old,new):
    if text.count(old)!=1:raise RuntimeError('Frozen template mismatch: '+old[:100])
    return text.replace(old,new,1)

def trainer():
    src=ROOT/'experiments/pure_dl/train_por_architecture_v18.py'
    sha=hashlib.sha256(src.read_bytes()).hexdigest()
    if sha!='296079009a69ae04bfdfd55e1de75f36257beef5213cdc1ad41d6670d3def93a':raise RuntimeError('v18 source changed')
    text=src.read_text(encoding='utf-8-sig')
    edits=[
        ('Bounded original-fold5 POR architecture training, private networks only.','v20 input features and fusion location; three private POR candidates.'),
        ('from por_architecture_v18 import Network,MODES','from feature_structure_v20 import Network,MODES,load_feature_data'),
        ('from por_data_v18 import ROOT,STUDY,REFERENCE,load_data,Evaluator','from por_data_v18 import ROOT,STUDY as DATA_STUDY,REFERENCE,Evaluator\nSTUDY=ROOT/"results/deep_learning/feature_structure_v20_20260928"'),
        ('d=load_data();meta=torch.load(STUDY/', 'd=load_feature_data(args.mode);meta=torch.load(DATA_STUDY/'),
        ("(STUDY/'data_manifest.json').read_text","(DATA_STUDY/'data_manifest.json').read_text"),
        ("(STUDY/'data_manifest.json').read_bytes","(DATA_STUDY/'data_manifest.json').read_bytes"),
        ("Path(__file__).with_name('por_architecture_v18.py'),Path(__file__).with_name('por_data_v18.py')","Path(__file__).with_name('feature_structure_v20.py'),Path(__file__).with_name('prepare_feature_structure_v20.py'),Path(__file__).with_name('por_architecture_v18.py'),Path(__file__).with_name('por_data_v18.py')"),
        ("network_module='por_architecture_v18'","network_module='feature_structure_v20'"),
        ("width=model.width,latent_tokens=16 if args.mode=='latent64' else 0,attention_heads=4 if args.mode=='latent64' else 0,", "width=model.width,side_features=63,side_view='amplitude' if args.mode=='amplitude_early' else 'shape',\n        fusion='late' if args.mode=='shape_late' else 'early',\n        feature_manifest_sha256=hashlib.sha256((STUDY/'feature_manifest.json').read_bytes()).hexdigest(),"),
        ("parameter_count_approximately_matched=True,", "parameter_matched=True,initial_state_sha256=hashlib.sha256(b''.join(k.encode()+v.detach().cpu().numpy().tobytes() for k,v in model.state_dict().items())).hexdigest(),")]
    for old,new in edits:text=replace_once(text,old,new)
    dest=ROOT/'experiments/pure_dl/train_feature_structure_v20.py'
    ast.parse(text)
    if dest.exists():
        if dest.read_text(encoding='utf-8')!=text:raise RuntimeError('Existing new trainer changed')
    else:dest.write_text(text,encoding='utf-8')
    evidence=STUDY/'trainer_adaptation.json'
    if not evidence.exists():evidence.write_text(json.dumps(dict(parent=str(src.relative_to(ROOT)),parent_sha256=sha,changes=[dict(old=o,new=n) for o,n in edits]),ensure_ascii=False,indent=2),encoding='utf-8')
    return dest

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--launch',choices=MODES,nargs='+',required=True);a=ap.parse_args()
    if len(set(a.launch))!=len(a.launch):raise ValueError('Duplicate mode')
    jobs=active_jobs()
    if len(jobs)+len(a.launch)>2:raise RuntimeError('GPU limit; active='+str(jobs))
    if not (STUDY/'feature_manifest.json').exists():raise RuntimeError('Prepare side features first')
    for mode in a.launch:
        if (ROOT/'outputs'/f'pure_dl_feature_structure_v20_{mode}_fold5').exists():raise RuntimeError('Output exists: '+mode)
    dest=trainer();rules=(STUDY/'AGENTS.md').read_text(encoding='utf-8')
    if not (STUDY/'METHOD.md').exists():(STUDY/'METHOD.md').write_text(METHOD,encoding='utf-8')
    marker='## v20特征与结构联合探索'
    append_once(ROOT.parent/'AGENTS.md',marker,marker+'\n本批实施三项amplitude_early/shape_early/shape_late；63维幅值或局部标准化形态侧视图，同参数双分支，分开检验特征和融合位置。仅POR，原第五折/Q=1/80-18/max2，旧206输入保留且统计只用训练井。显式分批，无其他折或交付覆盖；维护feature_structure_v20_20260928/PROJECTS.md。')
    records=[]
    for mode in a.launch:
        out=ROOT/'outputs'/f'pure_dl_feature_structure_v20_{mode}_fold5';out.mkdir()
        for name,content in [('AGENTS.md',rules),('README.md',METHOD+'\n本项模式：'+mode+'。\n'),('PROJECTS.md','# 实验进度\n目标：'+mode+'的POR特征/结构验证。\n已做：数据/代码/协议。\n计划：训练和最佳恢复。\n未做：最终收益和重载。\n完成后：按预定对照比较，不自动补折。\n')]:
            (out/name).write_text(content,encoding='utf-8')
        cmd=[sys.executable,'-X','utf8','-u',str(dest),'--mode',mode,'--out',str(out)]
        with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,close_fds=True,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        row=dict(mode=mode,pid=proc.pid,started_at=datetime.now().astimezone().isoformat(),command=cmd)
        (out/'launch.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8');records.append(row)
        append_once(ROOT/'EXPLORATIONS.md',out.name,f'- [v20 {mode}](outputs/{out.name}/README.md)：输入表征与融合位置消融，初步收益未证实。')
        print('LAUNCHED',json.dumps(row,ensure_ascii=False),flush=True)
    entry='## '+datetime.now().astimezone().isoformat()+' v20显式启动\n'+json.dumps(records,ensure_ascii=False)+'\n预定义三项之一或两项，最多2GPU；未启动的项目不在自动队列，需本次显式调用。'
    for p in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md'):
        with p.open('a',encoding='utf-8') as f:f.write('\n\n'+entry+'\n')

if __name__=='__main__':main()
