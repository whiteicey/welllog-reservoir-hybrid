"""Prepare and explicitly launch a bounded three-arm complex-head study.
No queue, auto-resume, previous experiment restart or remaining-fold continuation.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,subprocess,sys
from pathlib import Path
from datetime import datetime
import psutil

CODE=Path(__file__).resolve().parent
ROOT=CODE.parents[1]
STUDY=ROOT/'results/deep_learning/dual8222_attention_v6_20260928'
PREFIX='pure_dl_dual8222_v6_'
MODES=['linear','deep_mlp','attention']
PLAN='''# 复杂预测头：两层深MLP与局部多头注意力

## 探索重点
用户授权继续优化预测头，允许增加结构复杂度和attention。本批在已验证80.274135337791双专家上测试三个固定对照；先集中在有正面证据的POR位置读出，保持PERM/SW、门控、混合权重、尺度及原两专家不变。

## 预先定义的结构与比较
1. linear：仅原线性位置参数继续训练，2064参数。旧v5非线性支路固定。
2. deep_mlp：同一线性位置更新，加联合两专家的256维特征->64维->两个残差块->16维修正；每块含两个64->128->64逐点MLP及PreNorm。86496可训练参数。
3. attention：保持上述投影、层数、宽度和FFN；每块将第一个逐点MLP换成4头、每头16维的局部自注意力，允许前后16点，共33点，含相对位置偏置。86888参数；比deep_mlp多392（约0.45%），近参数预算但不宣称严格等参数。
两种复杂头使用相同基础投影/FFN初始化、输出末层为零，初始完整预测应同80.274135。输入为同一查询井内两位内部专家的隐藏特征，不引入外部模型、参考井或标签。
注意力屏蔽无效key；无效query只保留自身dummy key防止全遮挡softmax，然后输出清零。两个局部层可通过中间点形成更大有效上下文，但不跨井。显式float32 attention、无dropout。
新增修正限制为0.25倍固定分量标准差；只约束新增分支，共同线性位置更新不限幅。保留原v5冻结非线性修正，所有模型从完全相同的已存权重起步。

## 训练和验证协议
原九曲线/206特征、原第五折/seed20261010/训练Q50、单覆盖256格800动作解码保持。损失继续POR双专家BCE+非默认NLL再除3；本批只检验结构，不更换损失。
共同线性位置lr3e-5；新随机初始化深头lr3e-4，deep_mlp与attention相同；两组均cosine80*64至1e-5、wd.01、clip2、batch64、64步/轮、max80/patience18/min10。优化器重建，是新热启动实验而非原轨迹精确续跑。窗口使用独立CUDA生成器。
初始完整预测对齐、首批梯度范围和最终最佳恢复检查在正式训练中执行。不另开预检/代码审查训练。保存current heads/optimizer/scheduler/RNG/best/history，只有显式--resume才恢复。续跑等价性尚未实测。
最多2GPU；三个固定对照分批明确启动，无自动队列。全部完成后核对保存重载和注意力实际学习，再写结论。没有旧ABD重启、补折或交付覆盖。

## 收益状态与关闭依据
准备时无训练成绩。初始80.274135不是本批收益；增加训练预算需由linear控制，attention的收益须与近容量deep_mlp比较。无改进时保留原模型，不把增加复杂度直接等同于进步。完整OOF>82与第五折>80.5尚未达成。
本批只固定上述三组，不按早期验证结果追加宽度/头数/半径/LR搜索。若attention无新增收益，记录局部attention在本设置的限制，不泛化成attention普遍无效。
'''


def trainers():
    rows=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            d=p.info;a=d.get('cmdline') or []
            if 'python' in (d.get('name') or '').lower() and 'pure_dl' in ' '.join(a) and any(Path(x.replace('\\','/')).name.startswith(('train_','benchmark_')) for x in a):rows.append(d)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return rows


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--launch',choices=MODES,nargs='*',default=[]);args=ap.parse_args()
    if len(set(args.launch))!=len(args.launch):raise ValueError('Duplicate mode')
    target=CODE/'train_dual8222_attention_v6.py'
    if args.prepare:
        if STUDY.exists() or target.exists():raise RuntimeError('Study exists; refuse overwrite')
        source=CODE/'train_dual8222_location_v5.py'
        text=source.read_text(encoding='utf-8-sig')
        edits=[
            ('from dual8222_location_readout_v5 import Network,MODES','from dual8222_attention_head_v6 import Network,MODES',1),
            ("SELECTION=ROOT/'outputs/pure_dl_dual8222_v4_por_mean_only80_fold5'","SELECTION=ROOT/'outputs/pure_dl_dual8222_v5_location_point3280_fold5'",1),
            ("selected_path=SELECTION/'selected_heads.pt'","selected_path=SELECTION/'readout_checkpoint.pt'",1),
            ('80.26873450885158','80.27413533779057',1),
            ('POR location-readout comparison from the verified 80.268735 recipe.','Complex/attention readout comparison from verified80.274135; original factors frozen.',1),
            ("Path(__file__).with_name('dual8222_location_readout_v5.py'),", "Path(__file__).with_name('dual8222_attention_head_v6.py'), Path(__file__).with_name('dual8222_location_readout_v5.py'), Path(__file__).with_name('train_dual8222_location_v5.py'),",1),
            ("'location_readout':'linear base trainable in all arms; optional bounded32wide residual'","'location_readout':'old v5 residual frozen;common linear locations;new2block64wide joint readout',\n         'new_readout_learning_rate':3e-4,'new_readout_blocks':2,'head_width':64,'attention_heads':4,'attention_radius':16,\n         'attention_implementation':'explicit float32 dot products,relative bias,padding mask,no dropout',\n         'torch_version':str(torch.__version__)",1),
            ("'local_kernel':5 if args.mode=='local32' else 1","'local_kernel':None",1),
            ("'comparison':'same initial80.268735; same linear parameters trainable; point32 versus local32 differs only added192per-expert context parameters'","'comparison':'same80.274135;linear budget control;deepMLP86496 versus attention86888;392parameter difference'",1),
            ('    optimizer=torch.optim.AdamW(params,lr=3e-5,weight_decay=.01)',
             "    common=[p for n,p in model.named_parameters() if p.requires_grad and '.rich_location.' not in n]\n    rich=[p for n,p in model.named_parameters() if p.requires_grad and '.rich_location.' in n]\n    groups=[{'params':common,'lr':3e-5}]\n    if rich:groups.append({'params':rich,'lr':3e-4})\n    optimizer=torch.optim.AdamW(groups,weight_decay=.01)",1),
            ("'next_lr':optimizer.param_groups[0]['lr'],'seconds':time.perf_counter()-t","'next_lr':optimizer.param_groups[0]['lr'],'all_learning_rates':[g['lr'] for g in optimizer.param_groups],'seconds':time.perf_counter()-t",1),
            ("'warm_start':'v4_mean_only@12'","'warm_start':'v5_point32@6'",1),
        ]
        for old,new,n in edits:
            if text.count(old)!=n:raise RuntimeError('Unexpected parent content for '+old)
            text=text.replace(old,new)
        ast.parse(text);ast.parse((CODE/'dual8222_attention_head_v6.py').read_text())
        target.write_text(text,encoding='utf-8')
        STUDY.mkdir()
        (STUDY/'trainer_derivation.json').write_text(json.dumps({'source':source.name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'target':target.name,'edits':edits},ensure_ascii=False,indent=2),encoding='utf-8')
        agents='# 目标与约束\n本批比较复杂POR预测头与注意力，保持双专家/原数据/评分及2GPU上限。维护PROJECTS.md：目标、已做、计划、未做和达成后工作，每批更新。历史源码/模型/交付不改；无自动队列/补折。\n'
        project='# 项目进度\n目标：检验复杂预测头的真实结构增量。\n已做：核对80.274135来源；预定义三组结构、训练合同和记录入口。\n计划：正式训练、最佳恢复、按井分解和胜出权重重载；核对attention有效性。\n未做：尚无本批训练收益、完整OOF提升或中断续跑等价性证据。\n达成后：固定结构与通道策略评估泛化；当前无自动补折。\n'
        for p in [STUDY]+[ROOT/'outputs'/(PREFIX+m+'80_fold5') for m in MODES]:
            if p!=STUDY:p.mkdir(exist_ok=False)
            for fn,value in [('AGENTS.md',agents),('PROJECTS.md',project),('README.md',PLAN)]:
                (p/fn).write_text(value,encoding='utf-8')
        for m in MODES:
            (ROOT/'outputs'/(PREFIX+m+'80_fold5')/'runtime_status.json').write_text(json.dumps({'status':'planned','mode':m,'atlas_current_trial':False}),encoding='utf-8')
        rule='\n\n## 2026-09-28 用户授权复杂预测头和attention\n在当前双专家80.274135基础上，本批预定义linear/deep_mlp/attention三个头对照；先优化POR位置，PERM/SW/骨干/旧残差/门控/混合尺度固定。新复杂头可使用同井两专家联合隐藏特征和两层四头局部attention，64维、33点窗口。共同线性lr3e-5，新深头lr3e-4；其余原数据/评分/Q/80-18/最多2GPU不变。三组同起点，明确分批启动，无旧路线或补折；维护PROJECTS、独立记录与图谱，不宣称复杂度本身有收益。\n'
        for p in [ROOT.parent/'AGENTS.md',ROOT/'AGENTS.md',CODE/'AGENTS.md']:
            with p.open('a',encoding='utf-8') as f:f.write(rule)
        with (ROOT/'EXPLORATIONS.md').open('a',encoding='utf-8') as f:
            for m in MODES:f.write('\n- [复杂预测头v6 '+m+'](outputs/'+PREFIX+m+'80_fold5/README.md)：同80.274135起点；正式结果待验证。\n')
    if not STUDY.exists():raise RuntimeError('Prepare first')
    active=trainers()
    if len(active)+len(args.launch)>2:raise RuntimeError('Too many GPU trainers: '+str(active))
    for m in args.launch:
        out=ROOT/'outputs'/(PREFIX+m+'80_fold5')
        if (out/'config.json').exists() or (out/'launch.json').exists():raise RuntimeError('Duplicate launch '+m)
    launches=[]
    for m in args.launch:
        out=ROOT/'outputs'/(PREFIX+m+'80_fold5')
        cmd=[sys.executable,'-u',target.as_posix(),'--out',out.as_posix(),'--mode',m]
        with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
            p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=err,stdin=subprocess.DEVNULL,close_fds=True,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        row={'mode':m,'pid':p.pid,'command':cmd,'started_at':datetime.now().astimezone().isoformat()}
        (out/'launch.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
        launches.append(row);print('LAUNCHED',json.dumps(row,ensure_ascii=False),flush=True)
    if launches:
        old=json.loads((STUDY/'launches.json').read_text()) if (STUDY/'launches.json').exists() else []
        (STUDY/'launches.json').write_text(json.dumps(old+launches,ensure_ascii=False,indent=2),encoding='utf-8')
        note='\n\n## '+datetime.now().astimezone().isoformat()+' 复杂预测头v6\n预定义linear/deep_mlp/attention三项；本次明确启动'+str([(a['mode'],a['pid']) for a in launches])+'。同80.274135、原双专家和评分；真实初始/训练/收益待日志。无旧路线或补折，目标仍未完成。详见results/deep_learning/dual8222_attention_v6_20260928。\n'
        for p in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md']:
            with p.open('a',encoding='utf-8') as f:f.write(note)

if __name__=='__main__':main()
