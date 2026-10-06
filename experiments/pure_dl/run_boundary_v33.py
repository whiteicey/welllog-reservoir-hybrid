"""Explicit two-target boundary experiment launcher; never queues or restarts."""
from pathlib import Path
from datetime import datetime
import argparse,ast,json,subprocess,sys
import psutil

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/boundary_context_v33_20260929'
EXPS=ROOT/'experiments/pure_dl'
TARGETS=('POR','SW')

def jobs():
    found=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            z=p.info;a=z.get('cmdline') or []
            if 'python' in (z.get('name') or '').lower() and '-c' not in a and any('pure_dl' in q and Path(q.replace('\\','/')).name.startswith(('train_','benchmark_','verify_')) for q in a):found.append(z)
        except(psutil.NoSuchProcess,psutil.AccessDenied):pass
    return found

def append_once(p,marker,text):
    p=Path(p);old=p.read_text(encoding='utf-8-sig') if p.exists() else ''
    if marker not in old:
        with p.open('a',encoding='utf-8') as f:f.write('\n\n'+text+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--targets',nargs='*',choices=TARGETS);ap.add_argument('--status',action='store_true');a=ap.parse_args()
    if a.status:
        for t in TARGETS:
            p=ROOT/'outputs'/f'pure_dl_boundary_v33_{t}_fold5'
            for n in ['runtime_status.json','fold_5_report.json','failure.json']:
                if (p/n).exists():
                    d=json.loads((p/n).read_text(encoding='utf-8'))
                    if n=='fold_5_report.json':d={k:v for k,v in d.items() if k!='epoch_timing'}
                    print(t,n,json.dumps(d,ensure_ascii=False))
        print('ACTIVE',json.dumps(jobs(),ensure_ascii=False));return
    if not a.targets:raise ValueError('Explicit --targets required')
    diagnostic=json.loads((STUDY/'diagnostic.json').read_text(encoding='utf-8'))
    for t in a.targets:
        s=diagnostic['summary'][t]
        assert s['numeric_near_score']<s['numeric_far_score'] and s['wells_with_higher_numeric_loss_near']>s['eligible_wells']/2
    current=jobs()
    if len(current)+len(a.targets)>2:raise RuntimeError('GPU task limit; existing '+str(current))
    src=(EXPS/'train_crosscurve_v32.py').read_text(encoding='utf-8-sig')
    replacements={
        'Matched redundant/cross-curve feature corrections on retained v31.':'Matched fixed51/boundary-truncated51 corrections on retained v31.',
        "module_name='crosscurve_v32'":"module_name='boundary_context_v33'",
        "('prepare_crosscurve_v32.py','por_reparam_v31.py'":"('prepare_boundary_v33.py','crosscurve_v32.py','por_reparam_v31.py'",
        "v32_feature_head_plus_frozen_v31_dependencies":"v33_boundary_head_plus_frozen_v31_dependencies"}
    for old,new in replacements.items():
        if src.count(old)!=1:raise RuntimeError('Trainer template mismatch '+old)
        src=src.replace(old,new)
    ast.parse(src)
    trainer=EXPS/'train_boundary_v33.py'
    if trainer.exists():
        assert trainer.read_text(encoding='utf-8')==src,'Do not mutate existing trainer'
    else:trainer.write_text(src,encoding='utf-8')
    marker='## v33 输入边界上下文有限对照'
    message=marker+'\n当前继续优化实施POR/SW各fixed/segmented共四项；PERM仅诊断不训练。冻结v31全预测器，原数据/Q50/评分/80-18/max2保持。相同小头、初始化和缩放，只改18维统计窗口是否跨输入边界。边界阈值训练输入拟合，非标签层位；无冷启动/扩容/其他折或交付覆盖。维护boundary_context_v33_20260929/PROJECTS.md。'
    for p in [ROOT.parent/'AGENTS.md',ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md']:
        append_once(p,marker,message)
    append_once(ROOT/'EXPLORATIONS.md','boundary_context_v33_20260929','- [v33 输入边界上下文](results/deep_learning/boundary_context_v33_20260929/README.md)：先输入边界失分诊断，再固定51点与截断统计有限对照。')
    readme='''# v33 输入边界上下文

## 探索重点
固定51点统计可能混合输入突变两侧信息。本批仅POR位置/SW分量权重的同容量小头对照，不修改已有269维输入路径或原骨干。

## 已探索与验证
读取当前v31配方并精确重算TOTAL80.336513436816；用训练输入q95阈值及固定11点峰间距定义边界。非默认POR近边界7.2157、远处42.9128；SW6.7080与17.6574，方向分别在14/14和12/13个有足够样本井上一致。属于开发诊断，不证明因果。边界/缩放没有使用标签。
两候选337->32->16，参数一致且同初始化；原父预测器冻结。fixed用普通51点mean/std，segmented用同一窗口截断到输入峰边界；两组共享缩放。相同Q50、loss、lr3e-4、80/pat18/min10、64批64步、原单覆盖解码。

## 收益
尚未完成新训练，不能将诊断低分或训练初始化当成改进。真实状态以runtime_status/fold_5_report为准，更新后的最好分与第0轮分开记录。

## 停止边界
本批只四项，不扫描阈值、窗口或曲线组合。若无超越当前预测的收益，停止此具体侧输入/头方案；不推论所有边界模型无效。无其他折或旧任务恢复，最终保持原交付。
'''
    append_once(STUDY/'README.md','# v33 输入边界上下文',readme)
    for t in a.targets:
        out=ROOT/'outputs'/f'pure_dl_boundary_v33_{t}_fold5'
        if out.exists():raise FileExistsError('Refuse duplicate launch '+str(out))
        out.mkdir()
        for name in ['AGENTS.md','PROJECTS.md']:(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
        (out/'README.md').write_text(readme+'\n本目录目标：'+t+'。\n',encoding='utf-8')
        args=[sys.executable,'-X','utf8','-u',str(trainer),'--target',t,'--out',str(out)]
        with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
            p=subprocess.Popen(args,cwd=str(ROOT),stdout=log,stderr=err,stdin=subprocess.DEVNULL,close_fds=True,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        info=dict(target=t,pid=p.pid,started_at=datetime.now().astimezone().isoformat(),command=args)
        (out/'launch.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        for path in [STUDY/'PROJECTS.md',ROOT/'PROJECTS.md']:
            append_once(path,'v33 launch '+t,'## v33 launch '+t+'\n'+json.dumps(info,ensure_ascii=False)+'\n无自动排队或补折。')
        print('LAUNCHED',json.dumps(info,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
