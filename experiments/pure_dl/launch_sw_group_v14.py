"""Explicit launch only; no queue and no automatic outer-fold escalation."""
from pathlib import Path
from datetime import datetime
import argparse,json,subprocess,sys
import psutil
from prepare_sw_group_v14 import ROOT,STUDY

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['inner','outer'],required=True);a=ap.parse_args()
    active=[]
    for proc in psutil.process_iter(['pid','name','cmdline']):
        try:
            d=proc.info;argv=d.get('cmdline') or []
            if 'python' in (d.get('name') or '').lower() and any('pure_dl/' in x.replace('\\','/') and Path(x.replace('\\','/')).name.startswith(('train_','benchmark_')) for x in argv):active.append(d['pid'])
        except(psutil.NoSuchProcess,psutil.AccessDenied):pass
    if active:raise RuntimeError('Two-run launch requires no existing GPU trainer: '+str(active))
    if not (STUDY/('data_'+a.phase+'.npz')).exists():raise RuntimeError('Prepare data first')
    outdirs=[ROOT/'outputs'/('pure_dl_sw_group_v14_'+a.phase+'_'+m+'80') for m in ['point','well']]
    for p in outdirs:
        if p.exists():raise RuntimeError('Do not overwrite '+str(p))
    readme='''# SW跨井泛化与井均衡对照v14

## 探索重点
原训练64井内部固定48训练/16监控，排除原第五折。检验训练内与未见井的差距，以及原按点平均损失是否受到井长主导。当前阶段不是原第五折实验，不给TOTAL，也不把内部监控称为独立盲测。

## 已经探索的内容与验证
两个候选使用相同TCN（82113参数）、初始化、随机窗口、学习率、原256格/800动作SW解码。point权重1；well权重N/(W*n_w)，N为训练有效点数、W为训练井数、n_w为该井有效点数。权重只依赖井样本数，均值1。两组均Q=1，不继承先前基于标签的质量诊断。全部预处理、支持格、标签先验仅由当前训练井拟合。先验初始化相同，权重只进入训练损失。
训练每轮同步用eval模式评价全部训练点与监控点，记录总SW、默认/非默认子集、逐井分数及BCE/条件CE。选模只用监控集按点官方SW，不用训练分数或井均分。max80/patience18/min10，bs64，64步，lr6e-4 cosine到1e-5，wd0.01。训练窗口未跨井，最后固定最佳参数复现预测。
原样本点均匀抽窗口，核心点重复覆盖及井边缘会使训练加权只是井均风险的近似，实际原SW评价不变。不宣称不同井标签应该一致。

## 取得的收益
启动时尚未证实。训练内得分和内层监控得分不能直接与原第五折SW73.622385比较，也不能组合出新的80.x总分。

## 不值得继续方向及边界
不复用看过内层监控井的预训练模型或旧Q，不重复增加attention宽度。只测试point/well两个预设模式，不扫描权重指数。内层候选若SW至少+0.05且至少9/16井提高，才考虑显式到原64井的匹配新训练；这不是自动授权补原其余4折。若无信号，不自动扩展训练。
'''
    (STUDY/'METHOD.md').write_text(readme,encoding='utf-8')
    (STUDY/'PROJECTS.md').write_text('# 目标与进展\n\n目标：定位未见井泛化瓶颈并检验井均衡监督。\n已完成：固定内层48/16切分与严格重拟合预处理；未用旧模型/Q/目标统计。\n计划：本批两组训练及训练/监控曲线比较。\n未完成：实测增益、原第五折评价、完整OOF>82。\n达成后：固定机制再依授权做原第五折比较；不自动补其他折或替换交付。\n',encoding='utf-8')
    rules=(STUDY/'AGENTS.md').read_text(encoding='utf-8');runs=[]
    for m,p in zip(['point','well'],outdirs):
        p.mkdir();(p/'AGENTS.md').write_text(rules,encoding='utf-8');(p/'README.md').write_text(readme+'\n本目录模式：'+m+'；阶段：'+a.phase+'。\n',encoding='utf-8')
        (p/'PROJECTS.md').write_text('# 进展\n\n目标：本模式训练与分组泛化诊断。\n已完成：来源和协议明确。\n计划：训练/监控轨迹与最终恢复。\n未完成：本模式分数和增益。\n达成后：与匹配对照比较，不自动补原折或改交付。\n',encoding='utf-8')
        argv=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_group_v14.py'),'--phase',a.phase,'--mode',m,'--out',str(p)]
        with (p/'train.log').open('wb') as log,(p/'train.err').open('wb') as err:
            child=subprocess.Popen(argv,cwd=str(ROOT),stdout=log,stderr=err,stdin=subprocess.DEVNULL,close_fds=True,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        run=dict(phase=a.phase,mode=m,pid=child.pid,started_at=datetime.now().astimezone().isoformat(),command=argv)
        (p/'launch.json').write_text(json.dumps(run,ensure_ascii=False,indent=2),encoding='utf-8');runs.append(run)
    note='\n\n## '+datetime.now().astimezone().isoformat()+' v14跨井诊断启动\n在继续优化授权下预设point/well两组，实际启动'+str([(x['mode'],x['pid']) for x in runs])+'。当前阶段'+a.phase+'；内层48训练/16监控，全部预处理重新拟合、Q=1、无旧权重。原第五折未用于当前内层训练或选模。真实收益待核实；不自动补折、旧交付不变。记录results/deep_learning/sw_group_generalization_v14_20260928。\n'
    for p in [STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md']:
        with p.open('a',encoding='utf-8') as f:f.write(note)
    with (ROOT/'EXPLORATIONS.md').open('a',encoding='utf-8') as f:f.write('\n- [v14训练井内部泛化与井均衡对照](results/deep_learning/sw_group_generalization_v14_20260928/METHOD.md)：严格内层48/16，原第五折未评估，收益待核实。\n')
    print(json.dumps(runs,ensure_ascii=False))

if __name__=='__main__':main()
