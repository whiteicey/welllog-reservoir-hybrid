"""Prepare six private model candidates; launches only explicitly supplied run IDs."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import argparse,ast,json,subprocess,sys
import psutil

CODE=Path(__file__).resolve().parent
ROOT=CODE.parents[1]
STUDY=ROOT/'results/deep_learning/independent_channels_v8_20260928'
RUNS={t+'_'+a:(t,a) for t in ('POR','PERM','SW') for a in ('dual','single')}
PREFIX='pure_dl_independent_v8_'
PLAN='''# 三通道独立模型：全骨干单任务优化

## 目标与新授权
用户明确允许三个通道采用不同模型、分别训练，不要求共享backbone。本批解除此前只训POR/冻结骨干的限制：POR、PERM、SW分别拥有独立实例、可训练特征提取器、门控、密度头、优化器、停止计数和checkpoint。允许最终选择不同结构，旧实验和已交付版本保持不变。

## 六个预定义候选
对每个目标分别比较：dual保留完整206维视图及校准68维视图两条私有多尺度U-Net+局部/全局注意力骨干，并训练私有输入路由；single仅保留完整206维视图的一条私有骨干，不使用第二专家或路由。共3目标×2结构，不再扫描宽度/种子。三个目标的模型没有共享Parameter、优化器或可复用的训练中隐藏缓存。
初始化来自同一训练折的既有多任务预训练权重和v5读出。这是独立单任务微调，不声称全流程从零训练时也完全无多任务预训练。dual初始应逐元素复现旧v5对应通道；single起点是原第一专家，需单独报告，不能冒称和dual初始函数相同。

## 训练协议
所有有效特征提取参数都能更新，包括stem、编码/解码、局部/全局attention；dual的校准和路由也更新。只有当前目标的门控和密度输出行可训练，其他输出行保留为冻结buffer以匹配原卷积计算，不计算其他目标的损失。只需一个目标的预测分布；无跨目标梯度、无外部B/D融合。
骨干/路由lr1e-5，目标头lr3e-5，AdamW、wd0.01、余弦80×64步至1e-6、clip2。原batch64、64步/轮、max80/patience18/min10及seed20261010；按自身通道官方分选最佳epoch（允许0）。骨干训练时启用原dropout，所有模型使用各自独立训练状态。
原九曲线/206特征、原64训练井16验证井、578891训练点144665验证点、Q50和单覆盖256格800动作保持。共享相同的确定性预处理定义不等于共享可学习骨干。每项目标用自身BCE+非默认NLL，POR Student-t3，PERM/SW Gaussian；除3保留历史loss尺度。标签监督只读当前通道。

## 收益与验证
启动前没有结果。每项报告自己的通道分，以及将该通道替换进旧参考时的诊断总分；后者不得当完整独立三模型成绩。六项完成后为每个目标固定一种模型，三份完整checkpoint真实重载推理，合并同一144665点再计算官方TOTAL。
正式训练检查初始、骨干梯度、实际骨干权重变化、最终最佳恢复和冻结packed行；不另开GPU预检任务。保存全模型current/best、optimizer/scheduler/RNG/停止状态，只有显式resume。首次训练真实错误单独记录，旧源码不改。

## 边界与成本
最多2GPU任务，显式分批启动，不设置自动队列；本批最多六项。未超过80.5也可记录单通道结构收益，但不补折；完整OOF>82仍未完成。单任务训练相对共享模型也改变了更新范围和总预算，因此不能将差异都归因为消除梯度冲突。不同结构/目标独立选型属于开发选择，需保留跨折确认限制。
最终3个模型需要3到6条骨干，计算和储存可能高于原共享两骨干；不把精度进展与推理成本混为一谈。无新数据、测试标签或旧ABD恢复，交付不覆盖。每批维护PROJECTS与演进图谱。
'''


def trainers():
    found=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            d=p.info;args=d.get('cmdline') or []
            if 'python' in (d.get('name') or '').lower() and 'pure_dl' in ' '.join(args) and any(Path(x.replace('\\','/')).name.startswith(('train_','benchmark_')) for x in args):found.append(d)
        except(psutil.NoSuchProcess,psutil.AccessDenied):pass
    return found


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--launch',nargs='*',choices=list(RUNS),default=[]);a=ap.parse_args()
    if len(a.launch)!=len(set(a.launch)):raise ValueError('Duplicate run')
    if a.prepare:
        if STUDY.exists():raise RuntimeError('Study already exists')
        for script in ('independent_channels_v8.py','train_independent_channels_v8.py'):
            ast.parse((CODE/script).read_text(encoding='utf-8'))
        if any((ROOT/'outputs'/(PREFIX+k+'80_fold5')).exists() for k in RUNS):raise RuntimeError('Existing outputs')
        STUDY.mkdir()
        agents='# 目标和约束\n三通道独立可训练骨干和读出，分别选型；保留原数据/分折/评分，最多2GPU。维护PROJECTS.md：目标、已做、计划、未做、达成后的工作，每批更新。模型检查点和旧交付不覆盖；不自动补折或重启旧任务。\n'
        projects='# 进度\n目标：独立单任务模型，按通道选择有效结构。\n已做：单/双私有骨干实现，六项有限对照已固定。\n计划：分批训练、实际骨干梯度与最佳恢复、固定三模型组装验证。\n未做：尚无本批精度收益或独立泛化结果，续跑等价性未实测。\n完成目标后：比较模型成本并在固定配方下验证原分组泛化；当前不补折。\n'
        for p in [STUDY]+[ROOT/'outputs'/(PREFIX+k+'80_fold5') for k in RUNS]:
            if p!=STUDY:p.mkdir(exist_ok=False)
            for name,text in [('README.md',PLAN),('AGENTS.md',agents),('PROJECTS.md',projects)]:
                (p/name).write_text(text,encoding='utf-8')
        for key,(t,arch) in RUNS.items():
            (ROOT/'outputs'/(PREFIX+key+'80_fold5')/'runtime_status.json').write_text(json.dumps({'status':'planned','target':t,'architecture':arch,'atlas_current_trial':False}),encoding='utf-8')
        rule='\n\n## 2026-09-28 最新授权：通道独立模型，无共享骨干\n用户允许不同模型分别预测与训练POR/PERM/SW，无须共享backbone。覆盖此前仅POR/骨干冻结限制。本批六项：各目标single完整视图和dual双视图私有模型；全骨干、当前目标门控/密度和私有路由可训练，各自优化器/早停/完整checkpoint，最后固定三模型共同验证。初始化来自原训练折预训练，不声称从零。原9曲线/分折/评分/80-18/max2保持，新低学习率全骨干微调配置须记录。无额外数据/自动补折/旧任务或交付覆盖，维护PROJECTS与图谱。\n'
        for p in [ROOT.parent/'AGENTS.md',ROOT/'AGENTS.md',ROOT/'experiments/AGENTS.md',CODE/'AGENTS.md']:
            with p.open('a',encoding='utf-8') as f:f.write(rule)
        with (ROOT/'EXPLORATIONS.md').open('a',encoding='utf-8') as f:
            for key in RUNS:f.write('\n- [独立通道模型v8 '+key+'](outputs/'+PREFIX+key+'80_fold5/README.md)：全私有骨干单任务训练，结果待验证。\n')
    if not STUDY.exists():raise RuntimeError('Prepare first')
    if len(trainers())+len(a.launch)>2:raise RuntimeError('Two-process GPU limit exceeded')
    for key in a.launch:
        out=ROOT/'outputs'/(PREFIX+key+'80_fold5')
        if (out/'config.json').exists() or (out/'launch.json').exists():raise RuntimeError('Refuse duplicate launch '+key)
    launches=[]
    for key in a.launch:
        target,architecture=RUNS[key];out=ROOT/'outputs'/(PREFIX+key+'80_fold5')
        cmd=[sys.executable,'-u',(CODE/'train_independent_channels_v8.py').as_posix(),'--target',target,'--architecture',architecture,'--out',out.as_posix()]
        with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
            p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=err,stdin=subprocess.DEVNULL,close_fds=True,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        row={'run':key,'target':target,'architecture':architecture,'pid':p.pid,'started_at':datetime.now().astimezone().isoformat(),'command':cmd}
        (out/'launch.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
        launches.append(row);print('LAUNCHED',json.dumps(row,ensure_ascii=False),flush=True)
    if launches:
        old=json.loads((STUDY/'launches.json').read_text()) if (STUDY/'launches.json').exists() else []
        (STUDY/'launches.json').write_text(json.dumps(old+launches,ensure_ascii=False,indent=2),encoding='utf-8')
        text='\n\n## '+datetime.now().astimezone().isoformat()+' 独立通道模型v8\n六项预定义单/双骨干候选；本次实际启动'+str([(v['run'],v['pid']) for v in launches])+'，各自端到端单通道训练。尚无本批已验证精度；保持原评分/分折，不自动补折或改变旧交付。计划和证据：results/deep_learning/independent_channels_v8_20260928。\n'
        for p in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md']:
            with p.open('a',encoding='utf-8') as f:f.write(text)

if __name__=='__main__':main()
