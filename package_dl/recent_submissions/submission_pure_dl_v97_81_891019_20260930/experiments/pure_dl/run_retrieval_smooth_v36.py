"""Explicit launch/status for the two-arm smooth retrieval-gate repair."""
from pathlib import Path
from datetime import datetime
import argparse,json,subprocess,sys,psutil

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/retrieval_smooth_gate_v36_20260929'
OUT=ROOT/'outputs/pure_dl_retrieval_smooth_v36_SW_fold5'

def jobs():
    result=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            z=p.info;a=z.get('cmdline') or []
            if 'python' in (z.get('name') or '').lower() and '-c' not in a and any('pure_dl' in s and (Path(s.replace('\\','/')).name.startswith(('train_','benchmark_','verify_')) or 'auxiliary_crossfit_v34.py' in s) for s in a):result.append(z)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return result

def append(path,marker,text):
    path=Path(path);old=path.read_text(encoding='utf-8-sig') if path.exists() else ''
    if marker not in old:
        with path.open('a',encoding='utf-8') as f:f.write('\n\n'+text+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--launch',action='store_true');ap.add_argument('--status',action='store_true');args=ap.parse_args()
    if args.status:
        for n in ('runtime_status.json','failure.json','fold_5_report.json'):
            p=OUT/n
            if p.exists():
                d=json.loads(p.read_text(encoding='utf-8'))
                if n=='fold_5_report.json':d={k:v for k,v in d.items() if k!='epoch_timing'}
                print(n,json.dumps(d,ensure_ascii=False))
        print('ACTIVE',json.dumps(jobs(),ensure_ascii=False));return
    if not args.launch:ap.error('Explicit --launch or --status required')
    if jobs():raise RuntimeError('Existing training jobs; do not launch another unexamined job')
    if OUT.exists():raise FileExistsError('No duplicate or automatic resume '+str(OUT))
    prior=ROOT/'results/deep_learning/retrieval_density_v35_20260929'
    diagnostic=json.loads((prior/'gate_diagnostic.json').read_text(encoding='utf-8'))
    assert all(p['active_fraction']==0 and all(v==0 for v in p['gradients'].values()) for p in diagnostic['probes'] if p['state']=='last')
    marker='## v35门控失活诊断'
    note=marker+'\n'+diagnostic['observed_at']+'。已保存第18轮状态在64固定训练锚点7973有效点上，matched/shuffled门控proposal均100%负值，alpha与梯度全部为0；初始output梯度非零，但query/key零。18轮验证分数不变。不能将此结果当作充分训练后的检索负证据。旧源码/权重不改，新v36仅修复平滑门控通路。'
    for p in (prior/'PROJECTS.md',prior/'README.md'):append(p,marker,note)
    project='''# v36 进度与固定方案

## 目标
保留80.336513原模型，修复v35检索混合门控的负半轴失活，完成有效matched/shuffled对照。

## 已完成
v35已完成但新分支失活：7973个固定训练探针点的末态proposal全部负、alpha和梯度为0。保存gate_diagnostic.json，历史训练结果不改。v36模块和独立训练器已实现，复用合法v35参考库与索引，不重做检索准备。

## 本批设计
只把clamp(u,0,.25)改为.25*sigmoid(u+logit(.04))，初始参与比例1%。记录真实初始候选分，单独关闭检索校验父预测。查询/键/输出层需要非零梯度证据；训练后候选最高分与回退父模型分分别记录。两个候选均11266参数，相同初始化/库/索引/带宽/数据/损失/80-18预算，仅值与键对应不同。严格整口查询井排除，shuffled不部署。只在matched超过基线与shuffled时采用。

## 待完成
执行两项训练、最佳候选恢复、默认/数值段和参与比例诊断、实际三列评分。新采用模型才额外磁盘重建。未取得新分数前保持原配方。

## 后续与约束
不扫描初始参与比例、K、库规模、带宽或学习率。原其他折/旧任务/交付不改。维护独立README、根PROJECTS/EXPLORATIONS与图谱；区分v35失活与v36有效实验结果。无自动排队或续跑。
'''
    (STUDY/'PROJECTS.md').write_text(project,encoding='utf-8')
    (STUDY/'README.md').write_text(project,encoding='utf-8')
    scope='## v36检索门控修正版\n当前继续授权下，v35末态检索分支失活，保存证据不改历史。v36采用平滑1%起点、最大25%的matched/shuffled两个SW候选；原模型单独回退，不冒称初始函数相同。原数据/Q50/解码/80-18/max2/整井参考排除保持，无其他折或交付覆盖。见retrieval_smooth_gate_v36_20260929/PROJECTS.md。'
    for p in (ROOT.parent/'AGENTS.md',ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md'):append(p,'## v36检索门控修正版',scope)
    append(ROOT/'EXPLORATIONS.md','retrieval_smooth_gate_v36_20260929','- [v36 检索门控修正](results/deep_learning/retrieval_smooth_gate_v36_20260929/README.md)：修复v35全关闭门控，初始1%平滑混合，保留基线独立评分。')
    OUT.mkdir()
    for n in ('AGENTS.md','PROJECTS.md','README.md'):(OUT/n).write_text((STUDY/n).read_text(encoding='utf-8'),encoding='utf-8')
    command=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_retrieval_smooth_v36.py'),'--out',str(OUT)]
    with (OUT/'train.log').open('wb') as log,(OUT/'train.err').open('wb') as err:
        p=subprocess.Popen(command,cwd=str(ROOT),stdout=log,stderr=err,stdin=subprocess.DEVNULL,close_fds=True,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=p.pid,started_at=datetime.now().astimezone().isoformat(),command=command,automatic_queue=False)
    (OUT/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md'):append(path,'## v36显式启动','## v36显式启动\n'+json.dumps(launch,ensure_ascii=False))
    print('LAUNCHED',json.dumps(launch),flush=True)

if __name__=='__main__':main()
