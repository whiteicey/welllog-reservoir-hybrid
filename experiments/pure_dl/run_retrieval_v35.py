"""Explicit v35 training launcher. No queue or implicit restart."""
from pathlib import Path
from datetime import datetime
import argparse,ast,json,subprocess,sys
from run_cross_target_v34 import jobs

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/retrieval_density_v35_20260929'
EXPS=ROOT/'experiments/pure_dl'
OUT=ROOT/'outputs/pure_dl_retrieval_density_v35_SW_fold5'

def append(p,marker,text):
    old=p.read_text(encoding='utf-8-sig') if p.exists() else ''
    if marker not in old:
        with p.open('a',encoding='utf-8') as f:f.write('\n\n'+text+'\n')

def make_trainer():
    src=(EXPS/'train_crosscurve_v32.py').read_text(encoding='utf-8-sig')
    changes={
        'Matched redundant/cross-curve feature corrections on retained v31.':'Matched versus within-reference-well shuffled SW retrieval density on frozen v31.',
        "module_name='crosscurve_v32'":"module_name='retrieval_density_v35'",
        "choices=('POR','PERM','SW')":"choices=('SW',)",
        "('prepare_crosscurve_v32.py','por_reparam_v31.py'":"('prepare_retrieval_v35.py','crosscurve_v32.py','por_reparam_v31.py'",
        'feature_count=206,amplitude_side=63,extra_view_features=18,cached_views=2':'feature_count=206,amplitude_side=63,reference_memory_count=4096,neighbor_count=16',
        'v32_feature_head_plus_frozen_v31_dependencies':'v35_memory_density_plus_frozen_v31_dependencies',
        "model.extract(data['x'][ix],rel,valid,data['absolute'][ix])":"model.extract(data['x'][ix],rel,valid,data['absolute'][ix],rows=ix)",
        "assert all(v>0 for v in group_norms.values())":"assert all(np.isfinite(v) and v>=0 for v in group_norms.values())"}
    for old,new in changes.items():
        if src.count(old)!=1:raise RuntimeError('Immutable trainer template mismatch '+old)
        src=src.replace(old,new)
    ast.parse(src);path=EXPS/'train_retrieval_v35.py'
    if path.exists():assert path.read_text(encoding='utf-8')==src
    else:path.write_text(src,encoding='utf-8')
    return path

def main():
    p=argparse.ArgumentParser();p.add_argument('--start',action='store_true');p.add_argument('--status',action='store_true');a=p.parse_args()
    if a.status:
        for name in ['runtime_status.json','fold_5_report.json','failure.json']:
            file=OUT/name
            if file.exists():
                value=json.loads(file.read_text(encoding='utf-8'))
                if name=='fold_5_report.json':value={k:v for k,v in value.items() if k!='epoch_timing'}
                print(name,json.dumps(value,ensure_ascii=False))
        print('ACTIVE',json.dumps(jobs(),ensure_ascii=False));return
    if not a.start:raise ValueError('Explicit --start required')
    if len(jobs())>=2:raise RuntimeError('GPU max2')
    if OUT.exists():raise FileExistsError('No repeated launch or overwrite')
    manifest=json.loads((STUDY/'memory_manifest.json').read_text())
    assert manifest['whole_query_well_exclusion_verified'] and not manifest['validation_labels_used_for_memory']
    trainer=make_trainer()
    marker='## v35 跨井参考密度有限实验'
    note=marker+'\n当前继续优化，固定v31 TOTAL80.336513；仅SW matched/shuffled两个同容量检索密度分支，参考库4096个训练非默认点，每查询排除自身整井，固定Top16后学习重排序。原默认概率/骨干/有效头不变，参考条件密度混合比例0至0.25。原Q50/评分/80-18/max2保持，不补折、不扫描库大小或恢复旧路线。维护retrieval_density_v35_20260929/PROJECTS.md。'
    for file in [ROOT.parent/'AGENTS.md',ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md']:append(file,marker,note)
    append(ROOT/'EXPLORATIONS.md','retrieval_density_v35_20260929','- [v35 跨井参考密度](results/deep_learning/retrieval_density_v35_20260929/PROJECTS.md)：固定候选、学习注意力和参与比例，训练参考标签与同井内打乱对照；SW有限试验。')
    OUT.mkdir()
    for file in ['AGENTS.md','PROJECTS.md']:(OUT/file).write_text((STUDY/file).read_text(encoding='utf-8'),encoding='utf-8')
    (OUT/'README.md').write_text('# v35 SW检索密度\n\n参见results/deep_learning/retrieval_density_v35_20260929/METHOD.md。两个独立优化器和停止状态共享冻结前向，负对照不部署。实际结果尚未完成。\n',encoding='utf-8')
    command=[sys.executable,'-X','utf8','-u',str(trainer),'--target','SW','--out',str(OUT)]
    with (OUT/'train.log').open('wb') as log,(OUT/'train.err').open('wb') as err:
        process=subprocess.Popen(command,cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=log,stderr=err,close_fds=True,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=process.pid,started_at=datetime.now().astimezone().isoformat(),command=command,automatic_queue=False)
    (OUT/'launch.json').write_text(json.dumps(launch,ensure_ascii=False,indent=2),encoding='utf-8')
    for file in [STUDY/'PROJECTS.md',ROOT/'PROJECTS.md']:append(file,'## v35显式启动','## v35显式启动\n'+json.dumps(launch,ensure_ascii=False))
    print('LAUNCHED',json.dumps(launch,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
