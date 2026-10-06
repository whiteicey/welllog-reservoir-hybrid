"""Once per completed source pair, CPU-only provenance readback."""
from pathlib import Path
import argparse,json,hashlib,psutil,numpy as np,torch,time

S=Path(__file__).resolve().parent;R=S.parents[2];O=R/'outputs/pure_dl_two_seed_fullcv_v370'
ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,choices=[2,3,4],required=True);fold=ap.parse_args().fold
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()
dest=S/f'FOLD{fold}_SOURCE_PAIR_PROVENANCE.json';assert not dest.exists(),'Already recorded unchanged source pair'
keys=['feature_median','feature_scale','regression_median','regression_scale','absolute_depth_median','absolute_depth_scale','selected_feature_names']
sources={};normalizations={};descriptors={};rowsets={}
for identity,base in [('original',20261006),('seed_20261013',20261009)]:
    own=O/f'fold_{fold}'/identity/'SOURCE';report=read(own/f'fold_{fold}_report.json');timing=read(own/'SOURCE_TIMING.json')
    assert timing['terminal'] and timing['exit_code'] in (0,None) and not psutil.pid_exists(timing['pid'])
    assert report['best_checkpoint_predictions_reproduced'] and report['all_validation_rows_covered_once']
    master=torch.load(own/f'fold_{fold}.pt',map_location='cpu',weights_only=False)
    descriptor=torch.load(own/'prepared_data.pt',map_location='cpu',weights_only=False)
    assert master['config']['seed']==base and descriptor['fold']==fold
    assert master['best_epoch']==report['best_epoch']
    assert descriptor['qinfo']['epoch']<master['best_epoch']
    assert descriptor['source_sha256']==sha(own/f'fold_{fold}.pt')
    train=np.asarray(master['training_raw_rows']);validation=np.asarray(master['validation_raw_rows'])
    assert len(np.intersect1d(train,validation))==0
    assert len(validation)==report['validation_rows'] if 'validation_rows' in report else len(validation)>0
    normalizations[identity]={k:master[k] for k in keys}
    rowsets[identity]=(train,validation);descriptors[identity]=descriptor
    sources[identity]=dict(actual_seed=base+fold-1,config_base_seed=base,source_sha256=descriptor['source_sha256'],Q=descriptor['qinfo'],scores=report['scores'],best_epoch=report['best_epoch'],stopped_epoch=report['stopped_epoch'],process_wall_seconds=timing['process_wall_seconds'],concurrency=timing['concurrency'],validation_rows=len(validation),training_raw_rows=len(train),training_validation_raw_disjoint=True)
    del master
assert normalizations['original']==normalizations['seed_20261013']
assert all(np.array_equal(a,b) for a,b in zip(rowsets['original'],rowsets['seed_20261013']))
a,b=descriptors.values();assert a['shared_input_path']==b['shared_input_path'] and a['shared_input_sha256']==b['shared_input_sha256']
assert Path(a['shared_input_path'])==O/f'fold_{fold}'/'shared_input_data.pt'
assert sha(Path(a['shared_input_path']))==a['shared_input_sha256']
assert a['source_sha256']!=b['source_sha256']
result=dict(observed_epoch=time.time(),fold=fold,sources=sources,shared_input_same=True,shared_input_sha256=a['shared_input_sha256'],normalization_keys=keys,normalization_exact_between_sources=True,normalization_sha256=hashlib.sha256(json.dumps(normalizations['original'],sort_keys=True).encode('utf-8')).hexdigest(),same_training_validation_rows=True,different_learned_sources=True,own_Q_strictly_before_best=True,score_scope='SOURCE prerequisites only; not complete8stage model or fullOOF',CPU_only=True,solo_benchmark=False)
dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
note='\nV370 fold'+str(fold)+' source-pair provenance complete: same raw training/validation rows, disjoint training versus validation; identical train-only normalization and shared input hash, different own-source weights and private Q strictly before each source best epoch. Original SOURCE '+str(sources['original']['scores']['TOTAL'])+', seed13 SOURCE '+str(sources['seed_20261013']['scores']['TOTAL'])+'. These are prerequisites, not full8-stage models/fullOOF. Evidence '+dest.name+'. Pending downstream heads, uniform native fusion and full original OOF.\n'
for p in [S/'README.md',S/'PROJECTS.md',R/'PROJECTS.md',R.parent/'PROJECTS.md']:
    with p.open('a',encoding='utf-8') as f:f.write(note)
print(json.dumps(dict(fold=fold,sources=sources,normalization_exact=True,own_Q_before_best=True)))
