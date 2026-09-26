"""Continue four folds only after verified screening; independently rescore all original OOF rows."""
import hashlib,importlib.util,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
out=Path('outputs/pure_dl_blur_ungated_sequence')
proof=json.loads((out/'screening_verification.json').read_text())
assert proof['passed'] and proof['first_fold_score']>82.3
assert shutil.disk_usage(out).free>5*1024**3
first={n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ['fold_1_pred.npy','fold_1.pt','fold_1_best_checkpoint.pt','config.json']}
(out/'continuation_first_fold_hashes.json').write_text(json.dumps(first,indent=2))
with (out/'continuation.log').open('w',encoding='utf-8') as log,(out/'continuation.err').open('w',encoding='utf-8') as error:
    process=subprocess.Popen([sys.executable,'-u','experiments/pure_dl/train_blur_ungated_sequence.py','--out',str(out),'--learning-rate','0.001','--schedule','cosine','--fold-count','5'],stdout=log,stderr=error)
    print('CONTINUATION_PID',process.pid,flush=True)
    result=process.wait()
assert result==0,'Continuation failed; inspect evidence, do not automatically restart'
assert all(hashlib.sha256((out/n).read_bytes()).hexdigest()==sha for n,sha in first.items())
subprocess.run([sys.executable,'experiments/pure_dl/verify_blur_ungated_sequence.py','--out',str(out),'--folds','5'],check=True)
import train_invariant as base
from sklearn.model_selection import GroupKFold
rec=base.load_split(base.ROOT.parent/'data','train')
truth=np.concatenate([frame.loc[np.isfinite(frame[base.TS].to_numpy(np.float64)).all(1),base.TS].to_numpy(np.float64) for _,frame in rec])
well=np.concatenate([np.full(np.isfinite(frame[base.TS].to_numpy(np.float64)).all(1).sum(),i) for i,(_,frame) in enumerate(rec)])
a=np.load(out/'oof.npz');pred=a['pred'];fold=a['fold']
assert truth.shape==pred.shape==(723556,3) and len(rec)==80
assert np.array_equal(a['truth'],truth) and np.array_equal(a['well'],well)
expected=np.full(len(truth),-1)
for f,(_,ix) in enumerate(GroupKFold(5).split(truth,groups=well)):expected[ix]=f
assert np.array_equal(fold,expected) and np.isfinite(pred).all()
assert (pred>=[0,.01,8.305]).all() and (pred<=[50,10000,99.9]).all()
spec=importlib.util.spec_from_file_location('original_metric',base.ROOT.parent/'catboost_hybrid_baseline/submission_code_sota/src/metrics.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
score=module.official_score(truth,pred)
report=json.loads((out/'report.json').read_text())
assert abs(score['TOTAL']-report['oof_scores']['TOTAL'])<1e-8
fold_scores=[]
for f in range(5):
    ix=fold==f;sc=module.official_score(truth[ix],pred[ix]);fr=report['folds'][f]
    assert abs(sc['TOTAL']-fr['scores']['TOTAL'])<1e-8
    assert np.array_equal(pred[ix],np.load(out/f'fold_{f+1}_pred.npy'))
    fold_scores.append(sc)
result={'passed':True,'complete_five_folds':True,'original_population_rows':len(truth),'original_wells':len(rec),'original_official_scores':score,'fold_scores':fold_scores,'screening_threshold':82.3,'first_fold_reused_unchanged':True,'goal_over82':score['TOTAL']>82,'selection':'Each fold selects its best official-validation checkpoint; validation is used for checkpoint selection'}
(out/'full_oof_original_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
