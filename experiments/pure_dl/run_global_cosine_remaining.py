"""Continue the SAME frozen configuration only after verified fifth-fold gate."""
import hashlib,json,re,subprocess,sys
from pathlib import Path
out=Path('outputs/pure_dl_global_cosine_sequence')
proof=json.loads((out/'screening_verification.json').read_text())
assert proof['passed'] and proof['screening_fold']==5 and proof['screening_fold_score']>79.8
assert json.loads((out/'global_point_table_verification.json').read_text())['passed']
cfg=json.loads((out/'config.json').read_text())
for name,digest in cfg['dependency_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
cached_hash=hashlib.sha256((out/'fold_5_pred.npy').read_bytes()).hexdigest()
with (out/'train_remaining.log').open('w',encoding='utf-8') as log,(out/'train_remaining.err').open('w',encoding='utf-8') as err:
    process=subprocess.Popen([sys.executable,'-u','experiments/pure_dl/train_global_cosine_sequence.py','--out',str(out),'--learning-rate','0.0006','--schedule','cosine','--fold-count','5'],stdout=log,stderr=err)
    print('REMAINING_TRAINING_PID',process.pid,flush=True)
    code=process.wait()
assert code==0,'Continuation failed; no automatic restart'
assert hashlib.sha256((out/'fold_5_pred.npy').read_bytes()).hexdigest()==cached_hash
assert json.loads((out/'config.json').read_text())==cfg
subprocess.run([sys.executable,'experiments/pure_dl/verify_global_cosine_sequence.py','--out',str(out),'--folds','5'],check=True)
# Record context from explicit fold-start markers; never attribute concatenated events to every fold.
logs={};current=None
for line in (out/'train_remaining.log').read_text(encoding='utf-8').splitlines():
    match=re.match(r'^FOLD ([1-5]) train-only preprocessing;',line)
    if match:current=int(match.group(1));assert current not in logs;logs[current]=[]
    if current is not None:logs[current].append(line)
assert set(logs)=={1,2,3,4}
for fold,lines in logs.items():
    (out/f'fold_{fold}_training.log').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    subprocess.run([sys.executable,'experiments/pure_dl/verify_t3_fold_table.py','--out',str(out),'--fold',str(fold)],check=True)
tables=[json.loads((out/f'fold_{f}_global_point_table_verification.json').read_text()) for f in [1,2,3,4]]
tables.append(dict(json.loads((out/'global_point_table_verification.json').read_text()),fold=5))
assert all(t['passed'] and t['unobserved_points']==0 for t in tables)
(out/'all_fold_tables_verification.json').write_text(json.dumps({'passed':True,'fifth_predictions_sha256_unchanged':cached_hash,'same_configuration':True,'folds':tables},indent=2),encoding='utf-8')
print('FULL_CV_AND_ALL_WEIGHT_TABLES_VERIFIED',flush=True)
