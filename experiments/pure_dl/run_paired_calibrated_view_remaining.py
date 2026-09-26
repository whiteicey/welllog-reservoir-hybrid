"""Continue promoted calibrated75 trial unchanged; never reinterpret as a new80 run."""
import hashlib,json,re,subprocess,sys
from pathlib import Path
out=Path('outputs/pure_dl_paired_calibrated_view_sequence')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
proof=json.loads((out/'calibrated_promotion_verification.json').read_text())
assert proof['passed'] and proof['screening_fold_score']>80.2
assert proof['original_budget']==75 and proof['full_q_replay']
assert json.loads((out/'screening_verification.json').read_text())['qualifies_for_remaining_folds']
assert json.loads((out/'fold_5_global_point_table_verification.json').read_text())['passed']
cfg=json.loads((out/'config.json').read_text())
assert cfg['epochs']==75 and cfg['early_stopping']['maximum_epochs']==75
assert cfg['dependency_sha256']==proof['dependency_sha256']
for name,digest in cfg['dependency_sha256'].items():assert sha(Path(name))==digest
for name,digest in proof['protected_sha256'].items():assert sha(out/name)==digest
assert not any((out/f'fold_{f}_pred.npy').exists() for f in range(1,5)), 'Partial continuation requires manual diagnosis'
assert not (out/'report.json').exists()
with (out/'train_remaining.log').open('w',encoding='utf-8') as log,(out/'train_remaining.err').open('w',encoding='utf-8') as error:
    process=subprocess.Popen([sys.executable,'-u','experiments/pure_dl/train_paired_calibrated_view_sequence.py','--out',str(out),'--learning-rate','0.0006','--schedule','cosine','--fold-count','5'],stdout=log,stderr=error)
    print('REMAINING_TRAINING_PID',process.pid,flush=True)
    code=process.wait()
assert code==0,'Continuation failed; no automatic restart'
for name,digest in proof['protected_sha256'].items():assert sha(out/name)==digest
assert json.loads((out/'config.json').read_text())==cfg
logs={};current=None
for line in (out/'train_remaining.log').read_text(encoding='utf-8').splitlines():
    match=re.match(r'^FOLD ([1-5]) train-only preprocessing;',line)
    if match:
        current=int(match.group(1));assert current not in logs;logs[current]=[]
    if current is not None:logs[current].append(line)
assert set(logs)=={1,2,3,4}
for f,lines in logs.items():(out/f'fold_{f}_training.log').write_text('\n'.join(lines)+'\n',encoding='utf-8')
subprocess.run([sys.executable,'experiments/pure_dl/verify_paired_calibrated_view_promotion.py','--out',str(out),'--folds','5'],check=True)
full=json.loads((out/'full_oof_original_verification.json').read_text())
tables=[json.loads((out/f'fold_{f}_global_point_table_verification.json').read_text()) for f in range(1,6)]
assert full['passed'] and all(t['passed'] and t['unobserved_points']==0 for t in tables)
(out/'calibrated_continuation_verification.json').write_text(json.dumps({'passed':True,'original75_configuration_unchanged':True,'fifth_artifacts_sha256_unchanged':proof['protected_sha256'],'original_oof_scores':full['original_official_scores'],'goal_over82':full['goal_over82'],'tables':tables},indent=2),encoding='utf-8')
print('FULL_CV_VERIFIED; goal_over82=',full['goal_over82'],flush=True)
