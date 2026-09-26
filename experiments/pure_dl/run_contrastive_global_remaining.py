"""Continue the SAME frozen configuration only after verified fifth-fold gate."""
import hashlib,json,re,subprocess,sys
from pathlib import Path
from fold5_screening_802 import qualifies, THRESHOLD

out=Path('outputs/pure_dl_contrastive_global_sequence')

def completed_fifth_preconditions():
    assert THRESHOLD==80.2
    required=['config.json','screening_report.json','screening_verification.json',
              'fifth_fold_verification.json','conflict_trace_verification.json',
              'global_point_table_verification.json','fold_5_report.json',
              'fold_5_validation_history.json','fold_5.pt','fold_5_best_checkpoint.pt',
              'fold_5_pred.npy','fold_5_distributions.npz','oof.npz']
    assert all((out/name).is_file() for name in required), 'Completed fifth-fold artifacts and ALL independent audits required before any launch'
    proof=json.loads((out/'screening_verification.json').read_text())
    assert proof['passed'] and proof['screening_fold']==5
    assert not proof['complete_five_folds'] and proof['qualifies_for_remaining_folds']
    assert proof['original_population_and_GroupKFold_verified']
    assert qualifies(proof['screening_fold_score']), 'Strict fifth-fold score >80.2 required'
    independent=json.loads((out/'fifth_fold_verification.json').read_text())
    assert independent['passed'] and not independent['complete_five_folds']
    assert len(independent['folds'])==1 and independent['folds'][0]['fold']==5
    assert independent['folds'][0]['same_audited_train_statistics']
    assert independent['folds'][0]['raw_loss_rows_exactly_training_fold']
    assert independent['folds'][0]['all_validation_rows_covered_once']
    point=json.loads((out/'global_point_table_verification.json').read_text())
    assert point['passed'] and point['channel_policy_replay_passed'] and point['unobserved_points']==0
    assert point['training_points']==point['evaluated_points']==578891
    trace=json.loads((out/'conflict_trace_verification.json').read_text())
    assert trace['passed'] and trace['fold']==5 and trace['exact_float32_policy_replay']
    assert trace['training_probe_rows_only'] and trace['default_channels_remain_one']
    assert trace['coverage']['probe_rows']==trace['coverage']['training_labeled_rows']==578891
    cfg=json.loads((out/'config.json').read_text())
    assert cfg['screening_threshold']==THRESHOLD and cfg['screening_fold']==5
    assert cfg['initial_lr']==.0006 and cfg['lr_schedule']=='cosine'
    assert cfg['epochs']==75 and cfg['seed']==20261006
    hashes=cfg['dependency_sha256']
    normalized_hash_names={Path(name).as_posix() for name in hashes}
    assert 'experiments/pure_dl/fold5_screening_802.py' in normalized_hash_names
    assert 'experiments/pure_dl/train_contrastive_global_sequence.py' in normalized_hash_names
    for name,digest in hashes.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest, name
    screen=json.loads((out/'screening_report.json').read_text())
    report=json.loads((out/'fold_5_report.json').read_text())
    history=json.loads((out/'fold_5_validation_history.json').read_text())
    assert screen['config']==cfg and not screen['complete_five_folds'] and screen['screening_fold']==5
    assert screen['screening_passed'] and qualifies(screen['screening_scores']['TOTAL'])
    assert abs(screen['screening_scores']['TOTAL']-proof['screening_fold_score'])<1e-8
    assert report['fold']==5 and report['best_checkpoint_predictions_reproduced']
    assert report['all_validation_rows_covered_once']
    assert abs(report['scores']['TOTAL']-proof['screening_fold_score'])<1e-8
    assert len(history)==report['stopped_epoch']
    assert history[-1]['epoch']==len(history)
    assert len(history)==75 or history[-1]['bad_epochs']>=18
    assert max(h['official_total'] for h in history)==report['scores']['TOTAL']
    assert history[-1]['best_epoch']==report['best_epoch']
    assert not (out/'report.json').exists(), 'Full run already exists; do not overwrite or restart'
    assert not any((out/f'fold_{f}_pred.npy').exists() for f in [1,2,3,4]), 'Partial continuation needs explicit recovery, not automatic restart'
    return cfg,hashlib.sha256((out/'fold_5_pred.npy').read_bytes()).hexdigest()

# This must complete before opening logs or creating ANY subprocess.
cfg,cached_hash=completed_fifth_preconditions()
with (out/'train_remaining.log').open('w',encoding='utf-8') as log,(out/'train_remaining.err').open('w',encoding='utf-8') as err:
    process=subprocess.Popen([sys.executable,'-u','experiments/pure_dl/train_contrastive_global_sequence.py','--out',str(out),'--learning-rate','0.0006','--schedule','cosine','--fold-count','5'],stdout=log,stderr=err)
    print('REMAINING_TRAINING_PID',process.pid,flush=True)
    code=process.wait()
assert code==0,'Continuation failed; no automatic restart'
assert hashlib.sha256((out/'fold_5_pred.npy').read_bytes()).hexdigest()==cached_hash
assert json.loads((out/'config.json').read_text())==cfg
subprocess.run([sys.executable,'experiments/pure_dl/verify_contrastive_global_sequence.py','--out',str(out),'--folds','5'],check=True)
full=json.loads((out/'full_oof_original_verification.json').read_text())
assert full['passed'] and full['complete_five_folds'] and full['rows']==723556 and full['wells']==80
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
assert hashlib.sha256((out/'fold_5_pred.npy').read_bytes()).hexdigest()==cached_hash
assert json.loads((out/'config.json').read_text())==cfg
for name,digest in cfg['dependency_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
print('FULL_CV_AND_ALL_WEIGHT_TABLES_VERIFIED',flush=True)
