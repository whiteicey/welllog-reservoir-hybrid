"""CPU-only inspect complete states and prepare explicit serialized recovery."""
from pathlib import Path
import hashlib,json,time,sys
import torch,psutil
ROOT=Path(__file__).resolve().parents[3];STUDY=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
state=read(STUDY/'BATCH_STATE.json');assert state['status']=='failed_or_stopped' and state['explicit_resume_only']
assert not psutil.pid_exists(state['controller_pid'])
prep=read(STUDY/'PREPARATION.json')
for p,digest in prep['code_sha256'].items():assert sha(ROOT/p)==digest,p
failed=[]
for identity,stage in (('original','PERM336'),('seed_20261013','PERM154')):
 out=ROOT/'outputs/pure_dl_two_seed_fullcv_v370/fold_1'/identity/stage
 runtime=read(out/'runtime_status.json');assert runtime['status']=='failed' and 'out of memory' in runtime['error']
 assert not psutil.pid_exists(runtime['pid']) and not (out/'report.json').exists()
 saved=torch.load(out/'last_state.pt',map_location='cpu',weights_only=False)
 required={'current','optimizer','scheduler','epoch','bad','best','best_epoch','best_state','best_pred','rng','windows','config','history'}
 assert required<=saved.keys();assert saved['config']==read(out/'config.json')
 assert saved['epoch']==saved['history'][-1]['epoch'] and saved['best_pred'].shape==(144621,)
 failed.append(dict(identity=identity,fold=1,stage=stage,old_pid=runtime['pid'],last_completed_epoch=saved['epoch'],
  bad=saved['bad'],best_epoch=saved['best_epoch'],best_score=saved['best'],last_state_path=str(out/'last_state.pt'),
  last_state_sha256=sha(out/'last_state.pt'),config_sha256=sha(out/'config.json'),
  full_optimizer_scheduler_RNG_window_states_present=True,
  command=['C:/Users/z5364/.conda/envs/normal/python.exe','-u','experiments/pure_dl/train_pair_fullcv_stage_v370.py','--stage',stage,'--resume']))
 del saved
result=dict(prepared_epoch=time.time(),authorization_pending=True,reason='Two PERM workers failed with CUDA OOM during validation',
 failed_jobs=failed,recovery_mode='one GPU task at a time; at least6GiB free before launch',
 no_training_source_or_recipe_changes=True,training_batch64_unchanged=True,validation_batch_unchanged=True,
 epoch_budget80_and_patience10_to14_unchanged=True,restore_complete_current_optimizer_scheduler_RNG_state=True,
 replays_only_work_after_last_complete_epoch=True,rerun_completed_stages=False,no_unattended_retry=True,
 user_confirmation_required_by='study AGENTS.md: 中止/失败不得自动重试或恢复',old_batch_state=state)
(STUDY/'OOM_RECOVERY_PREPARED.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
(STUDY/'OOM_FAILED_BATCH_SNAPSHOT.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(authorization_pending=True,failed_jobs=failed),ensure_ascii=False,indent=2))
