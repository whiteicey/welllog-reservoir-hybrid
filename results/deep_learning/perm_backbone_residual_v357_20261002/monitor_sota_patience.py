"""User-requested active-run 10/14 guard; no GPU, no restart or source mutation."""
from pathlib import Path
import json,time,psutil,os
R=Path(__file__).resolve().parents[3];S=Path(__file__).resolve().parent
O=R/'outputs/pure_dl_perm_backbone_residual_v357_fold5/local'
PID=46896;threshold=89.51838939879066;promoted=False
while True:
 try:
  process=psutil.Process(PID)
  live=any(Path(a).name=='train_perm_backbone_residual_v357.py' for a in process.cmdline())
 except psutil.Error:live=False
 try:state=json.loads((O/'runtime_status.json').read_text(encoding='utf-8-sig'))
 except (OSError,ValueError):
  if live:time.sleep(2);continue
  break
 promoted=promoted or float(state.get('best',state.get('target_score',-1e20)))>threshold
 patience=14 if promoted else 10
 evidence=dict(guard_pid=os.getpid(),training_pid=PID,training_live=live,threshold_target_SOTA=threshold,threshold_basis='PERM336 with fixedPOR341/SW351, equivalent to retainedassembledTOTAL80.706129723238',strict=True,SOTA_broken=promoted,effective_patience=patience,snapshot=state)
 (S/'ACTIVE_PATIENCE_CONTROL.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
 if not live or state.get('status','').startswith(('complete','failed','stopped')):break
 if int(state.get('epoch',0))>=10 and int(state.get('bad',0))>=patience:
  request=S/'local_early_stop_request.json'
  evidence['reason']='Latestuser patience10 untilstrictSOTA, then14; existing epoch-boundary request/bestrestoration'
  request.write_text(json.dumps(evidence,indent=2),encoding='utf-8')
  evidence['stop_request_written']=True
  (S/'ACTIVE_PATIENCE_CONTROL.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
  break
 time.sleep(2)
