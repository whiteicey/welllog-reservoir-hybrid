"""Explicitly start only the two declared v22 candidates; never queue restarts."""
from pathlib import Path
from datetime import datetime
import ast, hashlib, json, subprocess, sys
from launch_latent_gate_v17 import active_jobs, append_once

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/frozen_growth_v22_20260928'
MODES=('ffn2','ffn4')
METHOD='''# v22: only train newly added capacity

## Question
Local v21 already completed all-parameter POR finetuning at base1/ffn2/ffn4. Best post-update TOTAL was 80.195914612/80.195127649/80.189160920; every run selected epoch0. Current verified reference is still80.274135337791.
This batch keeps the exact expanded v21 architecture but freezes all original1,290,490 parameters. Only additional FFN/position-head parameters can update. The two new runs test whether original-weight drift is necessary for the observed degradation and whether additional capacity alone helps. The base1 model has no additional parameters, so its fixed prediction is the no-update reference; it is not presented as a separately trained new experiment.

## Capacity and initialization
18 blocks grow internal nonlinear units by2x/4x; the small POR location residual grows32 to64/128. Total parameters are1,981,498 and3,363,514; trainable added parameters691,008 and2,073,024. External channels, attention, views and routing unchanged. Both expanded sites change together. The user-visible prior ZIP proposed FFN-only expansion; this study deliberately reuses the actually executed local v21 model (including small-head growth) for the update-scope comparison.
All original state is loaded from the v5/v8 champion, added output projections are zero. Training must first reproduce all144665 original POR predictions exactly. No v21 finetuned weight is used. Frozen original parameters can operate on changed activations; output mixture scales/default probabilities need not stay numerically fixed.

## Matching
Reuse exact v21 cached features, labels, Q50, original64/16wells, seeds, objective and singlecoverage256bins/800actions decoder. Same train/eval module behavior and AdamW settings: new body1e-5,new head3e-5,wd.01,clip2,cosine80x64 to1e-6,bs64,64steps/epoch,max80,pat18,min10. Optimizer contains only trainable new parameters. No no_grad barrier across frozen original layers. Initialization, architecture, input and loss are matched to each corresponding v21 experiment; trainer bookkeeping is newly implemented and source provenance recorded. No claim of bitwise training reproducibility.

## Evidence and stopping
Save both selected checkpoint (may beepoch0) and best checkpoint after actual updates, their predictions, full resumable optimizer/scheduler/RNG/current state. Check old parameters before and after restoration, and verify added gradients/parameter changes. Final scores combine actual POR arrays with unchanged verified PERM/SW arrays. Fresh GPU reload is required for an adopted positive; unchanged old targets reuse prior evidence. Report post-update and selected scores separately.
Only these two explicit jobs, max2GPU. No other folds, dormant experiments, feature/LR/loss sweeps or delivery replacement. Per-task timing and actual concurrent wall time separate. Update PROJECTS/README/EXPLORATIONS/atlas after completion. One reused development fold and one seed do not establish independent generalization or a scaling law.
'''


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if len(sys.argv)!=1: raise RuntimeError('This entry point launches exactly ffn2 and ffn4')
    if active_jobs(): raise RuntimeError('Active trainer found; do not duplicate or compete')
    for mode in MODES:
        if (ROOT/'outputs'/f'pure_dl_frozen_growth_v22_{mode}_fold5').exists():
            raise RuntimeError('Existing output; inspect instead of overwriting')
    parent=ROOT/'experiments/pure_dl/champion_scaling_v21.py'
    if hashlib.sha256(parent.read_bytes()).hexdigest()!='5e10a7ebd62a861da32542fb49cc25e88ca73cd4b0eed0baa05d0f1bcebe6004':
        raise RuntimeError('v21 expansion implementation changed; inspect before executing')
    for name in ('frozen_growth_v22.py','train_frozen_growth_v22.py'):
        ast.parse(Path(__file__).with_name(name).read_text(encoding='utf-8'))
    (STUDY/'METHOD.md').write_text(METHOD,encoding='utf-8')
    rules=(STUDY/'AGENTS.md').read_text(encoding='utf-8')
    marker='## v22 fixed original weights, new capacity only'
    append_once(ROOT.parent/'AGENTS.md',marker,marker+'\nCurrent continued scaling study: two explicit POR candidates ffn2/ffn4, same actual v21 growth architecture and initialization; only .extra_ parameters train. All original weights frozen, original input/Q50/loss/LRs/80-18/max2 unchanged. No extra folds, old-task restart or delivery replacement. Maintain frozen_growth_v22_20260928/PROJECTS.md.')
    records=[]
    for mode in MODES:
        out=ROOT/'outputs'/f'pure_dl_frozen_growth_v22_{mode}_fold5';out.mkdir()
        (out/'AGENTS.md').write_text(rules,encoding='utf-8')
        (out/'README.md').write_text(METHOD+'\nMode: '+mode+'; training gain not established at launch.\n',encoding='utf-8')
        (out/'PROJECTS.md').write_text('# Progress\nGoal: new-capacity-only POR training.\nDone: plan, source and initial-function contract.\nPlanned: train, restore, compare against v21 and original champion.\nNot done: scores and final verification.\nAfter: keep only verified beneficial recipe; no automatic additional jobs.\n',encoding='utf-8')
        command=[sys.executable,'-X','utf8','-u',str(Path(__file__).with_name('train_frozen_growth_v22.py')),'--mode',mode,'--out',str(out)]
        with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
            proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,close_fds=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        record=dict(mode=mode,pid=proc.pid,started_at=datetime.now().astimezone().isoformat(),command=command)
        (out/'launch.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8');records.append(record)
        append_once(ROOT/'EXPLORATIONS.md',out.name,f'- [v22 {mode}, new parameters only](outputs/{out.name}/README.md): fixed original POR weights, same capacity as v21; launch is not evidence of gain.')
        print('LAUNCHED',json.dumps(record,ensure_ascii=False),flush=True)
    text='## '+datetime.now().astimezone().isoformat()+' v22 explicit launch\n'+json.dumps(records,ensure_ascii=False)+'\nOnly the two predefined jobs; no queue, other folds or prior-source modifications.'
    for p in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md'):
        with p.open('a',encoding='utf-8') as stream: stream.write('\n\n'+text+'\n')


if __name__=='__main__':main()
