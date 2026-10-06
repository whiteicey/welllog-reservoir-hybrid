"""Create isolated corrected entrypoints; historical and failed files are untouched."""
from pathlib import Path
import hashlib, json

ROOT=Path(__file__).resolve().parents[3]
CODE=ROOT/'experiments/pure_dl'
STUDY=Path(__file__).resolve().parent
model=(CODE/'complete_seed_models_v362.py').read_text(encoding='utf-8')
model=model.replace("('train_complete_seed_stage_v362.py','train_complete_seed_source_v362.py')", "('train_complete_seed_stage_v362.py','train_complete_seed_stage_v362r1.py','train_complete_seed_source_v362.py')")
worker=(CODE/'train_complete_seed_stage_v362.py').read_text(encoding='utf-8').replace('complete_seed_models_v362 import','complete_seed_models_v362r1 import')
worker=worker.replace("stage=a.stage;out=OUT/stage;", "stage=a.stage;out=OUT/('SW351_fixed' if stage=='SW351' else stage);")
old="loss_fn=lambda output,ix,mask:criterion(output,data['labels'][ix],data['default'][ix],mask,data['quality'][ix])[0]"
new="loss_fn=lambda output,ix,mask:criterion(output,data['labels'][ix],data['default'][ix],mask,data['quality'][ix],*([model] if stage=='SW351' else []))[0]"
assert worker.count(old)==1
worker=worker.replace(old,new)
controller=(CODE/'run_complete_seeds_v362.py').read_text(encoding='utf-8')
controller=controller.replace("data=torch.load(own/'SOURCE/prepared_data.pt'", "def phase(s):return own/('SW351_fixed' if s=='SW351' else s)\n data=torch.load(own/'SOURCE/prepared_data.pt'")
controller=controller.replace("np.load(own/s/'channel_pred.npy')", "np.load(phase(s)/'channel_pred.npy')")
controller=controller.replace("(own/s/('fold_5_report.json'", "(phase(s)/('fold_5_report.json'")
controller=controller.replace("sha(own/s/('fold_5.pt'", "sha(phase(s)/('fold_5.pt'")
controller=controller.replace("if (STUDY/'BATCH_STATE.json').exists():raise RuntimeError('Existing batch requires explicit scoped resumption; no auto restart')", "prior=json.loads((STUDY/'BATCH_STATE.json').read_text());assert prior=={'status':'failed_or_stopped','seed':20261010,'stage':'SW351','exit_code':1,'explicit_resume_only':True,'no_retry':True}\n assert not gpu_jobs()\n assert not (STUDY/'INTERFACE_FIX_LAUNCHED.json').exists()\n write(STUDY/'INTERFACE_FIX_LAUNCHED.json',dict(controller_pid=os.getpid(),reason='manual isolated missing-model interface correction, failed attempt zero updates',previous_batch=prior))")
controller=controller.replace("for stage in STAGES:\n", "for stage in (('SW351',) if seed==20261010 else STAGES):\n")
controller=controller.replace("dest=own/stage;", "dest=own/('SW351_fixed' if stage=='SW351' else stage);")
controller=controller.replace("experiments/pure_dl/train_complete_seed_stage_v362.py", "experiments/pure_dl/train_complete_seed_stage_v362r1.py")
controller=controller.replace("total_wall_seconds=time.time()-started", "total_wall_seconds=time.time()-1790913400.4120793,corrected_controller_wall_seconds=time.time()-started")
files={'complete_seed_models_v362r1.py':model,'train_complete_seed_stage_v362r1.py':worker,'run_complete_seeds_v362r1.py':controller}
for name,text in files.items():
    dest=CODE/name
    assert not dest.exists()
    compile(text,str(dest),'exec')
    dest.write_text(text,encoding='utf-8')
fix=dict(failed_stage='seed20261010/SW351',failed_pid=28792,failed_epoch=0,
         failed_reason='DecisionObjective.forward() missing required model argument',
         fixed_scope='SW351 loss call receives model exactly as historical trainer; other losses/architecture/budget unchanged',
         failed_files_preserved=True,previous_seven_stages_preserved=True,
         replacement_stage_directory='SW351_fixed',additional_training_phase=False,
         continuation='Manually corrected entrypoint after verified terminal failure; no automatic retry or user-stop resumption. Finite remaining 1+4x8 stages only.',
         source_sha256={n:hashlib.sha256((CODE/n).read_bytes()).hexdigest() for n in files})
(STUDY/'INTERFACE_FIX_PREPARED.json').write_text(json.dumps(fix,indent=2),encoding='utf-8')
note='\nV362 isolated manual interface correction: SW351 failed at epoch0 before first optimizer update; missing model argument now matches original351 trainer. Failed source/log/config/laststate and all seven completed stages preserved. Corrected r1 worker/model binding/controller use SW351_fixed directory. This is replacement of untrained failed phase, not ninth learned stage; no extra seed or old-stage rerun. Finite remaining1+4x8 dependency launches only, no automatic retries/stop resumption. Full elapsed time includes failed attempt and repair gap.\n'
for p in (STUDY/'AGENTS.md',STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
    with p.open('a',encoding='utf-8') as h:h.write(note)
print(json.dumps(fix,indent=2))
