"""Create an isolated, hash-recorded v30 trainer from the immutable v28 loop."""
from pathlib import Path
import ast,hashlib,json
ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/champion_coadapt_v30_20260929'
E=ROOT/'experiments/pure_dl'


def main():
    parent=E/'train_champion_route_v28.py';out=E/'train_coadapt_v30.py'
    if out.exists():raise FileExistsError(out)
    s=parent.read_text(encoding='utf-8-sig')
    def sub(a,b):
        nonlocal s
        if s.count(a)!=1:raise ValueError('Expected one exact match: '+a[:100])
        s=s.replace(a,b,1)
    sub('Train two prespecified internal-routing variants for one private target.',
        'Warm-start existing routes versus routes plus retained factor corrections.')
    sub('from champion_route_v28 import Network,Evaluator,MODES,ROOT,STUDY,PREVIOUS,load_data,reference_prediction,TARGETS,routing_scope',
        'from champion_coadapt_v30 import Network,Evaluator,MODES,ROOT,STUDY,PREVIOUS,load_data,reference_prediction,TARGETS,factor_scope')
    sub('factor_scope = routing_scope','')
    sub("initial['joint'][k],v) for k,v in initial['conditional'].items()", "initial['route_only'][k],v) for k,v in initial['coadapt'].items()")
    sub("('champion_route_v28.py','champion_posterior_v27.py'", "('champion_coadapt_v30.py','champion_split_route_v29.py','champion_route_v28.py','champion_posterior_v27.py'")
    start=s.index('    cfg=dict(');end=s.index('    if args.resume:assert json.loads',start)
    cfg="""    cfg=dict(target=target,factor=model.factor,modes=list(MODES),screening_fold=5,
        model_module='champion_coadapt_v30',source_checkpoint=model.source_path,retained_head=model.retained_path,
        parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),retained_v29=model.choice,
        per_candidate_trainable_parameters={m:sum(p.numel() for p in model.heads[m].parameters() if p.requires_grad) for m in MODES},
        per_candidate_bundle_parameters={m:sum(p.numel() for p in model.heads[m].parameters()) for m in MODES},
        architecture_growth=False,inference_extra_parameters=0,shared_trainable_parameters=False,
        backbone_shared_between_targets=False,head_optimizer_groups={m:[dict(name=g['name'],parameters=sum(p.numel() for p in g['params'])) for g in model.trainable_groups(m)] for m in MODES},
        inputs=dict(backbone=206,side=63,expert_hidden=128),
        matched_change='same architecture and initial weights; route_only updates v28/v29, coadapt also updates retained v25/v26 factor',
        old_main_readouts_and_backbone_frozen=True,source_eval_only=True,
        epochs=80,patience=18,minimum_epochs=10,steps_per_epoch=64,batch_size=64,seed=20261010,
        learning_rate=3e-5,eta_min=1e-6,weight_decay=.01,clip_norm_per_parameter_group=2.,
        optimizer_reset=True,new_parameter_initialization=False,bounds='unchanged retained .25 and .125 paths',
        quality=data['qinfo'],reference_scores=ref,validation_rows=144665,training_rows=578891,
        objective='original own-target complete predictive likelihood/3; fixedQ50',
        decoder='original singlecoverage/256bins/800actions+default/batch32',
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        torch_version=str(torch.__version__),deterministic_mode='warn_only; not a bitwise retraining claim',
        complete_five_folds=False,remaining_folds_started=False)
"""
    s=s[:start]+cfg+s[end:]
    sub("opts={m:torch.optim.AdamW(model.heads[m].parameters(),lr=3e-4,weight_decay=.01) for m in MODES}",
        "opts={m:torch.optim.AdamW(model.trainable_groups(m),lr=3e-5,weight_decay=.01) for m in MODES}")
    sub('initial v26 function mismatch','initial v29 function mismatch')
    sub("retained_head=model.retained_path,checkpoint_format='v26_frozen_predictor_plus_v28_route_delta'",
        "retained_head=model.retained_path,parent_v29=model.choice,checkpoint_format='v30_existing_readout_and_routes_bundle'")
    sub("frozen_gradients_none=all(p.grad is None for n,p in model.named_parameters() if not n.startswith('heads.'))",
        "frozen_gradients_none=all(p.grad is None for p in model.parameters() if not p.requires_grad)")
    sub("assert proof['output_gradient_norm']>0 and proof['frozen_gradients_none']",
        "proof['parameter_group_gradient_norms']={g['name']:float(torch.stack([p.grad.detach().float().square().sum() for p in g['params'] if p.grad is not None]).sum().sqrt()) for g in opts[m].param_groups}\n                    assert all(v>0 for v in proof['parameter_group_gradient_norms'].values()) and proof['frozen_gradients_none']")
    sub("torch.nn.utils.clip_grad_norm_(model.heads[m].parameters(),2.,error_if_nonfinite=True)",
        "for group in opts[m].param_groups:\n                    torch.nn.utils.clip_grad_norm_(group['params'],2.,error_if_nonfinite=True)")
    ast.parse(s)
    out.write_text(s,encoding='utf-8')
    p=dict(template=str(parent.relative_to(ROOT)),template_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
        new_trainer=str(out.relative_to(ROOT)),new_trainer_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        historical_source_modified=False,changes=['retained v29 initialization','no new capacity','six fixed update-range trials','LR3e-5','separate gradient clipping per route/factor group'])
    (STUDY/'trainer_provenance.json').write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(p,ensure_ascii=False))


if __name__=='__main__':main()
