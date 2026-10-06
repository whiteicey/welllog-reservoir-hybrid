"""Create a new, provenance-recorded v29 trainer; never alter v28 sources."""
from pathlib import Path
import ast,hashlib,json

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/champion_split_route_v29_20260929'
E=ROOT/'experiments/pure_dl'


def main():
    old=E/'train_champion_route_v28.py';destination=E/'train_champion_split_route_v29.py'
    if destination.exists():raise FileExistsError(destination)
    src=old.read_text(encoding='utf-8-sig')
    def sub(a,b):
        nonlocal src
        if src.count(a)!=1:raise ValueError('Expected one match: '+a[:100])
        src=src.replace(a,b,1)
    sub('Train two prespecified internal-routing variants for one private target.',
        'Train equal-parameter coupled/split route increments from retained v28.')
    sub('from champion_route_v28 import Network,Evaluator,MODES,ROOT,STUDY,PREVIOUS,load_data,reference_prediction,TARGETS,routing_scope',
        'from champion_split_route_v29 import Network,Evaluator,MODES,ROOT,STUDY,PREVIOUS,load_data,reference_prediction,TARGETS,routing_scope,gradient_isolation')
    sub("initial['joint'][k],v) for k,v in initial['conditional'].items()", "initial['coupled'][k],v) for k,v in initial['split'].items()")
    sub("('champion_route_v28.py','champion_posterior_v27.py'", "('champion_split_route_v29.py','champion_route_v28.py','champion_posterior_v27.py'")
    sub("cfg=dict(target=target,factor='expert_routing'", "cfg=dict(target=target,factor='atom_conditional_routing'")
    sub("model_module='champion_route_v28',source_checkpoint=model.source_path,retained_head=model.retained_path,",
        "model_module='champion_split_route_v29',source_checkpoint=model.source_path,retained_head=model.retained_path,\n        retained_v28_route=model.route_path,retained_v28_mode=model.parent_mode,")
    sub("head_hidden=32,head_output=1)","head_hidden=32,head_output=1,branches=2)")
    sub("matched_change='joint routes atom and density; conditional keeps merged atom fixed, routes density only'",
        "matched_change='coupled averages both active branches for both routes; split assigns branch0 to atom and branch1 to conditional density'")
    sub('new_head_seed=20261104','new_head_seed=20261105')
    sub('log_odds_correction_bound=.25','log_odds_correction_bound=.125')
    sub('initial v26 function mismatch','initial v28 function mismatch')
    sub("retained_head=model.retained_path,checkpoint_format='v26_frozen_predictor_plus_v28_route_delta'",
        "retained_head=model.retained_path,retained_v28_route=model.route_path,retained_v28_mode=model.parent_mode,checkpoint_format='frozen_v28_plus_v29_route_delta'")
    sub("gradient_scope=s['proof'],factor_scope=factor_scope(model,data,m),last_delta_parameter_change_L2=changes[m],",
        "gradient_scope=s['proof'],factor_scope=factor_scope(model,data,m),gradient_isolation=gradient_isolation(model,data,m),last_delta_parameter_change_L2=changes[m],")
    ast.parse(src)
    destination.write_text(src,encoding='utf-8')
    provenance=dict(template=str(old.relative_to(ROOT)),template_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
        new_trainer=str(destination.relative_to(ROOT)),new_trainer_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        historical_source_modified=False,changes=['new module and modes','retain v28 route metadata','bound .125','branch seed20261105/6','gradient isolation checks'])
    (STUDY/'trainer_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(provenance,ensure_ascii=False))


if __name__=='__main__':main()
