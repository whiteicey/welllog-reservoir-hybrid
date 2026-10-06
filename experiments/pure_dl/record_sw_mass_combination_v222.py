"""Actual fixed-head assembly; optional recipes do not replace the five-phase default."""
import json,hashlib
import numpy as np
import torch
from sw_mass_width128_v222 import ROOT,STUDY,OUT,PARENT,PARENT_SHA
from train_dual8222_heads80 import atomic_json,atomic_array
import train_invariant as base

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def main():
    report=read(OUT/'mass/report.json')
    assert report['predictions_reproduced_exactly'] and report['frozen_parameters_exact']
    data=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
    control=read(ROOT/'results/deep_learning/perm_ranked_probability_v154_20261001/CURRENT_RECIPE.json')
    assembled=read(STUDY/'mass_assembled.json')
    recipe=dict(control)
    recipe.update(scores=assembled['scores'],total_training_stages=6,dependencies=control['dependencies']+['v222 SW likelihood-only mass patch'],
                  optional_not_default=True,total_stage_upper_bound=8,added_head_stages=dict(POR=0,PERM=0,SW=1),
                  previous_recipe_path='results/deep_learning/perm_ranked_probability_v154_20261001/CURRENT_RECIPE.json',
                  gain_vs_previous_TOTAL=assembled['scores']['TOTAL']-control['scores']['TOTAL'])
    recipe['SW']=dict(checkpoint=str((OUT/'mass/model_best.pt').relative_to(ROOT)),sha256=assembled['checkpoint_sha256'],
                      module='sw_mass_width128_v222',stage='mass',parent_checkpoint=str(PARENT.relative_to(ROOT)),parent_sha256=PARENT_SHA)
    atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    por=np.load(ROOT/control['POR']['checkpoint'].replace('model_best.pt','channel_pred.npy'))
    permout=ROOT/'outputs/pure_dl_perm_cross_context_slow_v207_fold5/attention'
    perm=np.load(permout/'channel_pred.npy');sw=np.load(OUT/'mass/channel_pred.npy')
    pred=np.column_stack((por,perm,sw));assert pred.shape==(144665,3) and np.isfinite(pred).all()
    scores=base.metric(data['truth'][data['validation']],pred)
    assert abs(scores['SW']-report['target_score'])<1e-9
    assert abs(scores['PERM']-read(permout/'report.json')['target_score'])<1e-9
    combined=dict(recipe)
    combined.update(scores=scores,total_training_stages=7,added_head_stages=dict(POR=0,PERM=1,SW=1),
                    dependencies=recipe['dependencies']+['v207 PERM context patch'],
                    gain_vs_previous_TOTAL=scores['TOTAL']-control['scores']['TOTAL'],
                    validation_rows=144665,actual_three_target_combination=True)
    combined['PERM']=read(ROOT/'results/deep_learning/perm_cross_context_slow_v207_20261001/CURRENT_RECIPE.json')['PERM']
    atomic_json(STUDY/'OPTIONAL_SEVEN_STAGE_RECIPE.json',combined)
    atomic_array(STUDY/'optional_seven_stage_pred.npy',pred)
    atomic_json(STUDY/'control_delta.json',dict(original104_SW=74.12175282815758,final_SW=report['target_score'],
        learned_gain_vs_initial=report['against_initial'],gain_vs104_SW=report['target_score']-74.12175282815758,
        initial_floor_normalization_delta_SW=report['initial_score']-74.12175282815758,
        six_stage_TOTAL=assembled['scores']['TOTAL'],seven_stage_TOTAL=scores['TOTAL'],default_five_stage_unchanged=True,
        strict_808_passed=False,complete_five_folds=False,remaining_folds_started=False))
    note=f"\nV211 actual optional6 TOTAL{assembled['scores']['TOTAL']:.12f}; actual optional7 (PERM207+SW222) TOTAL{scores['TOTAL']:.12f}, at mostone extra perhead. Both fixed144665assembly, no rowwise model choice. Default5 kept because gains small; below80.8, no remainingfolds/fullOOF claim.\n"
    for doc in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
        with doc.open('a',encoding='utf-8') as f:f.write(note)
    print(json.dumps(dict(optional6=assembled['scores'],optional7=scores)))

if __name__=='__main__':main()

