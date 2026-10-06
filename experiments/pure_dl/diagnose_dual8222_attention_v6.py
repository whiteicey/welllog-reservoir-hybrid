"""CPU-only attention masking and training-state diagnostics after completed runs.
No training, parameter search, prediction changes, or checkpoint mutation.
"""
from __future__ import annotations
import json,sys
from datetime import datetime
from pathlib import Path
import torch
from dual8222_attention_head_v6 import TokenMixer

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/dual8222_attention_v6_20260928'
PREFIX='pure_dl_dual8222_v6_'


def main():
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261010)
        mixer=TokenMixer(True).cpu().eval()
        x=torch.randn(2,40,64)
        valid=torch.ones(2,40,dtype=torch.bool);valid[0,:6]=False;valid[1,32:]=False
        with torch.no_grad():
            normal=mixer(x,valid)
            changed=x.clone();changed[~valid]=torch.randn_like(changed[~valid])*1000
            altered=mixer(changed,valid)
            blank=mixer(x,torch.zeros_like(valid))
            full_valid=torch.ones_like(valid);full=mixer(x,full_valid)
            far=x.clone();far[:,30:]+=100
            far_output=mixer(far,full_valid)
        assert torch.isfinite(normal).all() and torch.isfinite(blank).all()
        assert torch.count_nonzero(normal[~valid])==0 and torch.count_nonzero(blank)==0
        assert torch.equal(normal[valid],altered[valid]),'Invalid keys affected valid predictions'
        assert torch.equal(full[:,:4],far_output[:,:4]),'One-layer attention read beyond radius16'
    mask_test={'finite_with_all_keys_padded':True,'invalid_query_outputs_zero':True,
               'valid_outputs_invariant_to_padding_values':True,'one_layer_radius16_enforced':True,
               'scope':'synthetic CPU TokenMixer checks, not an accuracy benchmark'}
    rows=[]
    for mode in ['linear','deep_mlp','attention']:
        folder=ROOT/'outputs'/(PREFIX+mode+'80_fold5')
        path=folder/'fold_5_report.json'
        if not path.exists():continue
        report=json.loads(path.read_text(encoding='utf-8'))
        history=json.loads((folder/'fold_5_validation_history.json').read_text(encoding='utf-8'))
        last=torch.load(folder/'last_state.pt',map_location='cpu',weights_only=False)
        best=torch.load(folder/'heads_best.pt',map_location='cpu',weights_only=False)
        a=last['current_heads'];b=best['state_dict']
        delta_by_group={}
        for group,match in [('qkv','mixer.qkv'),('relative_bias','relative_bias'),('new_output','rich_location.output.'),('linear_mean','mean_weight')]:
            terms=[(a[k].double()-b[k].double()).square().sum() for k in a if match in k]
            delta_by_group[group]=float(torch.stack(terms).sum().sqrt()) if terms else None
        relative=[v.float().norm() for k,v in a.items() if 'relative_bias' in k]
        output=[v.float().norm() for k,v in a.items() if 'rich_location.output.weight' in k]
        rows.append({'mode':mode,'final_scores':report['scores'],'selected_epoch':report['best_epoch'],
            'stopped_epoch':report['stopped_epoch'],
            'best_trained_nonzero_epoch':max(history,key=lambda x:x['official_total'])['epoch'],
            'best_trained_nonzero_epoch_TOTAL':max(x['official_total'] for x in history),
            'last_epoch_TOTAL':history[-1]['official_total'],
            'first_epoch_training_NLL':history[0]['por_continuous_nll'],
            'last_epoch_training_NLL':history[-1]['por_continuous_nll'],
            'last_vs_selected_parameter_L2':delta_by_group,
            'last_relative_bias_L2':[float(x) for x in relative],
            'last_new_output_weight_L2':[float(x) for x in output],
            'relative_bias_started_at_zero':bool(relative),
            'source_predictions_restored_if_epoch_zero':report['best_epoch']==0,
            'interpretation':'Parameter movement does not prove useful generalization. Nonzero relative bias from zero initialization establishes actual attention-gradient updates; training losses are different stochastic epoch samples.'})
    result={'observed_at':datetime.now().astimezone().isoformat(),'mask_checks':mask_test,'runs':rows,
            'scope':'CPU inspection of completed runs only; no new model selection',
            'new_training':False}
    (STUDY/'attention_implementation_diagnostic.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
