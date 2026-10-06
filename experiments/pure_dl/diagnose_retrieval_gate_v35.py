"""Inspect the completed v35 gate using its saved last state, without updates."""
from pathlib import Path
from datetime import datetime
import json,sys,psutil
import numpy as np
import torch
from retrieval_density_v35 import Network,load_data,channel_objective,MODES,STUDY,ROOT
from multiscale_sequence import window_indices
from train_dual8222_heads80 import atomic_json


def main():
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(3)
    active=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            z=p.info;a=z.get('cmdline') or []
            if 'python' in (z.get('name') or '').lower() and '-c' not in a and any('pure_dl' in s and Path(s.replace('\\','/')).name.startswith(('train_','benchmark_')) for s in a):active.append(z)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    assert not active,active
    out=ROOT/'outputs/pure_dl_retrieval_density_v35_SW_fold5'
    saved=torch.load(out/'last_state.pt',map_location='cpu',weights_only=False)
    data,master=load_data();model=Network('SW',master).cuda()
    anchors=data['train_raw'][torch.linspace(0,len(data['train_raw'])-1,64,device='cuda').long()]
    ix,valid=window_indices(anchors,data['lower'],data['upper'])
    relative=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
    core=torch.zeros_like(valid);core[:,64:192]=True;mask=valid&core&data['labeled'][ix]
    h,parent,context,_=model.extract(data['x'][ix],relative,valid,data['absolute'][ix],rows=ix)
    rows=[]
    for state in ('initial','last'):
        if state=='last':model.heads.load_state_dict(saved['heads'],strict=True)
        for m in MODES:
            head=model.heads[m];head.zero_grad(set_to_none=True);captured={}
            handle=head.output.register_forward_hook(lambda _m,_a,o:captured.update(params=o.detach().float()))
            output=head(h,parent,context,data['x'][ix,...,206:],valid)
            loss,_,_=channel_objective(output,data['labels'][ix],data['default'][ix],mask,data['quality'][ix],2)
            loss.backward();handle.remove()
            proposal=captured['params'][...,0][mask];alpha=output['alpha'][mask].detach()
            gradients={n:float(p.grad.detach().float().norm()) if p.grad is not None else None for n,p in head.named_parameters()}
            rows.append(dict(state=state,mode=m,loss=float(loss.detach()),probe_points=int(mask.sum()),
                negative_proposal_fraction=float((proposal<0).float().mean()),proposal_min=float(proposal.min()),proposal_max=float(proposal.max()),
                alpha_mean=float(alpha.mean()),alpha_max=float(alpha.max()),active_fraction=float((alpha>0).float().mean()),gradients=gradients))
    history=[]
    for m in MODES:
        s=saved['states'][m]
        history.append(dict(mode=m,epochs=len(s['history']),unique_validation_scores=sorted(set(x['target_score'] for x in s['history'])),
            best_updated_equal_reference=bool(np.array_equal(np.load(out/m/'best_updated_pred.npy'),s['best_pred']))))
    result=dict(observed_at=datetime.now().astimezone().isoformat(),saved_epoch=saved['epoch'],active_training=active,
        scope='64 fixed evenly spaced training anchors; last saved heads, no optimizer update',probes=rows,history=history,
        interpretation='If last proposals are negative and gradients zero, hard clipping has an inactive region; parameter changes alone can include AdamW decay and do not prove useful learning.')
    atomic_json(STUDY/'gate_diagnostic.json',result);model.close()
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
