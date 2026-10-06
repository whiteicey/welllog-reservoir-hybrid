"""Read-only trained-model diagnosis. No optimizer, training, or model writes."""
from pathlib import Path
from datetime import datetime
import gc, json, sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pure_dl'))
from champion_coadapt_v30 import load_data, Network as CoModel
from champion_split_route_v29 import Network as SplitModel
from multiscale_sequence import window_indices
from launch_champion_posterior_v27 import jobs
import train_invariant as base


def stats(a,bound):
    a=np.concatenate(a).astype(np.float64)
    ratio=np.abs(a)/bound
    return dict(n=len(a),mean=float(a.mean()),std=float(a.std()),positive_fraction=float((a>0).mean()),
        abs_bound_ratio_quantiles=np.quantile(ratio,[.1,.5,.9,.99]).tolist(),
        fraction_over_90pct=float((ratio>.90).mean()),fraction_over_95pct=float((ratio>.95).mean()),
        mean_tanh_local_derivative=float(np.maximum(0,1-ratio**2).mean()))


@torch.no_grad()
def main():
    active=jobs()
    if active:raise RuntimeError('Read-only diagnosis postponed because GPU work is active')
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    prior=ROOT/'results/deep_learning/champion_coadapt_v30_20260929'
    recipe=json.loads((prior/'CURRENT_RECIPE.json').read_text(encoding='utf-8'))
    data,master=load_data()
    indices=data['validation'];truth=data['truth'][indices];pred=np.load(prior/'assembled_pred.npy')
    scores=base.metric(truth,pred)
    assert all(abs(scores[k]-v)<1e-9 for k,v in recipe['scores'].items())
    targets=('POR','PERM','SW');weights=(.3,.35,.35)
    dm=base.dmask(truth);pd=base.dmask(pred)
    errors=np.column_stack((np.abs(pred[:,0]-truth[:,0])/(.08*(np.abs(truth[:,0])+1e-5)),
        np.abs(np.log10(np.maximum(pred[:,1]/np.maximum(truth[:,1],1e-5),1e-5))),
        np.abs(pred[:,2]-truth[:,2])/(.05*(np.abs(truth[:,2])+1e-5))))
    point_score=np.maximum(0,1-errors)
    losses={}
    for c,t in enumerate(targets):
        entry={}
        masks={'default':dm[:,c], 'numeric':~dm[:,c],
            'numeric_predicted_default':~dm[:,c]&pd[:,c],
            'numeric_predicted_numeric':~dm[:,c]&~pd[:,c]}
        for name,mask in masks.items():
            entry[name]=dict(n=int(mask.sum()),score=float(100*point_score[mask,c].mean()),
                zero_score_fraction=float((errors[mask,c]>=1).mean()),
                total_lost_points=float(weights[c]*100*(1-point_score[mask,c]).sum()/len(truth)))
        numeric=~dm[:,c]
        entry['numeric_default_prediction_fraction']=float(pd[numeric,c].mean())
        entry['numeric_error_quantiles']=np.quantile(errors[numeric,c],[.1,.5,.9]).tolist()
        losses[t]=entry
    tr=data['train_raw'];q=data['quality'][tr].cpu().numpy();default=data['default'][tr].cpu().numpy()
    quality=dict(source=data['qinfo'],n=len(q),mean=float(q.mean()),downweighted_n=int((q<1).sum()),
        downweighted_fraction=float((q<1).mean()),by_target={})
    for c,t in enumerate(targets):
        n=~default[:,c]
        quality['by_target'][t]=dict(numeric_fraction=float(n.mean()),weighted_numeric_fraction=float(q[n].sum()/q.sum()),
            default_Q_mean=float(q[~n].mean()),numeric_Q_mean=float(q[n].mean()))
    depth=data['depth'].cpu().numpy();lower=data['lower'].cpu().numpy()
    step=np.diff(depth)[lower[1:]==lower[:-1]]
    spacing=dict(min=float(step.min()),median=float(np.median(step)),max=float(step.max()),
        gaps_above_3median=int((step>3*np.median(step)).sum()),nonpositive=int((step<=0).sum()))
    wells=np.unique(data['wells'][data['fold']!=4]);assert len(wells)==64
    anchors=[]
    starts=np.asarray(data['starts'])
    for w in wells:
        lo,hi=int(starts[w]),int(starts[w+1])
        anchors.extend(lo+int((hi-lo-1)*p) for p in (.2,.4,.6,.8))
    core=torch.zeros((1,256),dtype=torch.bool,device='cuda');core[:,64:192]=True
    saturation={}
    for choice in recipe['choices']:
        t=choice['target'];mode=choice['mode']
        cls=CoModel if choice['model_module']=='champion_coadapt_v30' else SplitModel
        model=cls(t,master,modes=(mode,)).cuda()
        cp=torch.load(ROOT/choice['checkpoint'],map_location='cpu',weights_only=False)
        model.heads[mode].load_state_dict(cp['delta_state'],strict=True)
        model.requires_grad_(False);model.eval()
        acc={k:[] for k in ['route28','route29_branch0','route29_branch1','route29_atom','route29_density']}
        for begin in range(0,len(anchors),32):
            anc=torch.as_tensor(anchors[begin:begin+32],device='cuda')
            ix,valid=window_indices(anc,data['lower'],data['upper'])
            rel=torch.tanh((data['depth'][ix]-data['depth'][anc,None])/10)
            h,o,context,_=model.extract(data['x'][ix],rel,valid,data['absolute'][ix])
            side=data['x'][ix,...,206:];take=valid&core&data['labeled'][ix]
            assert bool((data['raw_to_valid'][ix[take].cpu().numpy()]>=0).all())
            if cls is CoModel:
                route28=model.heads[mode].route28;route29=model.heads[mode].route29
            else:
                route28=model.retained_route;route29=model.heads[mode]
            values={'route28':route28.correction(h,side,valid)}
            x=torch.cat((h[0],h[1],side.transpose(1,2).to(h[0].dtype)),1).masked_fill(~valid[:,None],0)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                for j,branch in enumerate(route29.branches):values['route29_branch'+str(j)]=branch(x,valid)
            da,dc=route29.corrections(h,side,valid)
            values.update(route29_atom=da,route29_density=dc)
            for name,v in values.items():acc[name].append(v[take].float().cpu().numpy())
        saturation[t]={k:stats(v,.25 if k=='route28' else .125) for k,v in acc.items()}
        print('SATURATION',t,json.dumps(saturation[t]),flush=True)
        model.close();del model,cp;gc.collect();torch.cuda.empty_cache()
    result=dict(observed_at=datetime.now().astimezone().isoformat(),recipe=str((prior/'CURRENT_RECIPE.json').relative_to(ROOT)),
        scores=scores,validation_error_diagnosis=losses,training_quality=quality,depth_spacing=spacing,training_routing_saturation=saturation,
        saturation_sampling='64 training wells x 4 fixed raw-depth quantiles .2/.4/.6/.8; valid labeled central128 points; no validation inputs',
        raw_training_windows=len(anchors),new_training=False,new_model_score=False,weights_updated=False,
        caveats=['Routing saturation is observed on predetermined training windows only; it does not establish relaxed bounds improve generalization.',
                 'Lost-point contributions are descriptive headroom against perfect predictions, not achievable gains or additive treatment effects.',
                 'Existing development predictions are reused for errors; no additional validation-based model selection.'])
    (STUDY/'diagnostic.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DIAG',json.dumps(dict(scores=scores,losses=losses,quality=quality,spacing=spacing)),flush=True)


if __name__=='__main__':main()
